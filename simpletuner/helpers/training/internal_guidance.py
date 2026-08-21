from __future__ import annotations

import contextlib
import inspect
import itertools
import math
from collections.abc import Sequence
from typing import Optional

import torch
from torch import nn


def _config_value(config, name: str):
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _as_patch_shape(value, spatial_dims: int) -> tuple[int, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        patch_shape = tuple(int(part) for part in value)
    elif value is None:
        patch_shape = ()
    else:
        patch_shape = (int(value),)

    if len(patch_shape) == spatial_dims:
        return patch_shape
    if len(patch_shape) == 1:
        return patch_shape * spatial_dims
    return (1,) * spatial_dims


def infer_internal_guidance_output_features(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Internal Guidance requires a transformer config with channel and patch dimensions.")

    in_channels = _config_value(config, "in_channels")
    out_channels = _config_value(config, "out_channels")
    patch_latent_dim = _config_value(config, "patch_latent_dim")
    if patch_latent_dim is not None:
        return int(patch_latent_dim)
    candidates = [int(value) for value in (in_channels, out_channels) if value is not None and int(value) > 0]
    if not candidates:
        raise ValueError("Internal Guidance could not infer the transformer's latent channel count.")
    channels = min(candidates)

    patch_size = _config_value(config, "patch_size")
    patch_size_t = _config_value(config, "patch_size_t")
    if isinstance(patch_size, Sequence) and not isinstance(patch_size, (str, bytes)):
        patch_shape = tuple(int(part) for part in patch_size)
    elif patch_size is None:
        patch_shape = ()
    else:
        patch_shape = (int(patch_size), int(patch_size))
    if patch_size_t is not None and len(patch_shape) == 2:
        patch_shape = (int(patch_size_t), *patch_shape)

    patch_volume = math.prod(patch_shape) if patch_shape else 1
    return channels * patch_volume


def infer_internal_guidance_block_count(model: nn.Module) -> int:
    candidates = (
        ("transformer_blocks", "single_transformer_blocks"),
        ("joint_transformer_blocks", "single_transformer_blocks"),
        ("double_stream_layers", "single_stream_layers"),
        ("visual_transformer_blocks",),
        ("transformer_blocks",),
        ("layers",),
        ("blocks",),
    )
    roots = [model]
    core = getattr(model, "core", None)
    if isinstance(core, nn.Module):
        roots.append(core)

    for root in roots:
        for names in candidates:
            modules = [getattr(root, name, None) for name in names]
            if modules and all(isinstance(module, (nn.ModuleList, nn.Sequential)) for module in modules):
                count = sum(len(module) for module in modules)
                if count:
                    return count
    raise ValueError("Internal Guidance could not determine the transformer's block count.")


class InternalGuidanceHead(nn.Module):
    def __init__(self, hidden_size: int, output_features: int, block_index: int = 0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.proj = nn.Linear(hidden_size, output_features)
        self.register_buffer("block_index", torch.tensor(int(block_index), dtype=torch.int64), persistent=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim < 3:
            raise ValueError(
                "Internal Guidance expects transformer hidden states with batch, token, and feature dimensions."
            )
        tokens = hidden_states.reshape(hidden_states.shape[0], -1, hidden_states.shape[-1])
        return self.proj(self.norm(tokens))

    @staticmethod
    def _patch_candidates(spatial_shape: tuple[int, ...], patch_volume: int):
        divisors = [[candidate for candidate in range(1, size + 1) if size % candidate == 0] for size in spatial_shape]
        return [candidate for candidate in itertools.product(*divisors) if math.prod(candidate) == patch_volume]

    @classmethod
    def infer_patch_shape(
        cls,
        target: torch.Tensor,
        *,
        token_count: int,
        output_features: int,
        preferred_patch_size=None,
        preferred_patch_size_t=None,
    ) -> tuple[int, ...]:
        if target.ndim < 3:
            raise ValueError("Internal Guidance requires channel-first latent targets with at least one spatial axis.")
        channels = int(target.shape[1])
        if output_features % channels:
            raise ValueError(
                f"Internal Guidance output width {output_features} is not divisible by target channels {channels}."
            )
        patch_volume = output_features // channels
        spatial_shape = tuple(int(size) for size in target.shape[2:])
        if math.prod(spatial_shape) != token_count * patch_volume:
            raise ValueError(
                "Internal Guidance hidden-state token count does not match the diffusion target: "
                f"tokens={token_count}, patch_volume={patch_volume}, spatial_shape={spatial_shape}."
            )

        candidates = cls._patch_candidates(spatial_shape, patch_volume)
        if not candidates:
            raise ValueError(
                f"Internal Guidance could not factor patch volume {patch_volume} across target shape {spatial_shape}."
            )

        preferred = _as_patch_shape(preferred_patch_size, len(spatial_shape))
        if preferred_patch_size_t is not None and len(spatial_shape) == 3:
            preferred = (int(preferred_patch_size_t), preferred[-2], preferred[-1])

        def score(candidate: tuple[int, ...]) -> tuple[float, float]:
            preferred_distance = sum(abs(math.log2(value / expected)) for value, expected in zip(candidate, preferred))
            isotropy = max(candidate) - min(candidate)
            return preferred_distance, isotropy

        return min(candidates, key=score)

    @classmethod
    def unpatchify(
        cls,
        prediction_tokens: torch.Tensor,
        target: torch.Tensor,
        *,
        preferred_patch_size=None,
        preferred_patch_size_t=None,
    ) -> torch.Tensor:
        batch_size, token_count, output_features = prediction_tokens.shape
        patch_shape = cls.infer_patch_shape(
            target,
            token_count=token_count,
            output_features=output_features,
            preferred_patch_size=preferred_patch_size,
            preferred_patch_size_t=preferred_patch_size_t,
        )
        spatial_shape = tuple(int(size) for size in target.shape[2:])
        grid_shape = tuple(size // patch for size, patch in zip(spatial_shape, patch_shape))
        channels = int(target.shape[1])

        prediction = prediction_tokens.reshape(batch_size, *grid_shape, *patch_shape, channels)
        spatial_dims = len(spatial_shape)
        permutation = [0, 1 + 2 * spatial_dims]
        for axis in range(spatial_dims):
            permutation.extend((1 + axis, 1 + spatial_dims + axis))
        return prediction.permute(permutation).reshape(batch_size, channels, *spatial_shape)


class InternalGuidanceRegularizer:
    MODULE_NAME = "internal_guidance_head"

    def __init__(self, config, accelerator, hidden_size: int, output_features: int, block_count: int):
        self.config = config
        self.device = accelerator.device
        self.enabled = bool(getattr(config, "internal_guidance_enabled", False))
        self.weight = float(getattr(config, "internal_guidance_loss_weight", 0.5) or 0.0)
        if self.weight <= 0:
            raise ValueError("internal_guidance_loss_weight must be greater than zero.")

        configured_block = getattr(config, "internal_guidance_block_index", None)
        self.block_index = int(configured_block) if configured_block is not None else max(0, block_count // 4)
        if not 0 <= self.block_index < block_count:
            raise ValueError(f"internal_guidance_block_index must be within [0, {block_count - 1}], got {self.block_index}.")

        self.head = InternalGuidanceHead(hidden_size, output_features, block_index=self.block_index)
        self.model: Optional[nn.Module] = None

    def attach_to_model(self, model: nn.Module, dtype: torch.dtype) -> None:
        if hasattr(model, self.MODULE_NAME):
            self.head = getattr(model, self.MODULE_NAME)
        else:
            setattr(model, self.MODULE_NAME, self.head)
        self.model = model
        self.head.to(device=self.device, dtype=dtype)

    def wants_hidden_states(self) -> bool:
        return self.enabled

    def _attached_head(self) -> nn.Module:
        if self.model is None:
            raise RuntimeError("Internal Guidance head is not attached to a transformer.")
        return getattr(self.model, self.MODULE_NAME)

    def predict(self, hidden_states: torch.Tensor, target: torch.Tensor, model_config) -> torch.Tensor:
        head = self._attached_head()
        prediction_tokens = head(hidden_states)
        return InternalGuidanceHead.unpatchify(
            prediction_tokens,
            target,
            preferred_patch_size=_config_value(model_config, "patch_size")
            or _config_value(model_config, "latent_patch_size"),
            preferred_patch_size_t=_config_value(model_config, "patch_size_t"),
        )

    def compute_loss(self, hidden_states_buffer, prepared_batch: dict, model_foundation) -> tuple[torch.Tensor, dict]:
        if hidden_states_buffer is None:
            raise ValueError("Internal Guidance is enabled but the model did not return a hidden-state buffer.")
        hidden_states = hidden_states_buffer.get(f"layer_{self.block_index}")
        if hidden_states is None:
            raise ValueError(
                f"Internal Guidance requested layer {self.block_index}, but that layer was not captured by "
                f"{model_foundation.NAME}."
            )

        target = model_foundation.get_prediction_target(prepared_batch)
        if target is None:
            raise ValueError("Internal Guidance requires the model's diffusion prediction target.")
        model_component = model_foundation.unwrap_model(model=model_foundation.get_trained_component(unwrap_model=False))
        model_config = getattr(model_component, "config", None)
        intermediate_prediction = self.predict(hidden_states, target, model_config)
        intermediate_loss = model_foundation.loss(
            prepared_batch,
            {"model_prediction": intermediate_prediction},
            apply_conditioning_mask=True,
        )
        weighted_loss = intermediate_loss * self.weight
        return weighted_loss, {
            "internal_guidance_loss": weighted_loss.detach().item(),
            "internal_guidance_unweighted_loss": intermediate_loss.detach().item(),
        }

    def guided_prediction(
        self,
        final_prediction: torch.Tensor,
        hidden_states: torch.Tensor,
        model_config,
        scale: float,
    ) -> torch.Tensor:
        intermediate = self.predict(hidden_states, final_prediction, model_config)
        return intermediate + float(scale) * (final_prediction - intermediate)

    @staticmethod
    def _extract_prediction(model_output):
        if torch.is_tensor(model_output):
            return model_output, "tensor"
        if isinstance(model_output, tuple) and model_output and torch.is_tensor(model_output[0]):
            return model_output[0], "tuple"
        if isinstance(model_output, list) and len(model_output) == 1 and torch.is_tensor(model_output[0]):
            return model_output[0], "list"
        if (
            isinstance(model_output, tuple)
            and model_output
            and isinstance(model_output[0], list)
            and len(model_output[0]) == 1
            and torch.is_tensor(model_output[0][0])
        ):
            return model_output[0][0], "tuple_list"
        sample = getattr(model_output, "sample", None)
        if torch.is_tensor(sample):
            return sample, "sample"
        raise TypeError(
            "Internal Guidance inference requires a tensor prediction, a single-item prediction list, "
            "or an output with a tensor-valued sample attribute."
        )

    @staticmethod
    def _replace_prediction(model_output, prediction: torch.Tensor, output_kind: str):
        if output_kind == "tensor":
            return prediction
        if output_kind == "tuple":
            return (prediction, *model_output[1:])
        if output_kind == "list":
            return [prediction]
        if output_kind == "tuple_list":
            return ([prediction], *model_output[1:])
        model_output.sample = prediction
        return model_output

    @contextlib.contextmanager
    def inference_context(self, scale: float):
        scale = float(scale)
        if scale == 1.0:
            yield
            return
        if scale <= 0:
            raise ValueError("validation_internal_guidance_scale must be greater than zero.")
        if self.model is None:
            raise RuntimeError("Internal Guidance head is not attached to a transformer.")

        original_forward = self.model.forward
        had_instance_forward = "forward" in self.model.__dict__
        instance_forward = self.model.__dict__.get("forward")
        parameters = inspect.signature(original_forward).parameters.values()
        accepts_buffer = "hidden_states_buffer" in inspect.signature(original_forward).parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
        )
        if not accepts_buffer:
            raise TypeError(
                f"{type(self.model).__name__}.forward does not accept hidden_states_buffer, so Internal Guidance "
                "sampling is unavailable for this transformer implementation."
            )

        def guided_forward(*args, **kwargs):
            from simpletuner.helpers.utils.hidden_state_buffer import HiddenStateBuffer

            hidden_states_buffer = HiddenStateBuffer(capture_layers={self.block_index})
            kwargs["hidden_states_buffer"] = hidden_states_buffer
            model_output = original_forward(*args, **kwargs)
            hidden_states = hidden_states_buffer.get_layer(self.block_index)
            if hidden_states is None:
                raise ValueError(
                    f"Internal Guidance sampling requested layer {self.block_index}, but the transformer did not "
                    "capture it."
                )
            final_prediction, output_kind = self._extract_prediction(model_output)
            guided = self.guided_prediction(final_prediction, hidden_states, self.model.config, scale)
            return self._replace_prediction(model_output, guided, output_kind)

        self.model.forward = guided_forward
        try:
            yield
        finally:
            if had_instance_forward:
                self.model.forward = instance_forward
            else:
                del self.model.forward


def attach_internal_guidance_head_from_state_dict(model: nn.Module, state_dict: dict) -> InternalGuidanceHead:
    """Attach the auxiliary head before a vanilla Diffusers/PEFT adapter load."""
    projection_weights = [
        value
        for key, value in state_dict.items()
        if InternalGuidanceRegularizer.MODULE_NAME in key and key.endswith("proj.weight") and torch.is_tensor(value)
    ]
    if len(projection_weights) != 1:
        raise ValueError(
            "Expected exactly one internal_guidance_head projection weight in the adapter state dict, "
            f"found {len(projection_weights)}."
        )

    block_values = [
        value
        for key, value in state_dict.items()
        if InternalGuidanceRegularizer.MODULE_NAME in key and key.endswith("block_index") and torch.is_tensor(value)
    ]
    if len(block_values) != 1:
        raise ValueError(
            "Expected exactly one internal_guidance_head block index in the adapter state dict, "
            f"found {len(block_values)}."
        )

    output_features, hidden_size = projection_weights[0].shape
    head = InternalGuidanceHead(
        hidden_size=int(hidden_size),
        output_features=int(output_features),
        block_index=int(block_values[0].item()),
    )
    head_prefix = f"{InternalGuidanceRegularizer.MODULE_NAME}."
    head_state_dict = {
        key.split(head_prefix, maxsplit=1)[1]: value
        for key, value in state_dict.items()
        if head_prefix in key and torch.is_tensor(value)
    }
    head.load_state_dict(head_state_dict, strict=True)
    parameter = next(model.parameters())
    head.to(device=parameter.device, dtype=parameter.dtype)
    setattr(model, InternalGuidanceRegularizer.MODULE_NAME, head)
    return head


def internal_guidance_lora_state_dict(state_dict: dict) -> dict:
    """Remove auxiliary-head tensors before passing an adapter to a standard LoRA loader."""
    return {key: value for key, value in state_dict.items() if InternalGuidanceRegularizer.MODULE_NAME not in key}


def internal_guidance_inference(model: nn.Module, scale: float):
    """Return a context manager that applies a loaded auxiliary head during sampling."""
    head = getattr(model, InternalGuidanceRegularizer.MODULE_NAME, None)
    if head is None:
        raise ValueError("The transformer does not have an internal_guidance_head.")
    internal_head = next((module for module in head.modules() if isinstance(module, InternalGuidanceHead)), None)
    if internal_head is None:
        raise TypeError("The transformer's internal_guidance_head has an unsupported module type.")

    regularizer = object.__new__(InternalGuidanceRegularizer)
    regularizer.model = model
    regularizer.block_index = int(internal_head.block_index.item())
    return regularizer.inference_context(scale)

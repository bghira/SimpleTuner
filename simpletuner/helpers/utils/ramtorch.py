import importlib
import logging
from fnmatch import fnmatch
from functools import lru_cache
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
_peft_lora_config_patched = False
_peft_lora_model_patched = False
_peft_inject_patched = False


@lru_cache(maxsize=1)
def _ramtorch_imports():
    """
    Import RamTorch lazily to avoid hard dependencies when the feature is unused.
    """
    importlib.import_module("ramtorch")  # noqa: F401

    from simpletuner.helpers import ramtorch_workarounds

    ramtorch_workarounds.apply_ramtorch_workarounds()
    from ramtorch.helpers import replace_linear_with_ramtorch
    from ramtorch.modules.linear import Linear as RamTorchLinear
    from ramtorch.zero1 import broadcast_zero_params, create_zero_param_groups
    from ramtorch.zero2 import setup_grad_sharding_hooks

    return {
        "Linear": RamTorchLinear,
        "replace_all": replace_linear_with_ramtorch,
        "broadcast_zero_params": broadcast_zero_params,
        "create_zero_param_groups": create_zero_param_groups,
        "setup_grad_sharding_hooks": setup_grad_sharding_hooks,
    }


def is_available() -> bool:
    try:
        _ramtorch_imports()
        return True
    except Exception:
        return False


def ensure_available() -> dict:
    try:
        imports = _ramtorch_imports()
        _maybe_patch_peft_lora_config()
        _maybe_patch_peft_lora_model()
        _maybe_patch_peft_inject()
        return imports
    except ImportError as exc:
        raise ImportError(
            "RamTorch is required for --ramtorch but is not installed. Install from the RamTorch checkout (e.g. "
            "`pip install -e ~/src/ramtorch`) or disable --ramtorch."
        ) from exc
    except Exception as exc:
        raise ImportError(f"RamTorch import failed: {exc}") from exc


def normalize_patterns(patterns: Optional[Sequence[str]]) -> Optional[list[str]]:
    if patterns is None:
        return None
    if isinstance(patterns, str):
        patterns = [patterns]
    normalized = []
    for entry in patterns:
        text = str(entry).strip()
        if text:
            normalized.extend([segment.strip() for segment in text.split(",") if segment.strip()])
    return normalized or None


def _normalize_device(device: object) -> object:
    if isinstance(device, torch.device):
        return device
    if device is None:
        return torch.device("cuda") if torch.cuda.is_available() else "cuda"
    return device


def _matches_pattern(name: str, module: nn.Module, patterns: Iterable[str]) -> bool:
    class_name = module.__class__.__name__
    for pattern in patterns:
        candidates = [name]
        if "." in name:
            candidates.append(name.split(".", 1)[1])
        if any(fnmatch(candidate, pattern) for candidate in candidates) or fnmatch(class_name, pattern):
            return True
    return False


def _count_eligible_linear_layers(
    module: nn.Module,
    patterns: Optional[list[str]],
    name_prefix: str = "",
) -> list[tuple[nn.Module, str, nn.Linear, str]]:
    """
    Count and collect all Linear layers eligible for replacement.

    Returns:
        List of tuples: (parent_module, child_name, linear_layer, qualified_name)
    """
    eligible = []

    def _recurse(current: nn.Module, prefix: str = ""):
        for child_name, child in current.named_children():
            qualified_name = f"{prefix}.{child_name}" if prefix else child_name

            if isinstance(child, nn.Linear):
                if patterns is None or _matches_pattern(qualified_name, child, patterns):
                    eligible.append((current, child_name, child, qualified_name))
                continue

            _recurse(child, qualified_name)

    _recurse(module, name_prefix)
    return eligible


def replace_linear_layers_with_ramtorch(
    module: nn.Module,
    *,
    device: object,
    target_patterns: Optional[Sequence[str]] = None,
    name_prefix: str = "",
    percent: Optional[float] = None,
) -> int:
    """
    Replace Linear layers within a module with RamTorch equivalents.

    Args:
        module: Root module to process.
        device: Target CUDA/ROCm device identifier (torch.device or string).
        target_patterns: Optional list of glob patterns to filter which modules are replaced.
        name_prefix: Optional prefix applied to the qualified module names used for matching.
        percent: Optional percentage (0-100) of eligible Linear layers to replace.
                 If None or 100, all eligible layers are replaced.

    Returns:
        Number of Linear modules replaced.
    """
    import math

    imports = ensure_available()
    ramtorch_linear = imports["Linear"]
    replace_all = imports["replace_all"]
    patterns = normalize_patterns(target_patterns)
    resolved_device = _normalize_device(device)

    # Handle percentage: if not specified or 100, replace all
    use_percent = percent is not None and percent < 100

    if patterns is None and not use_percent:
        # Replace every Linear using RamTorch's helper.
        num_linear = sum(1 for _, child in module.named_modules() if isinstance(child, nn.Linear))
        replace_all(module, device=resolved_device)
        for _, child in module.named_modules():
            if isinstance(child, ramtorch_linear):
                for param in child.parameters(recurse=False):
                    setattr(param, "is_ramtorch", True)
        return num_linear

    # Collect all eligible layers first (needed for percentage calculation)
    eligible = _count_eligible_linear_layers(module, patterns, name_prefix)

    if not eligible:
        return 0

    # Calculate how many to replace based on percentage
    if use_percent:
        total_eligible = len(eligible)
        num_to_replace = math.ceil(total_eligible * percent / 100)
        if num_to_replace == 0 and percent > 0:
            num_to_replace = 1  # At least one if percent > 0
        eligible = eligible[:num_to_replace]
        logger.info(
            "RamTorch percentage mode: replacing %d of %d eligible Linear layers (%.1f%%).",
            num_to_replace,
            total_eligible,
            percent,
        )

    replaced = 0
    for parent, child_name, child, qualified_name in eligible:
        if isinstance(child, ramtorch_linear):
            continue

        new_layer = ramtorch_linear(
            child.in_features,
            child.out_features,
            bias=child.bias is not None,
            device=resolved_device,
            dtype=child.weight.dtype,
            skip_init=True,
        )
        new_layer.train(child.training)

        with torch.no_grad():
            new_layer.weight.copy_(child.weight.detach().to("cpu"))
            if child.bias is not None and new_layer.bias is not None:
                new_layer.bias.copy_(child.bias.detach().to("cpu"))

        new_layer.weight.requires_grad = child.weight.requires_grad
        if new_layer.bias is not None and child.bias is not None:
            new_layer.bias.requires_grad = child.bias.requires_grad
        setattr(new_layer.weight, "is_ramtorch", True)
        if new_layer.bias is not None:
            setattr(new_layer.bias, "is_ramtorch", True)

        setattr(parent, child_name, new_layer)
        replaced += 1

    return replaced


def replace_all_layers_with_ramtorch(
    module: nn.Module,
    *,
    device: object,
    include_linear: bool = True,
    include_embedding: bool = True,
    include_conv: bool = True,
    include_layernorm: bool = True,
    include_rmsnorm: bool = True,
    target_patterns: Optional[Sequence[str]] = None,
    name_prefix: str = "",
    percent: Optional[float] = None,
) -> dict:
    """
    Replace all supported layer types with CPU-bouncing RamTorch versions.

    This includes Linear (via ramtorch library) plus Embedding, Conv, and LayerNorm
    (via our extensions).

    Args:
        module: Root module to process.
        device: Target CUDA/ROCm device identifier.
        include_linear: Replace nn.Linear layers.
        include_embedding: Replace nn.Embedding layers.
        include_conv: Replace nn.Conv2d and nn.Conv3d layers.
        include_layernorm: Replace nn.LayerNorm layers.
        include_rmsnorm: Replace RMSNorm-like layers.
        target_patterns: Optional glob patterns to filter which modules are replaced.
        name_prefix: Optional prefix for module names.
        percent: Optional percentage (0-100) of eligible Linear layers to replace.

    Returns:
        Dict with counts of each layer type replaced.
    """
    from simpletuner.helpers.ramtorch_extensions import replace_module_with_ramtorch

    resolved_device = _normalize_device(device)
    counts = {"linear": 0, "embedding": 0, "conv": 0, "layernorm": 0}

    # Replace Linear layers using ramtorch library
    if include_linear:
        counts["linear"] = replace_linear_layers_with_ramtorch(
            module,
            device=resolved_device,
            target_patterns=target_patterns,
            name_prefix=name_prefix,
            percent=percent,
        )

    # Replace other layers using our extensions
    counts["other"] = replace_module_with_ramtorch(
        module,
        device=str(resolved_device) if isinstance(resolved_device, torch.device) else resolved_device,
        include_embedding=include_embedding,
        include_conv=include_conv,
        include_layernorm=include_layernorm,
        include_rmsnorm=include_rmsnorm,
    )

    return counts


def register_lora_custom_module(lora_config) -> bool:
    """
    Add RamTorch Linear to PEFT's custom module map so LoRA can wrap CPUBouncingLinear layers.
    """
    if lora_config is None:
        return False

    try:
        import torch
        from peft.tuners.lora.layer import Linear as PeftLinear
        from ramtorch.modules.linear import Linear as RamTorchLinear
    except Exception:
        return False

    class RamTorchPeftLinear(PeftLinear):  # type: ignore[misc]
        """
        PEFT Linear wrapper that ensures LoRA weights live on the input device for RamTorch layers.
        """

        def _ensure_lora_on_device(self, device):
            if device is None:
                return
            try:
                active = self.active_adapter
            except Exception:
                active = None

            def _active_adapters(candidate):
                if candidate is None:
                    return []
                if isinstance(candidate, (list, tuple, set)):
                    return list(candidate)
                return [candidate]

            def _move_layer(layer):
                try:
                    if hasattr(layer, "to") and layer.weight.device != device:
                        layer.to(device)
                except Exception:
                    pass

            actives = _active_adapters(active)
            keys_to_move = actives or list(self.lora_A.keys())

            for name in keys_to_move:
                try:
                    key = name
                    if key in self.lora_A:
                        _move_layer(self.lora_A[key])
                    if key in self.lora_B:
                        _move_layer(self.lora_B[key])
                    if key in getattr(self.lora_embedding_A, "_parameters", {}):
                        emb_a = self.lora_embedding_A[key]
                        emb_b = self.lora_embedding_B.get(key)
                        try:
                            if hasattr(emb_a, "to") and emb_a.device != device:
                                emb_a.data = emb_a.data.to(device)
                        except Exception:
                            pass
                        try:
                            if emb_b is not None and hasattr(emb_b, "to") and emb_b.device != device:
                                emb_b.data = emb_b.data.to(device)
                        except Exception:
                            pass
                except Exception:
                    continue

        def forward(self, x: torch.Tensor, *args, **kwargs):  # type: ignore[override]
            self._ensure_lora_on_device(x.device if torch.is_tensor(x) else None)
            return super().forward(x, *args, **kwargs)

    custom_modules = getattr(lora_config, "_custom_modules", None)
    if custom_modules is None:
        custom_modules = {}
        lora_config._custom_modules = custom_modules
    if not isinstance(custom_modules, dict):
        return False
    if RamTorchLinear in custom_modules:
        return False

    custom_modules[RamTorchLinear] = RamTorchPeftLinear
    logger.debug("Registered RamTorch Linear for PEFT LoRA custom module dispatch.")
    return True


def _maybe_patch_peft_lora_config() -> bool:
    """
    Ensure diffusers' LoRA config creation function attaches the RamTorch custom module mapping.
    """

    global _peft_lora_config_patched
    if _peft_lora_config_patched:
        return False

    def _wrap_create_lora_config(module, attribute) -> bool:
        create_fn = getattr(module, attribute, None)
        if create_fn is None:
            return False
        if getattr(create_fn, "_ramtorch_wrapped_custom_patch", False):
            return False

        def _wrapped_create_lora_config(*args, **kwargs):
            state_dict = args[0] if args else kwargs.get("state_dict")
            if isinstance(state_dict, dict):
                prefixed = {}
                for key, value in state_dict.items():
                    if not isinstance(key, str):
                        continue
                    if key.startswith("_orig_mod."):
                        continue
                    prefixed[f"_orig_mod.{key}"] = value
                if prefixed:
                    merged = dict(state_dict)
                    merged.update(prefixed)
                    state_dict = merged
                    if args:
                        args = (state_dict,) + tuple(args[1:])
                    else:
                        kwargs["state_dict"] = state_dict

            config = create_fn(*args, **kwargs)
            try:
                register_lora_custom_module(config)
                from peft.tuners.lora.layer import Linear as PeftLinear
                from ramtorch.modules.linear import Linear as RamTorchLinear

                custom_modules = getattr(config, "_custom_modules", None)
                if not isinstance(custom_modules, dict):
                    custom_modules = {}
                    config._custom_modules = custom_modules
                custom_modules.setdefault(RamTorchLinear, PeftLinear)

                # Handle torch.compile / OptimizedModule name prefixing: add _orig_mod targets.
                targets = getattr(config, "target_modules", None)
                if targets:
                    targets_list = [targets] if isinstance(targets, str) else list(targets)
                    prefixed_targets = [
                        f"_orig_mod.{t}" for t in targets_list if isinstance(t, str) and not t.startswith("_orig_mod.")
                    ]
                    if prefixed_targets:
                        config.target_modules = list(dict.fromkeys(targets_list + prefixed_targets))
            except Exception as exc:
                logger.debug("RamTorch LoRA custom module patch failed for %s.%s: %s", module.__name__, attribute, exc)
            return config

        _wrapped_create_lora_config._ramtorch_wrapped = True
        _wrapped_create_lora_config._ramtorch_wrapped_custom_patch = True
        setattr(module, attribute, _wrapped_create_lora_config)
        return True

    patched = False
    try:
        import diffusers.loaders.peft as diffusers_peft  # type: ignore

        patched |= _wrap_create_lora_config(diffusers_peft, "_create_lora_config")
    except Exception:
        pass

    try:
        import diffusers.utils.peft_utils as diffusers_peft_utils  # type: ignore

        patched |= _wrap_create_lora_config(diffusers_peft_utils, "_create_lora_config")
    except Exception:
        pass

    _peft_lora_config_patched = patched
    return patched


def _maybe_patch_peft_lora_model() -> bool:
    """
    Patch PEFT's LoRA dispatcher so CPUBouncingLinear is supported even if a config lacks custom module metadata.
    """

    global _peft_lora_model_patched
    if _peft_lora_model_patched:
        return False

    try:
        import torch
        from peft.tuners.lora.layer import Linear as PeftLinear
        from peft.tuners.lora.model import LoraModel
        from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
        from ramtorch.modules.linear import Linear as RamTorchLinear
    except Exception:
        return False

    class RamTorchPeftLinear(PeftLinear):  # type: ignore[misc]
        def _ensure_lora_on_device(self, device):
            if device is None:
                return
            try:
                active = self.active_adapter
            except Exception:
                active = None

            def _active_adapters(candidate):
                if candidate is None:
                    return []
                if isinstance(candidate, (list, tuple, set)):
                    return list(candidate)
                return [candidate]

            def _move_layer(layer):
                try:
                    if hasattr(layer, "to") and layer.weight.device != device:
                        layer.to(device)
                except Exception:
                    pass

            actives = _active_adapters(active)
            keys_to_move = actives or list(self.lora_A.keys())

            for name in keys_to_move:
                try:
                    key = name
                    if key in self.lora_A:
                        _move_layer(self.lora_A[key])
                    if key in self.lora_B:
                        _move_layer(self.lora_B[key])
                except Exception:
                    continue

        def forward(self, x: torch.Tensor, *args, **kwargs):  # type: ignore[override]
            self._ensure_lora_on_device(x.device if torch.is_tensor(x) else None)
            return super().forward(x, *args, **kwargs)

    # Patch target-module check to ignore _orig_mod. prefixes from torch.compile
    if not getattr(BaseTuner._check_target_module_exists, "_ramtorch_wrapped", False):
        orig_check = BaseTuner._check_target_module_exists

        @staticmethod  # type: ignore[misc]
        def _wrapped_check(peft_config, key: str):
            result = orig_check(peft_config, key)
            if result:
                return result
            if isinstance(key, str) and key.startswith("_orig_mod."):
                return orig_check(peft_config, key.removeprefix("_orig_mod."))
            return result

        _wrapped_check._ramtorch_wrapped = True
        BaseTuner._check_target_module_exists = _wrapped_check  # type: ignore[assignment]

    orig = LoraModel._create_new_module
    if getattr(orig, "_ramtorch_wrapped", False):
        _peft_lora_model_patched = True
        return False

    @staticmethod
    def _wrapped_create_new_module(lora_config, adapter_name, target, **kwargs):
        try:
            target_base = target.get_base_layer() if isinstance(target, BaseTunerLayer) else target
            if isinstance(target_base, RamTorchLinear):
                return RamTorchPeftLinear(target, adapter_name, **kwargs)
        except Exception as exc:
            logger.debug("RamTorch LoRA dispatcher fallback failed: %s", exc)
        return orig(lora_config, adapter_name, target, **kwargs)

    _wrapped_create_new_module._ramtorch_wrapped = True
    LoraModel._create_new_module = staticmethod(_wrapped_create_new_module)
    _peft_lora_model_patched = True
    return True


def _maybe_patch_peft_inject() -> bool:
    """
    Patch PEFT inject_adapter_in_model to duplicate LoRA state_dict keys with _orig_mod.
    This allows adapter loading on torch.compile/OptimizedModule models where module
    names are prefixed with _orig_mod.
    """

    global _peft_inject_patched
    if _peft_inject_patched:
        return False

    try:
        import copy

        from peft.mapping import inject_adapter_in_model as orig_inject
        from peft.tuners.tuners_utils import BaseTuner
    except Exception:
        return False

    if getattr(orig_inject, "_ramtorch_wrapped", False) and getattr(BaseTuner.inject_adapter, "_ramtorch_wrapped", False):
        _peft_inject_patched = True
        return False

    def _maybe_prefix_state_dict(state_dict, model):
        if not isinstance(state_dict, dict):
            return state_dict
        # Only act if the compiled prefix exists in the model.
        try:
            has_prefixed_modules = any(
                isinstance(name, str) and name.startswith("_orig_mod.") for name, _ in model.named_modules()
            )
        except Exception:
            has_prefixed_modules = False
        if not has_prefixed_modules:
            return state_dict
        patched = copy.copy(state_dict)
        for key, value in state_dict.items():
            if not isinstance(key, str):
                continue
            if key.startswith("_orig_mod."):
                continue
            prefixed_key = f"_orig_mod.{key}"
            if prefixed_key not in patched:
                patched[prefixed_key] = value
        return patched

    def _wrapped_inject(peft_config, model, adapter_name: str | None = None, *args, **kwargs):
        state_dict = kwargs.get("state_dict")
        if state_dict is None and args:
            # positional signature: (peft_config, model, adapter_name, state_dict, ...)
            state_dict = args[0]
            args = args[1:]
        patched_state = _maybe_prefix_state_dict(state_dict, model)
        if patched_state is not state_dict:
            kwargs["state_dict"] = patched_state
        elif state_dict is not None and "state_dict" not in kwargs:
            kwargs["state_dict"] = state_dict
        return orig_inject(peft_config, model, adapter_name=adapter_name, *args, **kwargs)

    _wrapped_inject._ramtorch_wrapped = True
    import peft.mapping as peft_mapping

    peft_mapping.inject_adapter_in_model = _wrapped_inject  # type: ignore

    orig_base_inject = BaseTuner.inject_adapter

    def _wrapped_base_inject(self, peft_config, model, adapter_name: str = "default", *args, **kwargs):
        # Prefixed targets for torch.compile
        targets = getattr(peft_config, "target_modules", None)
        if targets:
            targets_list = [targets] if isinstance(targets, str) else list(targets)
            prefixed_targets = [
                f"_orig_mod.{t}" for t in targets_list if isinstance(t, str) and not t.startswith("_orig_mod.")
            ]
            if prefixed_targets:
                peft_config.target_modules = list(dict.fromkeys(targets_list + prefixed_targets))
        # Prefixed state_dict if passed positionally
        state_dict = kwargs.get("state_dict")
        if state_dict is None and args:
            state_dict = args[0]
            args = args[1:]
        state_dict = _maybe_prefix_state_dict(state_dict, model)
        if state_dict is not None:
            kwargs["state_dict"] = state_dict
        return orig_base_inject(self, peft_config, model, adapter_name, *args, **kwargs)

    _wrapped_base_inject._ramtorch_wrapped = True
    BaseTuner.inject_adapter = _wrapped_base_inject  # type: ignore[assignment]

    _peft_inject_patched = True
    return True


def ramtorch_zero_utils():
    imports = ensure_available()
    return imports["broadcast_zero_params"], imports["create_zero_param_groups"], imports["setup_grad_sharding_hooks"]


def move_embeddings_to_device(module: nn.Module, device: object) -> int:
    """
    Move all non-RamTorch layers to the specified device.

    When ramtorch is applied to a model, only nn.Linear layers are converted
    to CPU-bouncing versions. All other layers (Embedding, LayerNorm, etc.)
    need to be on GPU because:
    1. Embeddings receive GPU input_ids
    2. Other layers receive GPU activations from ramtorch Linear outputs

    Args:
        module: Root module containing layers to move.
        device: Target device (e.g., "cuda", torch.device("cuda:0")).

    Returns:
        Number of modules moved.
    """
    moved = 0

    def _has_ramtorch_params(mod: nn.Module) -> bool:
        for param in mod.parameters(recurse=False):
            if getattr(param, "is_ramtorch", False):
                return True
        return False

    def _is_leaf_module(mod: nn.Module) -> bool:
        for child in mod.children():
            if any(True for _ in child.parameters(recurse=True)):
                return False
        return True

    for name, child in module.named_modules():
        # Move buffers from all modules (e.g., position_ids in CLIPTextEmbeddings)
        for buf_name, buf in child.named_buffers(recurse=False):
            if buf.device.type == "cpu":
                child.register_buffer(buf_name, buf.to(device))

        # Move non-ramtorch parameters to GPU
        # This handles both leaf modules AND parameters on parent modules (like pos_embed)
        for param_name, param in child.named_parameters(recurse=False):
            if getattr(param, "is_ramtorch", False):
                continue
            if param.device.type == "cpu":
                # Move just this parameter, not the whole module (to avoid moving ramtorch children)
                param.data = param.data.to(device)
                moved += 1

    return moved


# Backwards compatibility aliases - all point to move_embeddings_to_device above
def move_non_ramtorch_modules_to_device(module: nn.Module, device: object, **kwargs) -> int:
    return move_embeddings_to_device(module, device)


def move_non_linear_layers_to_device(module: nn.Module, device: object) -> int:
    return move_embeddings_to_device(module, device)


def mark_ddp_ignore_params(module: nn.Module) -> int:
    """
    Mark RamTorch parameters on a module to be ignored by DistributedDataParallel.

    Returns:
        Number of parameters added to the ignore list.
    """
    ramtorch_names = [name for name, param in module.named_parameters() if getattr(param, "is_ramtorch", False)]
    if not ramtorch_names:
        device_types = {param.device.type for _, param in module.named_parameters() if param.device is not None}
        if "cuda" in device_types and "cpu" in device_types:
            ramtorch_names = [name for name, param in module.named_parameters() if param.device.type == "cpu"]
    if not ramtorch_names:
        return 0

    existing = getattr(module, "_ddp_params_and_buffers_to_ignore", set())
    module._ddp_params_and_buffers_to_ignore = set(existing) | set(ramtorch_names)
    return len(ramtorch_names)

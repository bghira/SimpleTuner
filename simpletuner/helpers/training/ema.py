import contextlib
import copy
import logging
import os
from typing import Any, Dict, Iterable, Optional, Union

import torch
import transformers
from diffusers.utils import is_transformers_available
from diffusers.utils.deprecation_utils import deprecate

try:
    from torch.distributed.tensor import DTensor, Replicate
except ImportError:
    DTensor = None  # type: ignore[assignment]
    Replicate = None  # type: ignore[assignment]

from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger("EMAModel")
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


def should_update_ema(args, step):
    if args.ema_update_interval is None:
        # If the EMA update interval is not set, always update the EMA.
        return True
    else:
        should_update = step % args.ema_update_interval == 0
        if should_update:
            logger.debug("Updating EMA weights...")
        return should_update


class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    @staticmethod
    def _is_dtensor(tensor: torch.Tensor) -> bool:
        return DTensor is not None and isinstance(tensor, DTensor)

    @staticmethod
    def _gather_dtensor(tensor: torch.Tensor) -> torch.Tensor:
        if not EMAModel._is_dtensor(tensor):
            return tensor
        if Replicate is None:
            raise RuntimeError("DTensor support requires torch.distributed.tensor.Replicate")
        replicated = tensor.redistribute(tensor.device_mesh, placements=[Replicate()])
        return replicated.to_local()

    def __init__(
        self,
        args,
        accelerator,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: Union[float, int] = 1.0,
        power: Union[float, int] = 2 / 3,
        foreach: bool = True,
        model_cls: Optional[Any] = None,
        model_config: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        Args:
            parameters (Iterable[torch.nn.Parameter]): The parameters to track.
            decay (float): The decay factor for the exponential moving average.
            min_decay (float): The minimum decay factor for the exponential moving average.
            update_after_step (int): The number of steps to wait before starting to update the EMA weights.
            use_ema_warmup (bool): Whether to use EMA warmup.
            inv_gamma (float):
                Inverse multiplicative factor of EMA warmup. Default: 1. Only used if `use_ema_warmup` is True.
            power (float): Exponential factor of EMA warmup. Default: 2/3. Only used if `use_ema_warmup` is True.
            foreach (bool): Use torch._foreach functions for updating shadow parameters. Should be faster.
            device (Optional[Union[str, torch.device]]): The device to store the EMA weights on. If None, the EMA
                        weights will be stored on CPU.

        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        """

        if isinstance(parameters, torch.nn.Module):
            deprecation_message = (
                "Passing a `torch.nn.Module` to `ExponentialMovingAverage` is deprecated. "
                "Please pass the parameters of the module instead."
            )
            deprecate(
                "passing a `torch.nn.Module` to `ExponentialMovingAverage`",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            parameters = parameters.parameters()

            # set use_ema_warmup to True if a torch.nn.Module is passed for backwards compatibility
            use_ema_warmup = True

        if kwargs.get("max_value", None) is not None:
            deprecation_message = "The `max_value` argument is deprecated. Please use `decay` instead."
            deprecate("max_value", "1.0.0", deprecation_message, standard_warn=False)
            decay = kwargs["max_value"]

        if kwargs.get("min_value", None) is not None:
            deprecation_message = "The `min_value` argument is deprecated. Please use `min_decay` instead."
            deprecate("min_value", "1.0.0", deprecation_message, standard_warn=False)
            min_decay = kwargs["min_value"]

        parameters = list(parameters)
        self._tracked_param_ids = [id(param) for param in parameters]
        self.shadow_params = [p.clone().detach() for p in parameters]

        if kwargs.get("device", None) is not None:
            deprecation_message = "The `device` argument is deprecated. Please use `to` instead."
            deprecate("device", "1.0.0", deprecation_message, standard_warn=False)
            self.to(device=kwargs["device"])

        self.temp_stored_params = None
        self._temp_stored_param_ids: Optional[list[int]] = None

        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value = None  # set in `step()`
        self.foreach = foreach

        self.model_cls = model_cls
        self.model_config = model_config
        self.args = args
        self.accelerator = accelerator
        self.training = True  # To emulate nn.Module's training mode

    def _should_use_foreach(
        self, primary_tensors: list[torch.Tensor], secondary_tensors: Optional[list[torch.Tensor]] = None
    ) -> bool:
        """
        Decide whether foreach ops are safe for the provided tensors.

        Foreach kernels require homogeneous devices and dtypes and do not support DTensors.
        """
        if not self.foreach:
            return False
        if not primary_tensors:
            return False
        if any(self._is_dtensor(tensor) for tensor in primary_tensors):
            return False
        if secondary_tensors is not None:
            if len(secondary_tensors) != len(primary_tensors):
                return False
            if any(self._is_dtensor(tensor) for tensor in secondary_tensors):
                return False

        primary_devices = {tensor.device for tensor in primary_tensors}
        secondary_devices = primary_devices if secondary_tensors is None else {tensor.device for tensor in secondary_tensors}
        if len(primary_devices) != 1 or len(secondary_devices) != 1 or primary_devices != secondary_devices:
            return False

        primary_dtypes = {tensor.dtype for tensor in primary_tensors}
        secondary_dtypes = primary_dtypes if secondary_tensors is None else {tensor.dtype for tensor in secondary_tensors}
        if len(primary_dtypes) != 1 or len(secondary_dtypes) != 1 or primary_dtypes != secondary_dtypes:
            return False

        return True

    def _align_shadow_params(
        self, parameters: Iterable[torch.nn.Parameter], *, allow_subset: bool = False
    ) -> list[tuple[torch.nn.Parameter, torch.nn.Parameter]]:
        params = list(parameters)
        if not allow_subset and len(params) != len(self.shadow_params):
            raise RuntimeError(
                f"EMA parameter count mismatch: expected {len(self.shadow_params)} parameters but received {len(params)}."
            )

        if not allow_subset and all(id(param) == tracked for param, tracked in zip(params, self._tracked_param_ids)):
            return list(zip(self.shadow_params, params))

        id_to_shadow = {tracked: shadow for tracked, shadow in zip(self._tracked_param_ids, self.shadow_params)}
        aligned: list[tuple[torch.nn.Parameter, torch.nn.Parameter]] = []
        missing_tracked = 0

        for param in params:
            shadow = id_to_shadow.get(id(param))
            if shadow is None:
                # Ignore untracked parameters when allow_subset=True; otherwise treat as an error.
                if not allow_subset:
                    missing_tracked += 1
                continue
            aligned.append((shadow, param))

        if not allow_subset:
            if missing_tracked > 0:
                raise RuntimeError(
                    f"EMA parameter mapping failed: received {missing_tracked} untracked parameter(s). "
                    "This usually means the model parameters were recreated after EMA initialization."
                )
            if len(aligned) != len(self.shadow_params):
                raise RuntimeError(
                    f"EMA parameter alignment incomplete: aligned {len(aligned)} of {len(self.shadow_params)} parameters."
                )
        else:
            if len(aligned) != len(self.shadow_params):
                logger.warning(
                    "EMA copy received %s tracked parameter(s) but EMA tracks %s. Applying EMA to the tracked subset only.",
                    len(aligned),
                    len(self.shadow_params),
                )

        return aligned

    def save_state_dict(self, path: str) -> None:
        """
        Save the EMA model's state directly to a file.

        Args:
            path (str): The file path where the EMA state will be saved.
        """
        # if the folder containing the path does not exist, create it
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # grab state dict
        state_dict = self.state_dict()
        # save it using torch.save
        torch.save(state_dict, path)
        logger.info(f"EMA model state saved to {path}")

    def load_state_dict(self, path: str) -> None:
        """
        Load the EMA model's state from a file and apply it to this instance.

        Args:
            path (str): The file path from where the EMA state will be loaded.
        """
        state_dict = torch.load(path, map_location="cpu", weights_only=True)

        # Load metadata
        self.decay = state_dict.get("decay", self.decay)
        self.min_decay = state_dict.get("min_decay", self.min_decay)
        self.optimization_step = state_dict.get("optimization_step", self.optimization_step)
        self.update_after_step = state_dict.get("update_after_step", self.update_after_step)
        self.use_ema_warmup = state_dict.get("use_ema_warmup", self.use_ema_warmup)
        self.inv_gamma = state_dict.get("inv_gamma", self.inv_gamma)
        self.power = state_dict.get("power", self.power)

        # Load shadow parameters
        shadow_params = []
        idx = 0
        while f"shadow_params.{idx}" in state_dict:
            shadow_params.append(state_dict[f"shadow_params.{idx}"])
            idx += 1

        if len(shadow_params) != len(self.shadow_params):
            raise ValueError(
                f"Mismatch in number of shadow parameters: expected {len(self.shadow_params)}, "
                f"but found {len(shadow_params)} in the state dict."
            )

        for current_param, loaded_param in zip(self.shadow_params, shadow_params):
            current_param.data.copy_(loaded_param.data)

        logger.info(f"EMA model state loaded from {path}")

    @classmethod
    def from_pretrained(cls, path, model_cls) -> "EMAModel":
        _, ema_kwargs = model_cls.load_config(path, return_unused_kwargs=True)
        model = model_cls.from_pretrained(path)

        ema_model = cls(model.parameters(), model_cls=model_cls, model_config=model.config)

        ema_model.load_state_dict(ema_kwargs)
        return ema_model

    def save_pretrained(self, path, max_shard_size: str = "10GB"):
        if self.model_cls is None:
            raise ValueError("`save_pretrained` can only be used if `model_cls` was defined at __init__.")

        if self.model_config is None:
            raise ValueError("`save_pretrained` can only be used if `model_config` was defined at __init__.")

        model = self.model_cls.from_config(self.model_config)
        state_dict = self.state_dict(exclude_params=True)
        state_dict.pop("shadow_params", None)

        model.register_to_config(**state_dict)
        # Copy shadow params to the new model by position (not by ID).
        # from_config creates an identical architecture, so param order matches.
        model_params = list(model.parameters())
        if len(model_params) != len(self.shadow_params):
            raise RuntimeError(
                f"EMA save_pretrained failed: model has {len(model_params)} parameters "
                f"but EMA tracks {len(self.shadow_params)}."
            )
        for param, shadow in zip(model_params, self.shadow_params):
            param.data.copy_(shadow.to(device=param.device, dtype=param.dtype))
        model.save_pretrained(path, max_shard_size=max_shard_size)

    def get_decay(self, optimization_step: int = None) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        if optimization_step is None:
            optimization_step = self.optimization_step

        step = max(0, optimization_step - self.update_after_step - 1)

        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            cur_decay_value = (1 + step) / (10 + step)

        cur_decay_value = min(cur_decay_value, self.decay)
        # make sure decay is not smaller than min_decay
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter], global_step: int = None):
        if not should_update_ema(self.args, global_step):
            return

        if self.args.ema_device == "cpu" and not self.args.ema_cpu_only:
            # Move EMA to accelerator for faster update.
            self.to(device=self.accelerator.device, non_blocking=True)
        if isinstance(parameters, torch.nn.Module):
            deprecation_message = (
                "Passing a `torch.nn.Module` to `ExponentialMovingAverage.step` is deprecated. "
                "Please pass the parameters of the module instead."
            )
            deprecate(
                "passing a `torch.nn.Module` to `ExponentialMovingAverage.step`",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            parameters = parameters.parameters()

        parameters = list(parameters)
        aligned = self._align_shadow_params(parameters)
        params_grad = [param for _, param in aligned if param.requires_grad]
        s_params_grad = [s_param for s_param, param in aligned if param.requires_grad]
        use_foreach = self._should_use_foreach(params_grad, s_params_grad)

        if global_step is not None:
            # When we're updating the EMA periodically, we can't trust the counter.
            self.optimization_step = global_step
        else:
            self.optimization_step += 1

        # Compute the decay factor for the exponential moving average.
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay

        context_manager = contextlib.nullcontext
        if is_transformers_available() and transformers.integrations.deepspeed.is_deepspeed_zero3_enabled():
            import deepspeed

        if use_foreach:
            if is_transformers_available() and transformers.integrations.deepspeed.is_deepspeed_zero3_enabled():
                context_manager = deepspeed.zero.GatheredParameters(parameters, modifier_rank=None)

            with context_manager():
                if len(params_grad) < len(aligned):
                    torch._foreach_copy_(
                        [s_param for s_param, param in aligned if not param.requires_grad],
                        [param for _, param in aligned if not param.requires_grad],
                        non_blocking=True,
                    )

                torch._foreach_sub_(
                    s_params_grad,
                    torch._foreach_sub(s_params_grad, params_grad),
                    alpha=one_minus_decay,
                )

        else:
            for s_param, param in aligned:
                if is_transformers_available() and transformers.integrations.deepspeed.is_deepspeed_zero3_enabled():
                    context_manager = deepspeed.zero.GatheredParameters(param, modifier_rank=None)

                with context_manager():
                    if param.requires_grad:
                        target = param
                        if target.dtype != s_param.dtype:
                            target = target.to(dtype=s_param.dtype)
                        if not self._is_dtensor(s_param):
                            target = target.to(device=s_param.device)
                        s_param.sub_(one_minus_decay * (s_param - target))
                    else:
                        target = param
                        if not self._is_dtensor(s_param):
                            target = target.to(device=s_param.device)
                        if target.dtype != s_param.dtype:
                            target = target.to(dtype=s_param.dtype)
                        s_param.copy_(target)
        if self.args.ema_device == "cpu" and not self.args.ema_cpu_only:
            # Move back to CPU for safe-keeping.
            self.to(device=self.args.ema_device, non_blocking=True)

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        parameters = list(parameters)
        aligned = self._align_shadow_params(parameters, allow_subset=True)
        params_only = [param for _, param in aligned]
        shadows_only = [s_param for s_param, _ in aligned]
        use_foreach = self._should_use_foreach(params_only, shadows_only)

        if use_foreach:
            torch._foreach_copy_(
                [param.data for param in params_only],
                [s_param.to(param.device).data for s_param, param in aligned],
            )
            return

        for s_param, param in aligned:
            source = s_param
            if self._is_dtensor(param):
                if not self._is_dtensor(source):
                    raise RuntimeError("EMA shadow parameter does not match distributed layout.")
                if source.dtype != param.dtype:
                    source = source.to(dtype=param.dtype)
                param.copy_(source)
                continue

            if self._is_dtensor(source):
                source_tensor = self._gather_dtensor(source)
            else:
                source_tensor = source

            param.data.copy_(source_tensor.to(device=param.device, dtype=param.dtype))

    def pin_memory(self) -> None:
        r"""
        Move internal buffers of the ExponentialMovingAverage to pinned memory. Useful for non-blocking transfers for
        offloading EMA params to the host.
        """
        if torch.backends.mps.is_available():
            logger.warning("Apple silicon does not support pinned memory. Skipping.")
            return

        if self.args.ema_cpu_only:
            return

        # This probably won't work, but we'll do it anyway.
        self.shadow_params = [p.pin_memory() for p in self.shadow_params]

    def to(self, *args, **kwargs):
        for param in self.shadow_params:
            param.data = param.data.to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        return self.to(device="cuda" if device is None else f"cuda:{device}")

    def cpu(self):
        return self.to(device="cpu")

    def state_dict(self, destination=None, prefix="", keep_vars=False, exclude_params: bool = False):
        r"""
        Returns a dictionary containing a whole state of the EMA model.
        """
        state_dict = {
            "decay": self.decay,
            "min_decay": self.min_decay,
            "optimization_step": self.optimization_step,
            "update_after_step": self.update_after_step,
            "use_ema_warmup": self.use_ema_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
        }
        if exclude_params:
            return state_dict
        for idx, param in enumerate(self.shadow_params):
            value = param if keep_vars else param.detach()
            if self._is_dtensor(value):
                value = self._gather_dtensor(value)
            state_dict[f"{prefix}shadow_params.{idx}"] = value
        return state_dict

    def store(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        r"""
        Save the current parameters for restoring later.
        """
        parameters = list(parameters)
        clones = []
        for param in parameters:
            if self._is_dtensor(param):
                clones.append(param.detach().clone())
            else:
                clones.append(param.detach().cpu().clone())
        self.temp_stored_params = clones
        self._temp_stored_param_ids = [id(param) for param in parameters]

    def restore(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        r"""
        Restore the parameters stored with the `store` method.
        """
        if self.temp_stored_params is None:
            raise RuntimeError("This ExponentialMovingAverage has no `store()`ed weights " "to `restore()`")
        parameters = list(parameters)

        if len(parameters) != len(self.temp_stored_params):
            raise RuntimeError(
                f"EMA restore parameter count mismatch: expected {len(self.temp_stored_params)} parameters but received {len(parameters)}."
            )

        if self._temp_stored_param_ids and all(
            id(param) == stored_id for param, stored_id in zip(parameters, self._temp_stored_param_ids)
        ):
            aligned_temp = list(zip(self.temp_stored_params, parameters))
        else:
            id_to_temp = (
                {stored_id: temp for stored_id, temp in zip(self._temp_stored_param_ids or [], self.temp_stored_params)}
                if self._temp_stored_param_ids
                else None
            )
            aligned_temp = []
            missing = []
            for param in parameters:
                if id_to_temp is not None and id(param) in id_to_temp:
                    aligned_temp.append((id_to_temp[id(param)], param))
                else:
                    missing.append(param)

            if missing:
                raise RuntimeError(
                    f"EMA restore failed: received {len(missing)} untracked parameter(s). "
                    "This usually means the model parameters were recreated after EMA.store()."
                )

        use_foreach = self._should_use_foreach(
            [param for _, param in aligned_temp],
            [temp_param for temp_param, _ in aligned_temp],
        )

        if use_foreach:
            torch._foreach_copy_(
                [param.data for _, param in aligned_temp],
                [c_param.data for c_param, _ in aligned_temp],
            )
        else:
            for c_param, param in aligned_temp:
                source = c_param
                if self._is_dtensor(param):
                    if not self._is_dtensor(source):
                        raise RuntimeError("Stored EMA parameter is missing distributed layout information.")
                    if source.dtype != param.dtype:
                        source = source.to(dtype=param.dtype)
                    param.copy_(source)
                else:
                    if self._is_dtensor(source):
                        source_tensor = self._gather_dtensor(source)
                    else:
                        source_tensor = source
                    param.data.copy_(source_tensor.to(device=param.device, dtype=param.dtype))

        # Better memory-wise.
        self.temp_stored_params = None
        self._temp_stored_param_ids = None

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.shadow_params)

    # Implementing nn.Module methods to emulate its behavior

    def named_children(self):
        # No child modules
        return iter([])

    def children(self):
        return iter([])

    def modules(self):
        yield self

    def named_modules(self, memo=None, prefix=""):
        yield prefix, self

    def parameters(self, recurse=True):
        return iter(self.shadow_params)

    def named_parameters(self, prefix="", recurse=True):
        for i, param in enumerate(self.shadow_params):
            name = f"{prefix}shadow_params.{i}"
            yield name, param

    def buffers(self, recurse=True):
        return iter([])

    def named_buffers(self, prefix="", recurse=True):
        return iter([])

    def train(self, mode=True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        # No gradients to zero in EMA model
        pass

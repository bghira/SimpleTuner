import torch, copy, logging, os, contextlib, transformers
from time import time
from typing import Any, Dict, Iterable, Optional, Union
from diffusers.utils.deprecation_utils import deprecate
from diffusers.models import UNet2DConditionModel
from diffusers.utils import is_transformers_available
from tqdm import tqdm

logger = logging.getLogger("EMAModel")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


def should_update_ema(args, step):
    if args.ema_update_interval is None:
        # If the EMA update interval is not set, always update the EMA.
        return True
    else:
        should_update = step % args.ema_update_interval == 0
        if should_update:
            logger.debug(f"Updating EMA weights...")
        return should_update


class EMAModel:
    """
    Exponential Moving Average of models weights
    """

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
            deprecation_message = (
                "The `max_value` argument is deprecated. Please use `decay` instead."
            )
            deprecate("max_value", "1.0.0", deprecation_message, standard_warn=False)
            decay = kwargs["max_value"]

        if kwargs.get("min_value", None) is not None:
            deprecation_message = "The `min_value` argument is deprecated. Please use `min_decay` instead."
            deprecate("min_value", "1.0.0", deprecation_message, standard_warn=False)
            min_decay = kwargs["min_value"]

        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        if kwargs.get("device", None) is not None:
            deprecation_message = (
                "The `device` argument is deprecated. Please use `to` instead."
            )
            deprecate("device", "1.0.0", deprecation_message, standard_warn=False)
            self.to(device=kwargs["device"])

        self.temp_stored_params = None

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

    @classmethod
    def from_pretrained(cls, path, model_cls) -> "EMAModel":
        _, ema_kwargs = model_cls.load_config(path, return_unused_kwargs=True)
        model = model_cls.from_pretrained(path)

        ema_model = cls(
            model.parameters(), model_cls=model_cls, model_config=model.config
        )

        ema_model.load_state_dict(ema_kwargs)
        return ema_model

    def save_pretrained(self, path, max_shard_size: str = "10GB"):
        if self.model_cls is None:
            raise ValueError(
                "`save_pretrained` can only be used if `model_cls` was defined at __init__."
            )

        if self.model_config is None:
            raise ValueError(
                "`save_pretrained` can only be used if `model_config` was defined at __init__."
            )

        model = self.model_cls.from_config(self.model_config)
        state_dict = self.state_dict()
        state_dict.pop("shadow_params", None)

        model.register_to_config(**state_dict)
        self.copy_to(model.parameters())
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

        if global_step is not None:
            self.optimization_step = global_step
        else:
            self.optimization_step += 1
        tqdm.write(f"EMA Optimization step: {self.optimization_step}")

        # Compute the decay factor for the exponential moving average.
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay

        context_manager = contextlib.nullcontext
        if (
            is_transformers_available()
            and transformers.deepspeed.is_deepspeed_zero3_enabled()
        ):
            import deepspeed

        if self.foreach:
            if (
                is_transformers_available()
                and transformers.deepspeed.is_deepspeed_zero3_enabled()
            ):
                context_manager = deepspeed.zero.GatheredParameters(
                    parameters, modifier_rank=None
                )

            with context_manager():
                params_grad = [param for param in parameters if param.requires_grad]
                s_params_grad = [
                    s_param
                    for s_param, param in zip(self.shadow_params, parameters)
                    if param.requires_grad
                ]

                if len(params_grad) < len(parameters):
                    torch._foreach_copy_(
                        [
                            s_param
                            for s_param, param in zip(self.shadow_params, parameters)
                            if not param.requires_grad
                        ],
                        [param for param in parameters if not param.requires_grad],
                        non_blocking=True,
                    )

                torch._foreach_sub_(
                    s_params_grad,
                    torch._foreach_sub(s_params_grad, params_grad),
                    alpha=one_minus_decay,
                )

        else:
            for s_param, param in zip(self.shadow_params, parameters):
                if (
                    is_transformers_available()
                    and transformers.deepspeed.is_deepspeed_zero3_enabled()
                ):
                    context_manager = deepspeed.zero.GatheredParameters(
                        param, modifier_rank=None
                    )

                with context_manager():
                    if param.requires_grad:
                        s_param.sub_(
                            one_minus_decay * (s_param - param.to(s_param.device))
                        )
                    else:
                        s_param.copy_(param)
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
        if self.foreach:
            torch._foreach_copy_(
                [param.data for param in parameters],
                [
                    s_param.to(param.device).data
                    for s_param, param in zip(self.shadow_params, parameters)
                ],
            )
        else:
            for s_param, param in zip(self.shadow_params, parameters):
                param.data.copy_(s_param.to(param.device).data)

    def pin_memory(self) -> None:
        r"""
        Move internal buffers of the ExponentialMovingAverage to pinned memory. Useful for non-blocking transfers for
        offloading EMA params to the host.
        """
        if torch.backends.mps.is_available():
            logger.warning(f"Apple silicon does not support pinned memory. Skipping.")
            return

        if self.args.ema_cpu_only:
            return

        # This probably won't work, but we'll do it anyway.
        self.shadow_params = [p.pin_memory() for p in self.shadow_params]

    def to(self, device=None, dtype=None, non_blocking=False) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            (
                p.to(device=device, dtype=dtype, non_blocking=non_blocking)
                if p.is_floating_point()
                else p.to(device=device, non_blocking=non_blocking)
            )
            for p in self.shadow_params
        ]

    def state_dict(self) -> dict:
        r"""
        Returns the state of the ExponentialMovingAverage as a dict. This method is used by accelerate during
        checkpointing to save the ema state dict.
        """
        # Following PyTorch conventions, references to tensors are returned:
        # "returns a reference to the state and not its copy!" -
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        return {
            "decay": self.decay,
            "min_decay": self.min_decay,
            "optimization_step": self.optimization_step,
            "update_after_step": self.update_after_step,
            "use_ema_warmup": self.use_ema_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "shadow_params": self.shadow_params,
        }

    def store(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        r"""
        Args:
        Save the current parameters for restoring later.
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored.
        """
        self.temp_stored_params = [param.detach().cpu().clone() for param in parameters]

    def restore(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        r"""
        Args:
        Restore the parameters stored with the `store` method. Useful to validate the model with EMA parameters without:
        affecting the original optimization process. Store the parameters before the `copy_to()` method. After
        validation (or model saving), use this to restore the former parameters.
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        if self.temp_stored_params is None:
            raise RuntimeError(
                "This ExponentialMovingAverage has no `store()`ed weights "
                "to `restore()`"
            )
        if self.foreach:
            torch._foreach_copy_(
                [param.data for param in parameters],
                [c_param.data for c_param in self.temp_stored_params],
            )
        else:
            for c_param, param in zip(self.temp_stored_params, parameters):
                param.data.copy_(c_param.data)

        # Better memory-wise.
        self.temp_stored_params = None

    def load_state_dict(self, state_dict: dict) -> None:
        r"""
        Args:
        Loads the ExponentialMovingAverage state. This method is used by accelerate during checkpointing to save the
        ema state dict.
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)

        self.decay = state_dict.get("decay", self.decay)
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.min_decay = state_dict.get("min_decay", self.min_decay)
        if not isinstance(self.min_decay, float):
            raise ValueError("Invalid min_decay")

        self.optimization_step = state_dict.get(
            "optimization_step", self.optimization_step
        )
        if not isinstance(self.optimization_step, int):
            raise ValueError("Invalid optimization_step")

        self.update_after_step = state_dict.get(
            "update_after_step", self.update_after_step
        )
        if not isinstance(self.update_after_step, int):
            raise ValueError("Invalid update_after_step")

        self.use_ema_warmup = state_dict.get("use_ema_warmup", self.use_ema_warmup)
        if not isinstance(self.use_ema_warmup, bool):
            raise ValueError("Invalid use_ema_warmup")

        self.inv_gamma = state_dict.get("inv_gamma", self.inv_gamma)
        if not isinstance(self.inv_gamma, (float, int)):
            raise ValueError("Invalid inv_gamma")

        self.power = state_dict.get("power", self.power)
        if not isinstance(self.power, (float, int)):
            raise ValueError("Invalid power")

        shadow_params = state_dict.get("shadow_params", None)
        if shadow_params is not None:
            self.shadow_params = shadow_params
            if not isinstance(self.shadow_params, list):
                raise ValueError("shadow_params must be a list")
            if not all(isinstance(p, torch.Tensor) for p in self.shadow_params):
                raise ValueError("shadow_params must all be Tensors")

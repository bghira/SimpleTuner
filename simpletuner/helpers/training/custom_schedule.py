import logging
import math
import os

import accelerate
import torch
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

logger = logging.getLogger(__name__)
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


def segmented_timestep_selection(actual_num_timesteps, bsz, weights, config, use_refiner_range: bool = False):
    # Determine the range of timesteps to use
    num_timesteps = actual_num_timesteps
    if use_refiner_range or config.refiner_training:
        if config.refiner_training_invert_schedule:
            # Inverted schedule calculation: we start from the last timestep and move downwards
            start_timestep = actual_num_timesteps - 1  # Start from the last timestep, e.g., 999
            # Calculate the end of the range based on the inverse of the training strength
            end_timestep = int(config.refiner_training_strength * actual_num_timesteps)
        else:
            # Normal refiner training schedule
            start_timestep = int(actual_num_timesteps * config.refiner_training_strength) - 1
            end_timestep = 0
        num_timesteps = start_timestep - end_timestep + 1
    else:
        start_timestep = actual_num_timesteps - 1
        end_timestep = 0

    # logger.debug(
    #     f"{'Using SDXL refiner' if config.refiner_training else 'Training base model '} with {num_timesteps} timesteps from a full schedule of {actual_num_timesteps} and a segment size of {num_timesteps // bsz} timesteps."
    # )
    segment_size = max(num_timesteps // bsz, 1)
    selected_timesteps = []

    # Select one timestep from each segment based on the weights
    for i in range(bsz):
        start = start_timestep - i * segment_size
        end = max(start - segment_size, end_timestep) if i != bsz - 1 else end_timestep
        # logger.debug(f"Segment from {start} to {end}")
        segment_weights = weights[end : start + 1]

        # Normalize segment weights to ensure they sum to 1
        segment_weights /= segment_weights.sum()

        # Sample one timestep from the segment
        segment_timesteps = torch.arange(end, start + 1)
        selected_timestep = torch.multinomial(segment_weights, 1).item()
        selected_timesteps.append(segment_timesteps[selected_timestep])

    # logger.debug(f"Selected timesteps: {selected_timesteps}")
    return torch.tensor(selected_timesteps)


def generate_timestep_weights(args, num_timesteps):
    weights = torch.ones(num_timesteps)

    # Determine the indices to bias
    num_to_bias = int(args.timestep_bias_portion * num_timesteps)

    if args.timestep_bias_strategy == "later":
        bias_indices = slice(-num_to_bias, None)
    elif args.timestep_bias_strategy == "earlier":
        bias_indices = slice(0, num_to_bias)
    elif args.timestep_bias_strategy == "range":
        # Out of the possible 1000 timesteps, we might want to focus on eg. 200-500.
        range_begin = args.timestep_bias_begin
        range_end = args.timestep_bias_end
        if range_begin < 0:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero."
            )
        if range_end > num_timesteps:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide an ending timestep smaller than the number of timesteps."
            )
        bias_indices = slice(range_begin, range_end)
    else:  # 'none' or any other string
        return weights
    if args.timestep_bias_multiplier <= 0:
        raise ValueError(
            "The parameter --timestep_bias_multiplier is not intended to be used to disable the training of specific timesteps."
            " If it was intended to disable timestep bias, use `--timestep_bias_strategy none` instead."
            " A timestep bias multiplier less than or equal to 0 is not allowed."
        )

    # Apply the bias
    weights[bias_indices] *= args.timestep_bias_multiplier

    # Normalize
    weights /= weights.sum()

    return weights


def get_polynomial_decay_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float = 1e-7,
    power: float = 1.0,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by *lr_end*, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        lr_end (`float`, *optional*, defaults to 1e-7):
            The end LR.
        power (`float`, *optional*, defaults to 1.0):
            Power factor.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """

    lr_init = optimizer.defaults["lr"]
    if not (float(lr_init) > float(lr_end)):
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return float(lr_end) / float(lr_init)  # as LambdaLR multiplies by lr_init
        else:
            lr_range = float(lr_init) - float(lr_end)
            decay_steps = int(num_training_steps) - int(num_warmup_steps)
            pct_remaining = 1 - (current_step - int(num_warmup_steps)) / decay_steps
            decay = lr_range * pct_remaining**power + float(lr_end)
            return decay / float(lr_init)  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def enforce_zero_terminal_snr(betas):
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    # Shift so last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    # Scale so first timestep is back to old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas
    return betas


def patch_scheduler_betas(scheduler):
    scheduler.betas = enforce_zero_terminal_snr(scheduler.betas)


class _enable_get_lr_call:
    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_lr_called_within_step = False
        return self


class Cosine(LRScheduler):
    r"""Use a cosine schedule for the learning rate, without restarts.
    This makes a nice and pretty chart on the tensorboard.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self,
        optimizer,
        T_0,
        steps_per_epoch=-1,
        T_mult=1,
        eta_min=0,
        last_step=-1,
        last_epoch=-1,
        verbose=False,
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            logger.warning(
                f"Cosine learning rate expects to use warmup steps as its interval. Expected positive integer T_0, but got {T_0}. Defaulting to 1000."
            )
            T_0 = 1000
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        if last_epoch != -1 and last_step != -1:
            last_epoch = last_step
        elif last_epoch != -1 and last_step == -1:
            last_step = last_epoch
        self.T_0 = T_0
        self.steps_per_epoch = steps_per_epoch
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_step
        super().__init__(optimizer=optimizer, last_epoch=last_step)

    def get_lr(self):
        lrs = [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]
        return lrs

    def step(self, step=None):
        if step is None and self.last_epoch < 0:
            step = 0

        if step is None:
            step = self.last_epoch + 1
            self.T_cur = (step // self.steps_per_epoch) + (step % self.steps_per_epoch) / self.steps_per_epoch
        else:
            self.T_cur = (step // self.steps_per_epoch) + (step % self.steps_per_epoch) / self.steps_per_epoch

        if self.T_cur >= self.T_i:
            self.T_cur = self.T_cur - self.T_i
            self.T_i = self.T_i * self.T_mult

        self.last_epoch = step

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group["lr"] = math.floor(lr * 1e9) / 1e9
                self.print_lr(self.verbose, i, lr, step)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def print_lr(self, is_verbose, group, lr, epoch=None):
        """Display the current learning rate."""
        if is_verbose:
            if epoch is None:
                print("Adjusting learning rate" " of group {} to {:.8e}.".format(group, lr))
            else:
                epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
                print("Epoch {}: adjusting learning rate" " of group {} to {:.8e}.".format(epoch_str, group, lr))


class CosineAnnealingHardRestarts(LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self,
        optimizer,
        T_0,
        steps_per_epoch=-1,
        T_mult=1,
        eta_min=0,
        last_step=-1,
        last_epoch=-1,
        verbose=False,
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        if last_epoch != -1 and last_step != -1:
            last_epoch = last_step
        elif last_epoch != -1 and last_step == -1:
            last_step = last_epoch
        self.T_0 = T_0
        self.steps_per_epoch = steps_per_epoch
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_step
        self.last_step = last_step
        super().__init__(optimizer=optimizer, last_epoch=last_step)

    def get_lr(self):
        lrs = [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]
        return lrs

    def step(self, step=None):
        # Check if the step argument is provided, if not, increment the last_step counter
        if step is None:
            step = self.last_step + 1

        # Calculate T_cur: This represents the current step within the current cycle
        # % operator ensures T_cur is always within the range of the current cycle
        self.T_cur = step % self.steps_per_epoch

        # Check if T_cur has reached the end of the current cycle (T_i)
        # If so, it's time for a warm restart
        if self.T_cur >= self.T_i:
            self.T_cur = 0  # Reset T_cur to start a new cycle
            self.T_i *= self.T_mult  # Increase the length of the next cycle

        # Update the last step with the current step
        self.last_step = step

        # This context manager ensures that the learning rate is updated correctly
        with _enable_get_lr_call(self):
            # Loop through each parameter group and its corresponding learning rate
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                # Update the learning rate for this parameter group
                # We use math.floor to truncate the precision to avoid numerical issues
                param_group["lr"] = math.floor(lr * 1e9) / 1e9
                # Print the updated learning rate if verbose mode is enabled
                self.print_lr(self.verbose, i, lr, step)

        # Update the last learning rate values for each parameter group
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def print_lr(self, is_verbose, group, lr, epoch=None):
        """Display the current learning rate."""
        if is_verbose:
            if epoch is None:
                print("Adjusting learning rate" " of group {} to {:.8e}.".format(group, lr))
            else:
                epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
                print("Epoch {}: adjusting learning rate" " of group {} to {:.8e}.".format(epoch_str, group, lr))


class Sine(LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_step=-1, verbose=False):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Sine learning rate expects positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")

        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_i = T_0
        self.T_cur = last_step
        self.last_epoch = last_step
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.verbose = verbose
        self._last_lr = self.base_lrs
        self.total_steps = 0  # Track total steps for a continuous wave
        super().__init__(optimizer=optimizer, last_epoch=last_step)

    def get_lr(self):
        # Calculate learning rates using a continuous sine function based on total steps
        lrs = [
            self.eta_min + (base_lr - self.eta_min) * (0.5 * (1 + math.sin(math.pi * self.total_steps / self.T_0)))
            for base_lr in self.base_lrs
        ]
        return lrs

    def step(self, step=None):
        if step is None:
            step = self.last_epoch + 1

        self.total_steps = step  # Use total steps instead of resetting per interval
        self.last_epoch = step
        for i, (param_group, lr) in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
            param_group["lr"] = math.floor(lr * 1e9) / 1e9
            self.print_lr(self.verbose, i, lr, step)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def print_lr(self, is_verbose, group, lr, epoch=None):
        if is_verbose:
            epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
            print(f"Epoch {epoch_str}: adjusting learning rate of group {group} to {lr:.8e}.")


from diffusers.optimization import get_scheduler

try:
    from simpletuner.helpers.models.flux import calculate_shift_flux
except Exception:  # pragma: no cover - optional dependency
    calculate_shift_flux = None  # type: ignore[assignment]


def apply_flow_schedule_shift(args, noise_scheduler, sigmas, noise):
    # Resolution-dependent shifting of timestep schedules as per section 5.3.2 of SD3 paper
    shift = None
    if args.flow_schedule_shift is not None and args.flow_schedule_shift > 0:
        # Static shift value for every resolution
        shift = args.flow_schedule_shift
    elif args.flow_schedule_auto_shift:
        # Resolution-dependent shift value calculation used by official Flux inference implementation
        image_seq_len = (noise.shape[-1] * noise.shape[-2]) // 4
        if calculate_shift_flux is None:
            raise RuntimeError("Flux flow schedule shift requires flux models and their dependencies.")
        mu = calculate_shift_flux(
            (noise.shape[-1] * noise.shape[-2]) // 4,
            noise_scheduler.config.base_image_seq_len,
            noise_scheduler.config.max_image_seq_len,
            noise_scheduler.config.base_shift,
            noise_scheduler.config.max_shift,
        )
        shift = math.exp(mu)
    if shift is not None:
        sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
    return sigmas


def get_lr_scheduler(
    args,
    optimizer,
    accelerator,
    logger,
    global_step: int,
    use_deepspeed_scheduler=False,
):
    if use_deepspeed_scheduler:
        logger.info("Using DeepSpeed learning rate scheduler")
        lr_scheduler = accelerate.utils.DummyScheduler(
            optimizer,
            total_num_steps=args.max_train_steps,
            warmup_num_steps=args.lr_warmup_steps,
        )
    elif args.lr_scheduler == "cosine_with_restarts":
        logger.info("Using Cosine with Restarts learning rate scheduler.")
        logger.warning(
            "cosine_with_restarts is currently misbehaving, and may not do what you expect. sine is recommended instead."
        )
        from simpletuner.helpers.training.custom_schedule import CosineAnnealingHardRestarts

        lr_scheduler = CosineAnnealingHardRestarts(
            optimizer=optimizer,
            T_0=int(args.lr_warmup_steps * accelerator.num_processes),
            T_mult=int(1),
            eta_min=float(args.lr_end),
            last_step=-1,
            verbose=os.environ.get("SIMPLETUNER_SCHEDULER_VERBOSE", "false").lower() == "true",
        )
    elif args.lr_scheduler == "sine":
        logger.info("Using Sine learning rate scheduler.")
        from simpletuner.helpers.training.custom_schedule import Sine

        lr_scheduler = Sine(
            optimizer=optimizer,
            T_0=int(args.lr_warmup_steps * accelerator.num_processes),
            T_mult=int(1),
            eta_min=float(args.lr_end),
            last_step=-1,
            verbose=os.environ.get("SIMPLETUNER_SCHEDULER_VERBOSE", "false").lower() == "true",
        )
    elif args.lr_scheduler == "cosine":
        logger.info("Using Cosine learning rate scheduler.")
        from simpletuner.helpers.training.custom_schedule import Cosine

        lr_scheduler = Cosine(
            optimizer=optimizer,
            T_0=int(args.lr_warmup_steps * accelerator.num_processes),
            T_mult=int(1),
            eta_min=float(args.lr_end),
            last_step=-1,
            verbose=os.environ.get("SIMPLETUNER_SCHEDULER_VERBOSE", "false").lower() == "true",
        )
    elif args.lr_scheduler == "polynomial":
        logger.info(f"Using Polynomial learning rate scheduler with last epoch {global_step - 2}.")
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            lr_end=args.lr_end,
            power=args.lr_power,
            last_epoch=global_step - 1,
        )
    else:
        logger.info(f"Using generic '{args.lr_scheduler}' learning rate scheduler.")
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    return lr_scheduler


# from huggingface/diffusers#8449 (author: @leffff)
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DISCLAIMER: This code is strongly influenced by https://github.com/leffff/euler-scheduler

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput


@dataclass
class FlowMatchingEulerSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep (which in flow-matching notation should be noted as
            `(x_{t+h})`). `prev_sample` should be used as next model input in the denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` (which in flow-matching notation should be noted as
            `(x_{1})`) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


def get_time_coefficients(timestep: torch.Tensor, ndim: int) -> torch.Tensor:
    return timestep.reshape((timestep.shape[0], *([1] * (ndim - 1))))


class FlowMatchingEulerScheduler(SchedulerMixin, ConfigMixin):
    """
    `FlowMatchingEulerScheduler` is a scheduler for training and inferencing Conditional Flow Matching models (CFMs).

    Flow Matching (FM) is a novel, simulation-free methodology for training Continuous Normalizing Flows (CNFs) by
    regressing vector fields of predetermined conditional probability paths, facilitating scalable training and
    efficient sample generation through the utilization of various probability paths, including Gaussian and
    Optimal Transport (OT) paths, thereby enhancing model performance and generalization capabilities

    Args:
        num_inference_steps (`int`, defaults to 100):
            The number of steps on inference.
    """

    @register_to_config
    def __init__(self, num_inference_steps: int = 100):
        self.timesteps = None
        self.num_inference_steps = None
        self.h = None

        if num_inference_steps is not None:
            self.set_timesteps(num_inference_steps)

    @staticmethod
    def add_noise(original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Add noise to the given sample

        Args:
            original_samples (`torch.Tensor`):
                The original sample that is to be noised
            noise (`torch.Tensor`):
                The noise that is used to noise the image
            timestep (`torch.Tensor`):
                Timestep used to create linear interpolation `x_t = t * x_1 + (1 - t) * x_0`.
                Where x_1 is a target distribution, x_0 is a source distribution and t (timestep) ∈ [0, 1]
        """

        t = get_time_coefficients(timestep, original_samples.ndim)

        noised_sample = t * original_samples + (1 - t) * noise

        return noised_sample

    def set_timesteps(self, num_inference_steps: int = 100) -> None:
        """
        Set number of inference steps (Euler intagration steps)

        Args:
            num_inference_steps (`int`, defaults to 100):
                The number of steps on inference.
        """

        self.num_inference_steps = num_inference_steps
        self.h = 1 / num_inference_steps
        self.timesteps = torch.arange(0, 1, self.h)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[FlowMatchingEulerSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                Timestep used to perform Euler Method `x_t = h * f(x_t, t) + x_{t-1}`.
                Where x_1 is a target distribution, x_0 is a source distribution and t (timestep) ∈ [0, 1]
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """

        step = FlowMatchingEulerSchedulerOutput(
            prev_sample=sample + self.h * model_output,
            pred_original_sample=sample + (1 - get_time_coefficients(timestep, model_output.ndim)) * model_output,
        )

        if return_dict:
            return step

        return (step.prev_sample,)

    @staticmethod
    def get_velocity(original_samples: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            original_samples (`torch.Tensor`):
                The original sample that is to be noised
            noise (`torch.Tensor`):
                The noise that is used to noise the image

        Returns:
            `torch.Tensor`
        """

        return original_samples - noise

    @staticmethod
    def scale_model_input(sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
         Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
         current timestep.

        Args:
            sample (`torch.Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """

        return sample


import logging
import math
import os
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint


class PFODESolverSD1x:
    def __init__(
        self,
        scheduler,
        t_initial=1,
        t_terminal=0,
    ) -> None:
        self.t_initial = t_initial
        self.t_terminal = t_terminal
        self.scheduler = scheduler

        train_step_terminal = 0
        train_step_initial = train_step_terminal + self.scheduler.config.num_train_timesteps  # 0+1000

        self.stepsize = (t_terminal - t_initial) / (train_step_terminal - train_step_initial)  # 1/1000

    def get_timesteps(self, t_start, t_end, num_steps):
        # (b,) -> (b,1)
        t_start = t_start[:, None]
        t_end = t_end[:, None]
        assert t_start.dim() == 2

        timepoints = torch.arange(0, num_steps, 1).expand(t_start.shape[0], num_steps).to(device=t_start.device)
        interval = (t_end - t_start) / (torch.ones([1], device=t_start.device) * num_steps)
        timepoints = t_start + interval * timepoints

        timesteps = (self.scheduler.num_train_timesteps - 1) + (
            timepoints - self.t_initial
        ) / self.stepsize  # correspondint to StableDiffusion indexing system, from 999 (t_init) -> 0 (dt)
        return timesteps.round().long()
        # return timesteps.floor().long()

    def solve(
        self,
        latents,
        unet,
        t_start,
        t_end,
        prompt_embeds,
        negative_prompt_embeds,
        guidance_scale=1.0,
        num_steps=2,
        num_windows=1,
    ):
        assert t_start.dim() == 1
        assert guidance_scale >= 1 and torch.all(torch.gt(t_start, t_end))

        do_classifier_free_guidance = True if guidance_scale > 1 else False
        bsz = latents.shape[0]

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        timestep_cond = None
        if unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(bsz)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=unet.config.time_cond_proj_dim
            ).to(device=latents.device, dtype=latents.dtype)

        timesteps = self.get_timesteps(t_start, t_end, num_steps).to(device=latents.device)
        timestep_interval = self.scheduler.config.num_train_timesteps // (num_windows * num_steps)

        # 7. Denoising loop
        with torch.no_grad():
            # for i in tqdm(range(num_steps)):
            for i in range(num_steps):
                print(f"Step {i} latents: mean={latents.mean().item()}, std={latents.std().item()}")
                t = torch.cat([timesteps[:, i]] * 2) if do_classifier_free_guidance else timesteps[:, i]
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # STEP: compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.step(noise_pred, timesteps[:, i].cpu(), latents, return_dict=False)[0]

                batch_timesteps = timesteps[:, i].cpu()
                prev_timestep = batch_timesteps - timestep_interval
                # prev_timestep = batch_timesteps - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

                alpha_prod_t = self.scheduler.alphas_cumprod[batch_timesteps]
                alpha_prod_t_prev = torch.zeros_like(alpha_prod_t)
                for ib in range(prev_timestep.shape[0]):
                    alpha_prod_t_prev[ib] = (
                        self.scheduler.alphas_cumprod[prev_timestep[ib]]
                        if prev_timestep[ib] >= 0
                        else self.scheduler.final_alpha_cumprod
                    )
                beta_prod_t = 1 - alpha_prod_t

                alpha_prod_t = alpha_prod_t.to(device=latents.device, dtype=latents.dtype)
                alpha_prod_t_prev = alpha_prod_t_prev.to(device=latents.device, dtype=latents.dtype)
                beta_prod_t = beta_prod_t.to(device=latents.device, dtype=latents.dtype)

                # 3. compute predicted original sample from predicted noise also called
                # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                if self.scheduler.config.prediction_type == "epsilon":
                    pred_original_sample = (latents - beta_prod_t[:, None, None, None] ** (0.5) * noise_pred) / alpha_prod_t[
                        :, None, None, None
                    ] ** (0.5)
                    pred_epsilon = noise_pred
                # elif self.scheduler.config.prediction_type == "sample":
                #     pred_original_sample = noise_pred
                #     pred_epsilon = (latents - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
                elif self.scheduler.config.prediction_type == "v_prediction":
                    pred_original_sample = (alpha_prod_t[:, None, None, None] ** 0.5) * latents - (
                        beta_prod_t[:, None, None, None] ** 0.5
                    ) * noise_pred
                    pred_epsilon = (alpha_prod_t[:, None, None, None] ** 0.5) * noise_pred + (
                        beta_prod_t[:, None, None, None] ** 0.5
                    ) * latents
                else:
                    raise ValueError(
                        f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
                        " `v_prediction`"
                    )

                pred_sample_direction = (1 - alpha_prod_t_prev[:, None, None, None]) ** (0.5) * pred_epsilon
                latents = alpha_prod_t_prev[:, None, None, None] ** (0.5) * pred_original_sample + pred_sample_direction

        return latents


class PFODESolverSDXL:
    def __init__(
        self,
        scheduler,
        t_initial=1,
        t_terminal=0,
    ) -> None:
        self.t_initial = t_initial
        self.t_terminal = t_terminal
        self.scheduler = scheduler

        train_step_terminal = 0
        train_step_initial = train_step_terminal + self.scheduler.config.num_train_timesteps  # 0+1000

        self.stepsize = (t_terminal - t_initial) / (train_step_terminal - train_step_initial)  # 1/1000

    def get_timesteps(self, t_start, t_end, num_steps):
        # (b,) -> (b,1)
        t_start = t_start[:, None]
        t_end = t_end[:, None]
        assert t_start.dim() == 2

        timepoints = torch.arange(0, num_steps, 1).expand(t_start.shape[0], num_steps).to(device=t_start.device)
        interval = (t_end - t_start) / (torch.ones([1], device=t_start.device) * num_steps)
        timepoints = t_start + interval * timepoints

        timesteps = (self.scheduler.num_train_timesteps - 1) + (
            timepoints - self.t_initial
        ) / self.stepsize  # correspondint to StableDiffusion indexing system, from 999 (t_init) -> 0 (dt)
        return timesteps.round().long()
        # return timesteps.floor().long()

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def solve(
        self,
        latents,
        unet,
        t_start,
        t_end,
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
        guidance_scale=1.0,
        num_steps=10,
        num_windows=4,
        resolution=1024,
    ):
        assert t_start.dim() == 1
        assert guidance_scale >= 1 and torch.all(torch.gt(t_start, t_end))
        dtype = latents.dtype
        device = latents.device
        bsz = latents.shape[0]
        do_classifier_free_guidance = True if guidance_scale > 1 else False

        add_text_embeds = pooled_prompt_embeds
        add_time_ids = torch.cat(
            # [self._get_add_time_ids((1024, 1024), (0, 0), (1024, 1024), dtype) for _ in range(bsz)]
            [self._get_add_time_ids((resolution, resolution), (0, 0), (resolution, resolution), dtype) for _ in range(bsz)]
        ).to(device)
        negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            # prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        timestep_cond = None
        if unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(bsz)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=unet.config.time_cond_proj_dim
            ).to(device=latents.device, dtype=latents.dtype)

        timesteps = self.get_timesteps(t_start, t_end, num_steps).to(device=latents.device)
        timestep_interval = self.scheduler.config.num_train_timesteps // (num_windows * num_steps)

        # 7. Denoising loop
        with torch.no_grad():
            # for i in tqdm(range(num_steps)):
            for i in range(num_steps):
                # expand the latents if we are doing classifier free guidance
                t = torch.cat([timesteps[:, i]] * 2) if do_classifier_free_guidance else timesteps[:, i]
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids,
                }
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # STEP: compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.step(noise_pred, timesteps[:, i].cpu(), latents, return_dict=False)[0]

                batch_timesteps = timesteps[:, i].cpu()
                prev_timestep = batch_timesteps - timestep_interval
                # prev_timestep = batch_timesteps - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

                alpha_prod_t = self.scheduler.alphas_cumprod[batch_timesteps]
                alpha_prod_t_prev = torch.zeros_like(alpha_prod_t)
                for ib in range(prev_timestep.shape[0]):
                    alpha_prod_t_prev[ib] = (
                        self.scheduler.alphas_cumprod[prev_timestep[ib]]
                        if prev_timestep[ib] >= 0
                        else self.scheduler.final_alpha_cumprod
                    )
                beta_prod_t = 1 - alpha_prod_t

                alpha_prod_t = alpha_prod_t.to(device=latents.device, dtype=latents.dtype)
                alpha_prod_t_prev = alpha_prod_t_prev.to(device=latents.device, dtype=latents.dtype)
                beta_prod_t = beta_prod_t.to(device=latents.device, dtype=latents.dtype)

                # 3. compute predicted original sample from predicted noise also called
                # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                if self.scheduler.config.prediction_type == "epsilon":
                    pred_original_sample = (latents - beta_prod_t[:, None, None, None] ** (0.5) * noise_pred) / alpha_prod_t[
                        :, None, None, None
                    ] ** (0.5)
                    pred_epsilon = noise_pred
                # elif self.scheduler.config.prediction_type == "sample":
                #     pred_original_sample = noise_pred
                #     pred_epsilon = (latents - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
                # elif self.scheduler.config.prediction_type == "v_prediction":
                #     pred_original_sample = (alpha_prod_t**0.5) * latents - (beta_prod_t**0.5) * noise_pred
                #     pred_epsilon = (alpha_prod_t**0.5) * noise_pred + (beta_prod_t**0.5) * latents
                else:
                    raise ValueError(
                        f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
                        " `v_prediction`"
                    )

                pred_sample_direction = (1 - alpha_prod_t_prev[:, None, None, None]) ** (0.5) * pred_epsilon
                latents = alpha_prod_t_prev[:, None, None, None] ** (0.5) * pred_original_sample + pred_sample_direction

        return latents


import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin
from diffusers.utils import BaseOutput


class Time_Windows:
    def __init__(self, t_initial=1, t_terminal=0, num_windows=4, precision=1.0 / 1000) -> None:
        assert t_terminal < t_initial
        time_windows = [1.0 * i / num_windows for i in range(1, num_windows + 1)][::-1]

        self.window_starts = time_windows  # [1.0, 0.75, 0.5, 0.25]
        self.window_ends = time_windows[1:] + [t_terminal]  # [0.75, 0.5, 0.25, 0]
        self.precision = precision

    def get_window(self, tp):
        idx = 0
        while idx < len(self.window_ends) - 1 and (tp - 0.1 * self.precision) <= self.window_ends[idx]:
            idx += 1
        if idx >= len(self.window_ends):
            idx = len(self.window_ends) - 1  # clamp to last window
        return self.window_starts[idx], self.window_ends[idx]

    def lookup_window(self, timepoint):
        if timepoint.dim() == 0:
            t_start, t_end = self.get_window(timepoint)
            t_start = torch.ones_like(timepoint) * t_start
            t_end = torch.ones_like(timepoint) * t_end
        else:
            t_start = torch.zeros_like(timepoint)
            t_end = torch.zeros_like(timepoint)
            bsz = timepoint.shape[0]
            for i in range(bsz):
                tp = timepoint[i]
                ts, te = self.get_window(tp)
                t_start[i] = ts
                t_end[i] = te
        return t_start, t_end


@dataclass
class PeRFlowSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class PeRFlowScheduler(SchedulerMixin, ConfigMixin):
    """
    `ReFlowScheduler` extends the denoising procedure introduced in denoising diffusion probabilistic models (DDPMs) with
    non-Markovian guidance.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        set_alpha_to_one (`bool`, defaults to `True`):
            Each diffusion step uses the alphas product value at that step and at the previous one. For the final step
            there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
            otherwise it uses the alpha value at step 0.
        prediction_type (`str`, defaults to `epsilon`, *optional*)
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        set_alpha_to_one: bool = False,
        prediction_type: str = "ddim_eps",
        t_noise: float = 1,
        t_clean: float = 0,
        num_time_windows=4,
    ):
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear" or prediction_type == "flow_matching":
            # For flow_matching, use a linear schedule like the FlowMatchEulerDiscreteScheduler
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
                    num_train_timesteps,
                    dtype=torch.float32,
                )
                ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        self.time_windows = Time_Windows(
            t_initial=t_noise,
            t_terminal=t_clean,
            num_windows=num_time_windows,
            precision=1.0 / num_train_timesteps,
        )

        # Store the prediction type for reference
        self.prediction_type = prediction_type

        logger.info(f"Loaded distillation scheduler {self.__class__.__name__} with prediction type {self.prediction_type}")

        assert prediction_type in ["epsilon", "diff_eps", "flow_matching"]

    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        return sample

    def get_window_alpha(self, timepoints):
        time_windows = self.time_windows
        num_train_timesteps = self.config.num_train_timesteps

        t_win_start, t_win_end = time_windows.lookup_window(timepoints)
        t_win_len = t_win_end - t_win_start
        t_interval = timepoints - t_win_start  # NOTE: negative value

        idx_start = (t_win_start * num_train_timesteps - 1).long()
        alphas_cumprod_start = self.alphas_cumprod[idx_start]

        idx_end = torch.clamp((t_win_end * num_train_timesteps - 1).long(), min=0)
        alphas_cumprod_end = self.alphas_cumprod[idx_end]

        alpha_cumprod_s_e = alphas_cumprod_start / alphas_cumprod_end
        gamma_s_e = alpha_cumprod_s_e**0.5

        return (
            t_win_start,
            t_win_end,
            t_win_len,
            t_interval,
            gamma_s_e,
            alphas_cumprod_start,
            alphas_cumprod_end,
        )

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """
        # if self.prediction_type == "flow_matching":
        #     # For flow_matching, use a simpler timestep schedule similar to FlowMatchEulerDiscreteScheduler
        #     timesteps = np.linspace(
        #         self.time_windows.window_starts[0],
        #         self.time_windows.window_ends[-1],
        #         num=num_inference_steps,
        #         endpoint=False,
        #     )
        #     self.timesteps = torch.from_numpy(
        #         (timesteps * self.config.num_train_timesteps).astype(np.int64)
        #     ).to(device)
        #     return

        # Original window-based timestep setting for other prediction types
        if num_inference_steps < self.config.num_time_windows:
            num_inference_steps = self.config.num_time_windows
            logger.debug(
                "num_inference_steps was below num_time_windows; using %s steps instead.",
                self.config.num_time_windows,
            )

        timesteps = []
        for i in range(self.config.num_time_windows):
            if i < num_inference_steps % self.config.num_time_windows:
                num_steps_cur_win = num_inference_steps // self.config.num_time_windows + 1
            else:
                num_steps_cur_win = num_inference_steps // self.config.num_time_windows

            t_s = self.time_windows.window_starts[i]
            t_e = self.time_windows.window_ends[i]
            timesteps_cur_win = np.linspace(t_s, t_e, num=num_steps_cur_win, endpoint=False)
            logger.debug("Timesteps in window %s: %s", i, timesteps_cur_win)
            timesteps.append(timesteps_cur_win)

        timesteps = np.concatenate(timesteps)

        self.timesteps = torch.from_numpy((timesteps * self.config.num_train_timesteps).astype(np.int64)).to(device)
        logger.debug("Perflow scheduler using timesteps: %s", self.timesteps)

    def _resolve_timestep_index(self, timestep: torch.Tensor) -> Tuple[int, torch.Tensor]:
        timestep = torch.as_tensor(timestep, device=self.timesteps.device, dtype=self.timesteps.dtype)
        idx = (self.timesteps == timestep).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            raise ValueError(f"Timestep {int(timestep.item())} was not found in the configured schedule.")
        i = int(idx[0].item())
        next_timestep = self.timesteps[i + 1] if i + 1 < len(self.timesteps) else self.timesteps[i]
        return i, torch.as_tensor(next_timestep, device=self.timesteps.device, dtype=self.timesteps.dtype)

    # 3. Add a helper method for specifically handling flow_matching

    def _flow_matching_step(
        self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Special step function specifically for flow_matching prediction type.
        Uses a simpler approach similar to FlowMatchEulerDiscreteScheduler.
        """
        _, next_timestep = self._resolve_timestep_index(timestep)

        dt = (next_timestep - torch.as_tensor(timestep, device=next_timestep.device, dtype=next_timestep.dtype)) / (
            self.config.num_train_timesteps
        )
        # Timesteps descend, so dt is expected to be non-positive for backward integration.
        dt = dt.to(sample.device, sample.dtype)

        # For flow_matching, model_output directly gives us velocity
        pred_velocity = model_output

        # Simple Euler step
        prev_sample = sample + dt * pred_velocity

        return prev_sample

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[PeRFlowSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddim.PeRFlowSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.PeRFlowSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddim.PeRFlowSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """

        timestep = torch.as_tensor(timestep, device=self.timesteps.device, dtype=self.timesteps.dtype)

        if self.config.prediction_type in ["epsilon", "ddim_eps"]:
            pred_epsilon = model_output
            t_c = timestep / self.config.num_train_timesteps
            t_s, t_e, _, c_to_s, _, alphas_cumprod_start, alphas_cumprod_end = self.get_window_alpha(t_c)

            lambda_s = (alphas_cumprod_end / alphas_cumprod_start) ** 0.5
            eta_s = (1 - alphas_cumprod_end) ** 0.5 - (
                alphas_cumprod_end / alphas_cumprod_start * (1 - alphas_cumprod_start)
            ) ** 0.5

            lambda_t = (lambda_s * (t_e - t_s)) / (lambda_s * (t_c - t_s) + (t_e - t_c))
            eta_t = (eta_s * (t_e - t_c)) / (lambda_s * (t_c - t_s) + (t_e - t_c))

            pred_win_end = lambda_t * sample + eta_t * pred_epsilon
            pred_velocity = (pred_win_end - sample) / (t_e - (t_s + c_to_s))

        elif self.config.prediction_type == "diff_eps":
            pred_epsilon = model_output
            t_c = timestep / self.config.num_train_timesteps
            t_s, t_e, _, c_to_s, gamma_s_e, _, _ = self.get_window_alpha(t_c)

            lambda_s = 1 / gamma_s_e
            eta_s = -1 * (1 - gamma_s_e**2) ** 0.5 / gamma_s_e

            lambda_t = (lambda_s * (t_e - t_s)) / (lambda_s * (t_c - t_s) + (t_e - t_c))
            eta_t = (eta_s * (t_e - t_c)) / (lambda_s * (t_c - t_s) + (t_e - t_c))

            pred_win_end = lambda_t * sample + eta_t * pred_epsilon
            pred_velocity = (pred_win_end - sample) / (t_e - (t_s + c_to_s))

        elif self.config.prediction_type in ["flow_matching", "velocity"]:
            pred_velocity = model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of 'epsilon', 'ddim_eps', 'diff_eps', 'flow_matching', or 'velocity'."
            )

        _, next_timestep = self._resolve_timestep_index(timestep)
        dt = (next_timestep - timestep) / self.config.num_train_timesteps
        # Timesteps descend, so dt is expected to be non-positive for backward integration.
        dt = dt.to(sample.device, sample.dtype)

        prev_sample = sample + dt * pred_velocity

        if not return_dict:
            return (prev_sample,)
        return PeRFlowSchedulerOutput(prev_sample=prev_sample, pred_original_sample=None)

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device) - 1  # indexing from 0

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps

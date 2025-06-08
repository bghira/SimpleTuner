from torch.optim.lr_scheduler import LambdaLR
import torch
import math
import accelerate
import os
import logging
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


def segmented_timestep_selection(
    actual_num_timesteps, bsz, weights, config, use_refiner_range: bool = False
):
    # Determine the range of timesteps to use
    num_timesteps = actual_num_timesteps
    if use_refiner_range or config.refiner_training:
        if config.refiner_training_invert_schedule:
            # Inverted schedule calculation: we start from the last timestep and move downwards
            start_timestep = (
                actual_num_timesteps - 1
            )  # Start from the last timestep, e.g., 999
            # Calculate the end of the range based on the inverse of the training strength
            end_timestep = int(config.refiner_training_strength * actual_num_timesteps)
        else:
            # Normal refiner training schedule
            start_timestep = (
                int(actual_num_timesteps * config.refiner_training_strength) - 1
            )
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
        raise ValueError(
            f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})"
        )

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
            raise ValueError(
                f"Cosine learning rate expects to use warmup steps as its interval. Expected positive integer T_0, but got {T_0}"
            )
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
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
            for base_lr in self.base_lrs
        ]
        return lrs

    def step(self, step=None):
        if step is None and self.last_epoch < 0:
            step = 0

        if step is None:
            step = self.last_epoch + 1
            self.T_cur = (step // self.steps_per_epoch) + (
                step % self.steps_per_epoch
            ) / self.steps_per_epoch
        else:
            self.T_cur = (step // self.steps_per_epoch) + (
                step % self.steps_per_epoch
            ) / self.steps_per_epoch

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
                print(
                    "Adjusting learning rate"
                    " of group {} to {:.8e}.".format(group, lr)
                )
            else:
                epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
                print(
                    "Epoch {}: adjusting learning rate"
                    " of group {} to {:.8e}.".format(epoch_str, group, lr)
                )


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
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
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
                print(
                    "Adjusting learning rate"
                    " of group {} to {:.8e}.".format(group, lr)
                )
            else:
                epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
                print(
                    "Epoch {}: adjusting learning rate"
                    " of group {} to {:.8e}.".format(epoch_str, group, lr)
                )


class Sine(LRScheduler):
    def __init__(
        self, optimizer, T_0, T_mult=1, eta_min=0, last_step=-1, verbose=False
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(
                f"Sine learning rate expects positive integer T_0, but got {T_0}"
            )
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
            self.eta_min
            + (base_lr - self.eta_min)
            * (0.5 * (1 + math.sin(math.pi * self.total_steps / self.T_0)))
            for base_lr in self.base_lrs
        ]
        return lrs

    def step(self, step=None):
        if step is None:
            step = self.last_epoch + 1

        self.total_steps = step  # Use total steps instead of resetting per interval
        self.last_epoch = step
        for i, (param_group, lr) in enumerate(
            zip(self.optimizer.param_groups, self.get_lr())
        ):
            param_group["lr"] = math.floor(lr * 1e9) / 1e9
            self.print_lr(self.verbose, i, lr, step)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def print_lr(self, is_verbose, group, lr, epoch=None):
        if is_verbose:
            epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
            print(
                f"Epoch {epoch_str}: adjusting learning rate of group {group} to {lr:.8e}."
            )


from diffusers.optimization import get_scheduler
from helpers.models.flux import calculate_shift_flux


def apply_flow_schedule_shift(args, noise_scheduler, sigmas, noise):
    # Resolution-dependent shifting of timestep schedules as per section 5.3.2 of SD3 paper
    shift = None
    if args.flow_schedule_shift is not None and args.flow_schedule_shift > 0:
        # Static shift value for every resolution
        shift = args.flow_schedule_shift
    elif args.flow_schedule_auto_shift:
        # Resolution-dependent shift value calculation used by official Flux inference implementation
        image_seq_len = (noise.shape[-1] * noise.shape[-2]) // 4
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
        from helpers.training.custom_schedule import CosineAnnealingHardRestarts

        lr_scheduler = CosineAnnealingHardRestarts(
            optimizer=optimizer,
            T_0=int(args.lr_warmup_steps * accelerator.num_processes),
            T_mult=int(1),
            eta_min=float(args.lr_end),
            last_step=-1,
            verbose=os.environ.get("SIMPLETUNER_SCHEDULER_VERBOSE", "false").lower()
            == "true",
        )
    elif args.lr_scheduler == "sine":
        logger.info("Using Sine learning rate scheduler.")
        from helpers.training.custom_schedule import Sine

        lr_scheduler = Sine(
            optimizer=optimizer,
            T_0=int(args.lr_warmup_steps * accelerator.num_processes),
            T_mult=int(1),
            eta_min=float(args.lr_end),
            last_step=-1,
            verbose=os.environ.get("SIMPLETUNER_SCHEDULER_VERBOSE", "false").lower()
            == "true",
        )
    elif args.lr_scheduler == "cosine":
        logger.info("Using Cosine learning rate scheduler.")
        from helpers.training.custom_schedule import Cosine

        lr_scheduler = Cosine(
            optimizer=optimizer,
            T_0=int(args.lr_warmup_steps * accelerator.num_processes),
            T_mult=int(1),
            eta_min=float(args.lr_end),
            last_step=-1,
            verbose=os.environ.get("SIMPLETUNER_SCHEDULER_VERBOSE", "false").lower()
            == "true",
        )
    elif args.lr_scheduler == "polynomial":
        logger.info(
            f"Using Polynomial learning rate scheduler with last epoch {global_step - 2}."
        )
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
from typing import Tuple, Optional, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.schedulers.scheduling_utils import (
    SchedulerMixin,
)


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
    def add_noise(
        original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
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
            pred_original_sample=sample
            + (1 - get_time_coefficients(timestep, model_output.ndim)) * model_output,
        )

        if return_dict:
            return step

        return (step.prev_sample,)

    @staticmethod
    def get_velocity(
        original_samples: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
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
    def scale_model_input(
        sample: torch.Tensor, timestep: Optional[int] = None
    ) -> torch.Tensor:
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

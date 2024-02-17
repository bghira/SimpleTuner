from torch.optim.lr_scheduler import LambdaLR
import torch, math, warnings
from torch.optim.lr_scheduler import LRScheduler


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
        return ValueError(
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
        super().__init__(optimizer, last_step, verbose)

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
        super().__init__(optimizer, last_step, verbose)

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
        self,
        optimizer,
        T_0,
        steps_per_epoch=-1,
        T_mult=1,
        eta_min=0,
        last_step=-1,
        verbose=False,
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(
                f"Sine learning rate expects to use warmup steps as its interval. Expected positive integer T_0, but got {T_0}"
            )
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")

        self.T_0 = T_0
        self.steps_per_epoch = steps_per_epoch
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_step
        super(Sine, self).__init__(optimizer, last_step, verbose)

    def get_lr(self):
        lrs = [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 - math.cos(math.pi / 2 + math.pi * self.T_cur / self.T_i))
            / 2
            for base_lr in self.base_lrs
        ]
        return lrs

    def step(self, step=None):
        if step is None:
            step = self.last_epoch + 1
        self.T_cur = step % self.T_i

        if step != 0 and step % self.T_i == 0:
            self.T_i *= self.T_mult

        self.last_epoch = step
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

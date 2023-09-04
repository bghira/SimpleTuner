from torch.optim.lr_scheduler import LambdaLR
import torch


def get_polynomial_decay_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float=1e-7,
    power: float=1.0,
    last_epoch: int=-1,
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

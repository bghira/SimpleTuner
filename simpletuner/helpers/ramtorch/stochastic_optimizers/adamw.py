import torch
from torch.optim import Optimizer


# @torch.compile
def copy_stochastic_(target: torch.Tensor, source: torch.Tensor, seed=0):
    # thanks to Nerogar for fast stochastic pytorch implementation
    # https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
    with torch.no_grad():
        generator = torch.Generator(device=source.device)
        generator.manual_seed(seed)

        # create a random 16 bit integer using torch.randint with explicit shape
        result = torch.randint(
            low=0,
            high=(1 << 16),
            size=source.shape,
            dtype=torch.int32,
            device=source.device,
            generator=generator,
        )

        # add the random number to the lower 16 bit of the mantissa
        result.add_(source.view(dtype=torch.int32))

        # mask off the lower 16 bit of the mantissa
        result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

        # copy the higher 16 bit into the target tensor
        target.copy_(result.view(dtype=torch.float32), non_blocking=True)


class AdamW(Optimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.9, 0.999)).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        centralization (float):
            center model grad (default: 0).
        chunk_size (int):
            Number of parameters to process before synchronizing.
            A larger chunk size can improve performance but uses more
            temporary GPU memory. (default: 16)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        centralization=0,
        chunk_size=64,
        dtype=torch.bfloat16,
        storage_device="cpu",
        # device=None
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            centralization=centralization,
        )
        super(AdamW, self).__init__(params, defaults)

        self.chunk_size = chunk_size
        self.optim_state_dtype = dtype
        self.optim_state_device = storage_device
        # self.device = device

        # Initialize state in pinned memory for faster async transfers
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if not state:
                    state["step"] = 0

                    if self.optim_state_device == "cpu":
                        state["ema"] = torch.zeros_like(p.data, dtype=dtype, device=self.optim_state_device).pin_memory()
                        state["ema_squared"] = torch.zeros_like(
                            p.data, dtype=dtype, device=self.optim_state_device
                        ).pin_memory()
                    else:
                        state["ema"] = torch.zeros_like(p.data, dtype=dtype, device=self.optim_state_device)
                        state["ema_squared"] = torch.zeros_like(p.data, dtype=dtype, device=self.optim_state_device)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # Enumerate to keep track of the parameter index for chunking
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Compass does not support sparse gradients")

                state = self.state[p]
                device = p.device

                # Lazy state initialization
                if not state:
                    state["step"] = 0
                    if self.optim_state_device == "cpu":
                        state["ema"] = torch.zeros_like(
                            p.data,
                            dtype=self.optim_state_dtype,
                            device=self.optim_state_device,
                        ).pin_memory()
                        state["ema_squared"] = torch.zeros_like(
                            p.data,
                            dtype=self.optim_state_dtype,
                            device=self.optim_state_device,
                        ).pin_memory()
                    else:
                        state["ema"] = torch.zeros_like(
                            p.data,
                            dtype=self.optim_state_dtype,
                            device=self.optim_state_device,
                        )
                        state["ema_squared"] = torch.zeros_like(
                            p.data,
                            dtype=self.optim_state_dtype,
                            device=self.optim_state_device,
                        )

                # ========= Asynchronously queue all operations for this parameter =========
                # Determine target GPU device for computation
                if device.type == "cpu":
                    # If param is on CPU, use default GPU for computation
                    compute_device = torch.cuda.current_device()
                else:
                    # If param is on GPU, use its device
                    compute_device = device

                # 1. Queue Host-to-Device copy
                ema_fp32 = state["ema"].to(compute_device, non_blocking=True, dtype=torch.float32)
                ema_squared_fp32 = state["ema_squared"].to(compute_device, non_blocking=True, dtype=torch.float32)

                grad = grad.to(torch.float32).to(compute_device, non_blocking=True)
                p_fp32 = p.to(compute_device, dtype=torch.float32, non_blocking=True)

                # 2. Queue computations on the GPU
                beta1, beta2 = group["betas"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                centralization = group["centralization"]
                state["step"] += 1

                if centralization != 0:
                    grad.sub_(grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization))

                bias_correction = 1 - beta1 ** state["step"]
                bias_correction_sqrt = (1 - beta2 ** state["step"]) ** (1 / 2)
                step_size = lr / bias_correction

                ema_fp32.mul_(beta1).add_(grad, alpha=1 - beta1)
                ema_squared_fp32.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (ema_squared_fp32.sqrt() / bias_correction_sqrt).add_(group["eps"])

                if weight_decay != 0:
                    p_fp32.data.mul_(1 - step_size * weight_decay)

                p_fp32.data.addcdiv_(ema_fp32, denom, value=-step_size)

                # 3. Queue Device-to-Host copy
                # only use stochastic rounding if using bf16
                if device.type == "cpu":
                    if p.dtype == torch.bfloat16:
                        copy_stochastic_(p.data, p_fp32, state["step"] + 42)
                    else:
                        p.data.copy_(p_fp32)
                else:
                    # Original GPU path
                    if p.dtype == torch.bfloat16:
                        copy_stochastic_(p, p_fp32, state["step"] + 42)
                    else:
                        p.data.copy_(p_fp32, non_blocking=True)
                if self.optim_state_dtype == torch.bfloat16:
                    copy_stochastic_(state["ema"], ema_fp32, state["step"] + 69)
                    copy_stochastic_(state["ema_squared"], ema_squared_fp32, state["step"] + 420)
                else:
                    state["ema"].copy_(ema_fp32, non_blocking=True)
                    state["ema_squared"].copy_(ema_squared_fp32, non_blocking=True)

                # ========= Check if we need to synchronize =========
                # We synchronize after processing a chunk of parameters.
                # The (i + 1) ensures we sync after the 1st, 2nd, ... chunk.
                if (i + 1) % self.chunk_size == 0:
                    torch.cuda.synchronize()

            # Final synchronization to handle the last partial chunk
            # This ensures all operations for the group are complete before exiting.
            torch.cuda.synchronize()

        return loss

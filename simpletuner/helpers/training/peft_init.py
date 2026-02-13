import torch


def _to_dense_tensor(tensor: torch.Tensor) -> torch.Tensor:
    # TorchAO quantized-training tensor subclasses expose dequantize(), and many
    # reduction/random ops are not implemented via __torch_dispatch__.
    if hasattr(tensor, "dequantize"):
        dequantized = tensor.dequantize()
        if isinstance(dequantized, torch.Tensor):
            return dequantized
    return tensor


def approximate_normal_tensor(inp, target, scale=1.0):
    device = inp.device
    dense_inp = _to_dense_tensor(inp)
    dense_target = _to_dense_tensor(target)
    working_dtype = dense_target.dtype if dense_target.is_floating_point() else torch.float32
    tensor = torch.randn(dense_target.shape, device=device, dtype=working_dtype)
    desired_norm = dense_inp.norm().to(device=device, dtype=working_dtype)
    desired_mean = dense_inp.mean().to(device=device, dtype=working_dtype)
    desired_std = dense_inp.std().to(device=device, dtype=working_dtype)

    current_norm = tensor.norm()
    tensor = tensor * (desired_norm / current_norm)
    current_std = tensor.std()
    tensor = tensor * (desired_std / current_std)
    tensor = tensor - tensor.mean() + desired_mean
    tensor.mul_(scale)

    target.copy_(tensor.to(dtype=target.dtype))


def init_lokr_network_with_perturbed_normal(lycoris, scale=1e-3):
    with torch.no_grad():
        for lora in lycoris.loras:
            lora.lokr_w1.fill_(1.0)
            approximate_normal_tensor(lora.org_weight, lora.lokr_w2, scale=scale)

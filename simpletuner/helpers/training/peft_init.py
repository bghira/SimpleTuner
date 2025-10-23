import torch


def approximate_normal_tensor(inp, target, scale=1.0):
    device = inp.device
    tensor = torch.randn_like(target).to(device)
    desired_norm = inp.norm().to(device)
    desired_mean = inp.mean().to(device)
    desired_std = inp.std().to(device)

    current_norm = tensor.norm()
    tensor = tensor * (desired_norm / current_norm)
    current_std = tensor.std()
    tensor = tensor * (desired_std / current_std)
    tensor = tensor - tensor.mean() + desired_mean
    tensor.mul_(scale)

    target.copy_(tensor)


def init_lokr_network_with_perturbed_normal(lycoris, scale=1e-3):
    with torch.no_grad():
        for lora in lycoris.loras:
            lora.lokr_w1.fill_(1.0)
            approximate_normal_tensor(lora.org_weight, lora.lokr_w2, scale=scale)

# From Diffusers repository: examples/research_projects/onnxruntime/text_to_image/train_text_to_image.py


def compute_snr(timesteps, noise_scheduler, use_soft_min: bool = False, sigma_data=1.0):
    """
    Computes SNR using two different methods based on the `use_soft_min` flag.

    Args:
        timesteps (torch.Tensor): The timesteps at which SNR is computed.
        noise_scheduler (NoiseScheduler): An object that contains the alpha_cumprod values.
        use_soft_min (bool): If True, use the _weighting_soft_min_snr method to compute SNR.
        sigma_data (torch.Tensor or None): The standard deviation of the data used in the soft min weighting method.

    Returns:
        torch.Tensor: The computed SNR values.
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Choose the method to compute SNR
    if use_soft_min:
        if sigma_data is None:
            raise ValueError("sigma_data must be provided when using soft min SNR calculation.")
        snr = (sigma * sigma_data) ** 2 / (sigma**2 + sigma_data**2) ** 2
    else:
        # Default SNR computation
        snr = (alpha / sigma) ** 2

    return snr

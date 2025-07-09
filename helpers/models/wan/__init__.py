import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

"""
This module provides utility functions for normalizing latent representations (WAN latents)
and computing the posterior distribution for a variational autoencoder (VAE) using a diagonal
Gaussian distribution. The posterior is computed from latent tensors that encode both the mean
(mu) and log-variance (logvar) parameters of the Gaussian.

Functions:
    normalize_wan_latents(latents, latents_mean, latents_std):
        Normalizes the latent tensor using given mean and standard deviation values.

    compute_wan_posterior(latents, latents_mean, latents_std):
        Computes the posterior distribution from the latent tensor by splitting it into mean
        and log-variance components, normalizing each, and then constructing a DiagonalGaussianDistribution.
"""


def normalize_wan_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
) -> torch.Tensor:
    """
    Normalize latent representations (WAN latents) using provided mean and standard deviation.

    This function reshapes the provided mean and standard deviation tensors so they can be broadcasted
    across the dimensions of the `latents` tensor. It then normalizes the latents by subtracting
    the mean and scaling by the standard deviation.

    Args:
        latents (torch.Tensor): The input latent tensor to be normalized.
                                  Expected shape is (batch_size, channels, ...).
        latents_mean (torch.Tensor): The mean tensor for normalization.
                                     Expected to have shape (channels,).
        latents_std (torch.Tensor): The standard deviation tensor for normalization.
                                    Expected to have shape (channels,).

    Returns:
        torch.Tensor: The normalized latent tensor, with the same shape as the input `latents`.
    """
    # Reshape latents_mean to (1, channels, 1, 1, 1) to allow broadcasting across batch and spatial dimensions.
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)

    # Reshape latents_std similarly to ensure it is broadcastable and resides on the same device as latents.
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)

    # Convert latents to float (if not already) and apply normalization:
    # Subtract the mean and then multiply by the standard deviation.
    # print(f"Shapes: {latents.shape}, {latents_mean.shape}, {latents_std.shape}")
    latents = ((latents.float() - latents_mean) * latents_std).to(latents)

    return latents


def compute_wan_posterior(
    latents: torch.Tensor, latents_mean: list, latents_std: list
) -> DiagonalGaussianDistribution:
    """
    Compute the WAN posterior distribution from latent representations.

    This function splits the input `latents` tensor along the channel dimension into two halves:
    one for the mean (mu) and one for the log-variance (logvar) of a Gaussian distribution.
    Both components are normalized using the `normalize_wan_latents` function. The normalized
    parameters are then concatenated and used to instantiate a DiagonalGaussianDistribution,
    representing the approximate posterior q(z|x) in a VAE framework.

    Args:
        latents (torch.Tensor): A tensor containing concatenated latent representations.
                                It is assumed that the first half of the channels corresponds
                                to the mean (mu) and the second half corresponds to the log-variance (logvar).
        latents_mean (torch.Tensor): The mean tensor for normalization.
        latents_std (torch.Tensor): The standard deviation tensor for normalization.

    Returns:
        DiagonalGaussianDistribution: A diagonal Gaussian distribution representing the
                                      computed posterior distribution.
    """
    latents_mean = torch.tensor(latents_mean)
    latents_std = 1.0 / torch.tensor(latents_std)

    # Split the concatenated tensor into mu and logvar
    mu, logvar = torch.chunk(latents, 2, dim=1)

    # Normalize each component separately
    mu = normalize_wan_latents(mu, latents_mean, latents_std)
    logvar = normalize_wan_latents(logvar, latents_mean, latents_std)

    # Concatenate back
    latents = torch.cat([mu, logvar], dim=1)

    # Construct the posterior distribution
    posterior = DiagonalGaussianDistribution(latents)

    return posterior

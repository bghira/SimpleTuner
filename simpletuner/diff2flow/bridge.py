import torch


class DiffusionToFlowBridge:
    """
    Lightweight adapter that converts diffusion model predictions (eps or v) into
    flow-matching vector fields without changing the underlying UNet.
    """

    def __init__(self, *, alphas_cumprod: torch.Tensor):
        if alphas_cumprod is None or not torch.is_tensor(alphas_cumprod):
            raise ValueError("alphas_cumprod tensor is required to build a DiffusionToFlowBridge.")

        self.register_buffers(alphas_cumprod.to(dtype=torch.float32))

    def register_buffers(self, alphas_cumprod: torch.Tensor):
        alphas = alphas_cumprod
        self.num_timesteps = alphas.shape[0]

        self.sqrt_alphas_cumprod = torch.sqrt(alphas)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas - 1.0)

    def _extract(self, arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape) -> torch.Tensor:
        if arr.device != timesteps.device:
            arr = arr.to(device=timesteps.device)
        values = arr[timesteps.long()]
        while values.ndim < len(broadcast_shape):
            values = values[..., None]
        return values

    def flow_target(self, clean_latents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return noise - clean_latents

    def _flow_from_v(self, v_pred: torch.Tensor, noisy_latents: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, noisy_latents.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, noisy_latents.shape)

        z_pred = sqrt_alpha * noisy_latents - sqrt_one_minus * v_pred
        eps_pred = sqrt_alpha * v_pred + sqrt_one_minus * noisy_latents
        # Align with SimpleTuner flow target: noise - latents
        return eps_pred - z_pred

    def _flow_from_eps(self, eps_pred: torch.Tensor, noisy_latents: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas_cumprod, timesteps, noisy_latents.shape)
        sqrt_recipm1_alpha = self._extract(self.sqrt_recipm1_alphas_cumprod, timesteps, noisy_latents.shape)

        z_pred = sqrt_recip_alpha * noisy_latents - sqrt_recipm1_alpha * eps_pred
        # Align with SimpleTuner flow target: noise - latents
        return eps_pred - z_pred

    def to(self, device=None, dtype=None):
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        for name in [
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "sqrt_recip_alphas_cumprod",
            "sqrt_recipm1_alphas_cumprod",
        ]:
            tensor = getattr(self, name, None)
            if tensor is not None:
                setattr(self, name, tensor.to(**kwargs))
        return self

    def timesteps_to_sigma(self, timesteps: torch.Tensor, broadcast_shape=None) -> torch.Tensor:
        """
        Convert diffusion timesteps to flow-equivalent sigma (noise fraction).

        For diffusion: x_t = sqrt(α_t)*x_0 + sqrt(1-α_t)*ε
        For flow:      x_t = (1-σ)*x_0 + σ*ε

        The flow-equivalent sigma is sqrt(1-α_t) / (sqrt(α_t) + sqrt(1-α_t)),
        which normalizes so that the coefficients sum to 1.
        """
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, (timesteps.shape[0],))
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, (timesteps.shape[0],))
        # Normalize to get flow-equivalent sigma in [0, 1]
        sigma = sqrt_one_minus / (sqrt_alpha + sqrt_one_minus)
        if broadcast_shape is not None:
            while sigma.ndim < len(broadcast_shape):
                sigma = sigma[..., None]
        return sigma

    def sigma_to_timesteps(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Convert flow-equivalent sigma back to the nearest diffusion timestep.

        This is the inverse of timesteps_to_sigma. Given σ = sqrt(1-α) / (sqrt(α) + sqrt(1-α)),
        we find the timestep t whose alpha_cumprod[t] yields the closest sigma.

        Uses binary search over the precomputed sigma schedule for efficiency.
        """
        # Flatten sigma for lookup, remember original shape
        orig_shape = sigma.shape
        sigma_flat = sigma.view(-1)

        # Precompute sigma schedule if not cached
        if not hasattr(self, "_sigma_schedule"):
            all_timesteps = torch.arange(self.num_timesteps, device=self.sqrt_alphas_cumprod.device)
            sqrt_alpha = self.sqrt_alphas_cumprod
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod
            self._sigma_schedule = sqrt_one_minus / (sqrt_alpha + sqrt_one_minus)

        sigma_schedule = self._sigma_schedule.to(device=sigma.device, dtype=sigma.dtype)

        # sigma_schedule is monotonically increasing (sigma=0 at t=0, sigma→1 at t=max)
        # Use searchsorted to find nearest timestep
        # searchsorted finds insertion point; we want nearest, so check both sides
        indices = torch.searchsorted(sigma_schedule, sigma_flat.contiguous())
        indices = indices.clamp(0, self.num_timesteps - 1)

        # Check if the previous index is closer
        prev_indices = (indices - 1).clamp(0, self.num_timesteps - 1)
        curr_diff = torch.abs(sigma_schedule[indices] - sigma_flat)
        prev_diff = torch.abs(sigma_schedule[prev_indices] - sigma_flat)
        use_prev = prev_diff < curr_diff
        timesteps = torch.where(use_prev, prev_indices, indices)

        # Reshape to match input (but without spatial dims - just batch)
        if len(orig_shape) > 1:
            timesteps = timesteps.view(orig_shape[0], -1)[:, 0]
        return timesteps

    def prediction_to_flow(
        self,
        prediction: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        prediction_type: str,
    ) -> torch.Tensor:
        """
        Convert a diffusion prediction into a flow-matching vector field.
        """
        if prediction_type in ("v_prediction", "vpred", "v"):
            return self._flow_from_v(prediction, noisy_latents, timesteps)
        if prediction_type in ("epsilon", "eps"):
            return self._flow_from_eps(prediction, noisy_latents, timesteps)
        raise ValueError(f"Unsupported prediction_type for diff2flow bridge: {prediction_type}")

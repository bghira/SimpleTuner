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

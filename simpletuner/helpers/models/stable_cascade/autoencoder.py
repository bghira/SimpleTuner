import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

try:
    from torchvision import models
except ImportError as torchvision_import_error:  # pragma: no cover - runtime guard
    models = None
    TORCHVISION_IMPORT_ERROR = torchvision_import_error
else:
    TORCHVISION_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


class StableCascadeStageCAutoencoder(nn.Module):
    """
    EfficientNet-based image encoder used by Stable Cascade stage C training.
    """

    def __init__(self, latent_channels: int = 16, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        if TORCHVISION_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Stable Cascade stage C requires torchvision to load the EfficientNet encoder."
            ) from TORCHVISION_IMPORT_ERROR

        weights = getattr(models, "EfficientNet_V2_S_Weights", None)
        if weights is not None:
            backbone_weights = weights.IMAGENET1K_V1
        else:
            backbone_weights = "DEFAULT"

        backbone = models.efficientnet_v2_s(weights=backbone_weights)
        self.backbone = backbone.features.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        self.mapper = nn.Sequential(
            nn.Conv2d(1280, latent_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(latent_channels, affine=False),
        ).eval()
        for parameter in self.mapper.parameters():
            parameter.requires_grad_(False)

        self.register_buffer(
            "_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        self.latent_dtype = dtype if dtype is not None else torch.float32
        self.config = torch.nn.Module()  # lightweight container
        self.config.scaling_factor = 1.0
        self.config.shift_factor = None

    @property
    def dtype(self) -> torch.dtype:
        """
        Mirror the interface of diffusers autoencoders so training helpers can read the active dtype.
        """
        return self.latent_dtype

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "StableCascadeStageCAutoencoder":
        latent_channels = kwargs.get("latent_channels", 16)
        instance = cls(latent_channels=latent_channels, dtype=torch_dtype or torch.float32)

        if pretrained_model_name_or_path:
            checkpoint_path = Path(pretrained_model_name_or_path)
            if checkpoint_path.exists():
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                missing, unexpected = instance.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.warning("Missing keys when loading EfficientNet encoder: %s", missing)
                if unexpected:
                    logger.warning("Unexpected keys when loading EfficientNet encoder: %s", unexpected)
        return instance

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:

        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        if not torch.is_floating_point(pixel_values):
            pixel_values = pixel_values.float()

        # Convert [-1, 1] inputs back to [0, 1] if needed.
        if pixel_values.min() < 0.0:
            pixel_values = (pixel_values + 1.0) / 2.0

        pixel_values = pixel_values.clamp(0.0, 1.0)
        pixel_values = (pixel_values - self._mean) / self._std

        with torch.no_grad():
            features = self.backbone(pixel_values)
            latents = self.mapper(features)
        return latents.to(self.latent_dtype)


class StableCascadeAutoencoderOutput:
    def __init__(self, sample: torch.Tensor) -> None:
        self._sample = sample

    @property
    def sample(self) -> torch.Tensor:
        return self._sample

    def __getitem__(self, key: str):
        if key == "sample":
            return self.sample
        raise KeyError(key)

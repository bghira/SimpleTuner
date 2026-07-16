import json
import math
import os
from typing import Optional

import safetensors
import safetensors.torch
import torch
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import hf_hub_download


class LTX2PixelShuffleND(torch.nn.Module):
    def __init__(self, dims: int, upscale_factors: tuple[int, int, int] = (2, 2, 2)):
        super().__init__()
        if dims not in (1, 2, 3):
            raise ValueError("dims must be 1, 2, or 3")
        self.dims = dims
        self.upscale_factors = upscale_factors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dims == 3:
            return rearrange(
                x,
                "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
                p1=self.upscale_factors[0],
                p2=self.upscale_factors[1],
                p3=self.upscale_factors[2],
            )
        if self.dims == 2:
            return rearrange(
                x,
                "b (c p1 p2) h w -> b c (h p1) (w p2)",
                p1=self.upscale_factors[0],
                p2=self.upscale_factors[1],
            )
        return rearrange(x, "b (c p1) f h w -> b c (f p1) h w", p1=self.upscale_factors[0])


class LTX2UpsamplerResBlock(torch.nn.Module):
    def __init__(self, channels: int, mid_channels: Optional[int] = None, dims: int = 3):
        super().__init__()
        if mid_channels is None:
            mid_channels = channels

        conv = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d
        self.conv1 = conv(channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = torch.nn.GroupNorm(32, mid_channels)
        self.conv2 = conv(mid_channels, channels, kernel_size=3, padding=1)
        self.norm2 = torch.nn.GroupNorm(32, channels)
        self.activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.activation(x + residual)


class LTX2BlurDownsample(torch.nn.Module):
    def __init__(self, dims: int, stride: int, kernel_size: int = 5) -> None:
        super().__init__()
        if dims not in (2, 3):
            raise ValueError("dims must be 2 or 3")
        if stride < 1:
            raise ValueError("stride must be at least 1")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer greater than or equal to 3")
        self.dims = dims
        self.stride = stride
        self.kernel_size = kernel_size

        k = torch.tensor([math.comb(kernel_size - 1, k) for k in range(kernel_size)])
        k2d = k[:, None] @ k[None, :]
        self.register_buffer("kernel", (k2d / k2d.sum()).float()[None, None, :, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x
        if self.dims == 2:
            return self._apply_2d(x)
        b, _, f, _, _ = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self._apply_2d(x)
        h2, w2 = x.shape[-2:]
        return rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f, h=h2, w=w2)

    def _apply_2d(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        weight = self.kernel.expand(channels, 1, self.kernel_size, self.kernel_size)
        return F.conv2d(x, weight=weight, stride=self.stride, padding=self.kernel_size // 2, groups=channels)


def _ltx2_rational_scale(scale: float) -> tuple[int, int]:
    mapping = {0.75: (3, 4), 1.5: (3, 2), 2.0: (2, 1), 4.0: (4, 1)}
    if float(scale) not in mapping:
        raise ValueError(f"Unsupported LTX-2 upsampler spatial scale {scale}.")
    return mapping[float(scale)]


class LTX2SpatialRationalResampler(torch.nn.Module):
    def __init__(self, mid_channels: int, scale: float):
        super().__init__()
        self.scale = float(scale)
        self.num, self.den = _ltx2_rational_scale(self.scale)
        self.conv = torch.nn.Conv2d(mid_channels, (self.num**2) * mid_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = LTX2PixelShuffleND(2, upscale_factors=(self.num, self.num))
        self.blur_down = LTX2BlurDownsample(dims=2, stride=self.den)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, f, _, _ = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.blur_down(x)
        return rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)


class LTX2LatentUpsampler(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 512,
        num_blocks_per_stage: int = 4,
        dims: int = 3,
        spatial_upsample: bool = True,
        temporal_upsample: bool = False,
        spatial_scale: float = 2.0,
        rational_resampler: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_blocks_per_stage = num_blocks_per_stage
        self.dims = dims
        self.spatial_upsample = spatial_upsample
        self.temporal_upsample = temporal_upsample
        self.spatial_scale = float(spatial_scale)
        self.rational_resampler = rational_resampler

        conv = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d
        self.initial_conv = conv(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = torch.nn.GroupNorm(32, mid_channels)
        self.initial_activation = torch.nn.SiLU()
        self.res_blocks = torch.nn.ModuleList(
            [LTX2UpsamplerResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)]
        )

        if spatial_upsample and temporal_upsample:
            self.upsampler = torch.nn.Sequential(
                torch.nn.Conv3d(mid_channels, 8 * mid_channels, kernel_size=3, padding=1),
                LTX2PixelShuffleND(3),
            )
        elif spatial_upsample:
            if rational_resampler:
                self.upsampler = LTX2SpatialRationalResampler(mid_channels=mid_channels, scale=self.spatial_scale)
            else:
                self.upsampler = torch.nn.Sequential(
                    torch.nn.Conv2d(mid_channels, 4 * mid_channels, kernel_size=3, padding=1),
                    LTX2PixelShuffleND(2),
                )
        elif temporal_upsample:
            self.upsampler = torch.nn.Sequential(
                torch.nn.Conv3d(mid_channels, 2 * mid_channels, kernel_size=3, padding=1),
                LTX2PixelShuffleND(1),
            )
        else:
            raise ValueError("Either spatial_upsample or temporal_upsample must be True")

        self.post_upsample_res_blocks = torch.nn.ModuleList(
            [LTX2UpsamplerResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)]
        )
        self.final_conv = conv(mid_channels, in_channels, kernel_size=3, padding=1)

    @classmethod
    def from_config(cls, config: dict) -> "LTX2LatentUpsampler":
        return cls(
            in_channels=config.get("in_channels", 128),
            mid_channels=config.get("mid_channels", 512),
            num_blocks_per_stage=config.get("num_blocks_per_stage", 4),
            dims=config.get("dims", 3),
            spatial_upsample=config.get("spatial_upsample", True),
            temporal_upsample=config.get("temporal_upsample", False),
            spatial_scale=config.get("spatial_scale", 2.0),
            rational_resampler=config.get("rational_resampler", False),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        b, _, f, _, _ = latent.shape
        if self.dims == 2:
            x = rearrange(latent, "b c f h w -> (b f) c h w")
            x = self.initial_activation(self.initial_norm(self.initial_conv(x)))
            for block in self.res_blocks:
                x = block(x)
            x = self.upsampler(x)
            for block in self.post_upsample_res_blocks:
                x = block(x)
            x = self.final_conv(x)
            return rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)

        x = self.initial_activation(self.initial_norm(self.initial_conv(latent)))
        for block in self.res_blocks:
            x = block(x)
        if self.temporal_upsample:
            x = self.upsampler(x)
            x = x[:, :, 1:, :, :]
        elif isinstance(self.upsampler, LTX2SpatialRationalResampler):
            x = self.upsampler(x)
        else:
            x = rearrange(x, "b c f h w -> (b f) c h w")
            x = self.upsampler(x)
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
        for block in self.post_upsample_res_blocks:
            x = block(x)
        return self.final_conv(x)


def resolve_ltx2_upsampler_path(model_or_path: str, filename: Optional[str], revision: Optional[str] = None) -> str:
    model_or_path = os.path.expanduser(str(model_or_path))
    if os.path.isfile(model_or_path):
        return model_or_path
    if os.path.isdir(model_or_path):
        if not filename:
            raise ValueError("ltx2_validation_spatial_upsampler_filename is required when using a directory.")
        candidate = os.path.join(model_or_path, filename)
        if os.path.isfile(candidate):
            return candidate
        raise ValueError(f"LTX-2 spatial upsampler file not found: {candidate}")
    if not filename:
        raise ValueError("ltx2_validation_spatial_upsampler_filename is required when using a Hugging Face repo.")
    return hf_hub_download(repo_id=model_or_path, filename=filename, revision=revision)


def load_ltx2_latent_upsampler(path: str, device: torch.device, dtype: torch.dtype) -> LTX2LatentUpsampler:
    with safetensors.safe_open(path, framework="pt") as handle:
        metadata = handle.metadata() or {}
    if "config" not in metadata:
        raise ValueError(f"LTX-2 upsampler checkpoint {path} does not include safetensors config metadata.")
    config = json.loads(metadata["config"])
    upsampler = LTX2LatentUpsampler.from_config(config)
    state_dict = safetensors.torch.load_file(path, device="cpu")
    upsampler.load_state_dict({key: value.to(dtype=dtype) for key, value in state_dict.items()}, strict=True)
    return upsampler.to(device=device, dtype=dtype).eval()


def upsample_ltx2_video_latents(
    latent: torch.Tensor,
    vae,
    upsampler: LTX2LatentUpsampler,
) -> torch.Tensor:
    latents_mean = getattr(vae, "latents_mean", None)
    latents_std = getattr(vae, "latents_std", None)
    if latents_mean is None or latents_std is None:
        raise ValueError("LTX-2 VAE must expose latents_mean and latents_std for spatial latent upscaling.")
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latent.device, dtype=latent.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latent.device, dtype=latent.dtype)
    scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", 1.0)

    latent = latent * latents_std / scaling_factor + latents_mean
    latent = upsampler(latent)
    return (latent - latents_mean) * scaling_factor / latents_std

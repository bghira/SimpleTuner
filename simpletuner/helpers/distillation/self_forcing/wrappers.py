from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from .scheduler import FlowMatchingSchedulerAdapter


def _flatten_video(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    if tensor.ndim != 5:
        raise ValueError(f"Expected 5D tensor (B, C, T, H, W); received shape {tuple(tensor.shape)}.")
    b, c, t, h, w = tensor.shape
    return tensor.reshape(b * t, c, h, w), torch.Size((b, c, t, h, w))


def _build_prepared_batch(
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    conditional_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    added_cond_kwargs = conditional_dict.get("added_cond_kwargs") or {}
    prepared: Dict[str, torch.Tensor] = {
        "noisy_latents": noisy_latents,
        "encoder_hidden_states": conditional_dict["prompt_embeds"],
        "timesteps": timesteps,
        "added_cond_kwargs": added_cond_kwargs,
    }
    for key in (
        "conditioning_image_embeds",
        "conditioning_pixel_values",
        "conditioning_latents",
        "force_keep_mask",
        "encoder_attention_mask",
    ):
        if key in conditional_dict and conditional_dict[key] is not None:
            prepared[key] = conditional_dict[key]
    return prepared


@dataclass
class FoundationModelWrapper:
    """
    Thin wrapper around a SimpleTuner ModelFoundation that exposes the WAN-style
    `(flow_pred, pred_x0)` API used in realtime's self-forcing code.
    """

    foundation: object
    scheduler: FlowMatchingSchedulerAdapter

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        *,
        return_dict: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Dict[str, torch.Tensor]:
        prepared = _build_prepared_batch(noisy_latents, timesteps, conditional_dict)
        prediction = self.foundation.model_predict(prepared)["model_prediction"]

        flow_pred = prediction
        latent_shape = flow_pred.shape

        flat_flow, _ = _flatten_video(flow_pred)
        flat_xt, _ = _flatten_video(noisy_latents)
        flat_timestep = timesteps.reshape(-1)

        flat_x0 = self.scheduler.convert_flow_to_x0(flat_flow, flat_xt, flat_timestep)
        pred_x0 = flat_x0.reshape(latent_shape)

        if return_dict:
            return {
                "flow_pred": flow_pred,
                "pred_x0": pred_x0,
            }
        return flow_pred, pred_x0


@dataclass
class ModuleWrapper:
    """
    Wraps a raw Wan transformer module (teacher/fake score) to expose the same API.
    """

    module: torch.nn.Module
    scheduler: FlowMatchingSchedulerAdapter
    weight_dtype: torch.dtype

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        *,
        return_dict: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Dict[str, torch.Tensor]:
        latents = noisy_latents.to(dtype=self.weight_dtype)
        timestep = timesteps
        encoder_hidden_states = conditional_dict["prompt_embeds"].to(dtype=self.weight_dtype)

        kwargs = {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "return_dict": False,
        }

        for key in (
            "conditioning_image_embeds",
            "conditioning_pixel_values",
            "conditioning_latents",
            "encoder_attention_mask",
        ):
            if key in conditional_dict and conditional_dict[key] is not None:
                kwargs[f"{key}"] = conditional_dict[key].to(dtype=self.weight_dtype)

        outputs = self.module(**kwargs)
        flow_pred = outputs[0]
        latent_shape = flow_pred.shape

        flat_flow, _ = _flatten_video(flow_pred)
        flat_xt, _ = _flatten_video(latents)
        flat_timestep = timesteps.reshape(-1)

        flat_x0 = self.scheduler.convert_flow_to_x0(flat_flow, flat_xt, flat_timestep)
        pred_x0 = flat_x0.reshape(latent_shape)

        if return_dict:
            return {
                "flow_pred": flow_pred,
                "pred_x0": pred_x0,
            }
        return flow_pred, pred_x0


@dataclass
class TextEncoderWrapper:
    model: object

    def encode(self, prompts: list[str], *, negative: bool = False) -> torch.Tensor:
        return self.model.encode_text_batch(prompts, is_negative_prompt=negative)


@dataclass
class VAEWrapper:
    vae: object

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.vae, "decode"):
            raise AttributeError("VAE does not implement `decode`.")
        return self.vae.decode(latents)

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.vae, "encode"):
            raise AttributeError("VAE does not implement `encode`.")
        posterior = self.vae.encode(pixels)
        if hasattr(posterior, "latent_dist"):
            return posterior.latent_dist.sample()
        if hasattr(posterior, "sample"):
            return posterior.sample()
        raise AttributeError("Unsupported encoder output on VAE.")

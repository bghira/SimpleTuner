import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrepaRegularizer:
    """Implements Cross-frame Representation Alignment (CREPA) as defined in Eq. (6) of the paper."""

    def __init__(self, config, accelerator, hidden_size: int):
        self.config = config
        self.device = accelerator.device
        self.hidden_size = hidden_size

        self.enabled = bool(getattr(config, "crepa_enabled", False))
        self.block_index = getattr(config, "crepa_block_index", None)
        raw_distance = getattr(config, "crepa_adjacent_distance", 1)
        self.distance = 1 if raw_distance is None else int(raw_distance)
        if self.distance < 0:
            raise ValueError("crepa_adjacent_distance must be non-negative.")
        raw_tau = getattr(config, "crepa_adjacent_tau", 1.0)
        self.tau = 1.0 if raw_tau is None else float(raw_tau)
        if self.tau <= 0:
            raise ValueError("crepa_adjacent_tau must be greater than zero.")
        self.weight = float(getattr(config, "crepa_lambda", 0.5) or 0.0)
        # Prefer explicit crepa_model, fall back to legacy crepa_encoder name.
        raw_encoder = getattr(config, "crepa_model", None) or getattr(config, "crepa_encoder", None)
        self.encoder_name = self._resolve_encoder_name(raw_encoder)
        self.encoder_image_size = int(getattr(config, "crepa_encoder_image_size", 518) or 518)
        self.normalize_by_frames = bool(getattr(config, "crepa_normalize_by_frames", True))
        # If false, fall back to global pooling when spatial token counts do not match.
        self.spatial_align = bool(getattr(config, "crepa_spatial_align", True))
        # Adjacent mode (default): only use exact distance d neighbors per paper's K = {f-d, f+d}
        # Cumulative mode (opt-in): use all distances from 1 to d for smoother alignment
        self.cumulative_neighbors = bool(getattr(config, "crepa_cumulative_neighbors", False))

        self.use_backbone_features = bool(getattr(config, "crepa_use_backbone_features", False))
        self.teacher_block_index = getattr(config, "crepa_teacher_block_index", None)
        self.encoder: Optional[torch.nn.Module] = None
        self.encoder_dim: Optional[int] = None
        self.projector: Optional[torch.nn.Module] = None

    # --------------------------- public helpers ---------------------------
    def attach_to_model(self, model: nn.Module):
        """Attach the projection head to the diffusion backbone so it is optimised."""

        if not self.enabled:
            return

        if self.block_index is None:
            raise ValueError("crepa_block_index must be set when CREPA is enabled.")

        if self.use_backbone_features:
            target_dim = self.hidden_size
            self.encoder_dim = target_dim
        else:
            self._load_encoder()
            target_dim = self.encoder_dim
            if target_dim is None:
                raise RuntimeError("CREPA failed to determine encoder output dimension.")

        if self.projector is None:
            self.projector = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, target_dim),
            )
            # Register as part of the model to ensure optimizer coverage.
            setattr(model, "crepa_projector", self.projector)

        # Ensure the projector lives on the correct device/dtype.
        self.projector.to(device=self.device, dtype=torch.float32)

    def wants_hidden_states(self) -> bool:
        return self.enabled

    def compute_loss(
        self,
        hidden_states: Optional[torch.Tensor],
        latents: Optional[torch.Tensor],
        vae: Optional[nn.Module],
        *,
        frame_features: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[dict]]:
        if not self.enabled:
            return None, None
        if hidden_states is None:
            raise ValueError("CREPA is enabled but no intermediate hidden states were provided.")
        if not self.use_backbone_features:
            if latents is None:
                raise ValueError("CREPA requires access to clean latents for decoding.")
            if vae is None:
                raise ValueError("CREPA requires a VAE to decode latents back to pixel space.")
        else:
            if frame_features is None:
                raise ValueError("CREPA backbone feature mode requires frame_features from the model.")
        if self.projector is None:
            raise RuntimeError("CREPA projector was not initialised on the diffusion model.")
        if self.weight == 0:
            return None, None

        if not self.use_backbone_features:
            video_pixels = self._decode_latents(latents, vae)
            frame_features = self._encode_frames(video_pixels)  # (B, T_pixel, N_enc, D_enc)
        else:
            frame_features = self._normalize_frame_features(frame_features)

        projected = self._project_hidden_states(hidden_states)  # (B, T_dit, N_dit, D_enc)

        # Temporal alignment: match frame count between projected and frame_features
        projected, frame_features = self._maybe_align_temporal(projected, frame_features)

        # Spatial alignment: match token count per frame
        projected, frame_features = self._maybe_align_tokens(projected, frame_features)

        projected = F.normalize(projected, dim=-1)
        frame_features = F.normalize(frame_features, dim=-1)

        # Per-patch cosine similarity then mean over patches: (B, T)
        self_sim = (projected * frame_features).sum(dim=-1).mean(dim=-1)

        total_sim = self_sim.clone()
        bsz, num_frames = total_sim.shape
        d = min(self.distance, num_frames - 1)
        tau = max(self.tau, 1e-8)

        if d > 0:
            if self.cumulative_neighbors:
                # Cumulative mode: use all distances from 1 to d
                for offset in range(1, d + 1):
                    weight = math.exp(-float(offset) / tau)
                    # f -> f+offset (align hidden state of frame f with features of frame f+offset)
                    fwd = (projected[:, :-offset, ...] * frame_features[:, offset:, ...]).sum(dim=-1).mean(dim=-1)
                    total_sim[:, :-offset] += weight * fwd
                    # f -> f-offset (align hidden state of frame f with features of frame f-offset)
                    back = (projected[:, offset:, ...] * frame_features[:, :-offset, ...]).sum(dim=-1).mean(dim=-1)
                    total_sim[:, offset:] += weight * back
            else:
                # Adjacent mode (default): only exact distance d per paper's K = {f-d, f+d}
                weight = math.exp(-float(d) / tau)
                # f -> f+d
                fwd = (projected[:, :-d, ...] * frame_features[:, d:, ...]).sum(dim=-1).mean(dim=-1)
                total_sim[:, :-d] += weight * fwd
                # f -> f-d
                back = (projected[:, d:, ...] * frame_features[:, :-d, ...]).sum(dim=-1).mean(dim=-1)
                total_sim[:, d:] += weight * back

        per_video_sum = total_sim.sum(dim=1)
        if self.normalize_by_frames:
            per_video_sum = per_video_sum / float(num_frames)

        align_loss = -per_video_sum.mean() * self.weight

        log_data = {
            "crepa_loss": align_loss.detach().item(),
            "crepa_similarity": total_sim.mean().detach().item(),
        }
        return align_loss, log_data

    # --------------------------- private helpers ---------------------------
    def _maybe_align_temporal(
        self, projected: torch.Tensor, frame_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align temporal dimensions between projected hidden states and frame features.

        DiT models may have different temporal resolution than pixel-space frames due to
        VAE temporal compression and patch embedding. This method subsamples the larger
        tensor to match the smaller one using representative frame selection.

        Args:
            projected: (B, T_dit, N_tokens, D) from DiT hidden states
            frame_features: (B, T_pixel, N_tokens, D) from encoder

        Returns:
            Tuple of aligned tensors with matching temporal dimension
        """
        t_proj = projected.shape[1]
        t_feat = frame_features.shape[1]

        if t_proj == t_feat:
            return projected, frame_features

        # Subsample the larger tensor to match the smaller
        if t_feat > t_proj:
            # More pixel frames than DiT temporal tokens - subsample frame_features
            indices = torch.linspace(0, t_feat - 1, t_proj, device=frame_features.device).long()
            frame_features = frame_features.index_select(1, indices)
        else:
            # More DiT temporal tokens than pixel frames - subsample projected
            indices = torch.linspace(0, t_proj - 1, t_feat, device=projected.device).long()
            projected = projected.index_select(1, indices)

        return projected, frame_features

    def _project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Expected shapes:
        # - (B, T, D) legacy global tokens
        # - (B, T, P, D) patch tokens (preferred)
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.unsqueeze(2)
        elif hidden_states.ndim != 4:
            raise ValueError(f"CREPA expected hidden states with 3 or 4 dims, got {hidden_states.shape}")

        b, t, p, d = hidden_states.shape
        projector_dtype = next(self.projector.parameters()).dtype
        flattened = hidden_states.to(dtype=projector_dtype).reshape(b * t * p, d)
        projected = self.projector(flattened)
        projected = projected.view(b, t, p, -1)
        return projected

    def _maybe_align_tokens(
        self, projected: torch.Tensor, frame_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Align spatial token counts. If disabled, fall back to global pooling.
        proj_tokens = projected.shape[2]
        enc_tokens = frame_features.shape[2]
        if proj_tokens == enc_tokens:
            return projected, frame_features

        if not self.spatial_align:
            projected = projected.mean(dim=2, keepdim=True)
            frame_features = frame_features.mean(dim=2, keepdim=True)
            return projected, frame_features

        target_tokens = min(proj_tokens, enc_tokens)
        projected = self._interpolate_tokens(projected, target_tokens)
        frame_features = self._interpolate_tokens(frame_features, target_tokens)
        return projected, frame_features

    def _interpolate_tokens(self, tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
        if tokens.shape[2] == target_tokens:
            return tokens
        b, t, n, d = tokens.shape
        flat = tokens.reshape(b * t, n, d).permute(0, 2, 1)  # (BT, D, N)

        src_size = int(math.sqrt(n))
        tgt_size = int(math.sqrt(target_tokens))
        if src_size * src_size == n and tgt_size * tgt_size == target_tokens:
            flat = flat.view(b * t, d, src_size, src_size)
            interpolated = F.interpolate(flat, size=(tgt_size, tgt_size), mode="bilinear", align_corners=False)
            interpolated = interpolated.view(b * t, d, target_tokens)
        else:
            interpolated = F.interpolate(flat, size=target_tokens, mode="linear", align_corners=False)

        interpolated = interpolated.permute(0, 2, 1).reshape(b, t, target_tokens, d)
        return interpolated

    def _load_encoder(self):
        if self.encoder is not None:
            return
        # Dinov2 torch hub exports lightweight models; user confirmed network usage is acceptable.
        self.encoder = torch.hub.load("facebookresearch/dinov2", self.encoder_name)
        self.encoder.eval().requires_grad_(False).to(self.device, dtype=torch.float32)

        dummy = torch.zeros(1, 3, self.encoder_image_size, self.encoder_image_size, device=self.device)
        with torch.no_grad():
            encoded = self._forward_encoder(dummy)
        self.encoder_dim = int(encoded.shape[-1])

    def _forward_encoder(self, images: torch.Tensor) -> torch.Tensor:
        enc_dtype = next(self.encoder.parameters()).dtype
        images = images.to(dtype=enc_dtype)
        with torch.no_grad():
            output = self.encoder(images)
        if isinstance(output, dict):
            # Prefer patch tokens; fall back to class token if needed.
            if "x_norm_patchtokens" in output:
                tokens = output["x_norm_patchtokens"]
            elif "x_norm_clstoken" in output:
                tokens = output["x_norm_clstoken"].unsqueeze(1)
            else:
                tokens = next(iter(output.values()))
        elif torch.is_tensor(output):
            tokens = output
        elif isinstance(output, (list, tuple)):
            tokens = output[0]
        else:
            raise TypeError(f"Unsupported encoder output type: {type(output)}")

        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(1)
        if tokens.ndim != 3:
            raise ValueError(f"Unexpected encoder token shape: {tokens.shape}")
        return tokens

    def _decode_latents(self, latents: torch.Tensor, vae: nn.Module) -> torch.Tensor:
        vae_dtype = next(vae.parameters()).dtype
        latents = latents.to(device=self.device, dtype=vae_dtype)
        scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", 1.0)
        shift_factor = getattr(getattr(vae, "config", None), "shift_factor", None)

        if shift_factor is not None:
            latents = latents / scaling_factor + shift_factor
        else:
            latents = latents / scaling_factor

        if hasattr(vae.config, "latents_mean") and hasattr(vae.config, "latents_std"):
            view_shape = [1, latents.shape[1]] + [1] * (latents.ndim - 2)
            mean = torch.tensor(vae.config.latents_mean, device=self.device, dtype=latents.dtype).view(view_shape)
            std = torch.tensor(vae.config.latents_std, device=self.device, dtype=latents.dtype).view(view_shape)
            latents = latents * std + mean

        with torch.no_grad():
            decoded = vae.decode(latents).sample
        decoded = decoded.clamp(-1, 1)
        # Convert to [0,1] and reshape to (B, T, C, H, W)
        decoded = (decoded + 1.0) * 0.5
        if decoded.ndim != 5:
            raise ValueError(f"Expected decoded video to be 5D, got {decoded.shape}")
        return decoded.permute(0, 2, 1, 3, 4).contiguous()

    def _encode_frames(self, video: torch.Tensor) -> torch.Tensor:
        # video: (B, T, C, H, W) in [0,1]
        b, t, c, h, w = video.shape
        frames = video.reshape(b * t, c, h, w)
        frames = F.interpolate(
            frames, size=(self.encoder_image_size, self.encoder_image_size), mode="bilinear", align_corners=False
        )
        enc_dtype = next(self.encoder.parameters()).dtype
        frames = frames.to(dtype=enc_dtype)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=enc_dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=enc_dtype).view(1, 3, 1, 1)
        frames = (frames - mean) / std

        tokens = self._forward_encoder(frames)  # (BT, N_tokens, D)
        tokens = tokens.view(b, t, tokens.shape[1], -1)
        return tokens

    def _normalize_frame_features(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        Ensure backbone-provided frame features are in (B, T, N, D) format.
        """
        if frame_features.ndim == 3:
            frame_features = frame_features.unsqueeze(1)
        if frame_features.ndim != 4:
            raise ValueError(f"Unexpected frame feature shape: {frame_features.shape}")
        return frame_features

    def _resolve_encoder_name(self, value: Optional[str]) -> str:
        if not value:
            return "dinov2_vitg14"
        value = str(value).strip()
        aliases = {
            "dino_v2_g": "dinov2_vitg14",
            "dinov2_g": "dinov2_vitg14",
            "dinov2-vitg14": "dinov2_vitg14",
            "dino_v2_s": "dinov2_vits14",
            "dinov2_s": "dinov2_vits14",
            "dinov2-vitb14": "dinov2_vitb14",
        }
        return aliases.get(value.lower(), value)

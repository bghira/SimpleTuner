import logging
import math
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from simpletuner.helpers.models.common import ModelFoundation

logger = logging.getLogger(__name__)


class CrepaMode(Enum):
    """Determines how hidden state shapes are interpreted for CREPA/REPA alignment.

    Different model architectures produce hidden states with different semantic layouts:
    - IMAGE: For image models where hidden states are (B, S, D) with S spatial tokens.
             Reshaped to (B, 1, S, D) treating the image as a single "frame".
    - VIDEO: For video models where hidden states are (B, T, D) with T temporal tokens.
             Reshaped to (B, T, 1, D) treating each frame as having one global token.

    Individual model foundations can override crepa_mode to handle unique architectures.
    """

    IMAGE = auto()
    VIDEO = auto()


class CrepaScheduler:
    """Schedules the CREPA coefficient (crepa_lambda) over training with warmup, decay, and cutoff support."""

    def __init__(self, config, max_train_steps: int):
        self.scheduler_type = str(getattr(config, "crepa_scheduler", "constant") or "constant").lower()
        self.base_weight = float(getattr(config, "crepa_lambda", 0.5) or 0.0)
        self.warmup_steps = int(getattr(config, "crepa_warmup_steps", 0) or 0)
        raw_decay_steps = getattr(config, "crepa_decay_steps", 0) or 0
        self.decay_steps = int(raw_decay_steps) if int(raw_decay_steps) > 0 else max_train_steps
        self.lambda_end = float(getattr(config, "crepa_lambda_end", 0.0) or 0.0)
        self.cutoff_step = int(getattr(config, "crepa_cutoff_step", 0) or 0)
        self.similarity_threshold = getattr(config, "crepa_similarity_threshold", None)
        if self.similarity_threshold is not None:
            self.similarity_threshold = float(self.similarity_threshold)
        raw_ema_decay = getattr(config, "crepa_similarity_ema_decay", None)
        self.similarity_ema_decay = float(raw_ema_decay) if raw_ema_decay is not None else 0.99
        self.threshold_mode = str(getattr(config, "crepa_threshold_mode", "permanent") or "permanent").lower()
        self.power = float(getattr(config, "crepa_power", 1.0) or 1.0)

        self._similarity_ema: Optional[float] = None
        self._cutoff_triggered = False

    def _compute_scheduled_weight(self, step: int) -> float:
        """Compute the scheduled weight at the given step (without cutoff logic)."""
        # Warmup phase: linear ramp from 0 to base_weight (applies to all scheduler types)
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_weight * (step / self.warmup_steps)

        # Constant scheduler: no decay after warmup
        if self.scheduler_type == "constant":
            return self.base_weight

        # Decay phase: step relative to end of warmup
        decay_step = step - self.warmup_steps
        total_decay_steps = max(self.decay_steps - self.warmup_steps, 1)
        progress = min(decay_step / total_decay_steps, 1.0)

        if self.scheduler_type == "linear":
            return self.base_weight + (self.lambda_end - self.base_weight) * progress
        elif self.scheduler_type == "cosine":
            return self.lambda_end + (self.base_weight - self.lambda_end) * (1 + math.cos(math.pi * progress)) / 2
        elif self.scheduler_type == "polynomial":
            return (self.base_weight - self.lambda_end) * ((1 - progress) ** self.power) + self.lambda_end
        else:
            return self.base_weight

    def _update_similarity_ema(self, similarity: Optional[float]) -> None:
        """Update the exponential moving average of similarity."""
        if similarity is None:
            return
        if self._similarity_ema is None:
            self._similarity_ema = similarity
        else:
            self._similarity_ema = (
                self.similarity_ema_decay * self._similarity_ema + (1 - self.similarity_ema_decay) * similarity
            )

    def _check_cutoff(self, step: int) -> bool:
        """Check if cutoff conditions are met."""
        # Step-based cutoff
        if self.cutoff_step > 0 and step >= self.cutoff_step:
            return True

        # Similarity threshold cutoff
        if self.similarity_threshold is not None and self._similarity_ema is not None:
            if self._similarity_ema >= self.similarity_threshold:
                return True

        return False

    def get_weight(self, step: int, similarity: Optional[float] = None) -> float:
        """
        Get the scheduled weight for the given step.

        Args:
            step: Current training step (global step from trainer/accelerator).
            similarity: Current similarity value for EMA tracking (optional).

        Returns:
            Scheduled CREPA coefficient weight (0.0 if cutoff is active).
        """
        # Update similarity EMA
        self._update_similarity_ema(similarity)

        # Handle cutoff logic
        if self._cutoff_triggered and self.threshold_mode == "permanent":
            return 0.0

        cutoff_active = self._check_cutoff(step)

        if cutoff_active:
            if self.threshold_mode == "permanent":
                self._cutoff_triggered = True
            return 0.0

        # Recoverable mode: reset trigger if cutoff is no longer active
        if self.threshold_mode == "recoverable" and self._cutoff_triggered:
            self._cutoff_triggered = False

        return self._compute_scheduled_weight(step)

    def is_cutoff(self) -> bool:
        """Check if CREPA is currently cut off."""
        return self._cutoff_triggered

    def get_similarity_ema(self) -> Optional[float]:
        """Get the current similarity EMA value."""
        return self._similarity_ema


class CrepaRegularizer:
    """Implements Cross-frame Representation Alignment (CREPA) as defined in Eq. (6) of the paper."""

    def __init__(
        self,
        config,
        accelerator,
        hidden_size: int,
        *,
        model_foundation: Optional["ModelFoundation"] = None,
        max_train_steps: int = 0,
    ):
        self.config = config
        self.device = accelerator.device
        self.hidden_size = hidden_size
        self.model_foundation = model_foundation

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
        self.base_weight = float(getattr(config, "crepa_lambda", 0.5) or 0.0)

        # Initialize scheduler for coefficient scheduling
        self.scheduler = CrepaScheduler(config, max_train_steps) if self.enabled else None
        # Prefer explicit crepa_model, fall back to legacy crepa_encoder name.
        raw_encoder = getattr(config, "crepa_model", None) or getattr(config, "crepa_encoder", None)
        self.encoder_name = self._resolve_encoder_name(raw_encoder)
        self.encoder_image_size = int(getattr(config, "crepa_encoder_image_size", 518) or 518)
        self.encoder_frames_batch_size = int(getattr(config, "crepa_encoder_frames_batch_size", -1) or -1)
        self.normalize_by_frames = bool(getattr(config, "crepa_normalize_by_frames", True))
        # If false, fall back to global pooling when spatial token counts do not match.
        self.spatial_align = bool(getattr(config, "crepa_spatial_align", True))
        # Adjacent mode (default): only use exact distance d neighbors per paper's K = {f-d, f+d}
        # Cumulative mode (opt-in): use all distances from 1 to d for smoother alignment
        self.cumulative_neighbors = bool(getattr(config, "crepa_cumulative_neighbors", False))

        self.use_backbone_features = bool(getattr(config, "crepa_use_backbone_features", False))
        self.use_tae = bool(getattr(config, "crepa_use_tae", False))
        self.teacher_block_index = getattr(config, "crepa_teacher_block_index", None)
        self.encoder: Optional[torch.nn.Module] = None
        self.encoder_dim: Optional[int] = None
        self.projector: Optional[torch.nn.Module] = None

        # Determine shape interpretation mode from model foundation (duck-typed)
        self.mode: CrepaMode = (
            model_foundation.crepa_mode
            if model_foundation and hasattr(model_foundation, "crepa_mode")
            else CrepaMode.VIDEO  # backward compat default
        )

        # Validate TAE availability if requested
        if self.use_tae and self.enabled and not self.use_backbone_features:
            if model_foundation is not None and not model_foundation.supports_validation_preview():
                logger.warning(
                    f"crepa_use_tae=True but {model_foundation.NAME} does not support TAE. "
                    "Falling back to full VAE decoding."
                )
                self.use_tae = False

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
        vae: Optional[nn.Module] = None,
        *,
        frame_features: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> Tuple[Optional[torch.Tensor], Optional[dict]]:
        if not self.enabled:
            return None, None
        if hidden_states is None:
            raise ValueError("CREPA is enabled but no intermediate hidden states were provided.")
        if not self.use_backbone_features:
            if latents is None:
                raise ValueError("CREPA requires access to clean latents for decoding.")
            # Only require VAE if not using unified decode via model_foundation
            if self.model_foundation is None and vae is None:
                raise ValueError("CREPA requires a VAE to decode latents back to pixel space.")
        else:
            if frame_features is None:
                raise ValueError("CREPA backbone feature mode requires frame_features from the model.")
        if self.projector is None:
            raise RuntimeError("CREPA projector was not initialised on the diffusion model.")
        if self.base_weight == 0:
            return None, None

        if not self.use_backbone_features:
            video_pixels = self._decode_latents_unified(latents, vae)
            frame_features = self._encode_frames(video_pixels)  # (B, T_pixel, N_enc, D_enc)
        else:
            frame_features = self._normalize_frame_features(frame_features)
            # Treat backbone features as a frozen teacher to avoid over-regularization.
            frame_features = frame_features.detach()

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

        # Get current similarity for EMA tracking
        current_similarity = total_sim.mean().detach().item()

        # Get scheduled weight (handles warmup, decay, and cutoff)
        if self.scheduler is not None:
            scheduled_weight = self.scheduler.get_weight(step, similarity=current_similarity)
        else:
            scheduled_weight = self.base_weight

        # Early exit if weight is zero (cutoff active or decayed to zero)
        if scheduled_weight == 0:
            log_data = {
                "crepa_loss": 0.0,
                "crepa_similarity": current_similarity,
                "crepa_weight": 0.0,
                "crepa_cutoff": True,
            }
            if self.scheduler is not None and self.scheduler.get_similarity_ema() is not None:
                log_data["crepa_similarity_ema"] = self.scheduler.get_similarity_ema()
            return None, log_data

        align_loss = -per_video_sum.mean() * scheduled_weight

        log_data = {
            "crepa_loss": align_loss.detach().item(),
            "crepa_similarity": current_similarity,
            "crepa_weight": scheduled_weight,
            "crepa_cutoff": False,
        }
        if self.scheduler is not None and self.scheduler.get_similarity_ema() is not None:
            log_data["crepa_similarity_ema"] = self.scheduler.get_similarity_ema()
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
        # Expected shapes depend on mode:
        # VIDEO mode: (B, T, D) -> (B, T, 1, D) or already (B, T, P, D)
        # IMAGE mode: (B, S, D) -> (B, 1, S, D) or already (B, 1, S, D)
        if hidden_states.ndim == 3:
            if self.mode == CrepaMode.VIDEO:
                # Video: (B, T, D) -> (B, T, 1, D) - T frames, 1 global token each
                hidden_states = hidden_states.unsqueeze(2)
            else:
                # Image: (B, S, D) -> (B, 1, S, D) - 1 frame, S spatial tokens
                hidden_states = hidden_states.unsqueeze(1)
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

    def _decode_latents_unified(self, latents: torch.Tensor, vae: Optional[nn.Module]) -> torch.Tensor:
        """
        Decode latents to pixel space using either the unified model interface or legacy VAE path.

        Uses model_foundation.decode_latents_to_pixels when available, which handles
        TAE vs VAE selection and all normalization concerns. Falls back to direct VAE
        decode for backward compatibility when model_foundation is not set.

        Returns:
            Video pixels in (B, T, C, H, W) format, [0, 1] range.
        """
        if self.model_foundation is not None:
            # Use unified decode interface - handles TAE/VAE and normalization
            return self.model_foundation.decode_latents_to_pixels(latents, use_tae=self.use_tae)

        # Legacy fallback: direct VAE decode (for backward compatibility)
        return self._decode_latents_legacy(latents, vae)

    def _decode_latents_legacy(self, latents: torch.Tensor, vae: nn.Module) -> torch.Tensor:
        """Legacy VAE decode path for backward compatibility."""
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

        frames_batches = (
            list(torch.split(frames, self.encoder_frames_batch_size, dim=0))
            if self.encoder_frames_batch_size > 0
            else [frames]
        )
        tokens = torch.cat(
            [self._forward_encoder(frames_batch) for frames_batch in frames_batches], dim=0
        )  # (BT, N_tokens, D)
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


class UrepaRegularizer:
    """
    Implements U-REPA (Universal Representation Alignment) for UNet-based diffusion models.

    U-REPA adapts the REPA concept for UNet architectures by:
    1. Extracting features from the mid-block (not early layers like DiT-based REPA)
    2. Using manifold loss to align relative similarity structure
    3. Handling convolutional (B, C, H, W) hidden states

    Reference: "U-REPA: Training-free Representation Alignment for UNet" (NeurIPS 2025)
    """

    def __init__(
        self,
        config,
        accelerator,
        hidden_size: int,
        *,
        model_foundation: Optional["ModelFoundation"] = None,
        max_train_steps: int = 0,
    ):
        self.config = config
        self.device = accelerator.device
        self.hidden_size = hidden_size
        self.model_foundation = model_foundation

        self.enabled = bool(getattr(config, "urepa_enabled", False))
        self.base_weight = float(getattr(config, "urepa_lambda", 0.5) or 0.5)
        self.manifold_weight = float(getattr(config, "urepa_manifold_weight", 3.0) or 3.0)

        # Prefer explicit urepa_model, fall back to crepa_model for shared encoder config.
        raw_encoder = (
            getattr(config, "urepa_model", None)
            or getattr(config, "crepa_model", None)
            or getattr(config, "crepa_encoder", None)
        )
        self.encoder_name = self._resolve_encoder_name(raw_encoder)
        self.encoder_image_size = int(getattr(config, "urepa_encoder_image_size", 518) or 518)

        self.use_tae = bool(getattr(config, "urepa_use_tae", False))

        # Scheduler for coefficient scheduling (reuse CrepaScheduler)
        self.scheduler = self._create_scheduler(config, max_train_steps) if self.enabled else None

        self.encoder: Optional[torch.nn.Module] = None
        self.encoder_dim: Optional[int] = None
        self.projector: Optional[torch.nn.Module] = None

        # Validate TAE availability if requested
        if self.use_tae and self.enabled:
            if model_foundation is not None and not model_foundation.supports_validation_preview():
                logger.warning(
                    f"urepa_use_tae=True but {model_foundation.NAME} does not support TAE. "
                    "Falling back to full VAE decoding."
                )
                self.use_tae = False

    def _create_scheduler(self, config, max_train_steps: int) -> CrepaScheduler:
        """Create a scheduler using U-REPA specific config values."""

        class UrepaConfig:
            pass

        sched_config = UrepaConfig()
        sched_config.crepa_scheduler = getattr(config, "urepa_scheduler", "constant")
        sched_config.crepa_lambda = getattr(config, "urepa_lambda", 0.5)
        sched_config.crepa_warmup_steps = getattr(config, "urepa_warmup_steps", 0)
        sched_config.crepa_decay_steps = getattr(config, "urepa_decay_steps", 0)
        sched_config.crepa_lambda_end = getattr(config, "urepa_lambda_end", 0.0)
        sched_config.crepa_cutoff_step = getattr(config, "urepa_cutoff_step", 0)
        sched_config.crepa_similarity_threshold = getattr(config, "urepa_similarity_threshold", None)
        sched_config.crepa_similarity_ema_decay = getattr(config, "urepa_similarity_ema_decay", 0.99)
        sched_config.crepa_threshold_mode = getattr(config, "urepa_threshold_mode", "permanent")
        sched_config.crepa_power = getattr(config, "urepa_power", 1.0)
        return CrepaScheduler(sched_config, max_train_steps)

    def attach_to_model(self, model: nn.Module):
        """Attach the projection head to the diffusion backbone so it is optimized."""
        if not self.enabled:
            return

        self._load_encoder()
        target_dim = self.encoder_dim
        if target_dim is None:
            raise RuntimeError("U-REPA failed to determine encoder output dimension.")

        if self.projector is None:
            self.projector = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, target_dim),
            )
            setattr(model, "urepa_projector", self.projector)

        self.projector.to(device=self.device, dtype=torch.float32)

    def wants_hidden_states(self) -> bool:
        return self.enabled

    def compute_loss(
        self,
        hidden_states: Optional[torch.Tensor],
        latents: Optional[torch.Tensor],
        vae: Optional[nn.Module] = None,
        *,
        step: int = 0,
    ) -> Tuple[Optional[torch.Tensor], Optional[dict]]:
        """
        Compute U-REPA loss combining cosine alignment and manifold loss.

        Args:
            hidden_states: Mid-block features in (B, C, H, W) format
            latents: Clean latents for decoding to pixel space
            vae: VAE for decoding (optional if model_foundation is set)
            step: Current training step for scheduling

        Returns:
            Tuple of (loss tensor, logging dict)
        """
        if not self.enabled:
            return None, None
        if hidden_states is None:
            raise ValueError("U-REPA is enabled but no intermediate hidden states were provided.")
        if latents is None:
            raise ValueError("U-REPA requires access to clean latents for decoding.")
        if self.model_foundation is None and vae is None:
            raise ValueError("U-REPA requires a VAE to decode latents back to pixel space.")
        if self.projector is None:
            raise RuntimeError("U-REPA projector was not initialized on the diffusion model.")
        if self.base_weight == 0:
            return None, None

        # Decode latents to pixel space and encode with vision encoder
        pixels = self._decode_latents(latents, vae)
        encoder_features = self._encode_pixels(pixels)  # (B, N_enc, D_enc)

        # Project hidden states from (B, C, H, W) to (B, N_proj, D_enc)
        projected = self._project_hidden_states(hidden_states)

        # Align spatial dimensions
        projected, encoder_features = self._align_spatial(projected, encoder_features)

        # Normalize for cosine similarity
        projected_norm = F.normalize(projected, dim=-1)
        encoder_norm = F.normalize(encoder_features, dim=-1)

        # Cosine alignment loss (per-token, averaged)
        cosine_sim = (projected_norm * encoder_norm).sum(dim=-1).mean(dim=-1)  # (B,)
        alignment_loss = -cosine_sim.mean()

        # Manifold loss: align relative similarity structure
        manifold_loss = self._compute_manifold_loss(projected_norm, encoder_norm)

        # Get scheduled weight
        current_similarity = cosine_sim.mean().detach().item()
        if self.scheduler is not None:
            scheduled_weight = self.scheduler.get_weight(step, similarity=current_similarity)
        else:
            scheduled_weight = self.base_weight

        # Early exit if weight is zero
        if scheduled_weight == 0:
            log_data = {
                "urepa_loss": 0.0,
                "urepa_alignment_loss": 0.0,
                "urepa_manifold_loss": 0.0,
                "urepa_similarity": current_similarity,
                "urepa_weight": 0.0,
                "urepa_cutoff": True,
            }
            return None, log_data

        # Combine losses: alignment + weighted manifold
        total_loss = scheduled_weight * (alignment_loss + self.manifold_weight * manifold_loss)

        log_data = {
            "urepa_loss": total_loss.detach().item(),
            "urepa_alignment_loss": alignment_loss.detach().item(),
            "urepa_manifold_loss": manifold_loss.detach().item(),
            "urepa_similarity": current_similarity,
            "urepa_weight": scheduled_weight,
            "urepa_cutoff": False,
        }
        return total_loss, log_data

    def _project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project UNet mid-block features from (B, C, H, W) to (B, H*W, D_enc).

        The mid-block produces convolutional features that need to be:
        1. Reshaped to sequence format (B, H*W, C)
        2. Projected to encoder dimension
        """
        if hidden_states.ndim != 4:
            raise ValueError(f"U-REPA expected 4D hidden states (B, C, H, W), got {hidden_states.shape}")

        b, c, h, w = hidden_states.shape
        # Reshape: (B, C, H, W) -> (B, H*W, C)
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(b, h * w, c)

        # Project to encoder dimension
        projector_dtype = next(self.projector.parameters()).dtype
        hidden_states = hidden_states.to(dtype=projector_dtype)
        projected = self.projector(hidden_states)
        return projected

    def _align_spatial(self, projected: torch.Tensor, encoder_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align spatial token counts between projected features and encoder features.

        Uses bilinear interpolation to match the smaller spatial dimension.
        """
        n_proj = projected.shape[1]
        n_enc = encoder_features.shape[1]

        if n_proj == n_enc:
            return projected, encoder_features

        target_tokens = min(n_proj, n_enc)
        projected = self._interpolate_tokens(projected, target_tokens)
        encoder_features = self._interpolate_tokens(encoder_features, target_tokens)
        return projected, encoder_features

    def _interpolate_tokens(self, tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
        """Interpolate token sequence to target length using bilinear interpolation."""
        if tokens.shape[1] == target_tokens:
            return tokens
        b, n, d = tokens.shape

        # Try 2D interpolation if tokens form a square grid
        src_size = int(math.sqrt(n))
        tgt_size = int(math.sqrt(target_tokens))
        if src_size * src_size == n and tgt_size * tgt_size == target_tokens:
            tokens = tokens.permute(0, 2, 1).view(b, d, src_size, src_size)
            tokens = F.interpolate(tokens, size=(tgt_size, tgt_size), mode="bilinear", align_corners=False)
            tokens = tokens.view(b, d, target_tokens).permute(0, 2, 1)
        else:
            # Fallback to 1D interpolation
            tokens = tokens.permute(0, 2, 1)  # (B, D, N)
            tokens = F.interpolate(tokens, size=target_tokens, mode="linear", align_corners=False)
            tokens = tokens.permute(0, 2, 1)  # (B, N, D)

        return tokens

    def _compute_manifold_loss(self, projected: torch.Tensor, encoder_features: torch.Tensor) -> torch.Tensor:
        """
        Compute manifold loss: ||sim(y[i],y[j]) - sim(h[i],h[j])||²_F

        This aligns the relative similarity structure between encoder features (y)
        and projected hidden states (h), rather than strict tokenwise alignment.

        For efficiency, we compute pairwise similarities within each sample.
        """
        # Compute pairwise similarity matrices
        # projected @ projected.T gives (B, N, N) similarity matrix
        proj_sim = torch.bmm(projected, projected.transpose(1, 2))  # (B, N, N)
        enc_sim = torch.bmm(encoder_features, encoder_features.transpose(1, 2))  # (B, N, N)

        # Frobenius norm of difference, normalized by number of elements
        diff = proj_sim - enc_sim
        manifold_loss = (diff**2).mean()

        return manifold_loss

    def _decode_latents(self, latents: torch.Tensor, vae: Optional[nn.Module]) -> torch.Tensor:
        """Decode latents to pixel space."""
        if self.model_foundation is not None:
            return self.model_foundation.decode_latents_to_pixels(latents, use_tae=self.use_tae)

        # Legacy VAE decode
        vae_dtype = next(vae.parameters()).dtype
        latents = latents.to(device=self.device, dtype=vae_dtype)
        scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", 1.0)
        shift_factor = getattr(getattr(vae, "config", None), "shift_factor", None)

        if shift_factor is not None:
            latents = latents / scaling_factor + shift_factor
        else:
            latents = latents / scaling_factor

        with torch.no_grad():
            decoded = vae.decode(latents).sample
        decoded = decoded.clamp(-1, 1)
        decoded = (decoded + 1.0) * 0.5  # Convert to [0, 1]
        return decoded

    def _encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """
        Encode pixels using vision encoder.

        Args:
            pixels: (B, C, H, W) in [0, 1] range for images, or (B, T, C, H, W) for video

        Returns:
            Encoded features (B, N, D) for images, or (B, T, N, D) for video
        """
        # Handle video format by taking first frame (U-REPA is for image models)
        if pixels.ndim == 5:
            # Video: (B, T, C, H, W) -> (B, C, H, W) using first frame
            pixels = pixels[:, 0]

        # Resize to encoder input size
        pixels = F.interpolate(
            pixels, size=(self.encoder_image_size, self.encoder_image_size), mode="bilinear", align_corners=False
        )

        # Normalize with ImageNet stats
        enc_dtype = next(self.encoder.parameters()).dtype
        pixels = pixels.to(dtype=enc_dtype)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=enc_dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=enc_dtype).view(1, 3, 1, 1)
        pixels = (pixels - mean) / std

        # Encode
        with torch.no_grad():
            output = self.encoder(pixels)

        if isinstance(output, dict):
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

        return tokens

    def _load_encoder(self):
        """Load the vision encoder (DINOv2 by default)."""
        if self.encoder is not None:
            return

        self.encoder = torch.hub.load("facebookresearch/dinov2", self.encoder_name)
        self.encoder.eval().requires_grad_(False).to(self.device, dtype=torch.float32)

        # Probe encoder dimension
        dummy = torch.zeros(1, 3, self.encoder_image_size, self.encoder_image_size, device=self.device)
        with torch.no_grad():
            output = self.encoder(dummy)
            if isinstance(output, dict):
                if "x_norm_patchtokens" in output:
                    self.encoder_dim = output["x_norm_patchtokens"].shape[-1]
                elif "x_norm_clstoken" in output:
                    self.encoder_dim = output["x_norm_clstoken"].shape[-1]
                else:
                    self.encoder_dim = next(iter(output.values())).shape[-1]
            else:
                self.encoder_dim = output.shape[-1]

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

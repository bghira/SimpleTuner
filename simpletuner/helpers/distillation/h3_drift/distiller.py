from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from simpletuner.helpers.distillation.common import DistillationBase
from simpletuner.helpers.distillation.registry import DistillationRegistry


@dataclass(frozen=True)
class _H3Prediction:
    video: Optional[torch.Tensor]
    audio: Optional[torch.Tensor]


@dataclass(frozen=True)
class _H3JointLoss:
    loss: torch.Tensor
    video_loss: torch.Tensor
    audio_loss: torch.Tensor
    video_elements: int
    audio_elements: int


class H3DriftDistiller(DistillationBase):
    """Regularize MiniMax-H3 LoRA/LyCORIS training against the frozen base prediction."""

    _DEFAULTS: Dict[str, Any] = {
        "distillation_type": "h3_drift",
        "loss_weight": 1.0,
        "sft_loss_weight": 1.0,
        "balance": "token",
        "video_weight": 1.0,
        "audio_weight": 1.0,
    }

    def __init__(
        self,
        teacher_model,
        student_model=None,
        *,
        noise_scheduler=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        merged_config = dict(self._DEFAULTS)
        if config:
            merged_config.update(config)

        super().__init__(teacher_model, student_model, merged_config)
        self.noise_scheduler = noise_scheduler

        model_family = str(self.config.get("model_family") or "").lower().replace("_", "")
        model_name = str(getattr(teacher_model, "NAME", "") or "").lower().replace(" ", "").replace("-", "")
        if model_family not in {"minimaxh3", ""} and model_name != "minimaxh3":
            raise ValueError("H3 drift distillation only supports MiniMax-H3.")
        if not self.is_flow_matching:
            raise ValueError("H3 drift distillation requires a flow-matching model.")
        if not self.low_rank_distillation or self.config.get("model_type") != "lora":
            raise ValueError("H3 drift distillation only supports low-rank LoRA/LyCORIS training.")

        balance = str(self.config.get("balance", "token")).lower()
        if balance not in {"token", "modality"}:
            raise ValueError("H3 drift balance must be one of: token, modality.")
        self.config["balance"] = balance

        video_weight = float(self.config.get("video_weight", 1.0))
        audio_weight = float(self.config.get("audio_weight", 1.0))
        if video_weight < 0 or audio_weight < 0 or video_weight + audio_weight <= 0:
            raise ValueError("H3 drift video/audio weights must be non-negative and not both zero.")
        self.config["video_weight"] = video_weight
        self.config["audio_weight"] = audio_weight

    def compute_distill_loss(
        self,
        prepared_batch: Dict[str, Any],
        model_output: Dict[str, Any],
        original_loss: torch.Tensor,
    ):
        prediction = self._prediction_from_output(model_output)
        try:
            self.toggle_adapter(enable=False)
            with torch.no_grad():
                reference_output = self.teacher_model.model_predict(prepared_batch)
        finally:
            self.toggle_adapter(enable=True)

        reference = self._prediction_from_output(reference_output)
        self._clear_reference_buffers(reference_output)

        joint_loss = self._joint_prediction_loss(
            prediction,
            reference,
            video_mask=self._video_mask_for_loss(prepared_batch, prediction.video),
            audio_mask=prepared_batch.get("audio_latent_mask"),
            sample_weight=prepared_batch.get("sample_weight"),
            balance=self.config["balance"],
            video_weight=self.config["video_weight"],
            audio_weight=self.config["audio_weight"],
        )

        drift_loss = joint_loss.loss * float(self.config.get("loss_weight", 1.0))
        sft_loss_weight = float(self.config.get("sft_loss_weight", 1.0))
        loss = drift_loss + original_loss * sft_loss_weight

        logs = {
            "h3_drift_loss": float(joint_loss.loss.detach()),
            "h3_drift_video_loss": float(joint_loss.video_loss.detach()),
            "h3_drift_audio_loss": float(joint_loss.audio_loss.detach()),
            "h3_drift_video_elements": float(joint_loss.video_elements),
            "h3_drift_audio_elements": float(joint_loss.audio_elements),
            "h3_drift_weighted_loss": float(drift_loss.detach()),
            "total": float(loss.detach()),
        }
        if sft_loss_weight != 0.0:
            logs["h3_drift_sft_loss"] = float((original_loss * sft_loss_weight).detach())
        return loss, logs

    @staticmethod
    def _prediction_from_output(output: Dict[str, Any]) -> _H3Prediction:
        video = output.get("model_prediction")
        audio = output.get("audio_prediction")
        if video is not None and not torch.is_tensor(video):
            raise ValueError(f"H3 drift video prediction must be a tensor or None, got {type(video)}.")
        if audio is not None and not torch.is_tensor(audio):
            raise ValueError(f"H3 drift audio prediction must be a tensor or None, got {type(audio)}.")
        return _H3Prediction(video=video, audio=audio)

    @staticmethod
    def _clear_reference_buffers(output: Dict[str, Any]) -> None:
        hidden_states_buffer = output.get("hidden_states_buffer")
        if isinstance(hidden_states_buffer, dict):
            hidden_states_buffer.clear()

    @staticmethod
    def _broadcast_mask(mask: Optional[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return torch.ones_like(target, dtype=torch.bool)
        if not torch.is_tensor(mask):
            raise ValueError(f"H3 drift loss mask must be a tensor, got {type(mask)}.")
        mask = mask.to(device=target.device, dtype=torch.bool)
        if mask.shape == target.shape:
            return mask
        if mask.ndim == 1:
            if mask.shape[0] != target.shape[0]:
                raise ValueError(f"H3 drift loss mask batch {mask.shape[0]} does not match target batch {target.shape[0]}.")
            mask = mask.view(mask.shape[0], *([1] * (target.ndim - 1)))
        elif mask.ndim < target.ndim:
            if mask.shape[0] != target.shape[0]:
                raise ValueError(f"H3 drift loss mask batch {mask.shape[0]} does not match target batch {target.shape[0]}.")
            mask = mask.view(mask.shape[0], *([1] * (target.ndim - mask.ndim)), *mask.shape[1:])
        try:
            return mask.expand_as(target)
        except RuntimeError as exc:
            raise ValueError(
                f"H3 drift loss mask shape {tuple(mask.shape)} cannot broadcast to {tuple(target.shape)}."
            ) from exc

    @staticmethod
    def _modality_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
        sample_weight: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if prediction.shape != target.shape:
            raise ValueError(
                f"H3 drift prediction shape {tuple(prediction.shape)} does not match reference {tuple(target.shape)}."
            )
        valid = H3DriftDistiller._broadcast_mask(mask, target)
        elements = int(valid.sum().item())
        if elements == 0:
            zero = prediction.sum() * 0.0
            return zero, zero, 0

        squared = (prediction.float() - target.detach().float()).square()
        if sample_weight is not None:
            if not torch.is_tensor(sample_weight):
                raise ValueError(f"H3 drift sample_weight must be a tensor, got {type(sample_weight)}.")
            if sample_weight.shape != (target.shape[0],):
                raise ValueError("H3 drift sample_weight must contain one value per batch item.")
            weight = sample_weight.to(device=squared.device, dtype=squared.dtype)
            squared = squared * weight.view(weight.shape[0], *([1] * (squared.ndim - 1)))
        total = squared.masked_select(valid).sum()
        return total / elements, total, elements

    @staticmethod
    def _joint_prediction_loss(
        prediction: _H3Prediction,
        target: _H3Prediction,
        *,
        video_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
        balance: str = "token",
        video_weight: float = 1.0,
        audio_weight: float = 1.0,
    ) -> _H3JointLoss:
        if balance not in {"token", "modality"}:
            raise ValueError(f"Unsupported H3 drift loss balance: {balance}.")
        if video_weight < 0 or audio_weight < 0 or video_weight + audio_weight <= 0:
            raise ValueError("H3 drift video/audio loss weights must be non-negative and not both zero.")

        zero_source = prediction.video if prediction.video is not None else prediction.audio
        if zero_source is None:
            raise ValueError("H3 drift prediction contains no target modality.")
        zero = zero_source.sum() * 0.0

        if prediction.video is None or target.video is None:
            if prediction.video is not None or target.video is not None:
                raise ValueError("H3 drift video prediction and reference presence differ.")
            video_mean, video_total, video_elements = zero, zero, 0
        else:
            video_mean, video_total, video_elements = H3DriftDistiller._modality_loss(
                prediction.video, target.video, video_mask, sample_weight
            )

        if prediction.audio is None or target.audio is None:
            if prediction.audio is not None or target.audio is not None:
                raise ValueError("H3 drift audio prediction and reference presence differ.")
            audio_mean, audio_total, audio_elements = zero, zero, 0
        else:
            audio_mean, audio_total, audio_elements = H3DriftDistiller._modality_loss(
                prediction.audio, target.audio, audio_mask, sample_weight
            )

        active_video_weight = video_weight if video_elements else 0.0
        active_audio_weight = audio_weight if audio_elements else 0.0
        if active_video_weight + active_audio_weight == 0:
            raise ValueError("H3 drift loss masks exclude every video and audio element.")

        if balance == "modality":
            loss = (active_video_weight * video_mean + active_audio_weight * audio_mean) / (
                active_video_weight + active_audio_weight
            )
        else:
            weighted_elements = active_video_weight * video_elements + active_audio_weight * audio_elements
            loss = (active_video_weight * video_total + active_audio_weight * audio_total) / weighted_elements

        return _H3JointLoss(loss, video_mean, audio_mean, video_elements, audio_elements)

    @staticmethod
    def _video_mask_for_loss(
        prepared_batch: Dict[str, Any],
        prediction: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if prediction is None or prepared_batch.get("loss_mask_type") not in {"mask", "segmentation"}:
            return None
        mask_image = prepared_batch.get("conditioning_pixel_values")
        if isinstance(mask_image, list):
            mask_image = mask_image[-1] if mask_image else None
        if not torch.is_tensor(mask_image):
            return None

        mask_image = mask_image.to(device=prediction.device, dtype=prediction.dtype)
        if mask_image.dim() == 3:
            mask_image = mask_image.unsqueeze(1)
        if mask_image.dim() == 4:
            if prepared_batch.get("loss_mask_type") == "segmentation":
                mask_image = torch.sum(mask_image, dim=1, keepdim=True) / mask_image.shape[1]
            elif mask_image.shape[1] > 1:
                mask_image = mask_image[:, 0:1]
            if prediction.dim() == 5:
                mask_image = mask_image.unsqueeze(2)
        elif mask_image.dim() == 5:
            if prepared_batch.get("loss_mask_type") == "segmentation":
                mask_image = torch.sum(mask_image, dim=1, keepdim=True) / mask_image.shape[1]
            elif mask_image.shape[1] > 1:
                mask_image = mask_image[:, 0:1]

        if mask_image.dim() != prediction.dim():
            raise ValueError(
                f"H3 drift video mask rank must match prediction rank. Got {mask_image.dim()} vs {prediction.dim()}."
            )
        mask_image = F.interpolate(mask_image, size=prediction.shape[2:], mode="area")
        mask_image = (mask_image / 2 + 0.5).clamp(0.0, 1.0)
        if prepared_batch.get("loss_mask_type") == "segmentation":
            mask_image = (mask_image > 0).to(dtype=prediction.dtype)
        return mask_image


DistillationRegistry.register(
    "h3_drift",
    H3DriftDistiller,
    requires_distillation_cache=False,
    requirement_notes="MiniMax-H3 LoRA/LyCORIS-only base-prediction drift regularizer.",
)

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.distillation.common import DistillationBase
from simpletuner.helpers.distillation.registry import DistillationRegistry


class FlowDPODistiller(DistillationBase):
    """Flow-DPO for paired preferred/rejected samples using low-rank adapters."""

    _DEFAULTS: Dict[str, Any] = {
        "distillation_type": "flow_dpo",
        "beta": 1.0,
        "loss_weight": 1.0,
        "sft_loss_weight": 0.0,
        "anchor_alpha": 0.0,
        "norm_type": "sum",
        "mask_dilate": 0,
        "auto_beta": True,
        "auto_beta_target_gf": 0.2,
        "auto_beta_decay": 0.99,
        "auto_beta_min": 1e-3,
        "auto_beta_max": 1.0,
        "auto_beta_eps": 1e-6,
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
        self.margin_ema: Optional[float] = None

        if not self.is_flow_matching:
            raise ValueError("Flow-DPO requires a flow-matching model.")
        if not self.low_rank_distillation or self.config.get("model_type") != "lora":
            raise ValueError("Flow-DPO only supports low-rank LoRA/LyCORIS training.")

        norm_type = str(self.config.get("norm_type", "sum")).lower()
        if norm_type not in {"sum", "mean", "masked_mean"}:
            raise ValueError("Flow-DPO norm_type must be one of: sum, mean, masked_mean.")
        self.config["norm_type"] = norm_type
        target_gf = float(self.config.get("auto_beta_target_gf", 0.0) or 0.0)
        if bool(self.config.get("auto_beta", True)) and target_gf >= 0.5:
            raise ValueError("Flow-DPO auto_beta_target_gf must be less than 0.5.")

    def compute_distill_loss(
        self,
        prepared_batch: Dict[str, Any],
        model_output: Dict[str, Any],
        original_loss: torch.Tensor,
    ):
        self._validate_batch(prepared_batch)

        policy_win = model_output["model_prediction"]
        win_latents = prepared_batch["latents"]
        lose_latents = self._conditioning_latents(prepared_batch).to(device=win_latents.device, dtype=win_latents.dtype)
        if lose_latents.shape != win_latents.shape:
            raise ValueError(
                f"Flow-DPO rejected latents must match preferred latents. Got {lose_latents.shape} vs {win_latents.shape}."
            )

        lose_batch = self._build_rejected_batch(prepared_batch, lose_latents)
        policy_lose = self.student_model.model_predict(lose_batch)["model_prediction"]

        try:
            self.toggle_adapter(enable=False)
            with torch.no_grad():
                ref_win = self.teacher_model.model_predict(prepared_batch)["model_prediction"]
                ref_lose = self.teacher_model.model_predict(lose_batch)["model_prediction"]
        finally:
            self.toggle_adapter(enable=True)

        win_target = prepared_batch["noise"] - win_latents
        lose_target = prepared_batch["noise"] - lose_latents
        mask = self._mask_for_loss(prepared_batch, policy_win)

        policy_win_err = self._per_sample_error(policy_win, win_target, mask)
        ref_win_err = self._per_sample_error(ref_win, win_target, mask)
        policy_lose_err = self._per_sample_error(policy_lose, lose_target, mask)
        ref_lose_err = self._per_sample_error(ref_lose, lose_target, mask)

        win_adv = ref_win_err - policy_win_err
        lose_adv = policy_lose_err - ref_lose_err
        margin = win_adv + lose_adv
        beta = self._beta_for_margin(margin)

        logits = 0.5 * beta * margin.float()
        dpo_loss = -F.logsigmoid(logits).mean()

        anchor_loss = policy_win.new_zeros(())
        anchor_alpha = float(self.config.get("anchor_alpha", 0.0) or 0.0)
        if anchor_alpha != 0.0:
            anchor_loss = (
                0.5
                * (F.mse_loss(policy_win.float(), ref_win.float()) + F.mse_loss(policy_lose.float(), ref_lose.float()))
                * anchor_alpha
            )

        loss = dpo_loss * float(self.config.get("loss_weight", 1.0)) + anchor_loss
        sft_loss_weight = float(self.config.get("sft_loss_weight", 0.0) or 0.0)
        if sft_loss_weight != 0.0:
            loss = loss + original_loss * sft_loss_weight

        gradient_factor = torch.sigmoid(-logits.detach()).mean()
        negative_margin_pct = (margin.detach() < 0).float().mean() * 100.0
        logs = {
            "flow_dpo_loss": float(dpo_loss.detach()),
            "flow_dpo_beta": float(beta.detach()),
            "flow_dpo_margin": float(margin.detach().mean()),
            "flow_dpo_win_adv": float(win_adv.detach().mean()),
            "flow_dpo_lose_adv": float(lose_adv.detach().mean()),
            "flow_dpo_policy_win_err": float(policy_win_err.detach().mean()),
            "flow_dpo_policy_lose_err": float(policy_lose_err.detach().mean()),
            "flow_dpo_ref_win_err": float(ref_win_err.detach().mean()),
            "flow_dpo_ref_lose_err": float(ref_lose_err.detach().mean()),
            "flow_dpo_negative_margin_pct": float(negative_margin_pct),
            "flow_dpo_gradient_factor": float(gradient_factor),
            "total": float(loss.detach()),
        }
        if anchor_alpha != 0.0:
            logs["flow_dpo_anchor_loss"] = float(anchor_loss.detach())

        return loss, logs

    def _validate_batch(self, prepared_batch: Dict[str, Any]) -> None:
        if prepared_batch.get("conditioning_type") != "reference_strict":
            raise ValueError(
                "Flow-DPO requires a rejected-sample conditioning dataset with conditioning_type=reference_strict."
            )
        if prepared_batch.get("conditioning_latents") is None:
            raise ValueError("Flow-DPO requires conditioning_latents from the rejected-sample dataset.")
        required_keys = ("latents", "noise", "input_noise", "sigmas", "timesteps", "noisy_latents")
        missing = [key for key in required_keys if key not in prepared_batch or prepared_batch[key] is None]
        if missing:
            raise ValueError(f"Flow-DPO prepared batch is missing required fields: {', '.join(missing)}.")

    @staticmethod
    def _conditioning_latents(prepared_batch: Dict[str, Any]) -> torch.Tensor:
        latents = prepared_batch["conditioning_latents"]
        if isinstance(latents, list):
            latent_types = prepared_batch.get("conditioning_latents_type")
            if isinstance(latent_types, list) and "reference_strict" in latent_types:
                latents = latents[latent_types.index("reference_strict")]
            elif len(latents) == 1:
                latents = latents[0]
            else:
                raise ValueError("Flow-DPO requires one reference_strict conditioning latent set.")
        if not torch.is_tensor(latents):
            raise ValueError("Flow-DPO conditioning_latents must be a tensor.")
        return latents

    @staticmethod
    def _build_rejected_batch(prepared_batch: Dict[str, Any], rejected_latents: torch.Tensor) -> Dict[str, Any]:
        rejected_batch = dict(prepared_batch)
        rejected_batch["latents"] = rejected_latents
        rejected_batch["clean_latents"] = rejected_latents
        rejected_batch["noisy_latents"] = (1 - prepared_batch["sigmas"]) * rejected_latents + prepared_batch[
            "sigmas"
        ] * prepared_batch["input_noise"]
        return rejected_batch

    def _mask_for_loss(self, prepared_batch: Dict[str, Any], prediction: torch.Tensor) -> Optional[torch.Tensor]:
        loss_mask_type = prepared_batch.get("loss_mask_type")
        if loss_mask_type not in {"mask", "segmentation"}:
            return None

        mask_image = prepared_batch.get("conditioning_pixel_values")
        if isinstance(mask_image, list):
            if not mask_image:
                return None
            mask_image = mask_image[-1]
        if not torch.is_tensor(mask_image):
            raise ValueError("Flow-DPO loss masking requires conditioning_pixel_values to be a tensor.")

        mask_image = mask_image.to(device=prediction.device, dtype=prediction.dtype)
        if mask_image.dim() == 3:
            mask_image = mask_image.unsqueeze(1)
        if mask_image.dim() == 4:
            if loss_mask_type == "segmentation":
                mask_image = torch.sum(mask_image, dim=1, keepdim=True) / mask_image.shape[1]
            elif mask_image.shape[1] > 1:
                mask_image = mask_image[:, 0:1]
            if prediction.dim() == 5:
                mask_image = mask_image.unsqueeze(2)
        elif mask_image.dim() == 5:
            if loss_mask_type == "segmentation":
                mask_image = torch.sum(mask_image, dim=1, keepdim=True) / mask_image.shape[1]
            elif mask_image.shape[1] > 1:
                mask_image = mask_image[:, 0:1]

        if mask_image.dim() != prediction.dim():
            raise ValueError(f"Flow-DPO mask rank must match prediction rank. Got {mask_image.dim()} vs {prediction.dim()}.")

        mask_image = F.interpolate(mask_image, size=prediction.shape[2:], mode="area")
        mask_image = mask_image / 2 + 0.5
        mask_image = mask_image.clamp(0.0, 1.0)
        if loss_mask_type == "segmentation":
            mask_image = (mask_image > 0).to(dtype=prediction.dtype)

        dilation = int(self.config.get("mask_dilate", 0) or 0)
        if dilation > 0:
            kernel_size = 2 * dilation + 1
            if mask_image.dim() == 4:
                mask_image = F.max_pool2d(mask_image, kernel_size=kernel_size, stride=1, padding=dilation)
            elif mask_image.dim() == 5:
                mask_image = F.max_pool3d(
                    mask_image,
                    kernel_size=(1, kernel_size, kernel_size),
                    stride=1,
                    padding=(0, dilation, dilation),
                )
        return mask_image

    def _per_sample_error(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        error = (prediction.float() - target.float()).pow(2)
        mask_float = None
        if mask is not None:
            mask_float = mask.float()
            error = error * mask_float

        reduction_dims = tuple(range(1, error.dim()))
        norm_type = self.config["norm_type"]
        if norm_type == "sum":
            return error.sum(dim=reduction_dims)
        if norm_type == "mean":
            return error.mean(dim=reduction_dims)

        if mask is None:
            return error.mean(dim=reduction_dims)
        denominator_mask = mask_float
        if denominator_mask.shape != error.shape:
            denominator_mask = denominator_mask.expand_as(error)
        denominator = denominator_mask.sum(dim=reduction_dims)
        return error.sum(dim=reduction_dims) / denominator.clamp_min(1.0)

    def _beta_for_margin(self, margin: torch.Tensor) -> torch.Tensor:
        fixed_beta = float(self.config.get("beta", 1.0))
        target_gf = float(self.config.get("auto_beta_target_gf", 0.0) or 0.0)
        if not bool(self.config.get("auto_beta", True)) or target_gf <= 0.0:
            return margin.new_tensor(fixed_beta, dtype=torch.float32)

        margin_value = margin.detach().abs().mean()
        accelerator = getattr(self.teacher_model, "accelerator", None)
        if accelerator is not None and int(getattr(accelerator, "num_processes", 1)) > 1:
            margin_value = accelerator.gather(margin_value.reshape(1)).mean()

        margin_mean = float(margin_value.detach().cpu())
        if self.margin_ema is None:
            self.margin_ema = margin_mean
        else:
            decay = float(self.config.get("auto_beta_decay", 0.99))
            self.margin_ema = decay * self.margin_ema + (1.0 - decay) * margin_mean

        eps = float(self.config.get("auto_beta_eps", 1e-6))
        target = min(max(target_gf, eps), 1.0 - eps)
        beta_value = (-2.0 * torch.logit(torch.tensor(target)).item()) / max(self.margin_ema, eps)

        beta_min = self.config.get("auto_beta_min")
        beta_max = self.config.get("auto_beta_max")
        if beta_min is not None:
            beta_value = max(float(beta_min), beta_value)
        if beta_max is not None:
            beta_value = min(float(beta_max), beta_value)

        return margin.new_tensor(beta_value, dtype=torch.float32)


DistillationRegistry.register(
    "flow_dpo",
    FlowDPODistiller,
    requires_distillation_cache=False,
    data_requirements=[[DatasetType.IMAGE, DatasetType.VIDEO], DatasetType.CONDITIONING],
    requirement_notes="Requires a paired conditioning dataset with conditioning_type=reference_strict for rejected samples.",
)

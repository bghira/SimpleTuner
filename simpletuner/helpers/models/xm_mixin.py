from dataclasses import fields, is_dataclass, replace

import torch
import torch.nn.functional as F

from simpletuner.helpers.training.explorative_modeling import (
    reduce_loss_to_samples,
    reshape_candidate_batch,
    route_usage_histogram,
    select_min_candidate_loss,
    select_winning_candidates,
)
from simpletuner.helpers.training.min_snr_gamma import compute_snr


class ExplorativeModelingMixin:
    def _prediction_type_value(self) -> str:
        prediction_type = self.PREDICTION_TYPE
        return str(getattr(prediction_type, "value", prediction_type))

    def _validate_xm_diffusion_support(
        self,
        *,
        family_name: str | None = None,
        support_twinflow: bool = False,
        support_scheduled_sampling: bool = False,
        support_input_perturbation: bool = False,
        support_crepa_self_flow: bool = False,
        support_block_size: bool = False,
    ) -> None:
        xm_config = getattr(self, "xm_config", None)
        if xm_config is None or not xm_config.enabled:
            return
        label = family_name or self.NAME
        if xm_config.training_target != "noise":
            raise ValueError(f"{label} XM currently supports only xm_training_target='noise'.")
        if xm_config.selection_scope != "sample":
            raise ValueError(f"{label} XM noise-candidate training requires xm_selection_scope='sample'.")
        if not support_block_size and int(getattr(xm_config, "block_size", 0) or 0) != 0:
            raise ValueError(f"{label} XM noise-candidate training requires xm_block_size=0.")
        if not support_twinflow and getattr(self.config, "twinflow_enabled", False):
            raise ValueError(f"{label} XM noise-candidate training is not compatible with TwinFlow.")
        if not support_scheduled_sampling and (
            getattr(self.config, "scheduled_sampling_reflexflow", False)
            or int(getattr(self.config, "scheduled_sampling_max_step_offset", 0) or 0) > 0
        ):
            raise ValueError(f"{label} XM noise-candidate training is not compatible with scheduled sampling.")
        if not support_input_perturbation and float(getattr(self.config, "input_perturbation", 0.0) or 0.0) != 0.0:
            raise ValueError(f"{label} XM noise-candidate training is not compatible with input_perturbation.")
        if not support_crepa_self_flow and (
            getattr(self.config, "crepa_self_flow", False)
            or getattr(self.config, "crepa_feature_source", None) == "self_flow"
        ):
            raise ValueError(f"{label} XM noise-candidate training is not compatible with CREPA self-flow.")

    def _validate_xm_support(self) -> None:
        self._validate_xm_diffusion_support(family_name=self.NAME)

    def _xm_noise_candidates_enabled(self, prepared_batch: dict | None = None) -> bool:
        xm_config = getattr(self, "xm_config", None)
        if not xm_config or not xm_config.enabled:
            return False
        self._validate_xm_support()
        if prepared_batch is not None and (
            prepared_batch.get("xm_candidate_count") or prepared_batch.get("xm_winner_indices") is not None
        ):
            return False
        return xm_config.training_target == "noise"

    @staticmethod
    def _repeat_xm_candidate_tensor(value: torch.Tensor, candidate_count: int) -> torch.Tensor:
        repeat_shape = (candidate_count,) + (1,) * (value.ndim - 1)
        return value.repeat(repeat_shape)

    def _repeat_xm_candidate_value(self, value, candidate_count: int, batch_size: int):
        if torch.is_tensor(value):
            if value.ndim == 0 or value.shape[0] != batch_size:
                return value
            return self._repeat_xm_candidate_tensor(value, candidate_count)
        if isinstance(value, list):
            return [self._repeat_xm_candidate_value(item, candidate_count, batch_size) for item in value]
        if isinstance(value, tuple):
            return tuple(self._repeat_xm_candidate_value(item, candidate_count, batch_size) for item in value)
        if isinstance(value, dict):
            return {key: self._repeat_xm_candidate_value(item, candidate_count, batch_size) for key, item in value.items()}
        if is_dataclass(value) and not isinstance(value, type):
            replacements = {
                field.name: self._repeat_xm_candidate_value(getattr(value, field.name), candidate_count, batch_size)
                for field in fields(value)
            }
            return replace(value, **replacements)
        return value

    def _prepare_xm_noise_candidates(self, prepared_batch: dict, *, family_name: str | None = None) -> dict:
        self._validate_xm_diffusion_support(family_name=family_name)
        candidate_count = self.xm_config.candidate_count
        latents = prepared_batch.get("latents")
        timesteps = prepared_batch.get("timesteps")
        if not torch.is_tensor(latents) or not torch.is_tensor(timesteps):
            label = family_name or self.NAME
            raise ValueError(f"{label} XM noise-candidate training requires latents and timesteps tensors.")
        if "noisy_latents" not in prepared_batch:
            label = family_name or self.NAME
            raise ValueError(f"{label} XM noise-candidate training requires prepared noisy_latents.")
        if prepared_batch.get("target") is not None:
            label = family_name or self.NAME
            raise ValueError(f"{label} XM noise-candidate training cannot be used with an explicit prepared target.")

        batch_size = latents.shape[0]
        expanded_batch = {
            key: self._repeat_xm_candidate_value(value, candidate_count, batch_size) for key, value in prepared_batch.items()
        }
        expanded_latents = expanded_batch["latents"]
        candidate_noise = torch.randn_like(expanded_latents)
        expanded_batch["noise"] = candidate_noise
        expanded_batch["input_noise"] = candidate_noise

        prediction_type = self._prediction_type_value()
        if prediction_type == "flow_matching":
            sigmas = expanded_batch.get("mixflow_interpolation_sigmas")
            if sigmas is None:
                sigmas = expanded_batch.get("sigmas")
            if not torch.is_tensor(sigmas):
                label = family_name or self.NAME
                raise ValueError(f"{label} XM noise-candidate training requires tensor sigmas for flow interpolation.")
            interpolation_grid = self._expand_sigma_values(sigmas, expanded_latents)
            expanded_batch["noisy_latents"] = (
                1.0 - interpolation_grid
            ) * expanded_latents + interpolation_grid * candidate_noise
            expanded_batch["flow_target"] = self.get_flow_matching_target(
                expanded_batch,
                latents=expanded_latents,
                noise=candidate_noise,
                prefer_explicit_target=False,
            ).to(device=expanded_latents.device, dtype=expanded_latents.dtype)
        elif prediction_type in ("epsilon", "v_prediction"):
            expanded_batch["noisy_latents"] = self.noise_schedule.add_noise(
                expanded_latents.float(),
                candidate_noise.float(),
                expanded_batch["timesteps"],
            ).to(device=expanded_latents.device, dtype=expanded_latents.dtype)
        elif prediction_type == "sample":
            sigmas = expanded_batch.get("sigmas")
            if torch.is_tensor(sigmas):
                interpolation_grid = self._expand_sigma_values(sigmas, expanded_latents)
                expanded_batch["noisy_latents"] = (
                    1.0 - interpolation_grid
                ) * expanded_latents + interpolation_grid * candidate_noise
            else:
                expanded_batch["noisy_latents"] = candidate_noise
        else:
            raise ValueError(f"{family_name or self.NAME} XM noise-candidate training does not support {prediction_type}.")

        expanded_batch["xm_candidate_count"] = candidate_count
        expanded_batch["xm_original_batch_size"] = batch_size
        prepared_batch.clear()
        prepared_batch.update(expanded_batch)
        return prepared_batch

    def _xm_diffusion_loss_tensor(
        self,
        prepared_batch: dict,
        model_output: dict,
        apply_conditioning_mask: bool,
        *,
        family_name: str | None = None,
    ) -> torch.Tensor:
        target = self.get_prediction_target(prepared_batch)
        model_pred = model_output["model_prediction"]
        if target is None:
            raise ValueError(f"Target is None. Cannot compute {family_name or self.NAME} XM loss.")

        loss_type = getattr(self.config, "loss_type", "l2")
        prediction_type = self._prediction_type_value()
        use_diff2flow_loss = (
            getattr(self.config, "diff2flow_loss", False)
            and getattr(self.config, "diff2flow_enabled", False)
            and self.diff2flow_bridge is not None
            and prediction_type in ("epsilon", "v_prediction")
        )

        if use_diff2flow_loss:
            flow_pred = self.diff2flow_bridge.prediction_to_flow(
                model_pred.float(),
                prepared_batch["noisy_latents"].float(),
                prepared_batch["timesteps"],
                prediction_type=prediction_type,
            )
            flow_target = self.get_flow_target(prepared_batch)
            if flow_target is None:
                raise ValueError("Flow target is None while diff2flow_loss is enabled.")
            loss = F.mse_loss(flow_pred.float(), flow_target.float(), reduction="none")
        elif prediction_type == "flow_matching":
            if loss_type in ["huber", "smooth_l1"]:
                loss = self._xm_huber_loss_tensor(model_pred, target, prepared_batch["timesteps"], loss_type)
            else:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        elif prediction_type in ("epsilon", "v_prediction"):
            if loss_type in ["huber", "smooth_l1"]:
                loss = self._xm_huber_loss_tensor(model_pred, target, prepared_batch["timesteps"], loss_type)
            else:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")

            snr_gamma = getattr(self.config, "snr_gamma", None)
            if snr_gamma is None or snr_gamma == 0:
                snr_weight = getattr(self.config, "snr_weight", 1.0)
                loss = snr_weight * loss
            else:
                snr = compute_snr(prepared_batch["timesteps"], self.noise_schedule)
                snr_divisor = snr
                if self.noise_schedule.config.prediction_type == "v_prediction":
                    snr_divisor = snr + 1
                mse_loss_weights = (
                    torch.stack(
                        [
                            snr,
                            snr_gamma * torch.ones_like(prepared_batch["timesteps"]),
                        ],
                        dim=1,
                    ).min(
                        dim=1
                    )[0]
                    / snr_divisor
                )
                loss = loss * mse_loss_weights.view(-1, *([1] * (loss.ndim - 1)))
        elif prediction_type == "sample":
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        else:
            raise ValueError(f"{family_name or self.NAME} XM loss is not implemented for prediction type {prediction_type}.")

        return self._apply_xm_conditioning_mask(
            prepared_batch,
            loss,
            apply_conditioning_mask=apply_conditioning_mask,
            family_name=family_name,
        )

    def _xm_huber_loss_tensor(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        loss_type: str,
    ) -> torch.Tensor:
        if getattr(self.config, "huber_schedule", "constant") != "constant":
            losses = []
            for idx in range(model_pred.shape[0]):
                huber_c = self.compute_scheduled_huber_c(timesteps[idx : idx + 1]).item()
                losses.append(
                    self.conditional_loss(
                        model_pred[idx : idx + 1].float(),
                        target[idx : idx + 1].float(),
                        reduction="none",
                        loss_type=loss_type,
                        huber_c=huber_c,
                    )
                )
            return torch.cat(losses, dim=0)
        return self.conditional_loss(
            model_pred.float(),
            target.float(),
            reduction="none",
            loss_type=loss_type,
            huber_c=getattr(self.config, "huber_c", 0.1),
        )

    def _apply_xm_conditioning_mask(
        self,
        prepared_batch: dict,
        loss: torch.Tensor,
        *,
        apply_conditioning_mask: bool,
        family_name: str | None = None,
    ) -> torch.Tensor:
        loss_mask_type = prepared_batch.get("loss_mask_type")
        if not loss_mask_type:
            legacy_type = prepared_batch.get("conditioning_type")
            if legacy_type in ("mask", "segmentation"):
                loss_mask_type = legacy_type
        if loss_mask_type == "mask" and apply_conditioning_mask:
            mask_image = (
                prepared_batch["conditioning_pixel_values"].to(dtype=loss.dtype, device=loss.device)[:, 0].unsqueeze(1)
            )
            mask_image = torch.nn.functional.interpolate(mask_image, size=loss.shape[2:], mode="area")
            mask_image = mask_image / 2 + 0.5
            return loss * mask_image
        if loss_mask_type == "segmentation" and apply_conditioning_mask:
            raise ValueError(
                f"{family_name or self.NAME} XM noise-candidate training does not support stochastic segmentation masked loss."
            )
        return loss

    def _select_xm_winners_in_place(
        self,
        prepared_batch: dict,
        model_output: dict,
        winner_indices: torch.Tensor,
        candidate_count: int,
    ) -> None:
        expanded_batch_size = candidate_count * winner_indices.shape[0]

        for key, value in list(prepared_batch.items()):
            prepared_batch[key] = self._select_xm_candidate_value(
                value,
                winner_indices,
                candidate_count,
                expanded_batch_size=expanded_batch_size,
                field_name=key,
            )
        prepared_batch.pop("xm_candidate_count", None)
        prepared_batch.pop("xm_original_batch_size", None)
        prepared_batch["xm_winner_indices"] = winner_indices.detach()

        for key, value in list(model_output.items()):
            if key in ("xm_candidate_count", "xm_winner_indices"):
                continue
            model_output[key] = self._select_xm_candidate_value(
                value,
                winner_indices,
                candidate_count,
                expanded_batch_size=expanded_batch_size,
                field_name=key,
            )
        model_output["xm_winner_indices"] = winner_indices.detach()
        model_output.pop("xm_candidate_count", None)

    def _select_xm_candidate_value(
        self,
        value,
        winner_indices: torch.Tensor,
        candidate_count: int,
        *,
        expanded_batch_size: int,
        field_name: str | None = None,
    ):
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == expanded_batch_size:
            return select_winning_candidates(value, winner_indices, candidate_count)
        if isinstance(value, list):
            if field_name not in getattr(self, "XM_SEQUENCE_LIST_KEYS", frozenset()) and len(value) == expanded_batch_size:
                return self._select_xm_candidate_sequence(value, winner_indices, candidate_count)
            return [
                self._select_xm_candidate_value(
                    item,
                    winner_indices,
                    candidate_count,
                    expanded_batch_size=expanded_batch_size,
                    field_name=field_name,
                )
                for item in value
            ]
        if isinstance(value, tuple):
            if field_name not in getattr(self, "XM_SEQUENCE_LIST_KEYS", frozenset()) and len(value) == expanded_batch_size:
                return tuple(self._select_xm_candidate_sequence(list(value), winner_indices, candidate_count))
            return tuple(
                self._select_xm_candidate_value(
                    item,
                    winner_indices,
                    candidate_count,
                    expanded_batch_size=expanded_batch_size,
                    field_name=field_name,
                )
                for item in value
            )
        if isinstance(value, dict):
            return {
                key: self._select_xm_candidate_value(
                    item,
                    winner_indices,
                    candidate_count,
                    expanded_batch_size=expanded_batch_size,
                    field_name=key,
                )
                for key, item in value.items()
            }
        if is_dataclass(value) and not isinstance(value, type):
            replacements = {
                field.name: self._select_xm_candidate_value(
                    getattr(value, field.name),
                    winner_indices,
                    candidate_count,
                    expanded_batch_size=expanded_batch_size,
                    field_name=field.name,
                )
                for field in fields(value)
            }
            return replace(value, **replacements)
        return value

    @staticmethod
    def _select_xm_candidate_sequence(value: list, winner_indices: torch.Tensor, candidate_count: int) -> list:
        batch_size = int(winner_indices.shape[0])
        if len(value) != batch_size * candidate_count:
            return value
        winners = winner_indices.detach().to(device="cpu", dtype=torch.long).tolist()
        return [value[int(winner) * batch_size + sample_idx] for sample_idx, winner in enumerate(winners)]

    def _select_xm_winning_sample_losses(
        self,
        sample_losses: torch.Tensor,
        prepared_batch: dict,
        model_output: dict,
        *,
        candidate_count: int,
        family_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if candidate_count < 2:
            raise ValueError(f"{family_name or self.NAME} XM candidate_count must be at least 2.")
        candidate_losses = reshape_candidate_batch(sample_losses, candidate_count)
        selected_loss, winner_indices = select_min_candidate_loss(candidate_losses)
        self._select_xm_winners_in_place(prepared_batch, model_output, winner_indices, candidate_count)
        return selected_loss, winner_indices, candidate_losses

    def _select_xm_winning_loss_tensor(
        self,
        loss_tensor: torch.Tensor,
        prepared_batch: dict,
        model_output: dict,
        *,
        candidate_count: int,
        family_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._select_xm_winning_sample_losses(
            reduce_loss_to_samples(loss_tensor),
            prepared_batch,
            model_output,
            candidate_count=candidate_count,
            family_name=family_name,
        )

    @staticmethod
    def _xm_candidate_logs(
        selected_loss: torch.Tensor,
        candidate_losses: torch.Tensor,
        winner_indices: torch.Tensor,
        candidate_count: int,
    ) -> dict:
        logs = {
            "xm_loss": selected_loss.detach().item(),
            "xm_candidate_loss_mean": candidate_losses.detach().float().mean().item(),
        }
        usage = route_usage_histogram(winner_indices, candidate_count)
        if usage is not None:
            usage = usage.to(device="cpu")
            for idx, count in enumerate(usage.tolist()):
                logs[f"xm_candidate_{idx}_wins"] = count
        return logs

    def _xm_noise_loss_with_logs(
        self,
        prepared_batch: dict,
        model_output: dict,
        *,
        candidate_count: int,
        apply_conditioning_mask: bool,
        family_name: str | None = None,
    ):
        loss_tensor = self._xm_diffusion_loss_tensor(
            prepared_batch,
            model_output,
            apply_conditioning_mask=apply_conditioning_mask,
        )
        selected_loss, winner_indices, candidate_losses = self._select_xm_winning_loss_tensor(
            loss_tensor,
            prepared_batch,
            model_output,
            candidate_count=candidate_count,
            family_name=family_name,
        )
        return selected_loss, self._xm_candidate_logs(
            selected_loss,
            candidate_losses,
            winner_indices,
            candidate_count,
        )

    def loss_with_logs(self, prepared_batch: dict, model_output, apply_conditioning_mask: bool = True):
        candidate_count = model_output.get("xm_candidate_count") if isinstance(model_output, dict) else None
        if candidate_count:
            return self._xm_noise_loss_with_logs(
                prepared_batch,
                model_output,
                candidate_count=int(candidate_count),
                apply_conditioning_mask=apply_conditioning_mask,
            )
        return self.loss(prepared_batch, model_output, apply_conditioning_mask=apply_conditioning_mask), None

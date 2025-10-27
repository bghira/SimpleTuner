from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.distillation.common import DistillationBase
from simpletuner.helpers.distillation.registry import DistillationRegistry

from ..self_forcing.pipeline import SelfForcingTrainingPipeline
from ..self_forcing.scheduler import FlowMatchingSchedulerAdapter
from ..self_forcing.wrappers import FoundationModelWrapper, ModuleWrapper

logger = logging.getLogger(__name__)

DMD_SAFETENSORS_DEFAULT_FILENAME = "fake_score_transformer.safetensors"
DMD_OPTIMIZER_DEFAULT_FILENAME = "fake_score_transformer_optim.pt"


class DMDDistiller(DistillationBase):
    """
    Distribution Matching Distillation (DMD) ported from the realtime-video stack.
    This implementation orchestrates generator rollout via the self-forcing pipeline,
    computes the KL-based generator loss, and updates a trainable fake score model.
    """

    DEFAULTS: Dict[str, Any] = {
        "dmd_denoising_steps": "1000,757,522",
        "generator_update_interval": 1,
        "real_score_guidance_scale": 3.0,
        "fake_score_guidance_scale": 0.0,
        "fake_score_lr": 1e-5,
        "fake_score_weight_decay": 0.01,
        "fake_score_betas": (0.9, 0.999),
        "fake_score_eps": 1e-8,
        "fake_score_grad_clip": 1.0,
        "min_timestep_ratio": 0.02,
        "max_timestep_ratio": 0.98,
        "num_frame_per_block": 3,
        "independent_first_frame": False,
        "same_step_across_blocks": False,
        "last_step_only": False,
        "num_training_frames": 21,
        "context_noise": 0,
        "ts_schedule": True,
        "ts_schedule_max": False,
        "min_score_timestep": 0,
        "timestep_shift": 1.0,
    }

    def __init__(
        self,
        teacher_model,
        student_model=None,
        *,
        noise_scheduler,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged = dict(self.DEFAULTS)
        if config:
            merged.update(config)
        super().__init__(teacher_model, student_model, merged)

        if not self.is_flow_matching:
            raise ValueError("DMD requires a flow-matching teacher.")

        self.device = teacher_model.accelerator.device
        self.weight_dtype = getattr(teacher_model.config, "weight_dtype", torch.float32)
        self.generator_update_interval = int(self.config["generator_update_interval"])
        self.generator_update_counter = 0

        self.scheduler_adapter = FlowMatchingSchedulerAdapter(noise_scheduler).to(device=self.device)

        self.student_model = student_model or teacher_model
        self.generator_wrapper = FoundationModelWrapper(self.student_model, self.scheduler_adapter)
        self.real_score_wrapper = FoundationModelWrapper(self.teacher_model, self.scheduler_adapter)

        self.fake_score_transformer, self.fake_score_optimizer = self._init_fake_score_transformer()
        self.fake_score_wrapper = ModuleWrapper(
            self.fake_score_transformer,
            self.scheduler_adapter,
            self.weight_dtype,
        )

        self.denoising_steps = self._parse_denoising_steps(self.config["dmd_denoising_steps"])
        self.pipeline = SelfForcingTrainingPipeline(
            denoising_step_list=self.denoising_steps,
            scheduler=self.scheduler_adapter,
            generator=self.generator_wrapper,
            num_frame_per_block=int(self.config["num_frame_per_block"]),
            independent_first_frame=bool(self.config["independent_first_frame"]),
            same_step_across_blocks=bool(self.config["same_step_across_blocks"]),
            last_step_only=bool(self.config["last_step_only"]),
            num_max_frames=int(self.config["num_training_frames"]),
            context_noise=int(self.config["context_noise"]),
        )

        num_train_timesteps = getattr(noise_scheduler.config, "num_train_timesteps", len(self.scheduler_adapter.sigmas))
        self.num_train_timestep = int(num_train_timesteps)
        self.min_step = int(self.config["min_timestep_ratio"] * self.num_train_timestep)
        self.max_step = int(self.config["max_timestep_ratio"] * self.num_train_timestep)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _parse_denoising_steps(self, raw: str | Sequence[int]) -> Tuple[int, ...]:
        if isinstance(raw, str):
            return tuple(int(step.strip()) for step in raw.split(",") if step.strip())
        return tuple(int(step) for step in raw)

    def _init_fake_score_transformer(self) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        try:
            teacher_component = self.teacher_model.get_trained_component(unwrap_model=True)
        except TypeError:
            teacher_component = self.teacher_model.get_trained_component()
        if not hasattr(teacher_component, "config"):
            raise ValueError("Teacher component is missing `config`, cannot clone fake score transformer.")
        config_dict = (
            teacher_component.config.to_dict()
            if hasattr(teacher_component.config, "to_dict")
            else dict(teacher_component.config)
        )
        try:
            fake_score = teacher_component.__class__(**config_dict)
        except TypeError:
            fake_score = teacher_component.__class__(teacher_component.config)
        fake_score.load_state_dict(teacher_component.state_dict())
        fake_score.to(device=self.device, dtype=self.weight_dtype)
        fake_score.train()

        optimizer = torch.optim.AdamW(
            fake_score.parameters(),
            lr=float(self.config["fake_score_lr"]),
            betas=tuple(self.config["fake_score_betas"]),
            weight_decay=float(self.config["fake_score_weight_decay"]),
            eps=float(self.config["fake_score_eps"]),
        )
        return fake_score, optimizer

    def _build_conditionals(self, prepared_batch: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        cond: Dict[str, Any] = {"prompt_embeds": prepared_batch["encoder_hidden_states"]}
        if prepared_batch.get("added_cond_kwargs") is not None:
            cond["added_cond_kwargs"] = prepared_batch["added_cond_kwargs"]
        for key in ("conditioning_image_embeds", "conditioning_pixel_values", "conditioning_latents"):
            if key in prepared_batch and prepared_batch[key] is not None:
                cond[key] = prepared_batch[key]

        negative = prepared_batch.get("negative_encoder_hidden_states")
        if negative is None:
            uncond = cond
        else:
            uncond = dict(cond)
            uncond["prompt_embeds"] = negative
        return cond, uncond

    def _sample_training_timestep(
        self,
        batch_size: int,
        num_frames: int,
        *,
        denoised_from: Optional[int],
        denoised_to: Optional[int],
    ) -> torch.Tensor:
        min_t = int(denoised_to if self.config["ts_schedule"] and denoised_to is not None else self.min_step)
        max_t = int(denoised_from if self.config["ts_schedule_max"] and denoised_from is not None else self.max_step)
        if max_t <= min_t:
            max_t = min_t + 1
        return torch.randint(
            low=min_t,
            high=max_t,
            size=(batch_size, num_frames),
            device=self.device,
            dtype=torch.long,
        )

    # ------------------------------------------------------------------
    # Core losses
    # ------------------------------------------------------------------
    def _compute_kl_grad(
        self,
        noisy_latents: torch.Tensor,
        clean_latents: torch.Tensor,
        timesteps: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        unconditional_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _, fake_pred = self.fake_score_wrapper.forward(
            noisy_latents,
            timesteps,
            conditional_dict,
        )

        _, real_cond = self.real_score_wrapper.forward(
            noisy_latents,
            timesteps,
            conditional_dict,
        )

        real_guidance = float(self.config["real_score_guidance_scale"])
        if real_guidance != 0.0 and unconditional_dict is not conditional_dict:
            _, real_uncond = self.real_score_wrapper.forward(
                noisy_latents,
                timesteps,
                unconditional_dict,
            )
            real_pred = real_uncond + real_guidance * (real_cond - real_uncond)
        else:
            real_pred = real_cond

        fake_guidance = float(self.config["fake_score_guidance_scale"])
        if fake_guidance != 0.0 and unconditional_dict is not conditional_dict:
            _, fake_uncond = self.fake_score_wrapper.forward(
                noisy_latents,
                timesteps,
                unconditional_dict,
            )
            fake_pred = fake_uncond + fake_guidance * (fake_pred - fake_uncond)

        grad = fake_pred - real_pred
        with torch.no_grad():
            normaliser = torch.abs(clean_latents - real_pred).mean(dim=[1, 2, 3, 4], keepdim=True).clamp(min=1e-6)
        grad = grad / normaliser

        logs = {
            "dmd_kl_grad_norm": torch.mean(torch.abs(grad)).detach(),
            "dmd_timestep_mean": torch.mean(timesteps.float()).detach(),
        }
        return grad, logs

    def _distribution_matching_loss(
        self,
        generated_latents: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        unconditional_dict: Dict[str, torch.Tensor],
        *,
        denoised_from: Optional[int],
        denoised_to: Optional[int],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, channels, num_frames, height, width = generated_latents.shape
        timesteps = self._sample_training_timestep(
            batch_size,
            num_frames,
            denoised_from=denoised_from,
            denoised_to=denoised_to,
        )

        flat_generated = generated_latents.reshape(batch_size * num_frames, channels, height, width)
        flat_noise = torch.randn_like(flat_generated)
        noisy_flat = self.scheduler_adapter.add_noise(flat_generated, flat_noise, timesteps.reshape(-1))
        noisy_latents = noisy_flat.reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)

        grad, logs = self._compute_kl_grad(
            noisy_latents,
            generated_latents,
            timesteps,
            conditional_dict,
            unconditional_dict,
        )
        target = (generated_latents - grad).detach()
        loss = 0.5 * F.mse_loss(generated_latents.double(), target.double())
        logs["dmd_loss"] = loss.detach()
        return loss, logs

    def _generator_step(self, prepared_batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        conditional_dict, unconditional_dict = self._build_conditionals(prepared_batch)
        latents_shape = prepared_batch["latents"].shape
        noise = torch.randn(latents_shape, device=self.device, dtype=self.weight_dtype)

        pipeline_output = self.pipeline.inference_with_trajectory(
            noise=noise,
            conditional_dict=conditional_dict,
        )
        generated_latents, denoised_from, denoised_to = pipeline_output

        loss, logs = self._distribution_matching_loss(
            generated_latents,
            conditional_dict,
            unconditional_dict,
            denoised_from=denoised_from,
            denoised_to=denoised_to,
        )
        logs["dmd_generator_loss"] = loss.detach()
        return loss, logs

    def _critic_step(self, prepared_batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self.fake_score_optimizer.zero_grad(set_to_none=True)

        conditional_dict, _ = self._build_conditionals(prepared_batch)
        latents_shape = prepared_batch["latents"].shape
        noise = torch.randn(latents_shape, device=self.device, dtype=self.weight_dtype)

        with torch.no_grad():
            pipeline_output = self.pipeline.inference_with_trajectory(
                noise=noise,
                conditional_dict=conditional_dict,
            )
            generated_latents, denoised_from, denoised_to = pipeline_output

        batch_size, channels, num_frames, height, width = generated_latents.shape
        timesteps = self._sample_training_timestep(
            batch_size,
            num_frames,
            denoised_from=denoised_from,
            denoised_to=denoised_to,
        )
        flat_generated = generated_latents.reshape(batch_size * num_frames, channels, height, width)
        flat_noise = torch.randn_like(flat_generated)
        noisy_flat = self.scheduler_adapter.add_noise(flat_generated, flat_noise, timesteps.reshape(-1))
        noisy_latents = noisy_flat.reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)

        _, pred_clean = self.fake_score_wrapper.forward(noisy_latents, timesteps, conditional_dict)
        pred_noise = self.scheduler_adapter.convert_x0_to_noise(
            pred_clean.reshape(batch_size * num_frames, channels, height, width),
            noisy_flat,
            timesteps.reshape(-1),
        ).reshape_as(pred_clean)

        flat_true_noise = flat_noise.reshape_as(pred_noise)
        loss = F.mse_loss(pred_noise.float(), flat_true_noise.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.fake_score_transformer.parameters(),
            float(self.config["fake_score_grad_clip"]),
        )
        self.fake_score_optimizer.step()

        logs = {
            "dmd_critic_loss": loss.detach(),
            "dmd_critic_timestep": torch.mean(timesteps.float()).detach(),
        }
        return loss, logs

    # ------------------------------------------------------------------
    # DistillationBase overrides
    # ------------------------------------------------------------------
    def compute_distill_loss(
        self,
        prepared_batch: Dict[str, Any],
        model_output: Dict[str, Any],
        original_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        self.generator_update_counter += 1
        if self.generator_update_counter % self.generator_update_interval != 0:
            return original_loss, {}

        generator_loss, logs = self._generator_step(prepared_batch)
        loss = original_loss + generator_loss

        logs["total"] = loss.detach()
        return loss, {k: float(v) for k, v in logs.items()}

    def discriminator_step(self, prepared_batch: Dict[str, Any], **_: Any) -> None:
        if self.fake_score_transformer is None:
            return
        try:
            critic_loss, logs = self._critic_step(prepared_batch)
            for key, value in logs.items():
                logger.debug("%s: %.5f", key, float(value))
            logger.debug("DMD critic loss: %.5f", float(critic_loss))
        except Exception:
            logger.exception("Failed to run DMD critic update.")

    def on_save_checkpoint(self, step: int, ckpt_dir: str) -> None:
        if self.fake_score_transformer is None:
            return
        os.makedirs(ckpt_dir, exist_ok=True)
        weight_path = os.path.join(ckpt_dir, DMD_SAFETENSORS_DEFAULT_FILENAME)
        tensor_dict = {k: v.detach().cpu() for k, v in self.fake_score_transformer.state_dict().items()}
        save_file(tensor_dict, weight_path)

        opt_path = os.path.join(ckpt_dir, DMD_OPTIMIZER_DEFAULT_FILENAME)
        torch.save(
            {
                "step": step,
                "state": self.fake_score_optimizer.state_dict(),
            },
            opt_path,
        )

    def on_load_checkpoint(self, ckpt_dir: str) -> None:
        if self.fake_score_transformer is None:
            return

        weight_path = os.path.join(ckpt_dir, DMD_SAFETENSORS_DEFAULT_FILENAME)
        if os.path.exists(weight_path):
            tensor_dict = load_file(weight_path, device="cpu")
            self.fake_score_transformer.load_state_dict(tensor_dict, strict=True)
            self.fake_score_transformer.to(device=self.device, dtype=self.weight_dtype)

        opt_path = os.path.join(ckpt_dir, DMD_OPTIMIZER_DEFAULT_FILENAME)
        if os.path.exists(opt_path):
            payload = torch.load(opt_path, map_location=self.device)
            self.fake_score_optimizer.load_state_dict(payload["state"])


DistillationRegistry.register(
    "dmd",
    DMDDistiller,
    requires_distillation_cache=False,
    data_requirements=[[DatasetType.IMAGE, DatasetType.VIDEO]],
)

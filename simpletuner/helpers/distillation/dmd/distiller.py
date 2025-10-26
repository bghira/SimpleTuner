# helpers/distillation/dmd/distiller.py
import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file, save_file

from simpletuner.helpers.distillation.common import DistillationBase
from simpletuner.helpers.distillation.registry import DistillationRegistry

logger = logging.getLogger(__name__)

DMD_SAFETENSORS_DEFAULT_FILENAME = "fake_score_transformer.safetensors"
DMD_OPTIMIZER_DEFAULT_FILENAME = "fake_score_transformer_optim.pt"


class DMDDistiller(DistillationBase):
    """
    Distribution Matching Distillation (DMD) implementation.

    Following the FastVideo approach with:
    - Generator (student) learning from real score transformer (teacher)
    - Fake score transformer as discriminator
    - Multi-step simulation for train-inference consistency
    """

    def __init__(
        self,
        teacher_model,
        student_model=None,
        *,
        noise_scheduler: FlowMatchEulerDiscreteScheduler,
        config: Optional[Dict[str, Any]] = None,
    ):
        default = {
            # DMD denoising steps (e.g., "1000,757,522" for 3-step)
            "dmd_denoising_steps": "1000,757,522",
            # Min/max timestep ratios for sampling
            "min_timestep_ratio": 0.02,
            "max_timestep_ratio": 0.98,
            # Generator update interval
            "generator_update_interval": 5,
            # Real score guidance scale
            "real_score_guidance_scale": 3.0,
            # Whether to simulate generator forward like inference
            "simulate_generator_forward": False,
            # Model family for fake score transformer
            "model_family": "wan",
            # Learning rates
            "fake_score_lr": 1e-5,
            "fake_score_lr_scheduler": "cosine_with_min_lr",
            "min_lr_ratio": 0.5,
        }
        if config:
            default.update(config)

        super().__init__(teacher_model, student_model, default)

        if not self.is_flow_matching:
            raise ValueError("DMD requires a flow-matching teacher.")

        self.noise_scheduler = noise_scheduler

        # Parse denoising steps
        self.denoising_step_list = [int(s) for s in self.config["dmd_denoising_steps"].split(",")]

        # Initialize fake score transformer
        self.fake_score_transformer = None
        self._init_fake_score_transformer()

        # Track generator updates
        self.generator_update_counter = 0

    def _init_fake_score_transformer(self):
        """Initialize fake score transformer based on model family."""
        fam = self.config["model_family"]

        if fam == "wan":
            # Clone teacher architecture for fake score transformer
            self.fake_score_transformer = self.teacher_model.get_trained_component().__class__(
                **self.teacher_model.get_trained_component().config
            )
            # Initialize with teacher weights
            self.fake_score_transformer.load_state_dict(self.teacher_model.get_trained_component().state_dict())
        else:
            raise NotImplementedError(f"Model family '{fam}' not implemented")

        # Optimizer for fake score transformer
        self.fake_score_optimizer = torch.optim.AdamW(
            self.fake_score_transformer.parameters(),
            lr=self.config["fake_score_lr"],
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8,
        )

    def prepare_batch(self, batch: Dict[str, Any], *_):
        """Prepare batch for DMD training."""
        latents = batch["latents"]  # Clean latents
        B, device = latents.shape[0], latents.device

        # Sample timestep from denoising steps
        index = torch.randint(0, len(self.denoising_step_list), [1], device=device, dtype=torch.long)
        target_timestep = self.denoising_step_list[index.item()]
        timestep = torch.tensor([target_timestep], device=device, dtype=torch.long)

        # Add noise to create noisy latents
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timestep)

        batch["noisy_latents"] = noisy_latents
        batch["timesteps"] = timestep.repeat(B)
        batch["clean_latents"] = latents
        batch["dmd_timestep_index"] = index

        # If simulating generator forward, prepare multi-step inputs
        if self.config["simulate_generator_forward"]:
            batch = self._prepare_multi_step_batch(batch)

        return batch

    def _prepare_multi_step_batch(self, batch: Dict[str, Any]):
        """Prepare batch for multi-step simulation."""
        latents = batch["latents"]
        device = latents.device
        target_idx = batch["dmd_timestep_index"].item()

        # Start from pure noise
        current_noise_latents = torch.randn_like(latents)

        # Simulate steps up to target
        with torch.no_grad():
            for step_idx in range(target_idx):
                current_timestep = self.denoising_step_list[step_idx]
                timestep_tensor = torch.tensor([current_timestep], device=device, dtype=torch.long)

                # Student prediction
                pred_noise = self.student_model.get_trained_component()(
                    current_noise_latents,
                    timestep_tensor.repeat(latents.shape[0]),
                    batch["encoder_hidden_states"],
                    return_dict=False,
                )[0]

                # Denoise
                pred_clean = self._pred_noise_to_pred_video(pred_noise, current_noise_latents, timestep_tensor)

                # Add noise for next step if not final
                if step_idx < target_idx - 1:
                    next_timestep = self.denoising_step_list[step_idx + 1]
                    next_timestep_tensor = torch.tensor([next_timestep], device=device, dtype=torch.long)
                    noise = torch.randn_like(pred_clean)
                    current_noise_latents = self.noise_scheduler.add_noise(pred_clean, noise, next_timestep_tensor)

        batch["noisy_latents"] = current_noise_latents
        return batch

    def compute_distill_loss(
        self,
        prepared_batch: Dict[str, Any],
        model_output: Dict[str, Any],
        _original_loss: torch.Tensor,
    ):
        """Compute DMD loss (only called when generator is being updated)."""
        self.generator_update_counter += 1

        # Skip if not generator update step
        if self.generator_update_counter % self.config["generator_update_interval"] != 0:
            return torch.tensor(0.0, device=model_output["model_prediction"].device), {}

        pred_noise = model_output["model_prediction"]
        noisy_latents = prepared_batch["noisy_latents"]
        timesteps = prepared_batch["timesteps"]

        # Convert to predicted clean
        generator_pred_video = self._pred_noise_to_pred_video(pred_noise, noisy_latents, timesteps[0:1])

        # Compute DMD loss
        loss = self._dmd_forward(generator_pred_video, prepared_batch)

        logs = {
            "dmd_loss": loss.item(),
            "total": loss.item(),
        }

        return loss, logs

    def _dmd_forward(self, generator_pred_video: torch.Tensor, batch: Dict[str, Any]):
        """Compute DMD loss with real and fake score transformers."""
        clean_latents = batch["clean_latents"]

        # Sample new timestep for DMD
        min_t = int(self.config["min_timestep_ratio"] * 1000)
        max_t = int(self.config["max_timestep_ratio"] * 1000)
        timestep = torch.randint(min_t, max_t, [1], device=clean_latents.device, dtype=torch.long)

        # Add noise to both clean and generated
        noise = torch.randn_like(clean_latents)
        noisy_clean = self.noise_scheduler.add_noise(clean_latents, noise, timestep)
        noisy_gen = self.noise_scheduler.add_noise(generator_pred_video, noise, timestep)

        # Get predictions from both transformers
        with torch.no_grad():
            # Real score (teacher) predictions with CFG
            real_cond = self.teacher_model.get_trained_component()(
                noisy_gen,
                timestep.repeat(noisy_gen.shape[0]),
                batch["encoder_hidden_states"],
                return_dict=False,
            )[0]

            if "negative_encoder_hidden_states" in batch:
                real_uncond = self.teacher_model.get_trained_component()(
                    noisy_gen,
                    timestep.repeat(noisy_gen.shape[0]),
                    batch["negative_encoder_hidden_states"],
                    return_dict=False,
                )[0]
                cfg_scale = self.config["real_score_guidance_scale"]
                real_score_pred = real_uncond + cfg_scale * (real_cond - real_uncond)
            else:
                real_score_pred = real_cond

        # Fake score predictions
        fake_score_pred = self.fake_score_transformer(
            noisy_gen,
            timestep.repeat(noisy_gen.shape[0]),
            batch["encoder_hidden_states"],
            return_dict=False,
        )[0]

        # Convert to clean predictions
        real_pred_clean = self._pred_noise_to_pred_video(real_score_pred, noisy_gen, timestep)
        fake_pred_clean = self._pred_noise_to_pred_video(fake_score_pred, noisy_gen, timestep)

        # DMD loss
        loss = F.mse_loss(fake_pred_clean, real_pred_clean.detach())

        return loss

    def fake_score_step(self, prepared_batch: Dict[str, Any]):
        """Update fake score transformer."""
        # Get generator prediction
        with torch.no_grad():
            pred_noise = self.student_model.get_trained_component()(
                prepared_batch["noisy_latents"],
                prepared_batch["timesteps"],
                prepared_batch["encoder_hidden_states"],
                return_dict=False,
            )[0]

            generator_pred_video = self._pred_noise_to_pred_video(
                pred_noise,
                prepared_batch["noisy_latents"],
                prepared_batch["timesteps"][0:1],
            )

        # Sample timestep for fake score training
        min_t = int(self.config["min_timestep_ratio"] * 1000)
        max_t = int(self.config["max_timestep_ratio"] * 1000)
        fake_score_timestep = torch.randint(min_t, max_t, [1], device=generator_pred_video.device, dtype=torch.long)

        # Add noise
        noise = torch.randn_like(generator_pred_video)
        noisy_fake = self.noise_scheduler.add_noise(generator_pred_video, noise, fake_score_timestep)

        # Fake score prediction
        fake_pred = self.fake_score_transformer(
            noisy_fake,
            fake_score_timestep.repeat(noisy_fake.shape[0]),
            prepared_batch["encoder_hidden_states"],
            return_dict=False,
        )[0]

        # Fake score loss (predict noise)
        loss = F.mse_loss(fake_pred, noise)

        # Update
        self.fake_score_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.fake_score_transformer.parameters(), 1.0)
        self.fake_score_optimizer.step()

        logger.debug(f"Fake score loss: {loss.item():.4f}")

    def _pred_noise_to_pred_video(self, pred_noise, noise_input, timestep):
        """Convert predicted noise to predicted clean video."""
        # Simple flow matching: x0 = (xt - σt * noise) / (1 - σt)
        sigma = self.noise_scheduler.sigmas[timestep.item()]
        pred_clean = (noise_input - sigma * pred_noise) / (1 - sigma)
        return pred_clean

    def on_save_checkpoint(self, step: int, ckpt_dir: str):
        """Save fake score transformer checkpoint."""
        if self.fake_score_transformer is None or (torch.distributed.is_initialized() and torch.distributed.get_rank() != 0):
            return

        # Model weights
        weight_path = os.path.join(ckpt_dir, DMD_SAFETENSORS_DEFAULT_FILENAME)
        tensor_dict = {k: v.detach().cpu() for k, v in self.fake_score_transformer.state_dict().items()}
        save_file(tensor_dict, weight_path)

        # Optimizer state
        opt_path = os.path.join(ckpt_dir, DMD_OPTIMIZER_DEFAULT_FILENAME)
        torch.save(
            {
                "step": step,
                "state": self.fake_score_optimizer.state_dict(),
            },
            opt_path,
        )

    def on_load_checkpoint(self, ckpt_dir: str):
        """Load fake score transformer checkpoint."""
        if self.fake_score_transformer is None:
            return

        weight_path = os.path.join(ckpt_dir, DMD_SAFETENSORS_DEFAULT_FILENAME)
        if not os.path.exists(weight_path):
            return

        # Load weights
        tensor_dict = load_file(weight_path, device="cpu")
        self.fake_score_transformer.load_state_dict(tensor_dict, strict=True)
        self.fake_score_transformer.to(self.teacher_model.get_trained_component().device, non_blocking=True)

        # Load optimizer
        opt_path = os.path.join(ckpt_dir, DMD_OPTIMIZER_DEFAULT_FILENAME)
        if os.path.exists(opt_path):
            if torch.cuda.is_available():
                map_location = {"cuda:0": f"cuda:{torch.cuda.current_device()}"}
            else:
                map_location = "cpu"
            payload = torch.load(opt_path, map_location=map_location)
            self.fake_score_optimizer.load_state_dict(payload["state"])


DistillationRegistry.register("dmd", DMDDistiller, requires_distillation_cache=False)

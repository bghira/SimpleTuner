# helpers/distillation/lcm/distiller.py
import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, FlowMatchEulerDiscreteScheduler, LCMScheduler

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.distillation.common import DistillationBase
from simpletuner.helpers.distillation.registry import DistillationRegistry

logger = logging.getLogger(__name__)


class DDIMSolver:
    """DDIM ODE solver for DDPM models."""

    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray([alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist())
        # Convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev


class FlowMatchingSolver:
    """ODE solver for flow-matching models."""

    def __init__(self, sigmas, timesteps=1000, euler_timesteps=50):
        # Create evenly spaced timesteps for consistency training
        step_ratio = timesteps // euler_timesteps
        self.timesteps = torch.linspace(0, timesteps - 1, euler_timesteps).long()
        self.sigmas = sigmas[self.timesteps]
        self.sigmas_prev = torch.cat([sigmas[0:1], self.sigmas[:-1]])

    def to(self, device):
        self.timesteps = self.timesteps.to(device)
        self.sigmas = self.sigmas.to(device)
        self.sigmas_prev = self.sigmas_prev.to(device)
        return self

    def euler_step(self, x_t, v_t, timestep_index):
        """Perform one Euler step in the flow ODE."""
        sigma_t = extract_into_tensor(self.sigmas, timestep_index, x_t.shape)
        sigma_prev = extract_into_tensor(self.sigmas_prev, timestep_index, x_t.shape)
        # For flow matching: dx/dt = v(x,t), so x_{t-1} = x_t + (t_{t-1} - t_t) * v_t
        # In sigma space: x_prev = x_t + (sigma_prev - sigma_t) * v_t
        x_prev = x_t + (sigma_prev - sigma_t) * v_t
        return x_prev


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    """Get c_skip and c_out scalings from LCM paper."""
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
    c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5
    return c_skip, c_out


class LCMDistiller(DistillationBase):
    """
    Latent Consistency Model distillation for both DDPM and Flow-Matching models.
    """

    def __init__(
        self,
        teacher_model,
        student_model=None,
        *,
        noise_scheduler: Union[DDPMScheduler, FlowMatchEulerDiscreteScheduler],
        config: Optional[Dict[str, Any]] = None,
    ):
        default = {
            # Number of DDIM/Euler timesteps for the ODE solver
            "num_ddim_timesteps": 50,
            # Guidance scale range for training
            "w_min": 1.0,
            "w_max": 15.0,
            # Loss type: "l2" or "huber"
            "loss_type": "l2",
            "huber_c": 0.001,
            # Timestep scaling factor for boundary conditions
            "timestep_scaling_factor": 10.0,
            # For flow-matching: shift parameter
            "shift": 7.0,
        }
        if config:
            default.update(config)

        super().__init__(teacher_model, student_model, default)

        # Store the scheduler
        self.noise_scheduler = noise_scheduler

        # Initialize the appropriate ODE solver
        if self.is_flow_matching:
            if hasattr(noise_scheduler, "sigmas"):
                sigmas = noise_scheduler.sigmas.cpu()
            else:
                # Create sigmas for flow matching if not provided
                timesteps = torch.linspace(0, 1, noise_scheduler.config.num_train_timesteps + 1)
                sigmas = timesteps.flip(0)

            self.solver = FlowMatchingSolver(
                sigmas,
                timesteps=noise_scheduler.config.num_train_timesteps,
                euler_timesteps=self.config["num_ddim_timesteps"],
            )
        else:
            # DDPM case
            self.solver = DDIMSolver(
                noise_scheduler.alphas_cumprod.cpu().numpy(),
                timesteps=noise_scheduler.config.num_train_timesteps,
                ddim_timesteps=self.config["num_ddim_timesteps"],
            )

            # Get alpha and sigma schedules for DDPM
            self.alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
            self.sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)

        # Move solver to device
        device = self.teacher_model.get_trained_component().device
        self.solver = self.solver.to(device)

        if hasattr(self, "alpha_schedule"):
            self.alpha_schedule = self.alpha_schedule.to(device)
            self.sigma_schedule = self.sigma_schedule.to(device)

    def prepare_batch(self, batch: Dict[str, Any], *_):
        """Prepare batch for LCM distillation."""
        latents = batch["latents"]
        B, device = latents.shape[0], latents.device

        # Sample random solver timestep indices
        if self.is_flow_matching:
            index = torch.randint(0, len(self.solver.timesteps), (B,), device=device).long()
            start_timesteps = self.solver.timesteps[index]
            timesteps = self.solver.timesteps[torch.clamp(index + 1, max=len(self.solver.timesteps) - 1)]
        else:
            # DDPM: use DDIM timesteps
            topk = self.noise_scheduler.config.num_train_timesteps // self.config["num_ddim_timesteps"]
            index = torch.randint(0, self.config["num_ddim_timesteps"], (B,), device=device).long()
            start_timesteps = self.solver.ddim_timesteps[index]
            timesteps = start_timesteps - topk
            timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

        # Get boundary condition scalings
        c_skip_start, c_out_start = scalings_for_boundary_conditions(
            start_timesteps, timestep_scaling=self.config["timestep_scaling_factor"]
        )
        c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]

        c_skip, c_out = scalings_for_boundary_conditions(timesteps, timestep_scaling=self.config["timestep_scaling_factor"])
        c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

        # Add noise to create starting point
        noise = torch.randn_like(latents)

        if self.is_flow_matching:
            # Flow matching: interpolate between noise and data
            sigma = extract_into_tensor(self.solver.sigmas, index, latents.shape)
            noisy_latents = sigma * noise + (1 - sigma) * latents
        else:
            # DDPM: use the standard noising process
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, start_timesteps)

        # Sample guidance scale
        w = (self.config["w_max"] - self.config["w_min"]) * torch.rand((B,)) + self.config["w_min"]
        w = w.reshape(B, 1, 1, 1).to(device=device, dtype=latents.dtype)

        # Get teacher predictions with CFG
        encoder_hidden_states = batch["encoder_hidden_states"]
        uncond_encoder_hidden_states = batch.get("negative_encoder_hidden_states")
        if uncond_encoder_hidden_states is None:
            uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

        self.toggle_adapter(enable=False)
        with torch.no_grad():
            # Conditional teacher prediction
            noise_pred_cond = self.teacher_model.get_trained_component()(
                noisy_latents,
                start_timesteps,
                encoder_hidden_states,
                return_dict=False,
            )[0]

            # Unconditional teacher prediction
            noise_pred_uncond = self.teacher_model.get_trained_component()(
                noisy_latents,
                start_timesteps,
                uncond_encoder_hidden_states,
                return_dict=False,
            )[0]

            # Apply CFG
            noise_pred = noise_pred_uncond + w * (noise_pred_cond - noise_pred_uncond)

            if self.is_flow_matching:
                # For flow matching, the model predicts velocity
                # Use the solver to take one ODE step
                x_prev = self.solver.euler_step(noisy_latents, noise_pred, index)

                # Get target prediction at the next timestep
                target_v = self.teacher_model.get_trained_component()(
                    x_prev,
                    timesteps,
                    encoder_hidden_states,
                    return_dict=False,
                )[0]

                # Consistency target using flow parameterization
                target = c_skip * x_prev + c_out * target_v
            else:
                # DDPM case: convert predictions to x0 and epsilon
                pred_x0 = self._get_predicted_original_sample(noise_pred, start_timesteps, noisy_latents)
                pred_epsilon = self._get_predicted_noise(noise_pred, start_timesteps, noisy_latents)

                # DDIM step to get x_prev
                x_prev = self.solver.ddim_step(pred_x0, pred_epsilon, index)

                # Get target prediction at the previous timestep
                target_noise_pred = self.teacher_model.get_trained_component()(
                    x_prev,
                    timesteps,
                    encoder_hidden_states,
                    return_dict=False,
                )[0]

                pred_x0_target = self._get_predicted_original_sample(target_noise_pred, timesteps, x_prev)
                target = c_skip * x_prev + c_out * pred_x0_target

        self.toggle_adapter(enable=True)

        # Store everything needed for loss computation
        batch.update(
            {
                "noisy_latents": noisy_latents,
                "timesteps": start_timesteps,
                "guidance_scale": w,
                "c_skip_start": c_skip_start,
                "c_out_start": c_out_start,
                "target": target,
                "index": index,
            }
        )

        return batch

    def compute_distill_loss(
        self,
        prepared_batch: Dict[str, Any],
        model_output: Dict[str, Any],
        _original_loss: torch.Tensor,
    ):
        """Compute the LCM consistency loss."""
        # Student prediction
        student_pred = model_output["model_prediction"]

        # Get scalings
        c_skip_start = prepared_batch["c_skip_start"]
        c_out_start = prepared_batch["c_out_start"]

        if self.is_flow_matching:
            # For flow matching, apply consistency parameterization
            noisy_latents = prepared_batch["noisy_latents"]
            model_pred = c_skip_start * noisy_latents + c_out_start * student_pred
        else:
            # For DDPM, convert to x0 prediction first
            pred_x0 = self._get_predicted_original_sample(
                student_pred,
                prepared_batch["timesteps"],
                prepared_batch["noisy_latents"],
            )
            model_pred = c_skip_start * prepared_batch["noisy_latents"] + c_out_start * pred_x0

        # Target from teacher
        target = prepared_batch["target"]

        # Compute loss
        if self.config["loss_type"] == "l2":
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        elif self.config["loss_type"] == "huber":
            loss = torch.mean(
                torch.sqrt((model_pred.float() - target.float()) ** 2 + self.config["huber_c"] ** 2) - self.config["huber_c"]
            )
        else:
            raise ValueError(f"Unknown loss type: {self.config['loss_type']}")

        logs = {
            "lcm_loss": loss.item(),
            "total": loss.item(),
        }

        return loss, logs

    def _get_predicted_original_sample(self, model_output, timesteps, sample):
        """Convert model output to x0 prediction for DDPM models."""
        alphas = extract_into_tensor(self.alpha_schedule, timesteps, sample.shape)
        sigmas = extract_into_tensor(self.sigma_schedule, timesteps, sample.shape)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            pred_x0 = (sample - sigmas * model_output) / alphas
        elif self.noise_scheduler.config.prediction_type == "sample":
            pred_x0 = model_output
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            pred_x0 = alphas * sample - sigmas * model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.noise_scheduler.config.prediction_type}")

        return pred_x0

    def _get_predicted_noise(self, model_output, timesteps, sample):
        """Convert model output to noise prediction for DDPM models."""
        alphas = extract_into_tensor(self.alpha_schedule, timesteps, sample.shape)
        sigmas = extract_into_tensor(self.sigma_schedule, timesteps, sample.shape)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            pred_epsilon = model_output
        elif self.noise_scheduler.config.prediction_type == "sample":
            pred_epsilon = (sample - alphas * model_output) / sigmas
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            pred_epsilon = alphas * model_output + sigmas * sample
        else:
            raise ValueError(f"Unknown prediction type: {self.noise_scheduler.config.prediction_type}")

        return pred_epsilon

    def get_scheduler(self, *_):
        """Return the LCM scheduler for inference."""
        if self.is_flow_matching:
            # For flow matching, we can use a modified flow scheduler
            # You might need to implement a custom LCMFlowMatchingScheduler
            logger.warning(
                "Using standard flow scheduler for LCM inference. Consider implementing LCMFlowMatchingScheduler."
            )
            return self.noise_scheduler
        else:
            # For DDPM models, use the standard LCMScheduler
            return LCMScheduler.from_config(
                self.noise_scheduler.config,
                timestep_scaling=self.config["timestep_scaling_factor"],
            )


DistillationRegistry.register(
    "lcm",
    LCMDistiller,
    requires_distillation_cache=False,
    data_requirements=[[DatasetType.IMAGE, DatasetType.VIDEO]],
)

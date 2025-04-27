import os, logging, random
import torch
import torch.nn.functional as F
from helpers.distillation.common import DistillationBase

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


class DDPMPeRFlowDistiller(DistillationBase):
    """Implementation of PeRFlow distillation for DDPM-based models."""

    def __init__(self, teacher_model, student_model=None, config=None):
        # PeRFlow specific config defaults
        perflow_config = {
            "loss_type": "velocity_matching",
            "pred_type": "velocity",
            "reweighting_scheme": None,
            "windows": 16,
            "solving_steps": 2,
            "support_cfg": False,
            "cfg_sync": False,
            "discrete_timesteps": -1,
            "is_regularisation_data": True if student_model is None else False,
        }

        # Update with user config
        if config:
            perflow_config.update(config)

        super().__init__(teacher_model, student_model, perflow_config)

        # Ensure the model is not flow matching
        if self.is_flow_matching:
            raise ValueError(
                "Teacher model must be a DDPM-based model for PeRFlowDistiller"
            )

        # Initialize PeRFlow-specific components
        self._init_perflow_components()

    def _init_perflow_components(self):
        """Initialize PeRFlow-specific components."""
        # Import required components for PeRFlow
        # This assumes you have these components available
        from helpers.training.custom_schedule import PeRFlowScheduler, PFODESolver

        # Create PeRFlow scheduler
        self.perflow_scheduler = PeRFlowScheduler(
            num_train_timesteps=self.teacher_scheduler.config.num_train_timesteps,
            beta_start=self.teacher_scheduler.config.beta_start,
            beta_end=self.teacher_scheduler.config.beta_end,
            beta_schedule=self.teacher_scheduler.config.beta_schedule,
            prediction_type=self.config["pred_type"],
            t_noise=1,
            t_clean=0,
            num_time_windows=self.config["windows"],
        )

        # Create ODE solver
        self.solver = PFODESolver(
            scheduler=self.teacher_scheduler,
            t_initial=1,
            t_terminal=0,
        )

        self.custom_schedulers["perflow"] = self.perflow_scheduler

    def prepare_batch(self, batch, model, state):
        """Prepare batch with PeRFlow-specific processing."""
        # Mark batch as regularization data for LoRA training
        if self.low_rank_distillation:
            batch["is_regularisation_data"] = self.config["is_regularisation_data"]

        # First prepare batch using student model's normal method
        prepared_batch = model.prepare_batch(batch, state)

        # If we're using the same model with adapters toggled, we're done
        # The model.prepare_batch method will generate the parent model prediction
        if self.low_rank_distillation and self.config["is_regularisation_data"]:
            return prepared_batch

        # Add PeRFlow-specific processing for separate teacher/student models
        bsz = prepared_batch["latents"].shape[0]
        device = prepared_batch["latents"].device

        with torch.no_grad():
            # Sample timesteps
            timepoints = torch.rand((bsz,), device=device)
            teacher_num_train_timesteps = self.perflow_scheduler.num_train_timesteps

            if self.config["discrete_timesteps"] == -1:
                timepoints = (
                    timepoints * teacher_num_train_timesteps
                ).floor() / teacher_num_train_timesteps
            else:
                assert isinstance(self.config["discrete_timesteps"], int)
                timepoints = (
                    timepoints * self.config["discrete_timesteps"]
                ).floor() / self.config["discrete_timesteps"]

            timepoints = 1 - timepoints  # [1, 1/1000] or [1, 1/40]
            prepared_batch["perflow_timepoints"] = timepoints

            # Generate noise
            noises = torch.randn_like(prepared_batch["latents"])
            prepared_batch["perflow_noises"] = noises

            # Get time window endpoints
            t_start, t_end = self.perflow_scheduler.time_windows.lookup_window(
                timepoints
            )
            prepared_batch["perflow_t_start"] = t_start
            prepared_batch["perflow_t_end"] = t_end

            # Get noisy latents at start time
            latents_start = self.teacher_scheduler.add_noise(
                prepared_batch["latents"],
                noises,
                torch.clamp((t_start * teacher_num_train_timesteps).long() - 1, min=0),
            )
            prepared_batch["perflow_latents_start"] = latents_start

            # Temporarily disable adapter for teacher predictions if using same model
            if self.low_rank_distillation:
                self.toggle_adapter(enable=False)

            # Get guidance scale based on config
            guidance_scale = 1.0 if self.config["cfg_sync"] else 7.5

            # Solve ODE to get latents at end time using teacher model
            latents_end = self.solver.solve(
                latents=latents_start,
                t_start=t_start,
                t_end=t_end,
                unet=self.teacher_model.model,
                prompt_embeds=(
                    prepared_batch["encoder_hidden_states"]
                    if self.config["cfg_sync"]
                    else prepared_batch["encoder_hidden_states"]
                ),
                negative_prompt_embeds=prepared_batch.get(
                    "negative_encoder_hidden_states"
                ),
                guidance_scale=guidance_scale,
                num_steps=self.config["solving_steps"],
                num_windows=self.config["windows"],
            )

            # Re-enable adapter if using same model
            if self.low_rank_distillation:
                self.toggle_adapter(enable=True)

            prepared_batch["perflow_latents_end"] = latents_end

            # Interpolate to get latents at the sampled timepoint
            latents_t = latents_start + (latents_end - latents_start) / (
                t_end[:, None, None, None] - t_start[:, None, None, None]
            ) * (timepoints[:, None, None, None] - t_start[:, None, None, None])
            prepared_batch["perflow_latents_t"] = latents_t

            # Prepare targets based on prediction type
            if (
                self.config["loss_type"] == "velocity_matching"
                and self.config["pred_type"] == "velocity"
            ):
                targets = (latents_end - latents_start) / (
                    t_end[:, None, None, None] - t_start[:, None, None, None]
                )
            elif (
                self.config["loss_type"] == "noise_matching"
                and self.config["pred_type"] == "diff_eps"
            ):
                _, _, _, _, gamma_s_e, _, _ = self.perflow_scheduler.get_window_alpha(
                    timepoints.float().cpu()
                )
                gamma_s_e = gamma_s_e[:, None, None, None].to(device=device)
                lambda_s = 1 / gamma_s_e
                eta_s = -1 * (1 - gamma_s_e**2) ** 0.5 / gamma_s_e
                targets = (latents_end - lambda_s * latents_start) / eta_s
            elif (
                self.config["loss_type"] == "noise_matching"
                and self.config["pred_type"] == "ddim_eps"
            ):
                _, _, _, _, _, alphas_cumprod_start, alphas_cumprod_end = (
                    self.perflow_scheduler.get_window_alpha(timepoints.float().cpu())
                )
                alphas_cumprod_start = alphas_cumprod_start[:, None, None, None].to(
                    device=device
                )
                alphas_cumprod_end = alphas_cumprod_end[:, None, None, None].to(
                    device=device
                )
                lambda_s = (alphas_cumprod_end / alphas_cumprod_start) ** 0.5
                eta_s = (1 - alphas_cumprod_end) ** 0.5 - (
                    alphas_cumprod_end
                    / alphas_cumprod_start
                    * (1 - alphas_cumprod_start)
                ) ** 0.5
                targets = (latents_end - lambda_s * latents_start) / eta_s
            else:
                raise NotImplementedError

            prepared_batch["perflow_targets"] = targets

        return prepared_batch

    def compute_distill_loss(self, prepared_batch, model_output, original_loss):
        """Compute PeRFlow-specific distillation loss."""
        # For PeRFlow, we have two scenarios:
        # 1. Using same model with adapters toggled (regularization)
        # 2. Using separate teacher/student models with PeRFlow targets

        if self.low_rank_distillation and self.config["is_regularisation_data"]:
            # In this case, the parent model prediction is already computed
            loss = original_loss
            logs = {"regularisation_loss": loss.item()}
        else:
            # Get model prediction
            model_pred = model_output["model_prediction"]

            # Get PeRFlow targets
            targets = prepared_batch["perflow_targets"]
            timepoints = prepared_batch["perflow_timepoints"]

            # Compute loss
            if self.config["reweighting_scheme"] is None:
                loss = F.mse_loss(model_pred.float(), targets.float(), reduction="mean")
            else:
                if self.config["reweighting_scheme"] == "reciprocal":
                    loss_weights = 1.0 / torch.clamp(1.0 - timepoints, min=0.1) / 2.3
                    loss = (
                        ((model_pred.float() - targets.float()) ** 2).mean(
                            dim=[1, 2, 3]
                        )
                        * loss_weights
                    ).mean()
                else:
                    raise NotImplementedError

            logs = {"perflow_loss": loss.item()}

        return loss, logs

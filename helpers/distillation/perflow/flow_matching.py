import os, logging
import torch
import torch.nn.functional as F
from helpers.distillation.common import DistillationBase
from helpers.training.custom_schedule import Time_Windows

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


class FlowMatchingPeRFlowDistiller(DistillationBase):
    """Implementation of PeRFlow distillation adapted for flow matching models."""

    def __init__(self, teacher_model, student_model=None, config=None):
        flow_perflow_config = {
            "loss_type": "flow_matching",
            "solving_steps": 6,
            "windows": 4,
            "support_cfg": True,
            "cfg_sync": False,
            "discrete_timesteps": -1,
            "velocity_norm_weight": 0.01,
            "debug_cosine": True,
        }

        if config:
            flow_perflow_config.update(config)
        super().__init__(teacher_model, student_model, flow_perflow_config)

        if not self.is_flow_matching:
            raise ValueError(
                "Teacher model must be a flow matching model for FlowMatchingPeRFlowDistiller"
            )

        self._init_flow_matching_components()

    def _init_flow_matching_components(self):
        self._create_time_windows()
        self.flow_scheduler = self.teacher_scheduler

    def _create_time_windows(self):
        self.time_windows = Time_Windows(num_windows=self.config["windows"])

    def solve_flow(self, prepared_batch, t_start, t_end, guidance_scale=1.0):
        logger.info(f"Solving flow with t_start: {t_start}, t_end: {t_end}")
        latents = prepared_batch["perflow_latents_start"]
        prompt_embeds = prepared_batch["encoder_hidden_states"]
        negative_prompt_embeds = prepared_batch.get("negative_encoder_hidden_states")
        device = latents.device

        do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None
        logger.info(f"Using classifier-free guidance: {do_cfg}")
        num_steps = self.config["solving_steps"]
        step_size = (t_start - t_end) / num_steps
        current_latents = latents
        current_t = t_start

        for i in range(num_steps):
            model_inputs = prepared_batch.copy()
            model_inputs.update(
                {
                    "latents": current_latents,
                    "noisy_latents": current_latents,
                    "timesteps": current_t,
                    "encoder_hidden_states": prompt_embeds,
                }
            )
            for key in ["encoder_attention_mask", "added_cond_kwargs"]:
                if key in prepared_batch:
                    model_inputs[key] = prepared_batch[key]

            if do_cfg:
                cond_inputs = model_inputs.copy()
                uncond_inputs = {
                    "latents": current_latents,
                    "noisy_latents": current_latents,
                    "timesteps": current_t,
                    "encoder_hidden_states": negative_prompt_embeds,
                }
                for key in ["encoder_attention_mask", "added_cond_kwargs"]:
                    if key in prepared_batch:
                        uncond_inputs[key] = prepared_batch[key]
                with torch.no_grad():
                    uncond_pred = self.teacher_model.model_predict(uncond_inputs)[
                        "model_prediction"
                    ]
                    cond_pred = self.teacher_model.model_predict(cond_inputs)[
                        "model_prediction"
                    ]
                v1 = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                with torch.no_grad():
                    v1 = self.teacher_model.model_predict(model_inputs)[
                        "model_prediction"
                    ]

            x_predict = (
                current_latents + step_size.view(-1, *([1] * (v1.dim() - 1))) * v1
            )
            next_t = current_t - step_size

            model_inputs["latents"] = x_predict
            model_inputs["noisy_latents"] = x_predict
            model_inputs["timesteps"] = next_t

            if do_cfg:
                cond_inputs = model_inputs.copy()
                uncond_inputs = {
                    "latents": x_predict,
                    "noisy_latents": x_predict,
                    "timesteps": next_t,
                    "encoder_hidden_states": negative_prompt_embeds,
                }
                for key in ["encoder_attention_mask", "added_cond_kwargs"]:
                    if key in prepared_batch:
                        uncond_inputs[key] = prepared_batch[key]
                with torch.no_grad():
                    uncond_pred = self.teacher_model.model_predict(uncond_inputs)[
                        "model_prediction"
                    ]
                    cond_pred = self.teacher_model.model_predict(cond_inputs)[
                        "model_prediction"
                    ]
                v2 = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                with torch.no_grad():
                    v2 = self.teacher_model.model_predict(model_inputs)[
                        "model_prediction"
                    ]

            avg_v = 0.5 * (v1 + v2)
            current_latents = (
                current_latents + step_size.view(-1, *([1] * (avg_v.dim() - 1))) * avg_v
            )
            current_t = next_t

            logger.info(
                f"Step {i+1}/{num_steps}: step_size={step_size.mean().item():.6f}, v1 mean={v1.mean().item():.6f}, v2 mean={v2.mean().item():.6f}, avg_v std={avg_v.std().item():.6f}"
            )
            logger.info(
                f"After step {i+1}/{num_steps}: latents mean={current_latents.mean().item():.6f}, std={current_latents.std().item():.6f}"
            )

        return current_latents

    def compute_distill_loss(self, prepared_batch, model_output, original_loss):
        model_pred = model_output["model_prediction"]
        targets = prepared_batch["perflow_targets"]
        logger.info(
            f"Computing loss - model_pred shape: {model_pred.shape}, targets shape: {targets.shape}"
        )
        loss = F.mse_loss(model_pred.float(), targets.float(), reduction="none")

        # Optional cosine debug logging
        if self.config.get("debug_cosine", False):
            cosine = F.cosine_similarity(model_pred.flatten(), targets.flatten(), dim=0)
            logger.info(
                f"Velocity cosine similarity (student vs. target): {cosine.item():.6f}"
            )

        # Optional velocity norm penalty
        if self.config.get("velocity_norm_weight", 0.0) > 0:
            vnorm = torch.norm(model_pred, dim=1).mean()
            loss += self.config["velocity_norm_weight"] * vnorm
            logger.info(f"Velocity norm penalty: {vnorm.item():.6f}")

        loss_mean = loss.mean().item()
        loss_max = loss.max().item()
        loss_min = loss.min().item()
        logger.info(f"Loss stats - mean: {loss_mean}, max: {loss_max}, min: {loss_min}")
        return loss.mean(), {
            "perflow_loss": loss_mean,
            "perflow_loss_max": loss_max,
            "perflow_loss_min": loss_min,
        }

    def prepare_batch(self, batch, model, state):
        if "is_regularisation_data" in batch:
            del batch["is_regularisation_data"]

        prepared_batch = (
            model.prepare_batch(batch, state) if "noisy_latents" not in batch else batch
        )

        logger.info(f"Prepared batch keys: {prepared_batch.keys()}")
        if "latents" in prepared_batch:
            logger.info(f"latents shape: {prepared_batch['latents'].shape}")
        if "noisy_latents" in prepared_batch:
            logger.info(f"noisy_latents shape: {prepared_batch['noisy_latents'].shape}")

        bsz = prepared_batch["latents"].shape[0]
        device = prepared_batch["latents"].device

        with torch.no_grad():
            timepoints = 1 - torch.rand((bsz,), device=device)
            timepoints = torch.clamp(timepoints, min=1e-6)
            if self.config["discrete_timesteps"] != -1:
                timepoints = (
                    timepoints * self.config["discrete_timesteps"]
                ).floor() / self.config["discrete_timesteps"]

            t_start, t_end = self.time_windows.lookup_window(timepoints)
            prepared_batch["perflow_timepoints"] = timepoints
            prepared_batch["perflow_t_start"] = t_start
            prepared_batch["perflow_t_end"] = t_end
            prepared_batch["perflow_latents_start"] = prepared_batch[
                "noisy_latents"
            ].clone()

            if self.low_rank_distillation:
                self.toggle_adapter(enable=False)

            guidance_scale = 1.0 if self.config["cfg_sync"] else 7.5
            latents_end = self.solve_flow(
                prepared_batch, t_start, t_end, guidance_scale=guidance_scale
            )

            if self.low_rank_distillation:
                self.toggle_adapter(enable=True)

            prepared_batch["perflow_latents_end"] = latents_end

            if len(prepared_batch["latents"].shape) == 5:
                latents_t = prepared_batch["perflow_latents_start"] + (
                    latents_end - prepared_batch["perflow_latents_start"]
                ) / (
                    t_end[:, None, None, None, None]
                    - t_start[:, None, None, None, None]
                ) * (
                    timepoints[:, None, None, None, None]
                    - t_start[:, None, None, None, None]
                )
            else:
                latents_t = prepared_batch["perflow_latents_start"] + (
                    latents_end - prepared_batch["perflow_latents_start"]
                ) / (t_end[:, None, None, None] - t_start[:, None, None, None]) * (
                    timepoints[:, None, None, None] - t_start[:, None, None, None]
                )

            prepared_batch["perflow_latents_t"] = latents_t
            prepared_batch["noisy_latents_original"] = prepared_batch[
                "noisy_latents"
            ].clone()
            prepared_batch["noisy_latents"] = latents_t

            if self.config["loss_type"] == "flow_matching":
                if len(prepared_batch["latents"].shape) == 5:
                    targets = (
                        latents_end - prepared_batch["perflow_latents_start"]
                    ) / (
                        t_end[:, None, None, None, None]
                        - t_start[:, None, None, None, None]
                    )
                else:
                    targets = (
                        latents_end - prepared_batch["perflow_latents_start"]
                    ) / (t_end[:, None, None, None] - t_start[:, None, None, None])
            else:
                raise ValueError(
                    f"Unsupported loss type for flow matching: {self.config['loss_type']}"
                )

            prepared_batch["perflow_targets"] = targets
            prepared_batch["perflow_targets"] *= -1  # reverse ODE for student

        logger.info(f"Final batch keys: {prepared_batch.keys()}")
        if "perflow_targets" in prepared_batch:
            logger.info(
                f"perflow_targets shape: {prepared_batch['perflow_targets'].shape}"
            )

        return prepared_batch

    def get_scheduler(self):
        from helpers.training.custom_schedule import PeRFlowScheduler

        return PeRFlowScheduler(
            num_time_windows=self.config["windows"],
            num_train_timesteps=self.teacher_scheduler.num_train_timesteps,
            prediction_type=self.teacher_model.PREDICTION_TYPE.value,
        )

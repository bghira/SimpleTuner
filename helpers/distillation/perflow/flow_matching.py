import os, logging
import torch
import torch.nn.functional as F
from copy import deepcopy
from helpers.distillation.common import DistillationBase
from helpers.training.custom_schedule import Time_Windows

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


def sigmoid_schedule(T, alpha=5.12):
    x = torch.linspace(1, 0, T + 1, device="cuda")
    ts = torch.sigmoid(x * alpha - alpha / 2)
    return (ts - ts.min()) / (ts.max() - ts.min())


def to_model_timestep(t):
    logger.info(f"Converting from {t} to {t * 1000.0}")
    return t * 1000.0


def compute_segment_vector(
    model,
    latents,
    t_start,
    t_end,
    steps=8,
    cfg=False,
    encoder_hidden_states=None,
    negative_encoder_hidden_states=None,
    encoder_attention_mask=None,
    added_cond_kwargs=None,
    guidance_scale=1.0,
    prepared_batch: dict = {},
):
    """
    Computes a flow-matching vector (delta) using Euler integration between t_start and t_end.
    Works batched, optionally with classifier-free guidance (CFG).

    Args:
        model: The teacher model.
        latents: [B, C, H, W] or [B, C, F, H, W] tensor.
        t_start: [B] float tensor.
        t_end: [B] float tensor.
        steps: Integer number of Euler steps.
        cfg: Whether to apply CFG.
        encoder_hidden_states: [B, ..., D] positive prompt embeds.
        negative_encoder_hidden_states: [B, ..., D] negative prompt embeds (if CFG enabled).
        encoder_attention_mask: Optional mask.
        added_cond_kwargs: Optional additional kwargs.

    Returns:
        delta: [B, ...] flow vector normalized by step duration.
    """
    assert latents.ndim in (4, 5), "Latents must be 4D or 5D"
    assert (
        t_start.shape == t_end.shape == (latents.shape[0],)
    ), "Timesteps must be batch-aligned"

    B = latents.shape[0]
    device = latents.device
    dtype = latents.dtype

    # Create time schedule [B, steps+1] linearly spaced from t_start to t_end
    t_steps = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)
    t_schedule = t_start[:, None] * (1 - t_steps) + t_end[:, None] * t_steps  # [B, S+1]

    x = latents.clone()
    for i in range(steps):
        t_curr = t_schedule[:, i]  # [B]
        if cfg:
            assert negative_encoder_hidden_states is not None
            inputs_pos = prepared_batch.copy()
            inputs_neg = prepared_batch.copy()
            inputs_pos.update(
                {
                    "latents": x,
                    "noisy_latents": x,
                    "timesteps": to_model_timestep(t_curr),
                    "encoder_hidden_states": encoder_hidden_states,
                }
            )
            inputs_neg.update(
                {
                    "latents": x,
                    "noisy_latents": x,
                    "timesteps": to_model_timestep(t_curr),
                    "encoder_hidden_states": negative_encoder_hidden_states,
                }
            )
            if encoder_attention_mask is not None:
                inputs_pos["encoder_attention_mask"] = encoder_attention_mask
                inputs_neg["encoder_attention_mask"] = encoder_attention_mask
            if added_cond_kwargs is not None:
                inputs_pos["added_cond_kwargs"] = added_cond_kwargs
                inputs_neg["added_cond_kwargs"] = added_cond_kwargs
            if (
                "add_text_embeds" in prepared_batch
                and prepared_batch.get("add_text_embeds") is not None
            ):
                inputs_pos["add_text_embeds"] = prepared_batch["add_text_embeds"]
                inputs_neg["add_text_embeds"] = torch.zeros_like(
                    prepared_batch["add_text_embeds"]
                )

            with torch.no_grad():
                v_pos = model.model_predict(inputs_pos)["model_prediction"]
                v_neg = model.model_predict(inputs_neg)["model_prediction"]
            v = (
                v_neg + (v_pos - v_neg) * guidance_scale
            )  # guidance_scale fixed at 1.0 for pure interpolation
        else:
            inputs = prepared_batch.copy()
            inputs.update(
                {
                    "latents": x,
                    "noisy_latents": x,
                    "timesteps": to_model_timestep(t_curr),
                    "encoder_hidden_states": encoder_hidden_states,
                }
            )
            if encoder_attention_mask is not None:
                inputs["encoder_attention_mask"] = encoder_attention_mask
            if added_cond_kwargs is not None:
                inputs["added_cond_kwargs"] = added_cond_kwargs
            if (
                "add_text_embeds" in prepared_batch
                and prepared_batch.get("add_text_embeds") is not None
            ):
                inputs["add_text_embeds"] = prepared_batch["add_text_embeds"]

            with torch.no_grad():
                v = model.model_predict(inputs)["model_prediction"]

        dt = t_schedule[:, i + 1] - t_schedule[:, i]  # [B]
        dt = dt.reshape(
            B, *((1,) * (latents.ndim - 1))
        )  # [B, 1, 1, 1] or [B, 1, 1, 1, 1]
        x = x + dt * v

    delta = (latents - x) / (t_start - t_end).reshape(B, *((1,) * (latents.ndim - 1)))
    return delta.detach()


class FlowMatchingPeRFlowDistiller(DistillationBase):
    def __init__(self, teacher_model, student_model=None, config=None):
        flow_perflow_config = {
            "loss_type": "flow_matching",
            "solving_steps": 30,
            "windows": 4,
            "support_cfg": False,
            "cfg_sync": True,
            "discrete_timesteps": -1,
            "segment_loss": False,
            "segment_steps": 8,
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
        if self.config.get("segment_loss", False):
            return compute_segment_vector(
                self.teacher_model,
                prepared_batch["perflow_latents_start"],
                t_start,
                t_end,
                steps=self.config.get("segment_steps", 8),
                # guidance_scale=guidance_scale,
                encoder_hidden_states=prepared_batch["encoder_hidden_states"],
                negative_encoder_hidden_states=prepared_batch.get(
                    "negative_encoder_hidden_states"
                ),
                # prepared_batch=prepared_batch,
            )

        logger.info(f"Solving flow with t_start: {t_start}, t_end: {t_end}")
        latents = prepared_batch["perflow_latents_start"]
        prompt_embeds = prepared_batch["encoder_hidden_states"]
        negative_prompt_embeds = prepared_batch.get(
            "negative_encoder_hidden_states"
        ) or torch.zeros_like(prompt_embeds)
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
                    "timesteps": to_model_timestep(current_t),
                    "encoder_hidden_states": prompt_embeds,
                }
            )
            for key in ["encoder_attention_mask", "added_cond_kwargs"]:
                if key in prepared_batch:
                    model_inputs[key] = prepared_batch[key]

            if do_cfg:
                cond_inputs = model_inputs.copy()
                uncond_inputs = model_inputs.copy()
                uncond_inputs.update(
                    {
                        "latents": current_latents,
                        "noisy_latents": current_latents,
                        "timesteps": to_model_timestep(current_t),
                        "encoder_hidden_states": negative_prompt_embeds,
                    }
                )
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
                current_latents + step_size[:, None, None, None] * v1
                if len(current_latents.shape) == 4
                else current_latents + step_size[:, None, None, None, None] * v1
            )

            next_t = current_t - step_size
            model_inputs["latents"] = x_predict
            model_inputs["noisy_latents"] = x_predict
            model_inputs["timesteps"] = next_t

            if do_cfg:
                cond_inputs = model_inputs.copy()
                uncond_inputs = model_inputs.copy()
                uncond_inputs.update(
                    {
                        "latents": x_predict,
                        "noisy_latents": x_predict,
                        "timesteps": to_model_timestep(next_t),
                        "encoder_hidden_states": negative_prompt_embeds,
                    }
                )
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
                current_latents + step_size[:, None, None, None] * avg_v
                if len(current_latents.shape) == 4
                else current_latents + step_size[:, None, None, None, None] * avg_v
            )
            current_t = next_t

            logger.info(
                f"Step {i+1}/{num_steps}: step_size={step_size.mean().item()}, v1 mean={v1.mean().item()}, v2 mean={v2.mean().item()}, avg_v std={avg_v.std().item()}"
            )
            logger.info(
                f"After step {i+1}/{num_steps}: latents mean={current_latents.mean().item()}, std={current_latents.std().item()}"
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

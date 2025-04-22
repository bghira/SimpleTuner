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
    # logger.info(f"Converting from {t} to {t * 1000.0}")
    return t * 1000.0


def compute_segment_vector(
    model,
    latents,
    t_start,
    t_end,
    steps=8,
    cfg: bool = False,
    encoder_hidden_states: torch.Tensor = None,
    negative_encoder_hidden_states: torch.Tensor = None,
    encoder_attention_mask: torch.Tensor = None,
    added_cond_kwargs: dict = None,
    guidance_scale: float = 1.0,
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
            add_text_embeds = prepared_batch.get("add_text_embeds", None)
            if add_text_embeds is not None:
                inputs_pos["add_text_embeds"] = add_text_embeds
                inputs_neg["add_text_embeds"] = torch.zeros_like(add_text_embeds)

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
            add_text_embeds = prepared_batch.get("add_text_embeds", None)
            if add_text_embeds is not None:
                inputs["add_text_embeds"] = add_text_embeds

            with torch.no_grad():
                v = model.model_predict(inputs)["model_prediction"]

        dt = t_schedule[:, i + 1] - t_schedule[:, i]  # [B]
        dt = dt.reshape(
            B, *((1,) * (latents.ndim - 1))
        )  # [B, 1, 1, 1] or [B, 1, 1, 1, 1]
        x = x + dt * v

    delta = (latents - x) / (t_start - t_end).reshape(B, *((1,) * (latents.ndim - 1)))
    return delta.detach()


def extract_velocity(noise_pred, latents, timesteps, scheduler):
    """
    Convert model output to velocity vector, depending on prediction_type.
    Args:
        noise_pred: Model output [B, C, H, W]
        latents: Current latents x_t [B, C, H, W]
        timesteps: Current timestep indices [B]
        scheduler: DDPM scheduler with alphas_cumprod
    Returns:
        velocity: [B, C, H, W]
    """
    alpha_t = (
        scheduler.alphas_cumprod[timesteps].reshape(-1, 1, 1, 1).to(latents.device)
    )
    beta_t = 1.0 - alpha_t

    if scheduler.config.prediction_type == "epsilon":
        return -noise_pred / (alpha_t.sqrt() * beta_t.sqrt())
    elif scheduler.config.prediction_type == "v_prediction":
        return noise_pred
    else:
        raise NotImplementedError(
            f"Unsupported prediction_type: {scheduler.config.prediction_type}"
        )


class PeRFlowDistiller(DistillationBase):
    def __init__(self, teacher_model, student_model=None, config=None):
        flow_perflow_config = {
            "loss_type": "flow_matching",
            "solving_steps": 8,
            "windows": 4,
            "cfg_transfer": True,
            "cfg_value": 7.0,
            "segment_loss": False,
            "segment_steps": 4,
            "discrete_timesteps": -1,
            "ode_direction": "reverse",
        }

        if config:
            flow_perflow_config.update(config)
        super().__init__(teacher_model, student_model, flow_perflow_config)

        if not self.is_flow_matching:
            raise ValueError(
                "Teacher model must be a flow matching model for PeRFlowDistiller"
            )

        self._init_flow_matching_components()

    def _init_flow_matching_components(self):
        self._create_time_windows()
        self.flow_scheduler = self.teacher_scheduler

    def _create_time_windows(self):
        self.time_windows = Time_Windows(num_windows=self.config["windows"])

    def solve_flow(self, prepared_batch, t_start, t_end, guidance_scale: float = 1.0):

        logger.info(f"Solving flow with t_start: {t_start}, t_end: {t_end}")
        latents = prepared_batch["perflow_latents_start"]
        prompt_embeds = prepared_batch["encoder_hidden_states"]
        negative_prompt_embeds = prepared_batch.get(
            "negative_encoder_hidden_states"
        ) or torch.zeros_like(prompt_embeds)
        device = latents.device

        if self.config.get("segment_loss", False):
            return compute_segment_vector(
                model=self.teacher_model,
                latents=prepared_batch["perflow_latents_start"],
                t_start=t_start,
                t_end=t_end,
                steps=self.config.get("segment_steps", 8),
                guidance_scale=guidance_scale,
                encoder_hidden_states=prepared_batch["encoder_hidden_states"],
                negative_encoder_hidden_states=negative_prompt_embeds,
                prepared_batch=prepared_batch,
            )

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
                    cond_pred = self.teacher_model.model_predict(model_inputs)[
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
            model_inputs["timesteps"] = to_model_timestep(next_t)

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

    def solve_ddpm(self, prepared_batch, t_start, t_end, guidance_scale=1.0):
        model = self.teacher_model
        scheduler = self.teacher_scheduler
        num_steps = self.config["solving_steps"]
        latents = prepared_batch["perflow_latents_start"]
        prompt_embeds = prepared_batch["encoder_hidden_states"]
        negative_prompt_embeds = prepared_batch.get("negative_encoder_hidden_states")
        pooled_prompt_embeds = prepared_batch.get("add_text_embeds")
        added_cond_kwargs = prepared_batch.get("added_cond_kwargs")
        prompt_attention_mask = prepared_batch.get("encoder_attention_mask")

        device = latents.device
        bsz = latents.shape[0]

        do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None
        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            if pooled_prompt_embeds is not None:
                pooled_prompt_embeds = torch.cat([torch.zeros_like(pooled_prompt_embeds), pooled_prompt_embeds], dim=0)
            if prompt_attention_mask is not None:
                prompt_attention_mask = torch.cat([prompt_attention_mask, prompt_attention_mask], dim=0)

        t_vec = torch.linspace(0, 1, num_steps + 1, device=device, dtype=latents.dtype)
        t_schedule = t_start[:, None] * (1 - t_vec) + t_end[:, None] * t_vec
        timesteps = (scheduler.config.num_train_timesteps - 1) * (1.0 - t_schedule)
        timesteps = timesteps.round().long()

        x = latents.clone()
        for i in range(num_steps):
            t_curr = timesteps[:, i]
            t_next = timesteps[:, i + 1]
            step = (t_curr - t_next).float().reshape(bsz, *[1] * (latents.ndim - 1)) / scheduler.config.num_train_timesteps

            t_input = t_curr
            if do_cfg:
                x = torch.cat([x, x], dim=0)
                t_input = torch.cat([t_curr, t_curr], dim=0)

            x_scaled = scheduler.scale_model_input(x, t_input)

            model_args = [x_scaled, t_input]
            kwargs = {}
            if prompt_embeds is not None:
                kwargs["encoder_hidden_states"] = prompt_embeds
            if pooled_prompt_embeds is not None:
                kwargs["add_text_embeds"] = pooled_prompt_embeds
            if prompt_attention_mask is not None:
                kwargs["encoder_attention_mask"] = prompt_attention_mask
            if added_cond_kwargs is not None:
                kwargs["added_cond_kwargs"] = added_cond_kwargs

            with torch.no_grad():
                noise_pred = model(*model_args, **kwargs)[0]

            if do_cfg:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            alpha_t = scheduler.alphas_cumprod[t_curr].reshape(bsz, 1, 1, 1).to(x.dtype)
            sigma_t = (1 - alpha_t).sqrt()
            velocity = -noise_pred / (alpha_t.sqrt() * sigma_t)

            x = x[:bsz] + step * velocity

        delta = (latents - x) / (t_start - t_end).reshape(bsz, *[1] * (latents.ndim - 1))
        return delta.detach()

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

    def prepare_batch(self, prepared_batch, model, state):
        print(f'Incoming timesteps: {prepared_batch["timesteps"]}')
        if "is_regularisation_data" in prepared_batch:
            del prepared_batch["is_regularisation_data"]

        logger.info(f"Prepared batch keys: {prepared_batch.keys()}")
        if "latents" in prepared_batch:
            logger.info(f"latents shape: {prepared_batch['latents'].shape}")
        if "noisy_latents" in prepared_batch:
            logger.info(f"noisy_latents shape: {prepared_batch['noisy_latents'].shape}")

        bsz = prepared_batch["latents"].shape[0]
        device = prepared_batch["latents"].device

        with torch.no_grad():
            sigmoid_ts = sigmoid_schedule(1000)
            timepoints = sigmoid_ts[
                torch.randint(0, len(sigmoid_ts), (bsz,), device=device)
            ]
            # timepoints = 1 - torch.rand((bsz,), device=device)
            # timepoints = torch.clamp(timepoints, min=0)
            if self.config["discrete_timesteps"] != -1:
                timepoints = (
                    timepoints * self.config["discrete_timesteps"]
                ).floor() / self.config["discrete_timesteps"]

            t_start, t_end = self.time_windows.lookup_window(timepoints)
            prepared_batch["timesteps"] = to_model_timestep(timepoints)
            prepared_batch["perflow_timepoints"] = timepoints
            prepared_batch["perflow_t_start"] = t_start
            prepared_batch["perflow_t_end"] = t_end
            prepared_batch["perflow_latents_start"] = prepared_batch[
                "noisy_latents"
            ].clone()

            if self.low_rank_distillation:
                self.toggle_adapter(enable=False)

            enable_cfg = self.config.get("cfg_transfer", True)
            guidance_scale = self.config.get("cfg_value", 7.0) if enable_cfg else 1.0

            if self.teacher_model.config.prediction_type in ["epsilon", "v_prediction"]:
                latents_end = self.solve_ddpm(
                    prepared_batch=prepared_batch,
                    t_start=t_start,
                    t_end=t_end,
                    guidance_scale=guidance_scale,
                )
            elif self.teacher_model.PREDICTION_TYPE.value == "flow_matching":
                latents_end = self.solve_flow(
                    prepared_batch=prepared_batch,
                    t_start=t_start,
                    t_end=t_end,
                    guidance_scale=guidance_scale,
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
            if self.config["ode_direction"] == "reverse":
                prepared_batch["perflow_targets"] = -prepared_batch["perflow_targets"]

        logger.info(f"Final batch keys: {prepared_batch.keys()}")
        if "perflow_targets" in prepared_batch:
            logger.info(
                f"perflow_targets shape: {prepared_batch['perflow_targets'].shape}"
            )

        print(f'Outgoing timesteps: {prepared_batch["timesteps"]}')
        return prepared_batch

    def get_scheduler(self):
        from helpers.training.custom_schedule import PeRFlowScheduler

        return PeRFlowScheduler(
            num_time_windows=self.config["windows"],
            num_train_timesteps=self.teacher_scheduler.num_train_timesteps,
            prediction_type=self.teacher_model.PREDICTION_TYPE.value,
        )

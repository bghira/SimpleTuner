import os
import logging
import torch
import torch.nn.functional as F
from copy import deepcopy

from helpers.distillation.common import DistillationBase
from helpers.training.custom_schedule import Time_Windows

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

EPS = 1e-7  # to avoid division by zero


def sigmoid_schedule(T: int, alpha: float = 5.12) -> torch.Tensor:
    """
    linspace from 1→0, apply sigmoid, then normalize to [0..1].
    """
    x = torch.linspace(1, 0, T + 1, device="cuda")
    ts = torch.sigmoid(x * alpha - alpha / 2)
    return (ts - ts.min()) / (ts.max() - ts.min())


def to_model_timestep(t: torch.Tensor) -> torch.Tensor:
    """
    Convert a fractional t in [0..1] → [0..1000]
    """
    return t * 1000.0


def compute_segment_vector(
    model,
    latents: torch.Tensor,
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    steps: int = 8,
    cfg: bool = False,
    encoder_hidden_states: torch.Tensor = None,
    negative_encoder_hidden_states: torch.Tensor = None,
    encoder_attention_mask: torch.Tensor = None,
    added_cond_kwargs: dict = None,
    guidance_scale: float = 1.0,
    prepared_batch: dict = {},
) -> torch.Tensor:
    """
    - Build a [B × (steps+1)] tensor of timesteps linearly from t_start→t_end
    - For i in [0..steps−1], do one Euler update x ← x + dt ⋅ v
    - v is computed with or without CFG
    - Finally return delta = (x_start - x_end)/(t_start - t_end + EPS)
    """

    assert latents.ndim in (4, 5), "Latents must be 4D or 5D"
    B = latents.shape[0]
    device = latents.device
    dtype = latents.dtype
    assert t_start.shape == t_end.shape == (B,), "t_start/t_end must each be [B]"

    # 1) Build t_schedule = [B, steps+1], linearly from t_start→t_end
    t_steps = torch.linspace(
        0.0, 1.0, steps + 1, device=device, dtype=dtype
    )  # [steps+1]
    t_schedule = (
        t_start[:, None] * (1.0 - t_steps) + t_end[:, None] * t_steps
    )  # [B, steps+1]

    # 2) Clone x = x_start
    x = latents.clone()

    for i in range(steps):
        t_curr = t_schedule[:, i]  # [B] in [0..1]

        if cfg and (negative_encoder_hidden_states is not None):
            # --- CFG branch ---
            inputs_pos = prepared_batch.copy()
            inputs_pos.update(
                {
                    "latents": x,
                    "noisy_latents": x,
                    "timesteps": to_model_timestep(t_curr),
                    "encoder_hidden_states": encoder_hidden_states,
                }
            )
            inputs_neg = prepared_batch.copy()
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
            v = v_neg + (v_pos - v_neg) * guidance_scale

        else:
            # --- No‐CFG branch ---
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

        # 3) dt = t_schedule[:, i+1] - t_schedule[:, i]; then Euler: x ← x + dt * v
        dt = (t_schedule[:, i + 1] - t_schedule[:, i]).reshape(
            B, *([1] * (latents.ndim - 1))
        )  # [B,1,1,1,...]
        x = x + dt * v

    # 4) After `steps` substeps, x is x_end. Compute delta = (x_start - x_end)/(t_start - t_end + EPS).
    denom = (t_start - t_end).reshape(B, *([1] * (latents.ndim - 1)))
    delta = (latents - x) / (denom + EPS)
    return delta.detach()


class PeRFlowDistiller(DistillationBase):
    """
    Key points:
      - ode_direction is always "reverse": we march t from t_start down to t_end.
      - If segment_loss=True, we call compute_segment_vector(t_start→t_end).
      - Otherwise we do a two‐stage Heun/Euler integration using solving_steps.
      - Logging of the first two substeps + last substep.
    """

    def __init__(self, teacher_model, student_model=None, config=None):
        flow_perflow_config = {
            "loss_type": "flow_matching",
            "solving_steps": 50,
            "windows": 50,
            "cfg_transfer": True,
            "cfg_value": 1.0,
            "segment_loss": True,
            "segment_steps": 4,
            "discrete_timesteps": -1,
            "ode_direction": "reverse",
            "debug_cosine": True,
            "velocity_norm_weight": 0.0,
        }

        if config:
            flow_perflow_config.update(config)
        super().__init__(teacher_model, student_model, flow_perflow_config)

        if not self.is_flow_matching:
            raise ValueError(
                "Teacher model must be a flow‐matching model for PeRFlowDistiller"
            )

        self._init_flow_matching_components()

    def _init_flow_matching_components(self):
        self._create_time_windows()
        self.flow_scheduler = self.teacher_scheduler

    def _create_time_windows(self):
        self.time_windows = Time_Windows(num_windows=self.config["windows"])

    def solve_flow(
        self,
        prepared_batch: dict,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Integrate the teacher’s flow from t_start → t_end “backwards” (ode_direction="reverse").
        If segment_loss=True, call compute_segment_vector and reconstruct x_end directly.
        Otherwise perform a two‐stage Heun/Euler over self.config["solving_steps"].
        """

        logger.info(f"Solving flow with t_start: {t_start}, t_end: {t_end}")

        latents = prepared_batch["perflow_latents_start"]
        prompt_embeds = prepared_batch["encoder_hidden_states"]
        negative_prompt_embeds = prepared_batch.get("negative_encoder_hidden_states")
        if negative_prompt_embeds is None:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)

        # If segment_loss → use compute_segment_vector to get delta, then x_end = x_start - delta*(t_start - t_end)
        if self.config.get("segment_loss", False):
            delta = compute_segment_vector(
                model=self.teacher_model,
                latents=latents,
                t_start=t_start,
                t_end=t_end,
                steps=self.config.get("segment_steps", 8),
                cfg=(guidance_scale > 1.0),
                encoder_hidden_states=prompt_embeds,
                negative_encoder_hidden_states=negative_prompt_embeds,
                encoder_attention_mask=prepared_batch.get(
                    "encoder_attention_mask", None
                ),
                added_cond_kwargs=prepared_batch.get("added_cond_kwargs", None),
                guidance_scale=guidance_scale,
                prepared_batch=prepared_batch,
            )
            # Reconstruct x_end for the student’s “latent_end”
            x_end = latents - delta * (t_start - t_end).reshape(
                latents.shape[0], *([1] * (latents.ndim - 1))
            )
            return x_end

        # Otherwise do a proper Heun/Euler integration over solving_steps
        do_cfg = guidance_scale > 1.0
        logger.info(f"Using classifier-free guidance: {do_cfg}")

        num_steps = self.config["solving_steps"]
        # step_size = (t_start - t_end)/num_steps, shape: [B]
        step_size = (t_start - t_end) / num_steps

        current_latents = latents
        current_t = t_start

        for i in range(num_steps):
            # ----- Stage 1: compute v1 at (current_latents, current_t) -----
            model_inputs = prepared_batch.copy()
            model_inputs.update(
                {
                    "latents": current_latents,
                    "noisy_latents": current_latents,
                    "timesteps": to_model_timestep(current_t),
                    "encoder_hidden_states": prompt_embeds,
                }
            )
            for key in [
                "encoder_attention_mask",
                "added_cond_kwargs",
                "add_text_embeds",
            ]:
                if key in prepared_batch:
                    model_inputs[key] = prepared_batch[key]

            with torch.no_grad():
                if do_cfg:
                    uncond_inputs = model_inputs.copy()
                    uncond_inputs["encoder_hidden_states"] = negative_prompt_embeds
                    v_uncond = self.teacher_model.model_predict(uncond_inputs)[
                        "model_prediction"
                    ]
                    v_cond = self.teacher_model.model_predict(model_inputs)[
                        "model_prediction"
                    ]
                    v1 = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    v1 = self.teacher_model.model_predict(model_inputs)[
                        "model_prediction"
                    ]

            step_broadcast = step_size.reshape(-1, *([1] * (latents.ndim - 1)))
            # one Euler provisional step:
            x_provisional = current_latents + step_broadcast * v1

            # ----- Stage 2: compute v2 at (x_provisional, next_t) -----
            next_t = current_t - step_size  # reverse direction
            model_inputs["latents"] = x_provisional
            model_inputs["noisy_latents"] = x_provisional
            model_inputs["timesteps"] = to_model_timestep(next_t)

            with torch.no_grad():
                if do_cfg:
                    uncond_inputs2 = model_inputs.copy()
                    uncond_inputs2["encoder_hidden_states"] = negative_prompt_embeds
                    v2_uncond = self.teacher_model.model_predict(uncond_inputs2)[
                        "model_prediction"
                    ]
                    v2_cond = self.teacher_model.model_predict(model_inputs)[
                        "model_prediction"
                    ]
                    v2 = v2_uncond + guidance_scale * (v2_cond - v2_uncond)
                else:
                    v2 = self.teacher_model.model_predict(model_inputs)[
                        "model_prediction"
                    ]

            avg_v = 0.5 * (v1 + v2)
            # full Heun update:
            current_latents = current_latents + step_broadcast * avg_v
            current_t = next_t

            if i < 2 or i == num_steps - 1:
                logger.info(
                    f"Step {i+1}/{num_steps}: step_size={step_size.mean().item():.4f}, "
                    f"v1 mean={v1.mean().item():.4f}, v2 mean={v2.mean().item():.4f}, avg_v std={avg_v.std().item():.4f}"
                )
                logger.info(
                    f" After step {i+1}/{num_steps}: latents mean={current_latents.mean().item():.4f}, "
                    f"std={current_latents.std().item():.4f}"
                )

        return current_latents

    def solve_ddpm(
        self,
        prepared_batch: dict,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        - Build a [B, num_steps+1] linear schedule of [0..1], then map to discrete timesteps
        - For i in [0..num_steps−1], do:
            1) Possibly duplicate x & t_curr if CFG
            2) noise_pred = model(x_scaled, t_input, …)
            3) velocity = − noise_pred / (sqrt(alpha_t)*sqrt(1−alpha_t))
            4) x ← x + step * velocity
        - Finally delta = (x_start − x_end)/(t_start − t_end + EPS).
        """
        model = self.teacher_model
        scheduler = self.teacher_scheduler
        num_steps = self.config["solving_steps"]

        latents = prepared_batch["perflow_latents_start"]
        prompt_embeds = prepared_batch["encoder_hidden_states"]
        negative_prompt_embeds = prepared_batch.get("negative_encoder_hidden_states")
        if negative_prompt_embeds is None:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)

        pooled_prompt_embeds = prepared_batch.get("add_text_embeds", None)
        added_cond_kwargs = prepared_batch.get("added_cond_kwargs", None)
        prompt_attention_mask = prepared_batch.get("encoder_attention_mask", None)

        bsz = latents.shape[0]
        device = latents.device

        do_cfg = guidance_scale > 1.0
        if do_cfg:
            # Duplicate conditionals along batch dimension
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            if pooled_prompt_embeds is not None:
                pooled_prompt_embeds = torch.cat(
                    [torch.zeros_like(pooled_prompt_embeds), pooled_prompt_embeds],
                    dim=0,
                )
            if prompt_attention_mask is not None:
                prompt_attention_mask = torch.cat(
                    [prompt_attention_mask, prompt_attention_mask], dim=0
                )

        # 1) Build [B, num_steps+1] schedule in [0..1], then turn to discrete timesteps
        t_vec = torch.linspace(
            0.0, 1.0, num_steps + 1, device=device, dtype=latents.dtype
        )
        t_schedule = (
            t_start[:, None] * (1.0 - t_vec) + t_end[:, None] * t_vec
        )  # [B, S+1]
        timesteps = (
            ((scheduler.config.num_train_timesteps - 1) * (1.0 - t_schedule))
            .round()
            .long()
        )  # [B, S+1]

        x = latents.clone()
        for i in range(num_steps):
            t_curr = timesteps[:, i]
            t_next = timesteps[:, i + 1]

            # step = (t_curr - t_next) / num_train_timesteps
            step = (t_curr - t_next).float().reshape(
                bsz, *([1] * (latents.ndim - 1))
            ) / scheduler.config.num_train_timesteps

            t_input = t_curr
            if do_cfg:
                # Duplicate x and t_input for CFG
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
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                x = x[:bsz]  # restore

            alpha_t = scheduler.alphas_cumprod[t_curr].reshape(bsz, 1, 1, 1).to(x.dtype)
            sigma_t = (1.0 - alpha_t).sqrt()
            velocity = -noise_pred / (alpha_t.sqrt() * sigma_t)

            # Euler update
            x = x[:bsz] + step * velocity

        # Final delta:
        denom = (t_start - t_end).reshape(bsz, *([1] * (latents.ndim - 1)))
        delta = (latents - x) / (denom + EPS)
        return delta.detach()

    def compute_distill_loss(
        self, prepared_batch: dict, model_output: dict, original_loss: torch.Tensor
    ):
        """
        Exactly MSE between student model_pred and perflow_targets, plus optional cosine or norm term.
        """
        model_pred = model_output["model_prediction"]
        targets = prepared_batch["perflow_targets"]

        loss = F.mse_loss(model_pred.float(), targets.float(), reduction="none")

        if self.config.get("debug_cosine", False):
            cosine = F.cosine_similarity(model_pred.flatten(), targets.flatten(), dim=0)
            logger.info(f"Velocity cosine sim (student vs target): {cosine.item():.6f}")

        vel_norm_wt = self.config.get("velocity_norm_weight", 0.0)
        if vel_norm_wt > 0:
            vnorm = model_pred.norm(dim=1).mean()
            loss = loss + vel_norm_wt * vnorm
            logger.info(f"Velocity norm penalty: {vnorm.item():.6f}")

        loss_mean = loss.mean().item()
        loss_max = loss.max().item()
        loss_min = loss.min().item()
        logger.info(
            f"Loss stats → mean: {loss_mean:.6f}, max: {loss_max:.6f}, min: {loss_min:.6f}"
        )

        return loss.mean(), {
            "perflow_loss": loss_mean,
            "perflow_loss_max": loss_max,
            "perflow_loss_min": loss_min,
        }

    def prepare_batch(self, prepared_batch: dict, model, state) -> dict:
        """
        1) Sample B random timepoints from sigmoid schedule of length 1000
        2) Find (t_start, t_end) via Time_Windows.lookup_window
        3) Compute x_end via solve_ddpm or solve_flow
        4) Form perflow_targets = (x_start - x_end)/(t_start - t_end + EPS)
        5) Pick a random interior (u) for student input: latents_t, t_mid
        """
        B = prepared_batch["latents"].shape[0]
        device = prepared_batch["latents"].device

        with torch.no_grad():
            # 1) Sample timepoints from sigmoid_schedule
            ts_all = sigmoid_schedule(1000)
            timepoints = ts_all[torch.randint(0, len(ts_all), (B,), device=device)]

            # Optional discrete quantization
            d = self.config["discrete_timesteps"]
            if d != -1:
                timepoints = (timepoints * d).floor() / d

            # 2) Find t_start, t_end for each timepoint
            t_start, t_end = self.time_windows.lookup_window(timepoints)

            prepared_batch.update(
                {
                    "timesteps": to_model_timestep(timepoints),
                    "perflow_timepoints": timepoints,
                    "perflow_t_start": t_start,
                    "perflow_t_end": t_end,
                    "perflow_latents_start": prepared_batch["noisy_latents"].clone(),
                }
            )

            # 3) Solve teacher to get x_end
            guidance_scale = (
                self.config["cfg_value"] if self.config["cfg_transfer"] else 1.0
            )

            if self.teacher_model.config.prediction_type in {"epsilon", "v_prediction"}:
                latents_end = self.solve_ddpm(
                    prepared_batch, t_start, t_end, guidance_scale
                )
            else:
                latents_end = self.solve_flow(
                    prepared_batch, t_start, t_end, guidance_scale
                )

            prepared_batch["perflow_latents_end"] = latents_end

            # 4) Compute velocity targets
            num = prepared_batch["perflow_latents_start"] - latents_end
            denom = (t_start - t_end).reshape(
                B, *([1] * (prepared_batch["perflow_latents_start"].ndim - 1))
            ) + EPS
            targets = num / denom
            prepared_batch["perflow_targets"] = targets

            # 5) Randomly pick interior (u) for student input
            u = torch.rand_like(t_start)  # [B]
            latents_t = prepared_batch["perflow_latents_start"] + u[
                :, None, None, None
            ] * (
                latents_end - prepared_batch["perflow_latents_start"]
            )  # [B, C, H, W]
            t_mid = t_start * (1.0 - u) + t_end * u  # [B]

            prepared_batch.update(
                {
                    "perflow_latents_t": latents_t,
                    "noisy_latents_original": prepared_batch["noisy_latents"].clone(),
                    "noisy_latents": latents_t,
                    "timesteps": to_model_timestep(t_mid),
                }
            )

        return prepared_batch

    def get_scheduler(self):
        """
        Return PeRFlowScheduler, initialized with number of windows = self.config["windows"].
        """
        from helpers.training.custom_schedule import PeRFlowScheduler

        return PeRFlowScheduler(
            num_time_windows=self.config["windows"],
            num_train_timesteps=self.teacher_scheduler.num_train_timesteps,
            prediction_type=self.teacher_model.PREDICTION_TYPE.value,
        )

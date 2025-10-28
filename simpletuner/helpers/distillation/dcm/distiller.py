# helpers/distillation/dcm/distiller.py
import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file, save_file

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.distillation.common import DistillationBase
from simpletuner.helpers.distillation.dcm.discriminator.wan import wan_forward, wan_forward_origin
from simpletuner.helpers.distillation.dcm.loss import gan_d_loss, gan_g_loss
from simpletuner.helpers.distillation.dcm.solver import EulerSolver, InferencePCMFMScheduler, extract_into_tensor
from simpletuner.helpers.distillation.registry import DistillationRegistry

logger = logging.getLogger(__name__)

EPS = 1e-7  # numerical guard

DCM_SAFETENSORS_DEFAULT_FILENAME = "discriminator.safetensors"
DCM_OPTIMIZER_DEFAULT_FILENAME = "discriminator_optim.pt"


class DCMDistiller(DistillationBase):
    """
    Dual-Expert Composite-Model distillation.

    ── semantic  : plain flow-matching MSE
    ── fine      : semantic loss  +  adversarial generator loss
                   (the discriminator is still updated outside the distiller)
    """

    # --------------------------------------------------------------------- #
    # initialization
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        teacher_model,
        student_model=None,
        *,
        noise_scheduler: FlowMatchEulerDiscreteScheduler,
        config: Optional[Dict[str, Any]] = None,
    ):
        default = {
            # Distillation mode: "semantic" (plain flow-matching MSE) or "fine" (semantic + adversarial loss)
            "mode": "semantic",
            # Number of parent timesteps in the normal inference schedule (Euler steps)
            "euler_timesteps": 50,
            # Shift parameter for DCM scheduler
            "dcm_shift": 17.0,
            # Number of multiphase steps for Euler-style prediction
            "multiphase": 4,
            # Classifier-free guidance weight for distillation
            "distill_cfg": 5.0,
            # Weight for adversarial (GAN) loss
            "adv_weight": 0.1,
            # Discriminator model family: "wan" or "hunyuan" must be provided
            "model_family": None,
            # Stride for discriminator head
            "discriminator_head_stride": 2,
            # Total number of layers in the discriminator
            "discriminator_total_layers": 30,
        }
        if config:
            default.update(config)

        super().__init__(teacher_model, student_model, default)

        if not self.is_flow_matching:
            raise ValueError("DCM requires a flow-matching teacher.")

        # stash the scheduler
        self.noise_sched = noise_scheduler

        # create the EulerSolver
        sigmas = noise_scheduler.sigmas.cpu().numpy()[::-1]
        self.solver = EulerSolver(
            sigmas,
            timesteps=noise_scheduler.config.num_train_timesteps,
            euler_timesteps=self.config["euler_timesteps"],
        )
        self.solver.training_mode = self.config["mode"]
        self.solver.to(self.teacher_model.get_trained_component().device)

        # build discriminator only if we’re in “fine” mode
        self.discriminator = None
        if self.config["mode"] == "fine":
            fam = self.config["model_family"]
            stride = self.config["discriminator_head_stride"]
            layers = self.config["discriminator_total_layers"]
            if fam == "wan":
                from simpletuner.helpers.distillation.dcm.discriminator.wan import Discriminator as D

                # Overload student model forward
                self.student_model.get_trained_component().wan_forward_origin = wan_forward_origin.__get__(
                    self.student_model.get_trained_component(),
                    type(self.student_model.get_trained_component()),
                )
                self.student_model.get_trained_component().forward = wan_forward.__get__(
                    self.student_model.get_trained_component(),
                    type(self.student_model.get_trained_component()),
                )
            else:
                raise NotImplementedError(f"Discriminator model family '{fam}' is not implemented. ")
            self.discriminator = D(stride, total_layers=layers)
            # Simple default D-optimiser (can be overridden later)
            self.disc_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=1e-5,
                betas=(0.0, 0.999),
                weight_decay=0.0,
            )

    def prepare_batch(self, batch: Dict[str, Any], *_):
        """
        Everything here is lifted 1-for-1 from `distill_one_step(_adv)` in the
        research scripts (but without gradient work).

        We stash **only** the tensors needed later by `compute_distill_loss`.
        """
        latents = batch["latents"]  # [B, C, T, H, W] or [B, C, H, W]
        B, device = latents.shape[0], latents.device

        # -------------------------------------------------------- #
        # 1.  pick an integer solver index  0 … solver.euler_steps-1
        # -------------------------------------------------------- #
        idx = torch.randint(0, len(self.solver.euler_timesteps), (B,), device=device, dtype=torch.long)
        sigma = extract_into_tensor(self.solver.sigmas, idx, latents.shape)
        sigma_prev = extract_into_tensor(self.solver.sigmas_prev, idx, latents.shape)
        timesteps = (sigma * self.noise_sched.config.num_train_timesteps).view(-1)

        # -------------------------------------------------------- #
        # 2.  forward-process (add noise) to obtain student input  x̃
        # -------------------------------------------------------- #
        noise = torch.randn_like(latents)
        noisy_latents = sigma * noise + (1.0 - sigma) * latents
        batch["noisy_latents"] = noisy_latents
        batch["timesteps"] = timesteps  # will be sent to the student

        # -------------------------------------------------------- #
        # 3.  teacher guidance  →  Euler step  →  target velocity
        # -------------------------------------------------------- #
        emb = batch["encoder_hidden_states"]
        uncond_emb = batch.get("negative_encoder_hidden_states")
        if uncond_emb is None:
            uncond_emb = torch.zeros_like(emb)

        cfg_w = self.config["distill_cfg"]
        self.toggle_adapter(enable=False)  # disable LoRA adapter if any
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            cond_out = self.teacher_model.get_trained_component()(noisy_latents, timesteps, emb, return_dict=False)[
                0
            ].float()
            uncond_out = self.teacher_model.get_trained_component()(
                noisy_latents,
                timesteps,
                uncond_emb,
                return_dict=False,
            )[0].float()
            teacher_out = cond_out + cfg_w * (cond_out - uncond_out)

        x_prev = self.solver.euler_step(noisy_latents, teacher_out, idx)

        #  target prediction at x_prev (multi-phase)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            target_pred = self.teacher_model.get_trained_component()(
                x_prev.float(),  # x_{i+1}
                (sigma_prev * self.noise_sched.config.num_train_timesteps).view(-1),
                emb,
                return_dict=False,
            )[0]

        target, end_idx = self.solver.euler_style_multiphase_pred(
            x_prev,
            target_pred,
            idx,
            multiphase=self.config["multiphase"],
            is_target=True,
        )

        batch["dcm_targets"] = target  # ground-truth velocity   (generator MSE)
        self.toggle_adapter(enable=True)  # re-enable student LoRA (if any)

        # ------------------------------------------------------------------ #
        # 4.  book-keeping for GAN generator loss (fine mode only)
        # ------------------------------------------------------------------ #
        if self.config["mode"] == "fine":
            # choose a second σ farther along the trajectory for GAN realism check
            adv_idx = torch.clamp(end_idx + 35, max=len(self.solver.sigmas) - 1)
            sigma_adv = extract_into_tensor(self.solver.sigmas_prev, adv_idx, latents.shape)
            batch.update(
                dict(
                    dcm_sigma=sigma,
                    dcm_sigma_prev=sigma_prev,
                    dcm_sigma_adv=sigma_adv,
                    dcm_x_prev=x_prev,  # teacher “real” sample at that stage
                    dcm_timesteps_adv=(sigma_adv * self.noise_sched.config.num_train_timesteps).view(-1),
                )
            )

        return batch

    def compute_distill_loss(
        self,
        prepared_batch: Dict[str, Any],
        model_output: Dict[str, Any],
        _original_loss: torch.Tensor,
    ):
        pred = model_output["model_prediction"]
        target = prepared_batch["dcm_targets"]

        loss = F.mse_loss(pred.float(), target.float())

        logs = {
            "mse": loss.item(),
            "total": loss.item(),
        }

        return loss, logs

    def get_scheduler(self, *_):
        return InferencePCMFMScheduler(
            num_train_timesteps=self.noise_sched.config.num_train_timesteps,
            shift=self.config["dcm_shift"],
            pcm_timesteps=self.config["euler_timesteps"],
        )

    def generator_loss_step(
        self,
        prepared_batch: Dict[str, Any],
        model_output: Dict[str, Any],
        current_loss: torch.Tensor,
    ):
        if self.config["mode"] != "fine":
            return current_loss, {}  # semantic mode – nothing to add

        pred = model_output["model_prediction"]
        target = prepared_batch["dcm_targets"]
        sigma = prepared_batch["dcm_sigma"]
        sigma_prev = prepared_batch["dcm_sigma_prev"]
        sigma_adv = prepared_batch["dcm_sigma_adv"]

        adv_noise = torch.randn_like(pred)
        real_adv = ((1 - sigma_adv) * target + (sigma_adv - sigma_prev) * adv_noise) / (1 - sigma_prev + EPS)
        fake_adv = ((1 - sigma_adv) * pred + (sigma_adv - sigma_prev) * adv_noise) / (1 - sigma_prev + EPS)

        # Freeze D for generator update
        self.discriminator.requires_grad_(False)
        gan_g = gan_g_loss(
            self.discriminator,
            self.student_model.get_trained_component(),  # <-- gradient flows into student
            fake_adv,
            real_adv,
            prepared_batch["dcm_timesteps_adv"],
            prepared_batch["encoder_hidden_states"],
            prepared_batch.get("encoder_attention_mask", None),
            weight=1.0,
            discriminator_head_stride=self.config["discriminator_head_stride"],
        )

        new_loss = current_loss + self.config["adv_weight"] * gan_g
        logs = {
            "gan_g": gan_g.detach().item(),
            "total": new_loss.detach().item(),
        }
        return new_loss, logs

    def discriminator_step(self, prepared_batch: Dict[str, Any]):
        if self.config["mode"] != "fine":
            return

        if getattr(self, "disc_optimizer", None) is None:
            return  # safety: no optimiser attached

        pred = prepared_batch["dcm_fake_pred"] if "dcm_fake_pred" in prepared_batch else None
        if pred is None:
            # recompute fake with Autocast-off to save VRAM; minimal overhead
            pred = self.student_model.get_trained_component()(
                prepared_batch["noisy_latents"],
                prepared_batch["timesteps"],
                prepared_batch["encoder_hidden_states"],
                return_dict=False,
            )[0]

        target = prepared_batch["dcm_targets"]
        sigma = prepared_batch["dcm_sigma"]
        sigma_prev = prepared_batch["dcm_sigma_prev"]
        sigma_adv = prepared_batch["dcm_sigma_adv"]

        adv_noise = torch.randn_like(pred)
        real_adv = ((1 - sigma_adv) * target + (sigma_adv - sigma_prev) * adv_noise) / (1 - sigma_prev + EPS)
        fake_adv = ((1 - sigma_adv) * pred.detach() + (sigma_adv - sigma_prev) * adv_noise) / (1 - sigma_prev + EPS)

        # Enable discriminator gradients for generator update
        self.discriminator.requires_grad_(True)
        d_loss = gan_d_loss(
            self.discriminator,
            self.teacher_model.get_trained_component(),  # used only for feature extraction, no grads
            fake_adv,
            real_adv,
            prepared_batch["dcm_timesteps_adv"],
            prepared_batch["encoder_hidden_states"],
            prepared_batch.get("encoder_attention_mask", None),
            weight=1.0,
            discriminator_head_stride=self.config["discriminator_head_stride"],
        )

        self.disc_optimizer.zero_grad(set_to_none=True)
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
        self.disc_optimizer.step()

        # optional: log via wandb if desired
        if hasattr(self, "logger"):
            self.logger.debug(f"D-loss {d_loss.item():.4f}")

    def on_save_checkpoint(self, step: int, ckpt_dir: str):
        if self.discriminator is None or torch.distributed.get_rank() != 0:
            return

        # -------- model weights  (tensor-only) ---------------
        weight_path = os.path.join(ckpt_dir, DCM_SAFETENSORS_DEFAULT_FILENAME)
        tensor_dict = {
            k: v.detach().cpu() for k, v in self.discriminator.state_dict().items()  # safetensors requires CPU tensors
        }
        save_file(tensor_dict, weight_path)

        # -------- optimizer state  (pickle OK) ---------------
        opt_path = os.path.join(ckpt_dir, DCM_OPTIMIZER_DEFAULT_FILENAME)
        torch.save(
            {
                "step": step,
                "state": self.disc_optimizer.state_dict(),
            },
            opt_path,
        )

    def on_load_checkpoint(self, ckpt_dir: str):
        if self.discriminator is None:
            return

        weight_path = os.path.join(ckpt_dir, DCM_SAFETENSORS_DEFAULT_FILENAME)
        if not os.path.exists(weight_path):
            return  # no discriminator checkpoint found

        # -------- load weights --------------------------------
        tensor_dict = load_file(weight_path, device="cpu")
        self.discriminator.load_state_dict(tensor_dict, strict=True)
        self.discriminator.to(self.teacher_model.get_trained_component().device, non_blocking=True)

        # -------- load optimizer ------------------------------
        opt_path = os.path.join(ckpt_dir, DCM_OPTIMIZER_DEFAULT_FILENAME)
        if os.path.exists(opt_path):
            payload = torch.load(opt_path, map_location={"cuda:0": f"cuda:{torch.cuda.current_device()}"})
            self.disc_optimizer.load_state_dict(payload["state"])


DistillationRegistry.register(
    "dcm",
    DCMDistiller,
    requires_distillation_cache=False,
    data_requirements=[[DatasetType.IMAGE, DatasetType.VIDEO]],
)

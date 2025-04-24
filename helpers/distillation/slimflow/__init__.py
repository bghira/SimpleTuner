import os, logging, random
import torch
import torch.nn.functional as F
import numpy as np
from diffusers import DiffusionPipeline
from helpers.distillation.common import DistillationBase


class SlimFlowDistiller(DistillationBase):
    """Implementation of SlimFlow distillation."""

    def __init__(self, teacher_model, student_model, config=None):
        # SlimFlow specific config defaults
        slimflow_config = {
            "annealing_schedule": "linear",  # Schedule for annealing parameter
            "beta_steps": 50000,  # Number of steps for annealing
            "use_horizontal_flip": True,  # Whether to use horizontal flip for data pairs
            "distill_loss_type": "lpips",  # Loss type for distillation ('l2' or 'lpips')
            "train_mode": "annealing_reflow",  # Current training mode
        }

        # Update with user config
        if config:
            slimflow_config.update(config)

        super().__init__(teacher_model, student_model, slimflow_config)

        # Initialize state for annealing
        self.current_beta = (
            1.0 if self.config["train_mode"] == "annealing_reflow" else 0.0
        )

    def _compute_beta(self, step):
        """Compute annealing parameter beta based on current step."""
        if self.config["train_mode"] != "annealing_reflow":
            return 0.0

        beta_steps = self.config["beta_steps"]
        schedule = self.config["annealing_schedule"]

        if schedule == "linear":
            return max(0, 1 - min(1, step / (6 * beta_steps)))
        elif schedule == "cosine":
            return (
                1
                + torch.cos(
                    torch.tensor(min(1, step / (2 * beta_steps)) * torch.pi)
                ).item()
            ) / 2
        elif schedule == "exp":
            return torch.exp(torch.tensor(-step / beta_steps)).item()
        else:
            raise ValueError(f"Unknown annealing schedule: {schedule}")

    def pre_training_step(self, model, step):
        """Update beta parameter before each training step."""
        self.current_beta = self._compute_beta(step)

    def prepare_batch(self, batch, model, state):
        """Prepare batch with SlimFlow-specific processing."""
        # First prepare the batch using the student model's normal method
        prepared_batch = model.prepare_batch(batch, state)

        if self.config["train_mode"] == "annealing_reflow":
            # For annealing reflow, we need to interpolate between random pairs and teacher pairs
            with torch.no_grad():
                # Get random noise sample x1'
                bsz = prepared_batch["latents"].shape[0]
                device = prepared_batch["latents"].device
                x1_prime = torch.randn_like(prepared_batch["latents"])

                # Create interpolated noise for annealing reflow
                # x1_beta = sqrt(1-β²)·x1 + β·x1'
                prepared_batch["x1_beta"] = (
                    torch.sqrt(1 - self.current_beta**2) * prepared_batch["latents"]
                    + self.current_beta * x1_prime
                )

                # For teacher model, we need to compute the target
                if (
                    self.current_beta < 1.0
                ):  # Only compute teacher target if we're using it
                    with self.teacher_model.get_trained_component().eval():
                        # Use the teacher model to predict the clean sample
                        teacher_output = self.teacher_model.model_predict(
                            prepared_batch
                        )
                        prepared_batch["teacher_target"] = teacher_output[
                            "model_prediction"
                        ]

        elif self.config["train_mode"] == "flow_guided_distillation":
            # For flow-guided distillation, we need both teacher predictions and intermediate steps
            with torch.no_grad():
                # Get the teacher's prediction
                with self.teacher_model.get_trained_component().eval():
                    teacher_output = self.teacher_model.model_predict(prepared_batch)
                    prepared_batch["teacher_target"] = teacher_output[
                        "model_prediction"
                    ]

                # Sample a timestep for 2-step simulation
                bsz = prepared_batch["latents"].shape[0]
                device = prepared_batch["latents"].device
                t = (
                    torch.rand(bsz, device=device)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
                prepared_batch["intermediate_t"] = t

        return prepared_batch

    def compute_distill_loss(self, prepared_batch, model_output, original_loss):
        """Compute SlimFlow-specific distillation loss."""
        loss = original_loss
        logs = {}

        if self.config["train_mode"] == "annealing_reflow":
            # For annealing reflow, interpolate between original loss and reflow loss
            if self.current_beta < 1.0 and "teacher_target" in prepared_batch:
                target = prepared_batch["teacher_target"]
                model_pred = model_output["model_prediction"]

                # Compute reflow loss
                reflow_loss = F.mse_loss(model_pred.float(), target.float())

                # Interpolate between original and reflow loss based on beta
                interpolated_loss = (
                    self.current_beta * original_loss
                    + (1 - self.current_beta) * reflow_loss
                )
                loss = interpolated_loss
                logs["beta"] = self.current_beta
                logs["reflow_loss"] = reflow_loss.item()

        elif self.config["train_mode"] == "flow_guided_distillation":
            # For flow-guided distillation, we use a combination of direct distillation and 2-step loss

            # 1. Direct distillation from teacher prediction
            if "teacher_target" in prepared_batch:
                target = prepared_batch["teacher_target"]
                model_pred = model_output["model_prediction"]

                if self.config["distill_loss_type"] == "l2":
                    distill_loss = F.mse_loss(model_pred.float(), target.float())
                else:  # 'lpips'
                    # In a real implementation, you would use a proper LPIPS loss
                    # For simplicity, we use MSE as a placeholder
                    distill_loss = F.mse_loss(model_pred.float(), target.float())

                loss = distill_loss
                logs["distill_loss"] = distill_loss.item()

            # 2. Two-step regularization
            if "intermediate_t" in prepared_batch:
                t = prepared_batch["intermediate_t"]
                x1 = prepared_batch["latents"]

                # Get the one-step prediction
                one_step_pred = model_output["model_prediction"]

                # Compute intermediate point x_t
                x_t = (1 - t) * one_step_pred + t * x1

                # Compute the velocity at the intermediate point
                with torch.no_grad():
                    intermediate_input = prepared_batch.copy()
                    intermediate_input["latents"] = x_t

                    # Use the teacher model for the intermediate prediction
                    teacher_intermediate = self.teacher_model.model_predict(
                        intermediate_input
                    )
                    v_xt = teacher_intermediate["model_prediction"]

                # Compute two-step prediction
                two_step_pred = x1 - (1 - t) * model_pred - t * v_xt

                # Compare with one-step prediction
                two_step_loss

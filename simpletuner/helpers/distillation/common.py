import copy
import logging
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn.functional as F


class DistillationBase:
    """Base class for model distillation techniques."""

    def __init__(
        self,
        teacher_model,
        student_model=None,  # Optional - can be None if same as teacher
        config=None,
    ):
        """
        Initialize distillation with teacher and student models.

        Args:
            teacher_model: Instance of ImageModelFoundation (teacher model)
            student_model: Instance of ImageModelFoundation (student model).
                           If None, assumes the teacher model will be used with adapters toggled
            config: Configuration dictionary for distillation-specific parameters
        """
        self.teacher_model = teacher_model
        self.student_model = student_model or teacher_model  # Use teacher if student not provided

        # Flag to check if we're using the same model with adapters
        self.low_rank_distillation = student_model is None

        # Default configuration for distillation-specific parameters
        default_config = {
            "distill_method": "standard",
            "teacher_weight": 1.0,
            "temperature": 1.0,
        }

        # Update default config with user-provided config
        self.config = default_config
        if config:
            self.config.update(config)

        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # Set up scheduler configurations
        self._init_scheduler_configs()

    def _init_scheduler_configs(self):
        """Initialize scheduler configurations based on teacher model type."""
        # Detect model type (flow matching vs. DDPM)
        from simpletuner.helpers.models.common import PredictionTypes

        self.is_flow_matching = self.teacher_model.PREDICTION_TYPE is PredictionTypes.FLOW_MATCHING

        # Store scheduler configurations for later use
        if hasattr(self.teacher_model, "noise_schedule"):
            self.teacher_scheduler = self.teacher_model.noise_schedule
            self.teacher_scheduler_config = self.teacher_scheduler.config
        else:
            self.teacher_scheduler = None
            self.teacher_scheduler_config = None

        # Store custom schedulers needed for specific distillation methods
        self.custom_schedulers = {}

    def get_scheduler(self, scheduler_name: str = None):
        """A child class can override this to provide a custom scheduler."""
        self.logger.warning("No distillation scheduler provided. Using default.")
        return None

    def toggle_adapter(self, enable=False):
        """
        Toggle the adapter on/off when using the same model for teacher and student.

        Args:
            enable: Whether to enable the adapter
        """
        if not self.low_rank_distillation:
            return

        if self.teacher_model.config.lora_type.lower() == "lycoris":
            # Handle LyCORIS adapter
            lycoris_wrapped_network = getattr(self.teacher_model.accelerator, "_lycoris_wrapped_network", None)
            if lycoris_wrapped_network:
                lycoris_wrapped_network.set_multiplier(1.0 if enable else 0.0)
        else:
            # Handle standard LoRA
            if enable:
                self.teacher_model.get_trained_component().enable_lora()
            else:
                self.teacher_model.get_trained_component().disable_lora()

    def prepare_batch(self, batch, model, state):
        """Process a batch for distillation training."""
        return batch

    def compute_distill_loss(self, prepared_batch, model_output, original_loss):
        """Compute the distillation loss to be combined with the original loss."""
        return original_loss, {}

    def pre_training_step(self, model, step):
        """Perform any setup needed before each training step."""
        pass

    def post_training_step(self, model, step):
        """Perform any cleanup or logging after each training step."""
        pass

    def generator_loss_step(
        self,
        prepared_batch: Dict[str, Any],
        model_output: Dict[str, Any],
        current_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Optionally add a generator-side adversarial term.

        Returns
        -------
        current_loss : torch.Tensor
            The (possibly) updated loss tensor that will be back-prop’d.
        logs : Dict[str, float]
            Extra items to merge into wandb / tensorboard logging.
        """
        return current_loss, {}

    def discriminator_step(
        self,
        prepared_batch: Dict[str, Any],
        **kwargs,
    ):
        """
        Optionally perform a discriminator update *after* the student
        optimizer.step().  Default is a no-op.
        """
        pass

    def on_load_checkpoint(self, ckpt_dir: str):
        pass

    def on_save_checkpoint(self, step: int, ckpt_dir: str):
        pass

    def on_epoch_end(self, epoch: int):
        pass

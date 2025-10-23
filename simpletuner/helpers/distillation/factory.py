# helpers/distillation/factory.py
import logging
from enum import Enum
from typing import Any, Dict, Optional, Union

from simpletuner.helpers.distillation.common import DistillationBase

logger = logging.getLogger(__name__)


class DistillationMethod(Enum):
    """Supported distillation methods."""

    DCM = "dcm"
    DMD = "dmd"
    LCM = "lcm"

    @classmethod
    def from_string(cls, method: str):
        """Convert string to enum, case-insensitive."""
        if method is None:
            return None
        method_upper = method.upper()
        for member in cls:
            if member.value.upper() == method_upper:
                return member
        raise ValueError(f"Unknown distillation method: {method}")


class DistillerFactory:
    """Factory class for creating distillers based on configuration."""

    @staticmethod
    def create_distiller(
        method: Union[str, DistillationMethod],
        teacher_model,
        noise_scheduler,
        config: Dict[str, Any],
        model_type: str = "lora",
        model_family: Optional[str] = None,
        prediction_type: Optional[str] = None,
        student_model=None,
    ) -> Optional[DistillationBase]:
        """
        Create a distiller instance based on the specified method.

        Args:
            method: Distillation method (dcm, dmd, lcm, etc.)
            teacher_model: The teacher model instance
            noise_scheduler: The noise scheduler instance
            config: Configuration dict from trainer
            model_type: Type of model training ("lora" or "full")
            model_family: Model family (e.g., "wan", "hunyuan")
            prediction_type: Model prediction type (e.g., "flow_matching", "epsilon")
            student_model: Optional separate student model (for full model distillation)

        Returns:
            Configured distiller instance or None if method is None
        """
        if isinstance(method, str):
            method = DistillationMethod.from_string(method)

        if method is None:
            return None

        distill_config = {}
        if config.get("distillation_config") is not None:
            # Check for method-specific config first
            if method.value in config["distillation_config"]:
                distill_config = config["distillation_config"][method.value]
            else:
                # Fall back to general distillation config
                distill_config = config["distillation_config"]

        if method == DistillationMethod.DCM:
            return DistillerFactory._create_dcm_distiller(
                teacher_model=teacher_model,
                noise_scheduler=noise_scheduler,
                distill_config=distill_config,
                model_type=model_type,
                model_family=model_family,
                prediction_type=prediction_type,
                student_model=student_model,
            )
        elif method == DistillationMethod.DMD:
            return DistillerFactory._create_dmd_distiller(
                teacher_model=teacher_model,
                noise_scheduler=noise_scheduler,
                distill_config=distill_config,
                model_type=model_type,
                model_family=model_family,
                prediction_type=prediction_type,
                student_model=student_model,
            )
        elif method == DistillationMethod.LCM:
            return DistillerFactory._create_lcm_distiller(
                teacher_model=teacher_model,
                noise_scheduler=noise_scheduler,
                distill_config=distill_config,
                model_type=model_type,
                prediction_type=prediction_type,
                student_model=student_model,
            )
        else:
            raise ValueError(f"Unsupported distillation method: {method}")

    @staticmethod
    def _create_dmd_distiller(
        teacher_model,
        noise_scheduler,
        distill_config: Dict[str, Any],
        model_type: str,
        model_family: Optional[str],
        prediction_type: Optional[str],
        student_model=None,
    ) -> DistillationBase:
        try:
            from simpletuner.helpers.distillation.dmd.distiller import DMDDistiller
        except ImportError:
            raise ImportError("DMD distiller not found. Please ensure simpletuner.helpers.distillation.dmd is available.")

        dmd_config = {
            "model_family": model_family or "wan",
            "dmd_denoising_steps": "1000,757,522",  # 3-step default
            "min_timestep_ratio": 0.02,
            "max_timestep_ratio": 0.98,
            "generator_update_interval": 5,
            "real_score_guidance_scale": 3.0,
            "simulate_generator_forward": False,
            "fake_score_lr": 1e-5,
            "fake_score_lr_scheduler": "cosine_with_min_lr",
            "min_lr_ratio": 0.5,
        }

        # flow-matching models need shift parameter
        if prediction_type and "flow" in prediction_type.lower():
            dmd_config["shift"] = 5.0

        dmd_config.update(distill_config)

        if model_type == "lora":
            return DMDDistiller(
                teacher_model=teacher_model,
                student_model=None,  # Use teacher with adapters
                noise_scheduler=noise_scheduler,
                config=dmd_config,
            )
        elif model_type == "full":
            if student_model is None:
                pass
            return DMDDistiller(
                teacher_model=teacher_model,
                student_model=student_model,
                noise_scheduler=noise_scheduler,
                config=dmd_config,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def _create_dcm_distiller(
        teacher_model,
        noise_scheduler,
        distill_config: Dict[str, Any],
        model_type: str,
        model_family: Optional[str],
        prediction_type: Optional[str],
        student_model=None,
    ) -> DistillationBase:
        try:
            from simpletuner.helpers.distillation.dcm.distiller import DCMDistiller
        except ImportError:
            raise ImportError("DCM distiller not found. Please ensure simpletuner.helpers.distillation.dcm is available.")

        dcm_config = {
            "model_family": model_family,
            "model_type": model_type,
            "loss_type": prediction_type,
            "pred_type": prediction_type,
            "is_regularisation_data": True,  # Default to regularization approach
        }

        dcm_config.update(distill_config)

        # dcm requires model_family for flow-matching setup
        if dcm_config.get("model_family") is None:
            raise ValueError("DCM requires 'model_family' to be specified (e.g., 'wan', 'hunyuan')")

        if model_type == "lora":
            return DCMDistiller(
                teacher_model=teacher_model,
                student_model=None,  # Use teacher with adapters
                noise_scheduler=noise_scheduler,
                config=dcm_config,
            )
        elif model_type == "full":
            if student_model is None:
                raise ValueError("Full model DCM distillation requires a separate student model.")
            pass
            return DCMDistiller(
                teacher_model=teacher_model,
                student_model=student_model,
                noise_scheduler=noise_scheduler,
                config=dcm_config,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def _create_lcm_distiller(
        teacher_model,
        noise_scheduler,
        distill_config: Dict[str, Any],
        model_type: str,
        prediction_type: Optional[str],
        student_model=None,
    ) -> DistillationBase:
        try:
            from simpletuner.helpers.distillation.lcm.distiller import LCMDistiller
        except ImportError:
            raise ImportError("LCM distiller not found. Please ensure simpletuner.helpers.distillation.lcm is available.")

        lcm_config = {
            "num_ddim_timesteps": 50,
            "w_min": 1.0,
            "w_max": 15.0,
            "loss_type": "l2",
            "huber_c": 0.001,
            "timestep_scaling_factor": 10.0,
        }

        # flow-matching models need shift parameter
        if prediction_type and "flow" in prediction_type.lower():
            lcm_config["shift"] = 7.0

        lcm_config.update(distill_config)

        if model_type == "lora":
            return LCMDistiller(
                teacher_model=teacher_model,
                student_model=None,  # Use teacher with adapters
                noise_scheduler=noise_scheduler,
                config=lcm_config,
            )
        elif model_type == "full":
            if student_model is None:
                raise ValueError("Full model LCM distillation requires a separate student model.")
            pass
            return LCMDistiller(
                teacher_model=teacher_model,
                student_model=student_model,
                noise_scheduler=noise_scheduler,
                config=lcm_config,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def init_distillation(trainer_instance):
    """
    Initialize distillation for a trainer instance.
    This is a convenience function that can be used as a drop-in replacement
    for the trainer's init_distillation method.

    Args:
        trainer_instance: The trainer instance with config and model attributes

    Returns:
        The created distiller instance or None
    """
    trainer_instance.distiller = None

    if trainer_instance.config.distillation_method is None:
        return None

    # Get prediction type from model if available
    prediction_type = None
    if hasattr(trainer_instance.model, "PREDICTION_TYPE"):
        prediction_type = trainer_instance.model.PREDICTION_TYPE.value

    trainer_instance.distiller = DistillerFactory.create_distiller(
        method=trainer_instance.config.distillation_method,
        teacher_model=trainer_instance.model,
        noise_scheduler=trainer_instance.noise_scheduler,
        config=vars(trainer_instance.config),
        model_type=trainer_instance.config.model_type,
        model_family=getattr(trainer_instance.config, "model_family", None),
        prediction_type=prediction_type,
        student_model=None,
    )

    return trainer_instance.distiller

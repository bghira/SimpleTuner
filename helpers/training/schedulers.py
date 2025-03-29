import os
from accelerate.logging import get_logger
from helpers.models.common import get_model_config_path, flow_matching_model_families

logger = get_logger(__name__, log_level=os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

target_level = os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
logger.setLevel(target_level)


def load_scheduler_from_args(args):
    flow_matching = False
    if (
        args.model_family == "sd3" and args.flow_matching_loss != "diffusion"
    ) or args.model_family in flow_matching_model_families:
        # Flow-matching models.
        flow_matching = True
        from diffusers import FlowMatchEulerDiscreteScheduler

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            get_model_config_path(
                args.model_family, args.pretrained_model_name_or_path
            ),
            subfolder="scheduler",
            shift=args.flow_schedule_shift,
        )
    else:
        if args.model_family == "legacy":
            args.rescale_betas_zero_snr = True
            args.training_scheduler_timestep_spacing = "trailing"

        from diffusers import DDPMScheduler

        noise_scheduler = DDPMScheduler.from_pretrained(
            get_model_config_path(
                args.model_family, args.pretrained_model_name_or_path
            ),
            subfolder="scheduler",
            rescale_betas_zero_snr=args.rescale_betas_zero_snr,
            timestep_spacing=args.training_scheduler_timestep_spacing,
        )
        args.prediction_type = noise_scheduler.config.prediction_type
        if flow_matching and args.flow_matching_loss == "diffusion":
            logger.warning(
                "Since --flow_matching_loss=diffusion, we will be reparameterising the model to v-prediction diffusion objective. This will break things for a while. Perhaps forever.."
            )

    return args, flow_matching, noise_scheduler

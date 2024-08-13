import os
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level=os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

target_level = os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
logger.setLevel(target_level)

def load_diffusion_model(args, weight_dtype):
    pretrained_load_args = {
        "revision": args.revision,
        "variant": args.variant,
    }
    unet = None
    transformer = None
    
    if args.sd3:
        # Stable Diffusion 3 uses a Diffusion transformer.
        logger.info("Loading Stable Diffusion 3 diffusion transformer..")
        try:
            from diffusers import SD3Transformer2DModel
        except Exception as e:
            logger.error(
                f"Can not load SD3 model class. This release requires the latest version of Diffusers: {e}"
            )
        transformer = SD3Transformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            **pretrained_load_args,
        )
    elif args.flux:
        from diffusers.models import FluxTransformer2DModel

        transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
            **pretrained_load_args,
        )
    elif args.pixart_sigma:
        from diffusers.models import PixArtTransformer2DModel

        transformer = PixArtTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
            **pretrained_load_args,
        )
    elif args.smoldit:
        logger.info("Loading SmolDiT model..")
        if args.validation_noise_scheduler is None:
            args.validation_noise_scheduler = "ddpm"
        transformer_variant = None
        from helpers.models.smoldit import SmolDiT2DModel, SmolDiTConfigurations

        if args.smoldit_config not in SmolDiTConfigurations:
            raise ValueError(
                f"Invalid SmolDiT size configuration: {args.smoldit_config}"
            )

        transformer = SmolDiT2DModel(**SmolDiTConfigurations[args.smoldit_config])
        if "lora" in args.model_type:
            raise ValueError("SmolDiT does not yet support LoRA training.")
    else:
        from diffusers import UNet2DConditionModel

        logger.info("Loading U-net..")
        unet_variant = args.variant
        if (
            args.kolors
            and args.pretrained_model_name_or_path.lower()
            == "kwai-kolors/kolors-diffusers"
        ):
            unet_variant = "fp16"
        pretrained_load_args["variant"] = unet_variant
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", **pretrained_load_args
        )

    return unet, transformer
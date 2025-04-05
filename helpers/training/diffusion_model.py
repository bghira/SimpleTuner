import os
from accelerate.logging import get_logger
from helpers.models.common import get_model_config_path

logger = get_logger(__name__, log_level=os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

target_level = os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
logger.setLevel(target_level)


def determine_subfolder(folder_value: str = None):
    if folder_value is None or str(folder_value).lower() == "none":
        return None
    return str(folder_value)


def load_diffusion_model(args, weight_dtype):
    pretrained_load_args = {
        "revision": args.revision,
        "variant": args.variant,
        "torch_dtype": weight_dtype,
        "use_safetensors": True,
    }
    unet = None
    transformer = None
    pretrained_transformer_path = (
        args.pretrained_transformer_model_name_or_path
        or args.pretrained_model_name_or_path
    )
    if "nf4-bnb" == args.base_model_precision:
        import torch
        from diffusers import BitsAndBytesConfig

        pretrained_load_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=weight_dtype,
        )

    if args.model_family == "ltxvideo":
        # LTXVideo uses a Diffusion transformer.
        logger.info("Loading LTX Video diffusion transformer..")
        try:
            from diffusers import LTXVideoTransformer3DModel
        except Exception as e:
            logger.error(
                f"Can not load LTXVideoTransformer3DModel model class. This release requires the latest version of Diffusers: {e}"
            )
        transformer = LTXVideoTransformer3DModel.from_pretrained(
            args.pretrained_transformer_model_name_or_path
            or args.pretrained_model_name_or_path,
            subfolder=determine_subfolder(args.pretrained_transformer_subfolder),
            **pretrained_load_args,
        )
    elif args.model_family == "pixart_sigma":
        from diffusers.models import PixArtTransformer2DModel

        transformer_load_fn = PixArtTransformer2DModel.from_pretrained
        if pretrained_transformer_path.lower().endswith(".safetensors"):
            # transformer_load_fn = PixArtTransformer2DModel.from_single_file
            raise ValueError("PixArt does not support single file loading.")

        transformer = transformer_load_fn(
            pretrained_transformer_path,
            subfolder=determine_subfolder(args.pretrained_transformer_subfolder),
            **pretrained_load_args,
        )
    else:
        from diffusers import UNet2DConditionModel

        logger.info("Loading U-net..")
        unet_variant = args.variant
        pretrained_load_args["variant"] = unet_variant
        unet_load_fn = UNet2DConditionModel.from_pretrained
        pretrained_unet_path = (
            args.pretrained_unet_model_name_or_path
            or args.pretrained_model_name_or_path
        )
        if pretrained_unet_path.lower().endswith(".safetensors"):
            unet_load_fn = UNet2DConditionModel.from_single_file
        unet = unet_load_fn(
            pretrained_unet_path,
            subfolder=determine_subfolder(args.pretrained_unet_subfolder),
            **pretrained_load_args,
        )
        if (
            args.gradient_checkpointing_interval is not None
            and args.gradient_checkpointing_interval > 0
        ):
            logger.warning(
                "Using experimental gradient checkpointing monkeypatch for a checkpoint interval of {}".format(
                    args.gradient_checkpointing_interval
                )
            )
            # monkey-patch the gradient checkpointing function for pytorch to run every nth call only.
            # definitely one of the more awful things I've ever done while programming, but it's easier than
            # modifying every one of the unet blocks' forward calls in Diffusers to make it work properly.
            from helpers.training.gradient_checkpointing_interval import (
                set_checkpoint_interval,
            )

            set_checkpoint_interval(int(args.gradient_checkpointing_interval))

    if (
        args.gradient_checkpointing_interval is not None
        and args.gradient_checkpointing_interval > 1
    ):
        if transformer is not None and hasattr(
            transformer, "set_gradient_checkpointing_interval"
        ):
            logger.info("Setting gradient checkpointing interval for transformer..")
            transformer.set_gradient_checkpointing_interval(
                int(args.gradient_checkpointing_interval)
            )
        if unet is not None and hasattr(unet, "set_gradient_checkpointing_interval"):
            logger.info("Checking gradient checkpointing interval for U-Net..")
            unet.set_gradient_checkpointing_interval(
                int(args.gradient_checkpointing_interval)
            )
    if args.pretrained_model_name_or_path.endswith(".safetensors"):
        args.pretrained_model_name_or_path = get_model_config_path(
            args.model_family, args.pretrained_model_name_or_path
        )
    return unet, transformer

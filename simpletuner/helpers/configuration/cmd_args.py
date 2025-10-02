import argparse
import json
import logging
import os
import random
import sys
import time
from collections.abc import Mapping
from datetime import timedelta
from typing import Any, Dict, List, Optional

import torch
from accelerate import InitProcessGroupKwargs
from accelerate.utils import ProjectConfiguration

from simpletuner.helpers.configuration.cli_utils import mapping_to_cli_args
from simpletuner.helpers.training.optimizer_param import is_optimizer_deprecated, is_optimizer_grad_fp32
from simpletuner.simpletuner_sdk.server.services.field_registry.types import ConfigField, FieldType, ValidationRuleType

logger = logging.getLogger("ArgsParser")
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")

if torch.cuda.is_available():
    os.environ["NCCL_SOCKET_NTIMEO"] = "2000000"


def print_on_main_thread(message):
    if should_log():
        print(message)


def info_log(message):
    if should_log():
        logger.info(message)


def warning_log(message):
    if should_log():
        logger.warning(message)


def error_log(message):
    if should_log():
        logger.error(message)


_ARG_PARSER_CACHE: Optional[argparse.ArgumentParser] = None

BOOL_TRUE_STRINGS = {"1", "true", "yes", "y", "on"}
BOOL_FALSE_STRINGS = {"0", "false", "no", "n", "off"}


def _parse_bool_flag(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    text_value = str(value).strip().lower()
    if text_value in BOOL_TRUE_STRINGS:
        return True
    if text_value in BOOL_FALSE_STRINGS:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _extract_choice_values(field: ConfigField) -> List[Any]:
    if not field.choices or field.dynamic_choices:
        return []
    values: List[Any] = []
    for choice in field.choices:
        if isinstance(choice, Mapping):
            values.append(choice.get("value"))
        else:
            values.append(choice)
    return values


def _infer_numeric_type(field: ConfigField, choice_values: List[Any]):
    candidates: List[Any] = []
    default = field.default_value
    if default not in (None, "==SUPPRESS=="):
        candidates.append(default)
    candidates.extend(choice_values)
    for rule in field.validation_rules:
        value = getattr(rule, "value", None)
        if value is not None:
            candidates.append(value)
    for candidate in candidates:
        if isinstance(candidate, float) and not float(candidate).is_integer():
            return float
        if isinstance(candidate, str):
            try:
                numeric_value = float(candidate)
            except ValueError:
                continue
            if "." in candidate or "e" in candidate.lower() or not numeric_value.is_integer():
                return float
    return int


def _is_required(field: ConfigField) -> bool:
    return any(rule.rule_type == ValidationRuleType.REQUIRED for rule in field.validation_rules)


def _add_argument_from_field(parser: argparse.ArgumentParser, field: ConfigField) -> None:
    choice_values = _extract_choice_values(field)
    cli_choices = [value for value in choice_values if value is not None]
    help_text = field.help_text or getattr(field, "cmd_args_help", "") or field.tooltip
    kwargs: Dict[str, Any] = {}
    if help_text:
        kwargs["help"] = help_text
    if _is_required(field):
        kwargs["required"] = True
    if field.field_type == FieldType.CHECKBOX:
        default = field.default_value
        if default == "==SUPPRESS==":
            return
        if default is None:
            default_bool = None
        else:
            default_bool = _parse_bool_flag(default)
        kwargs.update(
            {
                "nargs": "?",
                "const": True,
                "type": _parse_bool_flag,
                "default": default_bool,
            }
        )
        parser.add_argument(field.arg_name, **kwargs)
        return
    if cli_choices and not field.dynamic_choices:
        kwargs["choices"] = cli_choices
    default = field.default_value
    if default not in (None, "==SUPPRESS=="):
        kwargs["default"] = default
    if field.field_type == FieldType.NUMBER:
        kwargs["type"] = _infer_numeric_type(field, cli_choices)
    else:
        kwargs.setdefault("type", str)
    parser.add_argument(field.arg_name, **kwargs)


def _populate_parser_from_field_registry(parser: argparse.ArgumentParser) -> None:
    from simpletuner.simpletuner_sdk.server.services.field_registry.registry import field_registry

    seen: set[str] = set()
    for field in field_registry._fields.values():
        arg_name = field.arg_name
        if not arg_name or not arg_name.startswith("--") or arg_name == "--help":
            continue
        if field.default_value == "==SUPPRESS==":
            continue
        if arg_name in seen:
            continue
        seen.add(arg_name)
        _add_argument_from_field(parser, field)


def get_argument_parser():
    global _ARG_PARSER_CACHE
    if _ARG_PARSER_CACHE is not None:
        return _ARG_PARSER_CACHE

    parser = argparse.ArgumentParser(
        description="The following SimpleTuner command-line options are available:",
        exit_on_error=False,
    )
    _populate_parser_from_field_registry(parser)
    _ARG_PARSER_CACHE = parser
    return parser


def get_default_config():
    parser = get_argument_parser()
    default_config = {}
    for action in parser.__dict__["_actions"]:
        if action.dest:
            default_config[action.dest] = action.default

    return default_config


def parse_cmdline_args(input_args=None, exit_on_error: bool = False):
    parser = get_argument_parser()
    args = None
    try:
        args = parser.parse_args(input_args)
    except Exception:  # pragma: no cover - parser handles errors consistently
        logger.error(f"Could not parse input: {input_args}")
        import traceback

        logger.error(traceback.format_exc())

    if args is None and exit_on_error:
        sys.exit(1)

    if args is None:
        return None

    if args.controlnet_custom_config is not None and type(args.controlnet_custom_config) is str:
        if args.controlnet_custom_config.startswith("{"):
            try:
                import ast

                args.controlnet_custom_config = ast.literal_eval(args.controlnet_custom_config)
            except Exception as e:
                logger.error(f"Could not load controlnet_custom_config: {e}")
                raise
    if args.webhook_config is not None and type(args.webhook_config) is str:
        if args.webhook_config.startswith("{"):
            try:
                import ast

                args.webhook_config = ast.literal_eval(args.webhook_config)
            except Exception as e:
                logger.error(f"Could not load webhook_config: {e}")
                raise
        else:
            # try to load from file
            if os.path.isfile(args.webhook_config):
                try:
                    with open(args.webhook_config, "r") as f:
                        import json

                        args.webhook_config = json.load(f)
                except Exception as e:
                    logger.error(f"Could not load webhook_config from file: {e}")
                    raise
            else:
                logger.error(f"Could not find webhook_config file: {args.webhook_config}")

    if args.tread_config is not None and type(args.tread_config) is str:
        if args.tread_config.startswith("{"):
            try:
                import ast

                args.tread_config = ast.literal_eval(args.tread_config)
            except Exception as e:
                logger.error(f"Could not load tread_config: {e}")
                raise

    if args.optimizer == "adam_bfloat16" and args.mixed_precision != "bf16":
        if not torch.backends.mps.is_available():
            logging.error("You cannot use --adam_bfloat16 without --mixed_precision=bf16.")
            sys.exit(1)

    if args.mixed_precision == "fp8" and not torch.cuda.is_available():
        logging.error("You cannot use --mixed_precision=fp8 without a CUDA device. Please use bf16 instead.")
        sys.exit(1)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.seed is not None:
        if args.seed == 0:
            # the current time should be used if value is zero, providing a rolling seed.
            args.seed = int(time.time())
        elif args.seed == -1:
            # more random seed if value is -1, it will be very different on each startup.
            args.seed = int(random.randint(0, 2**30))

    if args.cache_dir is None or args.cache_dir == "":
        args.cache_dir = os.path.join(args.output_dir, "cache")

    if args.maximum_image_size is not None and not args.target_downsample_size:
        raise ValueError("When providing --maximum_image_size, you must also provide a value for --target_downsample_size.")
    if (
        args.maximum_image_size is not None
        and args.resolution_type == "area"
        and args.maximum_image_size > 5
        and not os.environ.get("SIMPLETUNER_MAXIMUM_IMAGE_SIZE_OVERRIDE", False)
    ):
        raise ValueError(
            f"When using --resolution_type=area, --maximum_image_size must be less than 5 megapixels. You may have accidentally entered {args.maximum_image_size} pixels, instead of megapixels."
        )
    elif args.maximum_image_size is not None and args.resolution_type == "pixel" and args.maximum_image_size < 512:
        raise ValueError(
            f"When using --resolution_type=pixel, --maximum_image_size must be at least 512 pixels. You may have accidentally entered {args.maximum_image_size} megapixels, instead of pixels."
        )
    if (
        args.target_downsample_size is not None
        and args.resolution_type == "area"
        and args.target_downsample_size > 5
        and not os.environ.get("SIMPLETUNER_MAXIMUM_IMAGE_SIZE_OVERRIDE", False)
    ):
        raise ValueError(
            f"When using --resolution_type=area, --target_downsample_size must be less than 5 megapixels. You may have accidentally entered {args.target_downsample_size} pixels, instead of megapixels."
        )
    elif args.target_downsample_size is not None and args.resolution_type == "pixel" and args.target_downsample_size < 512:
        raise ValueError(
            f"When using --resolution_type=pixel, --target_downsample_size must be at least 512 pixels. You may have accidentally entered {args.target_downsample_size} megapixels, instead of pixels."
        )

    model_is_bf16 = (
        args.base_model_precision == "no_change" and (args.mixed_precision == "bf16" or torch.backends.mps.is_available())
    ) or (args.base_model_precision != "no_change" and args.base_model_default_dtype == "bf16")
    model_is_quantized = args.base_model_precision != "no_change"
    if model_is_quantized and args.mixed_precision == "fp8" and args.base_model_precision != "fp8-torchao":
        raise ValueError(
            "You cannot use --mixed_precision=fp8 with a quantized base model. Please use bf16 or remove base_model_precision option from your configuration."
        )
    # check optimiser validity
    chosen_optimizer = args.optimizer
    is_optimizer_deprecated(chosen_optimizer)
    from simpletuner.helpers.training.optimizer_param import optimizer_parameters

    optimizer_cls, optimizer_details = optimizer_parameters(chosen_optimizer, args)
    using_bf16_optimizer = optimizer_details.get("default_settings", {}).get("precision") in ["any", "bf16"]
    if using_bf16_optimizer and not model_is_bf16:
        raise ValueError(f"Model is not using bf16 precision, but the optimizer {chosen_optimizer} requires it.")
    if is_optimizer_grad_fp32(args.optimizer):
        warning_log("Using an optimizer that requires fp32 gradients. Training will potentially run more slowly.")
        if args.gradient_precision != "fp32":
            args.gradient_precision = "fp32"
    else:
        if args.gradient_precision == "fp32":
            args.gradient_precision = "unmodified"

    if torch.backends.mps.is_available():
        if args.model_family.lower() not in ["sd3", "flux", "legacy"] and not args.unet_attention_slice:
            warning_log("MPS may benefit from the use of --unet_attention_slice for memory savings at the cost of speed.")
        if args.train_batch_size > 16:
            error_log(
                "An M3 Max 128G will use 12 seconds per step at a batch size of 1 and 65 seconds per step at a batch size of 12."
                " Any higher values will result in NDArray size errors or other unstable training results and crashes."
                "\nPlease reduce the batch size to 12 or lower."
            )
            sys.exit(1)

        if args.quantize_via == "accelerator":
            error_log(
                "MPS does not benefit from models being quantized on the accelerator device. Overriding --quantize_via to 'cpu'."
            )
            args.quantize_via = "cpu"

    if args.max_train_steps is not None and args.max_train_steps > 0 and args.num_train_epochs > 0:
        error_log("When using --max_train_steps (MAX_NUM_STEPS), you must set --num_train_epochs (NUM_EPOCHS) to 0.")
        sys.exit(1)

    if (
        args.pretrained_vae_model_name_or_path is not None
        # currently these are the only models we have using the SDXL VAE.
        and args.model_family not in ["sdxl", "pixart_sigma", "kolors"]
        and "sdxl" in args.pretrained_vae_model_name_or_path
        and "deepfloyd" not in args.model_type
    ):
        warning_log(
            f"The VAE model {args.pretrained_vae_model_name_or_path} is not compatible. Please use a compatible VAE to eliminate this warning. The baked-in VAE will be used, instead."
        )
        args.pretrained_vae_model_name_or_path = None
    if args.pretrained_vae_model_name_or_path == "" or args.pretrained_vae_model_name_or_path == "''":
        args.pretrained_vae_model_name_or_path = None

    if "deepfloyd" not in args.model_type:
        info_log(f"VAE Model: {args.pretrained_vae_model_name_or_path or args.pretrained_model_name_or_path}")
        info_log(f"Default VAE Cache location: {args.cache_dir_vae}")
        info_log(f"Text Cache location: {args.cache_dir_text}")

    elif "deepfloyd" in args.model_type:
        deepfloyd_pixel_alignment = 8
        if args.aspect_bucket_alignment != deepfloyd_pixel_alignment:
            warning_log(
                f"Overriding aspect bucket alignment pixel interval to {deepfloyd_pixel_alignment}px instead of {args.aspect_bucket_alignment}px."
            )
            args.aspect_bucket_alignment = deepfloyd_pixel_alignment

    if "deepfloyd-stage2" in args.model_type and args.resolution < 256:
        warning_log("DeepFloyd Stage II requires a resolution of at least 256. Setting to 256.")
        args.resolution = 256
        args.aspect_bucket_alignment = 64
        args.resolution_type = "pixel"

    validation_resolution_is_float = False
    if "." in str(args.validation_resolution):
        try:
            # this makes handling for int() conversion easier later.
            args.validation_resolution = float(args.validation_resolution)
            validation_resolution_is_float = True
        except ValueError:
            pass
    validation_resolution_is_digit = False
    try:
        int(args.validation_resolution)
        validation_resolution_is_digit = True
    except ValueError:
        pass

    if (
        (validation_resolution_is_digit or validation_resolution_is_float)
        and int(args.validation_resolution) < 128
        and "deepfloyd" not in args.model_type
    ):
        # Convert from megapixels to pixels:
        log_msg = f"It seems that --validation_resolution was given in megapixels ({args.validation_resolution}). Converting to pixel measurement:"
        if int(args.validation_resolution) == 1:
            args.validation_resolution = 1024
        else:
            args.validation_resolution = int(int(args.validation_resolution) * 1e3)
            # Make it divisible by 8:
            args.validation_resolution = int(int(args.validation_resolution) / 8) * 8
        info_log(f"{log_msg} {int(args.validation_resolution)}px")
    if args.timestep_bias_portion < 0.0 or args.timestep_bias_portion > 1.0:
        raise ValueError("Timestep bias portion must be between 0.0 and 1.0.")

    if args.metadata_update_interval < 60:
        raise ValueError("Metadata update interval must be at least 60 seconds.")

    args.vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )

    if args.use_ema and args.ema_cpu_only:
        args.ema_device = "cpu"

    if (args.optimizer_beta1 is not None and args.optimizer_beta2 is None) or (
        args.optimizer_beta1 is None and args.optimizer_beta2 is not None
    ):
        error_log("Both --optimizer_beta1 and --optimizer_beta2 should be provided.")
        sys.exit(1)

    if args.gradient_checkpointing:
        # enable torch compile w/ activation checkpointing :[ slows us down.
        torch._dynamo.config.optimize_ddp = False

    args.logging_dir = os.path.join(args.output_dir, args.logging_dir)
    args.accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.logging_dir)
    # Create the custom configuration
    args.process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))  # 1.5 hours

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if args.disable_tf32:
            warning_log("--disable_tf32 is provided, not enabling. Training will potentially be much slower.")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        else:
            info_log(
                "Enabled NVIDIA TF32 for faster training on Ampere GPUs. Use --disable_tf32 if this causes any problems."
            )

    args.is_quantized = False if (args.base_model_precision == "no_change" or "lora" not in args.model_type) else True
    args.weight_dtype = (
        torch.bfloat16
        if (args.mixed_precision == "bf16" or (args.base_model_default_dtype == "bf16" and args.is_quantized))
        else torch.float16 if args.mixed_precision == "fp16" else torch.float32
    )
    args.disable_accelerator = os.environ.get("SIMPLETUNER_DISABLE_ACCELERATOR", False)

    if "lycoris" == args.lora_type.lower():
        from lycoris import create_lycoris

        if args.lycoris_config is None:
            raise ValueError(
                "--lora_type=lycoris requires you to add a JSON " + "configuration file location with --lycoris_config"
            )
        # is it readable?
        if not os.path.isfile(args.lycoris_config) or not os.access(args.lycoris_config, os.R_OK):
            raise ValueError(f"Could not find the JSON configuration file at '{args.lycoris_config}'")
        import json

        with open(args.lycoris_config, "r") as f:
            lycoris_config = json.load(f)
        assert lycoris_config is not None, "lycoris_config could not be parsed as JSON"
        assert "algo" in lycoris_config, "lycoris_config JSON must contain algo key"
        assert "multiplier" in lycoris_config, "lycoris_config JSON must contain multiplier key"
        if "full_matrix" not in lycoris_config or lycoris_config.get("full_matrix") is not True:
            assert (
                "linear_dim" in lycoris_config
            ), "lycoris_config JSON must contain linear_dim key if full_matrix is not set."
        assert "linear_alpha" in lycoris_config, "lycoris_config JSON must contain linear_alpha key"

    elif "standard" == args.lora_type.lower():
        if hasattr(args, "lora_init_type") and args.lora_init_type is not None:
            if torch.backends.mps.is_available() and args.lora_init_type == "loftq":
                error_log("Apple MPS cannot make use of LoftQ initialisation. Overriding to 'default'.")
            elif args.is_quantized and args.lora_init_type == "loftq":
                error_log("LoftQ initialisation is not supported with quantised models. Overriding to 'default'.")
            else:
                args.lora_initialisation_style = args.lora_init_type if args.lora_init_type != "default" else True
        if args.use_dora:
            if "quanto" in args.base_model_precision:
                error_log("Quanto does not yet support DoRA training in PEFT. Disabling DoRA. 😴")
                args.use_dora = False
            else:
                warning_log("DoRA support is experimental and not very thoroughly tested.")
                args.lora_initialisation_style = "default"

    if args.distillation_config is not None:
        if args.distillation_config.startswith("{"):
            try:
                import ast

                args.distillation_config = ast.literal_eval(args.distillation_config)
            except Exception as e:
                logger.error(f"Could not load distillation_config: {e}")
                raise

    if not args.data_backend_config:
        from simpletuner.helpers.training.state_tracker import StateTracker

        args.data_backend_config = os.path.join(StateTracker.get_config_path(), "multidatabackend.json")
        warning_log(f"No data backend config provided. Using default config at {args.data_backend_config}.")

    if args.validation_num_video_frames is not None and args.validation_num_video_frames < 1:
        raise ValueError("validation_num_video_frames must be at least 1.")

    # Check if we have a valid gradient accumulation steps.
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps}, should be >= 1"
        )

    if args.validation_guidance_skip_layers is not None:
        if args.model_family not in ["sd3", "wan"]:
            raise ValueError("Currently, skip-layer guidance is not supported for {}".format(args.model_family))
        try:
            import json

            args.validation_guidance_skip_layers = json.loads(args.validation_guidance_skip_layers)
        except Exception as e:
            logger.error(f"Could not load validation_guidance_skip_layers: {e}")
            raise

    if (
        args.sana_complex_human_instruction is not None
        and type(args.sana_complex_human_instruction) is str
        and args.sana_complex_human_instruction not in ["", "None"]
    ):
        try:
            import json

            args.sana_complex_human_instruction = json.loads(args.sana_complex_human_instruction)
        except Exception as e:
            logger.error(f"Could not load complex human instruction ({args.sana_complex_human_instruction}): {e}")
            raise
    elif args.sana_complex_human_instruction == "None":
        args.sana_complex_human_instruction = None

    if args.attention_mechanism != "diffusers" and not torch.cuda.is_available():
        warning_log("For non-CUDA systems, only Diffusers attention mechanism is officially supported.")

    deprecated_options = {
        # how to deprecate options:
        # "flux_beta_schedule_alpha": "flow_beta_schedule_alpha",
    }

    for deprecated_option, replacement_option in deprecated_options.items():
        if (
            getattr(args, replacement_option) is not None
            and getattr(args, deprecated_option) is not None
            and type(getattr(args, deprecated_option)) is not object
        ):
            warning_log(f"The option --{deprecated_option} has been replaced with --{replacement_option}.")
            setattr(args, replacement_option, getattr(args, deprecated_option))
        elif getattr(args, deprecated_option) is not None:
            error_log(
                f"The option {deprecated_option} has been deprecated without a replacement option. Please remove it from your configuration."
            )
            sys.exit(1)

    info_log(f"Parsed command line arguments: {args}")
    return args

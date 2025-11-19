import argparse
import ast
import json
import logging
import os
import random
import sys
import time
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from accelerate import InitProcessGroupKwargs
from accelerate.utils import ProjectConfiguration

from simpletuner.helpers.configuration.cli_utils import mapping_to_cli_args
from simpletuner.helpers.logging import get_logger
from simpletuner.helpers.training.attention_backend import AttentionBackendMode
from simpletuner.helpers.training.multi_process import should_log
from simpletuner.helpers.training.optimizer_param import is_optimizer_deprecated, is_optimizer_grad_fp32
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.simpletuner_sdk.server.services.field_registry.types import (
    ConfigField,
    FieldType,
    ParserType,
    ValidationRuleType,
)
from simpletuner.simpletuner_sdk.server.utils.paths import resolve_config_path

logger = get_logger("ArgsParser")

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


def _configure_tf32(disable_tf32: bool) -> None:
    """Configure TF32/FP32 behavior for CUDA backends."""
    if not torch.cuda.is_available():
        return

    backend_root = getattr(torch, "backends", None)
    if backend_root is None:
        return

    cuda_backend = getattr(backend_root, "cuda", None)
    cudnn_backend = getattr(backend_root, "cudnn", None)
    matmul_backend = getattr(cuda_backend, "matmul", None)
    cudnn_conv_backend = getattr(cudnn_backend, "conv", None)
    cudnn_rnn_backend = getattr(cudnn_backend, "rnn", None)

    supports_precision_overrides = (
        hasattr(backend_root, "fp32_precision")
        and matmul_backend is not None
        and hasattr(matmul_backend, "fp32_precision")
        and cudnn_backend is not None
        and hasattr(cudnn_backend, "fp32_precision")
    )

    def _set_tf32(enabled: bool) -> None:
        if supports_precision_overrides:
            precision = "tf32" if enabled else "ieee"
            backend_root.fp32_precision = precision
            if matmul_backend is not None and hasattr(matmul_backend, "fp32_precision"):
                matmul_backend.fp32_precision = precision
            if cudnn_backend is not None and hasattr(cudnn_backend, "fp32_precision"):
                cudnn_backend.fp32_precision = precision
            for cudnn_op_backend in (cudnn_conv_backend, cudnn_rnn_backend):
                if cudnn_op_backend is not None and hasattr(cudnn_op_backend, "fp32_precision"):
                    cudnn_op_backend.fp32_precision = precision
        else:
            if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
                matmul_backend.allow_tf32 = enabled
            if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
                cudnn_backend.allow_tf32 = enabled

    if disable_tf32:
        warning_log("--disable_tf32 is provided, not enabling. Training will potentially be much slower.")
        _set_tf32(False)
    else:
        _set_tf32(True)
        info_log("Enabled NVIDIA TF32 for faster training on Ampere GPUs. Use --disable_tf32 if this causes any problems.")


def _configure_rocm_environment() -> None:
    """Enable ROCm-specific acceleration toggles when running on HIP builds."""
    if not torch.cuda.is_available():
        return
    hip_version = getattr(getattr(torch, "version", None), "hip", None)
    if not hip_version:
        return

    os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "1")

    if "HIPBLASLT_ALLOW_TF32" in os.environ:
        return

    if not _has_mi300_gpu():
        return

    os.environ["HIPBLASLT_ALLOW_TF32"] = "1"


def _has_mi300_gpu() -> bool:
    """Return True when at least one visible device exposes an MI300 (gfx94x) architecture."""
    try:
        device_count = torch.cuda.device_count()
    except Exception:
        device_count = 0

    for index in range(device_count):
        try:
            props = torch.cuda.get_device_properties(index)
        except Exception:
            continue
        if _device_is_mi300(props):
            return True
    return False


def _device_is_mi300(props: Any) -> bool:
    mi300_tokens = ("mi300", "gfx940", "gfx941", "gfx942", "gfx943", "gfx944", "gfx94")
    candidates = (
        str(getattr(props, "gcnArchName", "") or "").lower(),
        str(getattr(props, "name", "") or "").lower(),
    )
    return any(token in candidate for token in mi300_tokens for candidate in candidates)


_ARG_PARSER_CACHE: Optional[argparse.ArgumentParser] = None

BOOL_TRUE_STRINGS = {"1", "true", "yes", "y", "on"}
BOOL_FALSE_STRINGS = {"0", "false", "no", "n", "off"}


def _parse_json_like_option(raw_value, option_name: str):
    """
    Normalize config options that accept rich JSON structures or file references.
    """
    if raw_value in (None, "", "None"):
        return None

    if isinstance(raw_value, (dict, list)):
        return raw_value

    if isinstance(raw_value, str):
        candidate = raw_value.strip()
        if not candidate:
            return None

        if candidate.startswith("{") or candidate.startswith("["):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as json_error:
                try:
                    return ast.literal_eval(candidate)
                except (ValueError, SyntaxError) as ast_error:
                    raise ValueError(
                        f"Could not parse {option_name} as JSON."
                        f" json.loads error: {json_error}; ast.literal_eval error: {ast_error}"
                    ) from ast_error

        expanded_path = os.path.expanduser(candidate)
        if os.path.isfile(expanded_path):
            try:
                with open(expanded_path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except json.JSONDecodeError as file_error:
                raise ValueError(f"Could not load {option_name} from {expanded_path}: {file_error}") from file_error

        return candidate

    return raw_value


def _normalize_sana_complex_instruction(raw_value):
    """
    Normalize the Sana complex human instruction value so downstream code always receives a list of strings.
    """

    if raw_value in (None, "", "None"):
        return None

    if isinstance(raw_value, (list, tuple)):
        normalized = []
        for entry in raw_value:
            if entry in (None, "", "None"):
                continue
            entry_str = str(entry).strip()
            if entry_str:
                normalized.append(entry_str)
        return normalized or None

    if not isinstance(raw_value, str):
        raise ValueError(f"Unsupported type for sana_complex_human_instruction: {type(raw_value).__name__}")

    candidate = raw_value.strip()
    if not candidate or candidate == "None":
        return None

    expanded_path = os.path.expanduser(candidate)
    if os.path.isfile(expanded_path):
        with open(expanded_path, "r", encoding="utf-8") as handle:
            file_contents = handle.read()
        return _normalize_sana_complex_instruction(file_contents)

    if candidate.startswith("{") or candidate.startswith("["):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as json_error:
            logger.error(f"Could not parse sana_complex_human_instruction as JSON: {json_error}")
            raise
        return _normalize_sana_complex_instruction(parsed)

    instructions = [line.strip() for line in candidate.splitlines() if line.strip()]
    return instructions or [candidate]


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
        if isinstance(candidate, float):
            return float
        if isinstance(candidate, str):
            try:
                numeric_value = float(candidate)
            except ValueError:
                continue
            if "." in candidate or "e" in candidate.lower() or not numeric_value.is_integer():
                return float
    return int


def _determine_cli_type(field: ConfigField, choice_values: List[Any]):
    if field.parser_type is not None:
        parser_type_map = {
            ParserType.STRING: str,
            ParserType.INTEGER: int,
            ParserType.FLOAT: float,
            ParserType.BOOLEAN: _parse_bool_flag,
        }
        try:
            return parser_type_map[field.parser_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported parser type override: {field.parser_type}") from exc
    if field.field_type == FieldType.NUMBER:
        return _infer_numeric_type(field, choice_values)
    return str


def _is_required(field: ConfigField) -> bool:
    return any(rule.rule_type == ValidationRuleType.REQUIRED for rule in field.validation_rules)


def _add_argument_from_field(parser: argparse.ArgumentParser, field: ConfigField) -> None:
    choice_values = _extract_choice_values(field)
    cli_choices = [value for value in choice_values if value is not None]
    help_text = field.help_text or getattr(field, "cmd_args_help", "") or field.tooltip
    kwargs: Dict[str, Any] = {}
    option_strings: List[str] = []
    if isinstance(field.arg_name, str):
        option_strings.append(field.arg_name)
    elif isinstance(field.arg_name, (list, tuple)):
        option_strings.extend(field.arg_name)
    else:
        option_strings.append(str(field.arg_name))

    if field.aliases:
        option_strings.extend(field.aliases)

    # Deduplicate while preserving order
    seen_opts = set()
    option_strings = [opt for opt in option_strings if not (opt in seen_opts or seen_opts.add(opt))]

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
        parser.add_argument(*option_strings, **kwargs)
        return
    if field.field_type == FieldType.SELECT:
        cli_choices = [str(value) for value in cli_choices]

    if cli_choices and not field.dynamic_choices:
        kwargs["choices"] = cli_choices
    default = field.default_value
    if field.field_type == FieldType.SELECT and default is not None:
        default = str(default)
    if default is not None:
        kwargs["default"] = default
    kwargs["type"] = _determine_cli_type(field, cli_choices)
    parser.add_argument(*option_strings, **kwargs)


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
    from simpletuner.helpers.training.state_tracker import StateTracker

    parser_error = None

    def _normalize_model_family(value: str) -> str:
        normalized = (value or "").strip().lower().replace("-", "_")
        if not normalized:
            return normalized

        try:
            from simpletuner.helpers.models.registry import ModelRegistry

            families = list(ModelRegistry.model_families().keys())
        except Exception:
            return normalized

        if normalized in families:
            return normalized

        simplified = normalized.replace("_", "")
        for family in families:
            if simplified == family.replace("_", ""):
                return family

        return normalized

    def _normalize_input_args(raw_args):
        if raw_args is None:
            return None

        normalized_args = []
        skip_next = False

        for idx, arg in enumerate(raw_args):
            if skip_next:
                skip_next = False
                continue

            if arg.startswith(("--model_family=", "--model-family=")):
                prefix, value = arg.split("=", 1)
                normalized_args.append(f"{prefix}={_normalize_model_family(value)}")
                continue

            if arg in ("--model_family", "--model-family") and idx + 1 < len(raw_args):
                normalized_args.append(arg)
                normalized_args.append(_normalize_model_family(raw_args[idx + 1]))
                skip_next = True
                continue

            normalized_args.append(arg)

        return normalized_args

    try:
        normalized_args = _normalize_input_args(input_args)
        args = parser.parse_args(normalized_args)
    except Exception:  # pragma: no cover - parser handles errors consistently
        parser_error = sys.exc_info()[1]
        logger.error(f"Could not parse input: {input_args}")
        import traceback

        logger.error(traceback.format_exc())
        webhook_handler = StateTracker.get_webhook_handler()
        if webhook_handler is not None:
            try:
                logger.info(f"Sending error message to webhook: {webhook_handler.webhook_url}")
                # Sanitize error message - don't expose raw args in webhook
                webhook_handler.send(
                    message="Command Line Argument Error: Failed to parse command line arguments. Please check the server logs for details.",
                    message_level="error",
                )
            except Exception as exc:
                logger.error(f"Failed to send webhook error message: {exc}")
            logger.error(f"Argument parsing failed for input: {input_args}")
        else:
            logger.error("No webhook handler available to send error message.")

    if args is None and exit_on_error:
        raise ValueError(f"Could not parse command line arguments: {parser_error or 'see above logs for details'}")

    if args is None:
        return None

    if args.controlnet_custom_config is not None and type(args.controlnet_custom_config) is str:
        if args.controlnet_custom_config.startswith("{"):
            try:
                args.controlnet_custom_config = ast.literal_eval(args.controlnet_custom_config)
            except Exception as e:
                logger.error(f"Could not load controlnet_custom_config: {e}")
                raise
    if args.webhook_config is not None:
        logger.debug("webhook_config at start = %s (type: %s)", args.webhook_config, type(args.webhook_config))
        # Handle different types of webhook_config
        # First, check if it's an AST object using isinstance (the proper way)
        if isinstance(args.webhook_config, (ast.AST, ast.Name, ast.Call, ast.Dict, ast.List, ast.Constant)):
            # This is an AST object - this indicates a configuration parsing error
            logger.error("webhook_config is an AST object - this indicates a configuration error")
            logger.error("webhook_config should be a JSON string or file path, not an AST object")
            logger.error("This typically happens when the configuration is passed incorrectly from the command line")
            logger.error("Please ensure webhook_config is properly formatted as a JSON string")
            # Provide a helpful error message with the AST object info
            ast_repr = repr(args.webhook_config)
            logger.error(f"Received AST object: {ast_repr}")
            raise ValueError(
                f"webhook_config is an AST object ({ast_repr}) instead of a JSON string or file path. "
                f"Please check your configuration format. "
                f'Expected format: \'[{{"webhook_type": "raw", "callback_url": "https://..."}}]\' '
                f"or a file path to a JSON file."
            )
        elif hasattr(args.webhook_config, "__class__") and "ast" in str(type(args.webhook_config)):
            # This is an AST object - this indicates a configuration parsing error
            logger.error("webhook_config is an AST object - this indicates a configuration error")
            logger.error("webhook_config should be a JSON string or file path, not an AST object")
            logger.error("This typically happens when the configuration is passed incorrectly from the command line")
            logger.error("Please ensure webhook_config is properly formatted as a JSON string")
            # Provide a helpful error message with the AST object info
            ast_repr = repr(args.webhook_config)
            logger.error(f"Received AST object: {ast_repr}")
            raise ValueError(
                f"webhook_config is an AST object ({ast_repr}) instead of a JSON string or file path. "
                f"Please check your configuration format. "
                f'Expected format: \'[{{"webhook_type": "raw", "callback_url": "https://..."}}]\' '
                f"or a file path to a JSON file."
            )
        elif isinstance(args.webhook_config, str):
            # SAFETY CHECK FIRST: Check if the string contains AST object patterns
            # This catches the case where AST objects get converted to strings BEFORE any other processing
            config_str = str(args.webhook_config)

            if (
                "ast." in config_str
                or "<ast." in config_str
                or "ast object at" in config_str.lower()
                or (config_str.strip().startswith("<") and config_str.strip().endswith(">"))
            ):
                logger.error("webhook_config contains AST object patterns in string form")
                logger.error("This indicates a configuration parsing error where AST objects were converted to strings")
                logger.error(f"Received webhook_config string: {config_str}")
                raise ValueError(
                    f"webhook_config contains AST object patterns instead of valid JSON. "
                    f"Received: {config_str[:200]}... "
                    f"Please check your configuration format. "
                    f'Expected format: \'[{{"webhook_type": "raw", "callback_url": "https://..."}}]\' '
                    f"or a file path to a JSON file."
                )

            if args.webhook_config.startswith("{") or args.webhook_config.startswith("["):
                try:
                    import json

                    # FINAL SAFETY CHECK: Detect AST object patterns
                    if (
                        "ast." in args.webhook_config
                        or "<ast." in args.webhook_config
                        or "ast object at" in args.webhook_config.lower()
                        or (args.webhook_config.strip().startswith("<") and args.webhook_config.strip().endswith(">"))
                    ):
                        logger.error("webhook_config contains AST object patterns")
                        logger.error(f"Received webhook_config: {args.webhook_config}")
                        raise ValueError(
                            f"webhook_config contains AST object patterns instead of valid JSON. "
                            f"Received: {args.webhook_config[:200]}... "
                            f"Please check your configuration format."
                        )

                    # Use json.loads() instead of ast.literal_eval() since we're dealing with JSON
                    # This properly handles JSON booleans (true/false) vs Python (True/False)
                    parsed_config = json.loads(args.webhook_config)
                    # Normalize single dict to list for consistency
                    if isinstance(parsed_config, dict):
                        args.webhook_config = [parsed_config]
                    elif isinstance(parsed_config, list):
                        args.webhook_config = parsed_config
                    else:
                        logger.error(f"webhook_config must be dict or list, got {type(parsed_config)}")
                        raise ValueError(f"Invalid webhook_config type: {type(parsed_config)}")
                except json.JSONDecodeError as e:
                    logger.error(f"Could not load webhook_config (invalid JSON): {e}")
                    raise
                except Exception as e:
                    logger.error(f"Could not load webhook_config: {e}")
                    raise
            else:
                # try to load from file
                if os.path.isfile(args.webhook_config):
                    try:
                        with open(args.webhook_config, "r") as f:
                            import json

                            loaded_config = json.load(f)
                            # Normalize single dict to list for consistency
                            if isinstance(loaded_config, dict):
                                args.webhook_config = [loaded_config]
                            elif isinstance(loaded_config, list):
                                args.webhook_config = loaded_config
                            else:
                                logger.error(f"webhook_config must be dict or list, got {type(loaded_config)}")
                                raise ValueError(f"Invalid webhook_config type: {type(loaded_config)}")
                    except Exception as e:
                        logger.error(f"Could not load webhook_config from file: {e}")
                        raise
                else:
                    logger.error(f"Could not find webhook_config file: {args.webhook_config}")
        elif isinstance(args.webhook_config, (dict, list)):
            # Already a dict or list - normalize to list
            if isinstance(args.webhook_config, dict):
                args.webhook_config = [args.webhook_config]
            # list is already good
        else:
            logger.error(f"webhook_config has unsupported type: {type(args.webhook_config)}")
            raise ValueError(f"webhook_config must be string, dict, or list, got {type(args.webhook_config)}")

    if args.tread_config is not None and type(args.tread_config) is str:
        if args.tread_config.startswith("{"):
            try:
                args.tread_config = ast.literal_eval(args.tread_config)
            except Exception as e:
                logger.error(f"Could not load tread_config: {e}")
                raise

    if args.sla_config is not None and isinstance(args.sla_config, str):
        candidate = args.sla_config.strip()
        if candidate.startswith("{"):
            try:
                args.sla_config = ast.literal_eval(candidate)
            except Exception as e:
                logger.error(f"Could not load sla_config: {e}")
                raise

    if args.optimizer == "adam_bfloat16" and args.mixed_precision != "bf16":
        if not torch.backends.mps.is_available():
            raise ValueError("You cannot use --adam_bfloat16 without --mixed_precision=bf16.")

    if args.mixed_precision == "fp8" and not torch.cuda.is_available():
        raise ValueError("You cannot use --mixed_precision=fp8 without a CUDA device. Please use bf16 instead.")

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
            raise ValueError(
                "An M3 Max 128G will use 12 seconds per step at a batch size of 1 and 65 seconds per step at a batch size of 12."
                " Any higher values will result in NDArray size errors or other unstable training results and crashes."
                "\nPlease reduce the batch size to 12 or lower."
            )

        if args.quantize_via == "accelerator":
            error_log(
                "MPS does not benefit from models being quantized on the accelerator device. Overriding --quantize_via to 'cpu'."
            )
            args.quantize_via = "cpu"

    if args.max_train_steps is not None and args.max_train_steps > 0 and args.num_train_epochs > 0:
        raise ValueError("When using --max_train_steps (MAX_NUM_STEPS), you must set --num_train_epochs (NUM_EPOCHS) to 0.")

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
        raise ValueError("Both --optimizer_beta1 and --optimizer_beta2 should be provided.")

    if args.gradient_checkpointing:
        # enable torch compile w/ activation checkpointing :[ slows us down.
        torch._dynamo.config.optimize_ddp = False

    args.logging_dir = os.path.join(args.output_dir, args.logging_dir)
    args.accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.logging_dir)
    # Create the custom configuration
    args.process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))  # 1.5 hours

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    _configure_tf32(disable_tf32=args.disable_tf32)
    _configure_rocm_environment()

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
        # Use resolve_config_path for safe path resolution
        resolved_lycoris_path = None
        try:
            resolved_candidate = resolve_config_path(
                args.lycoris_config,
                config_dir=StateTracker.get_config_path(),
                check_cwd_first=True,
            )
            if resolved_candidate is not None:
                resolved_lycoris_path = resolved_candidate
        except Exception as e:
            logger.warning(f"Error resolving lycoris config path: {e}")

        if resolved_lycoris_path is None:
            # If resolution failed, check if it's a valid path within allowed directories
            expanded_candidate = os.path.expanduser(args.lycoris_config)
            if os.path.isabs(expanded_candidate):
                # For absolute paths, ensure they're within config directory only (security fix)
                try:
                    abs_path = Path(expanded_candidate).resolve(strict=True)
                    config_dir = Path(StateTracker.get_config_path()).resolve()

                    # Only allow paths within config directory for security
                    if abs_path.is_relative_to(config_dir):
                        resolved_lycoris_path = abs_path
                    else:
                        raise ValueError(
                            f"Lycoris config path '{args.lycoris_config}' is outside allowed directory. "
                            f"Must be within: {config_dir}"
                        )
                except (ValueError, FileNotFoundError, RuntimeError) as e:
                    raise ValueError(
                        f"Lycoris config path '{args.lycoris_config}' is invalid or outside allowed directory. "
                        f"Must be within: {StateTracker.get_config_path()}. Error: {e}"
                    )
            else:
                # For relative paths, try to resolve within config directory
                config_dir = Path(StateTracker.get_config_path())
                candidate = config_dir / expanded_candidate
                if candidate.exists() and candidate.is_file():
                    resolved_lycoris_path = candidate.resolve()
                else:
                    raise ValueError(f"Could not find lycoris config at '{args.lycoris_config}'. " f"Looked in: {candidate}")

        if resolved_lycoris_path is not None:
            args.lycoris_config = str(resolved_lycoris_path)

        # is it readable?
        lycoris_path_to_check = os.path.expanduser(str(args.lycoris_config))
        if not os.path.isfile(lycoris_path_to_check) or not os.access(lycoris_path_to_check, os.R_OK):
            raise ValueError(f"Could not find the JSON configuration file at '{args.lycoris_config}'")
        import json

        args.lycoris_config = lycoris_path_to_check
        with open(lycoris_path_to_check, "r") as f:
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
                args.distillation_config = ast.literal_eval(args.distillation_config)
            except Exception as e:
                logger.error(f"Could not load distillation_config: {e}")
                raise

    if hasattr(args, "deepspeed_config"):
        try:
            args.deepspeed_config = _parse_json_like_option(args.deepspeed_config, "deepspeed_config")
        except ValueError as parse_error:
            logger.error(str(parse_error))
            raise

    if getattr(args, "fsdp_enable", False):
        if args.deepspeed_config not in (None, "", "None", False):
            raise ValueError("Cannot enable FSDP when a DeepSpeed configuration is also provided.")

        try:
            args.fsdp_version = int(args.fsdp_version)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid --fsdp_version value: {args.fsdp_version}")

        if args.fsdp_version not in (1, 2):
            raise ValueError("--fsdp_version must be either 1 or 2.")
        if args.fsdp_version == 1:
            warning_log("FSDP v1 is deprecated. Please prefer --fsdp_version=2 for DTensor-based FSDP.")

        state_dict_type = str(args.fsdp_state_dict_type or "").strip().upper()
        if state_dict_type == "":
            state_dict_type = "SHARDED_STATE_DICT"
        valid_state_dict_types = {"SHARDED_STATE_DICT", "FULL_STATE_DICT"}
        if state_dict_type not in valid_state_dict_types:
            raise ValueError(
                f"Invalid --fsdp_state_dict_type '{args.fsdp_state_dict_type}'. "
                f"Expected one of {sorted(valid_state_dict_types)}."
            )
        args.fsdp_state_dict_type = state_dict_type

        auto_wrap_policy = str(args.fsdp_auto_wrap_policy or "").strip().upper()
        auto_wrap_mapping = {
            "TRANSFORMER_BASED_WRAP": "transformer_based_wrap",
            "SIZE_BASED_WRAP": "size_based_wrap",
            "NO_WRAP": "no_wrap",
            "NONE": None,
        }
        if auto_wrap_policy in auto_wrap_mapping:
            args.fsdp_auto_wrap_policy = auto_wrap_mapping[auto_wrap_policy]
        elif auto_wrap_policy:
            # Allow custom callables configured via dotted path without transformation.
            args.fsdp_auto_wrap_policy = args.fsdp_auto_wrap_policy
        else:
            args.fsdp_auto_wrap_policy = "transformer_based_wrap"

        transformer_cls = args.fsdp_transformer_layer_cls_to_wrap
        if transformer_cls in (None, "", "None"):
            args.fsdp_transformer_layer_cls_to_wrap = None
        else:
            if isinstance(transformer_cls, (list, tuple)):
                values = [str(entry).strip() for entry in transformer_cls]
            else:
                values = [entry.strip() for entry in str(transformer_cls).split(",")]
            filtered_values = [entry for entry in values if entry]
            args.fsdp_transformer_layer_cls_to_wrap = filtered_values or None
            info_log(f"FSDP transformer layer classes to wrap: {args.fsdp_transformer_layer_cls_to_wrap}")
    else:
        # When FSDP is disabled, normalise auxiliary options so downstream logic can rely on None/False.
        args.fsdp_transformer_layer_cls_to_wrap = None

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

    if args.model_family == "sana":
        try:
            args.sana_complex_human_instruction = _normalize_sana_complex_instruction(args.sana_complex_human_instruction)
        except Exception as exc:
            logger.error(f"Could not load complex human instruction ({args.sana_complex_human_instruction}): {exc}")
            raise
    elif args.sana_complex_human_instruction == "None":
        args.sana_complex_human_instruction = None

    if isinstance(getattr(args, "validation_adapter_path", None), str):
        candidate = args.validation_adapter_path.strip()
        args.validation_adapter_path = candidate or None

    if getattr(args, "validation_adapter_config", None):
        args.validation_adapter_config = _parse_json_like_option(
            args.validation_adapter_config,
            "--validation_adapter_config",
        )

    if args.validation_adapter_path and args.validation_adapter_config:
        raise ValueError("Provide either --validation_adapter_path or --validation_adapter_config, not both.")

    if isinstance(getattr(args, "validation_adapter_name", None), str):
        candidate = args.validation_adapter_name.strip()
        args.validation_adapter_name = candidate or None

    strength_value = getattr(args, "validation_adapter_strength", None)
    if strength_value is None or strength_value in ("", "None"):
        args.validation_adapter_strength = 1.0
    else:
        try:
            strength = float(strength_value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid --validation_adapter_strength value: {strength_value}") from None
        if strength <= 0:
            raise ValueError("--validation_adapter_strength must be greater than 0.")
        args.validation_adapter_strength = strength

    mode_value = getattr(args, "validation_adapter_mode", None)
    if mode_value in (None, "", "None"):
        args.validation_adapter_mode = "adapter_only"
    else:
        normalized_mode = str(mode_value).strip().lower()
        valid_modes = {"adapter_only", "comparison", "none"}
        if normalized_mode not in valid_modes:
            raise ValueError(
                f"Invalid --validation_adapter_mode '{mode_value}'. Expected one of: {', '.join(sorted(valid_modes))}."
            )
        args.validation_adapter_mode = normalized_mode

    if args.attention_mechanism != "diffusers" and not torch.cuda.is_available():
        warning_log("For non-CUDA systems, only Diffusers attention mechanism is officially supported.")

    if hasattr(args, "sageattention_usage"):
        args.sageattention_usage = AttentionBackendMode.from_raw(args.sageattention_usage)

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
            raise ValueError(
                f"The option {deprecated_option} has been deprecated without a replacement option. Please remove it from your configuration."
            )

    return args

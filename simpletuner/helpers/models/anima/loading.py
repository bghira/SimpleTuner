# Vendored from diffusers-anima: /src/diffusers-anima/src/diffusers_anima/pipelines/anima/loading.py
# Adapted for SimpleTuner local imports.

"""Anima pipeline loading utilities: from_pretrained, from_single_file, and component loading."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
    from .pipeline import AnimaPipeline

import torch
from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

from .constants import (
    ANIMA_VAE_CONFIG,
    DTYPE_MAP,
    DTYPE_NAME_MAP,
    HF_URL_PREFIXES,
    LOCAL_QWEN_TOKENIZER_DIR,
    LOCAL_T5_TOKENIZER_DIR,
    QWEN3_06B_CONFIG,
)
from .options import AnimaComponents, AnimaLoaderOptions, AnimaRuntimeOptions
from .scheduler import AnimaFlowMatchEulerDiscreteScheduler
from .text_encoding import AnimaPromptTokenizer
from .transformer import AnimaTransformerModel, _convert_anima_state_dict_to_diffusers
from .vae_conversion import convert_anima_vae_state_dict

# ---------------------------------------------------------------------------
# Internal component sources for from_single_file
# These are fixed to the Anima model architecture and not user-configurable.
# To use different components, build the pipeline manually.
# ---------------------------------------------------------------------------
_ANIMA_REPO = "hdae/diffusers-anima-preview"
_TEXT_ENCODER_WEIGHTS = f"{_ANIMA_REPO}::text_encoder/model.safetensors"
_TEXT_ENCODER_CONFIG_REPO = "Qwen/Qwen3-0.6B-Base"
_QWEN_TOKENIZER_SOURCE = f"{_ANIMA_REPO}::prompt_tokenizer_qwen"
_T5_TOKENIZER_SOURCE = f"{_ANIMA_REPO}::prompt_tokenizer_t5"
_VAE_SOURCE = f"{_ANIMA_REPO}::vae/diffusion_pytorch_model.safetensors"


def coerce_anima_scheduler(
    scheduler: FlowMatchEulerDiscreteScheduler,
) -> AnimaFlowMatchEulerDiscreteScheduler:
    """Upgrade a plain ``FlowMatchEulerDiscreteScheduler`` to an Anima-aware one."""
    if isinstance(scheduler, AnimaFlowMatchEulerDiscreteScheduler):
        return scheduler
    if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        return AnimaFlowMatchEulerDiscreteScheduler.from_config(scheduler.config)
    raise ValueError("`scheduler` must be a FlowMatchEulerDiscreteScheduler-compatible instance.")


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_dtype_name(value: str | torch.dtype, *, name: str) -> str:
    if isinstance(value, str):
        if value not in DTYPE_MAP:
            raise ValueError(f"Unsupported {name}: {value}")
        return value

    mapped = DTYPE_NAME_MAP.get(value)
    if mapped is None:
        raise ValueError(f"Unsupported {name}: {value}")
    return mapped


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    mapped = DTYPE_MAP.get(dtype)
    if mapped is not None:
        return mapped
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def resolve_text_encoder_dtype(
    *,
    model_dtype: torch.dtype,
    text_encoder_dtype: str,
    execution_device: str,
) -> torch.dtype:
    mapped = DTYPE_MAP.get(text_encoder_dtype)
    if text_encoder_dtype != "auto" and mapped is None:
        raise ValueError(f"Unsupported text_encoder_dtype: {text_encoder_dtype}")

    resolved_dtype = model_dtype if mapped is None else mapped
    if execution_device == "cpu" and resolved_dtype == torch.float16:
        return torch.float32
    return resolved_dtype


def warn_if_unsafe_fp16(*, resolved_device: str, resolved_dtype: torch.dtype) -> None:
    if resolved_device == "cuda" and resolved_dtype == torch.float16:
        warnings.warn(
            "dtype=float16 may cause NaN/Inf with Anima." " Use --dtype auto or --dtype bfloat16.",
            stacklevel=2,
        )


def normalize_proxies(proxies: dict[str, Any] | None) -> dict[str, str] | None:
    if proxies is None:
        return None
    if not isinstance(proxies, dict):
        raise ValueError("`proxies` must be a dictionary with string keys and values.")
    normalized: dict[str, str] = {}
    for key, value in proxies.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("`proxies` must be a dictionary with string keys and values.")
        normalized[key] = value
    return normalized


def loader_options_from_kwargs(kwargs: dict[str, Any], *, consume: bool) -> AnimaLoaderOptions:
    get_value = kwargs.pop if consume else kwargs.get
    cache_dir = get_value("cache_dir", None)
    revision = get_value("revision", None)
    return AnimaLoaderOptions(
        local_files_only=bool(get_value("local_files_only", False)),
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        force_download=bool(get_value("force_download", False)),
        token=get_value("token", None),
        revision=str(revision) if revision is not None else None,
        proxies=normalize_proxies(get_value("proxies", None)),
    )


def runtime_options_from_kwargs(kwargs: dict[str, Any], *, consume: bool) -> AnimaRuntimeOptions:
    get_value = kwargs.pop if consume else kwargs.get
    dtype_arg = get_value("dtype", "auto")
    torch_dtype_arg = get_value("torch_dtype", None)
    if torch_dtype_arg is not None:
        if dtype_arg != "auto":
            raise ValueError("Specify only one of `dtype` or `torch_dtype` for custom loading.")
        dtype_arg = torch_dtype_arg
    return AnimaRuntimeOptions(
        device=str(get_value("device", "auto")),
        dtype=normalize_dtype_name(dtype_arg, name="dtype"),
        text_encoder_dtype=normalize_dtype_name(
            get_value("text_encoder_dtype", "auto"),
            name="text_encoder_dtype",
        ),
    )


def scheduler_from_kwargs(kwargs: dict[str, Any], *, consume: bool) -> AnimaFlowMatchEulerDiscreteScheduler | None:
    value = kwargs.pop("scheduler", None) if consume else kwargs.get("scheduler", None)
    if value is None:
        return None
    return coerce_anima_scheduler(value)


def extract_hf_repo_id_and_filename(model_url: str) -> tuple[str, str]:
    import re

    stripped = model_url
    for prefix in HF_URL_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :]
            break

    match = re.match(r"([^/]+)/([^/]+)/(?:blob/main/)?(.+)", stripped)
    if match is None:
        raise ValueError(
            "URL model_path must be a Hugging Face file URL, for example "
            "'https://huggingface.co/<repo_id>/blob/main/<filename>'."
        )
    repo_id = f"{match.group(1)}/{match.group(2)}"
    filename = match.group(3)
    return repo_id, filename


def download_model_file_via_diffusers(
    *,
    repo_id: str,
    filename: str,
    options: AnimaLoaderOptions,
) -> str:
    # hf_hub_download (huggingface_hub ≥ 1.0) dropped the `proxies` parameter.
    # Proxy configuration should be done via the HTTPS_PROXY / HTTP_PROXY
    # environment variables or huggingface_hub's global HTTP backend settings.
    normalized_token = None if options.token is False else options.token
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=options.cache_dir,
        force_download=options.force_download,
        local_files_only=options.local_files_only,
        token=normalized_token,
        revision=options.revision,
    )


def resolve_single_file_path(
    model_path: str,
    *,
    options: AnimaLoaderOptions,
    input_label: str = "model_path",
    allow_remote_url: bool = False,
) -> str:
    if "::" in model_path and not model_path.startswith(("http://", "https://")):
        repo_id, filename = model_path.split("::", maxsplit=1)
        repo_id = repo_id.strip()
        filename = filename.strip()
        if not repo_id or not filename:
            raise ValueError(f"{input_label} in 'repo_id::filename' format requires both repo_id and filename.")
        return download_model_file_via_diffusers(
            repo_id=repo_id,
            filename=filename,
            options=options,
        )

    path = Path(model_path)
    if path.exists() and path.is_file():
        return str(path)

    if model_path.startswith(("http://", "https://")):
        if not allow_remote_url:
            raise ValueError(
                f"Remote URL source is not supported for {input_label}. " "Use local path or 'repo_id::filename'."
            )
        if options.local_files_only:
            raise ValueError(f"`local_files_only=True` does not allow remote URL downloads for {input_label}.")
        repo_id, filename = extract_hf_repo_id_and_filename(model_path)
        return download_model_file_via_diffusers(
            repo_id=repo_id,
            filename=filename,
            options=options,
        )

    if allow_remote_url:
        raise ValueError(f"{input_label} must be a local file path, Hugging Face URL, or 'repo_id::filename'.")
    raise ValueError(f"{input_label} must be a local file path or 'repo_id::filename'.")


def resolve_split_file_path(
    source: str,
    *,
    options: AnimaLoaderOptions,
    component_name: str,
) -> str:
    if source.startswith(("http://", "https://")):
        raise ValueError(
            f"Remote URL source is not supported for {component_name}. " "Use local path or 'repo_id::filename'."
        )
    return resolve_single_file_path(
        source,
        options=options,
        input_label=component_name,
    )


def parse_repo_and_subfolder(source: str) -> tuple[str, str | None]:
    if "::" in source:
        repo, subfolder = source.split("::", maxsplit=1)
        repo = repo.strip()
        subfolder = subfolder.strip()
        if repo:
            return repo, (subfolder or None)
    return source, None


def load_tokenizer_from_source(
    source: str,
    *,
    options: AnimaLoaderOptions,
) -> "PreTrainedTokenizer" | "PreTrainedTokenizerFast":
    repo_or_path, subfolder = parse_repo_and_subfolder(source)
    kwargs: dict[str, Any] = {
        "local_files_only": options.local_files_only,
        "cache_dir": options.cache_dir,
        "force_download": options.force_download,
    }
    if subfolder is not None:
        kwargs["subfolder"] = subfolder
    if options.token is not None:
        kwargs["token"] = options.token
    if options.revision is not None:
        kwargs["revision"] = options.revision
    if options.proxies is not None:
        kwargs["proxies"] = options.proxies
    return AutoTokenizer.from_pretrained(repo_or_path, **kwargs)


def load_prompt_tokenizer(
    *,
    qwen_tokenizer_source: str,
    t5_tokenizer_source: str,
    options: AnimaLoaderOptions,
) -> AnimaPromptTokenizer:
    qwen_tokenizer = load_tokenizer_from_source(qwen_tokenizer_source, options=options)
    t5_tokenizer = load_tokenizer_from_source(t5_tokenizer_source, options=options)
    return AnimaPromptTokenizer(
        qwen_tokenizer=qwen_tokenizer,
        t5_tokenizer=t5_tokenizer,
    )


def save_prompt_tokenizers_to_local_dir(*, prompt_tokenizer: AnimaPromptTokenizer, save_directory: Path) -> None:
    qwen_tokenizer = getattr(prompt_tokenizer, "qwen_tokenizer", None)
    t5_tokenizer = getattr(prompt_tokenizer, "t5_tokenizer", None)

    if qwen_tokenizer is None or not hasattr(qwen_tokenizer, "save_pretrained"):
        warnings.warn(
            "Skipping bundled qwen tokenizer save because save_pretrained is not available.",
            stacklevel=2,
        )
    else:
        qwen_dir = save_directory / LOCAL_QWEN_TOKENIZER_DIR
        qwen_dir.mkdir(parents=True, exist_ok=True)
        qwen_tokenizer.save_pretrained(str(qwen_dir))

    if t5_tokenizer is None or not hasattr(t5_tokenizer, "save_pretrained"):
        warnings.warn(
            "Skipping bundled t5 tokenizer save because save_pretrained is not available.",
            stacklevel=2,
        )
    else:
        t5_dir = save_directory / LOCAL_T5_TOKENIZER_DIR
        t5_dir.mkdir(parents=True, exist_ok=True)
        t5_tokenizer.save_pretrained(str(t5_dir))


def resolve_prompt_tokenizer_sources_for_local_dir(
    *,
    pipeline_dir: Path,
) -> tuple[str, str, bool]:
    """Return (qwen_source, t5_source, uses_local) for a Diffusers pipeline directory.

    If both tokenizer subdirectories are present in the directory, use them
    directly (local, no network required). Otherwise, fall back to the fixed
    Anima HF repository sources so that tokenizers are fetched on demand.
    """
    local_qwen = pipeline_dir / LOCAL_QWEN_TOKENIZER_DIR
    local_t5 = pipeline_dir / LOCAL_T5_TOKENIZER_DIR
    if local_qwen.is_dir() and local_t5.is_dir():
        return str(local_qwen), str(local_t5), True

    return _QWEN_TOKENIZER_SOURCE, _T5_TOKENIZER_SOURCE, False


def _strip_net_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if all(key.startswith("net.") for key in state_dict):
        return {key[4:]: value for key, value in state_dict.items()}
    return dict(state_dict)


def _strip_model_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if all(key.startswith("model.") for key in state_dict):
        return {key[6:]: value for key, value in state_dict.items()}
    return dict(state_dict)


def load_vae_single_file(file_path: str, device: str, dtype: torch.dtype) -> AutoencoderKLQwenImage:
    state_dict = load_file(file_path, device="cpu")
    state_dict = convert_anima_vae_state_dict(state_dict)

    vae = AutoencoderKLQwenImage.from_config(ANIMA_VAE_CONFIG)
    missing, unexpected = vae.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Single-file VAE does not match expected AutoencoderKLQwenImage architecture. "
            f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )
    vae.to(dtype=dtype)
    vae.eval().requires_grad_(False)
    vae.to(device)
    return vae


def load_text_encoder(
    *,
    device: str,
    dtype: torch.dtype,
    options: AnimaLoaderOptions,
) -> Qwen3Model:
    file_path = resolve_split_file_path(
        _TEXT_ENCODER_WEIGHTS,
        options=options,
        component_name="text_encoder",
    )
    return load_text_encoder_single_file(file_path=file_path, device=device, dtype=dtype)


def load_text_encoder_single_file(
    *,
    file_path: str,
    device: str,
    dtype: torch.dtype,
) -> Qwen3Model:
    state_dict = load_file(file_path, device="cpu")
    state_dict = _strip_model_prefix(state_dict)

    config = Qwen3Config(**QWEN3_06B_CONFIG)

    text_encoder = Qwen3Model(config)
    missing, unexpected = text_encoder.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Text encoder weights do not match expected Qwen3-0.6B architecture. "
            f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )

    text_encoder.to(dtype=dtype)
    text_encoder.eval().requires_grad_(False)
    text_encoder.to(device=device, dtype=dtype)
    return text_encoder


def load_vae(
    device: str,
    dtype: torch.dtype,
    options: AnimaLoaderOptions,
) -> AutoencoderKLQwenImage:
    file_path = resolve_split_file_path(
        _VAE_SOURCE,
        options=options,
        component_name="vae",
    )
    return load_vae_single_file(file_path, device=device, dtype=dtype)


def load_transformer_native(model_path: str, device: str, dtype: torch.dtype) -> AnimaTransformerModel:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Anima checkpoint not found: {model_path}")

    state_dict = load_file(model_path, device="cpu")
    state_dict = _strip_net_prefix(state_dict)
    core_state_dict, llm_adapter_state_dict = _convert_anima_state_dict_to_diffusers(state_dict)

    transformer = AnimaTransformerModel()
    merged_state = {**core_state_dict, **llm_adapter_state_dict}

    missing, unexpected = transformer.load_state_dict(merged_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Anima checkpoint does not match native transformer architecture. "
            f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )

    transformer.to(dtype=dtype)
    transformer.eval().requires_grad_(False)
    transformer.to(device=device, dtype=dtype)
    return transformer


def _recast_module_to_parameter_dtype(module: torch.nn.Module | None) -> None:
    """Recast non-persistent buffers to the module parameter dtype/device.

    Some upstream loaders leave runtime buffers (for example Transformers RoPE
    buffers) in float32 even when parameters are loaded in bfloat16. The single-file
    loader path applies `.to(dtype=...)` after construction, so we mirror that here
    for Diffusers-loaded components to preserve reproducibility.

    Sharded/multi-device modules are skipped intentionally.
    """
    if module is None or not hasattr(module, "to"):
        return

    devices: set[torch.device] = set()
    dtypes: set[torch.dtype] = set()
    parameters = getattr(module, "parameters", None)
    if parameters is None:
        return

    for parameter in parameters():
        if not isinstance(parameter, torch.nn.Parameter):
            continue
        if parameter.device.type == "meta":
            return
        devices.add(parameter.device)
        dtypes.add(parameter.dtype)
        if len(devices) > 1 or len(dtypes) > 1:
            return

    if not devices or not dtypes:
        return

    module.to(device=next(iter(devices)), dtype=next(iter(dtypes)))


def module_parameter_dtype(module: torch.nn.Module | None) -> torch.dtype | None:
    """Return the first parameter dtype for a module."""
    if module is None:
        return None
    parameters = getattr(module, "parameters", None)
    if parameters is None:
        return None
    for parameter in parameters():
        if isinstance(parameter, torch.nn.Parameter):
            return parameter.dtype
    return None


def normalize_loaded_component_buffers(pipe: "DiffusionPipeline") -> None:
    """Normalize component runtime buffers after Diffusers ``from_pretrained``."""
    for component_name in ("text_encoder", "transformer", "vae"):
        _recast_module_to_parameter_dtype(getattr(pipe, component_name, None))
    transformer_dtype = module_parameter_dtype(getattr(pipe, "transformer", None))
    if transformer_dtype is not None:
        pipe.model_dtype = transformer_dtype
    text_encoder_dtype = module_parameter_dtype(getattr(pipe, "text_encoder", None))
    if text_encoder_dtype is not None:
        pipe.text_encoder_dtype = text_encoder_dtype


def resolve_vae_scale_factor(*, vae: AutoencoderKLQwenImage) -> int:
    block_channels = getattr(vae.config, "block_out_channels", None)
    if isinstance(block_channels, (list, tuple)) and len(block_channels) > 0:
        return 2 ** (len(block_channels) - 1)
    return 8


def resolve_patch_size(*, transformer: AnimaTransformerModel) -> int:
    import numbers

    patch_size = getattr(getattr(transformer, "config", None), "patch_size", None)
    if isinstance(patch_size, (list, tuple)) and len(patch_size) > 0:
        spatial = patch_size[-1]
        if isinstance(spatial, numbers.Integral) and int(spatial) > 0:
            return int(spatial)
    if isinstance(patch_size, numbers.Integral) and int(patch_size) > 0:
        return int(patch_size)
    return 2


def _enable_vae_method(
    vae: AutoencoderKLQwenImage,
    *,
    enabled: bool,
    method_name: str,
    unsupported_feature_name: str,
) -> None:
    if not enabled:
        return
    method = getattr(vae, method_name, None)
    if method is None:
        _warn_unsupported_vae_feature(unsupported_feature_name)
        return
    method()


def _disable_vae_method(
    vae: AutoencoderKLQwenImage,
    *,
    method_name: str,
    unsupported_feature_name: str,
) -> None:
    method = getattr(vae, method_name, None)
    if method is None:
        _warn_unsupported_vae_feature(unsupported_feature_name)
        return
    method()


def _warn_unsupported_vae_feature(feature_name: str) -> None:
    warnings.warn(f"{feature_name} is not supported by this VAE.", stacklevel=2)


def build_anima_pipeline(
    components: AnimaComponents,
    *,
    device: str = "auto",
    dtype: str = "auto",
    text_encoder_dtype: str = "auto",
    local_files_only: bool = False,
    cache_dir: str | None = None,
    force_download: bool = False,
    token: str | bool | None = None,
    revision: str | None = None,
    proxies: dict[str, str] | None = None,
    scheduler: FlowMatchEulerDiscreteScheduler | None = None,
) -> "AnimaPipeline":
    """Construct an ``AnimaPipeline`` from a raw transformer checkpoint.

    Used by ``AnimaPipeline.from_single_file``. All auxiliary components
    (text encoder, VAE, tokenizers) are loaded from fixed Anima sources.
    """
    # Import here to avoid circular imports.
    from .pipeline import AnimaPipeline

    resolved_device = resolve_device(device)
    load_options = AnimaLoaderOptions(
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        force_download=force_download,
        token=token,
        revision=revision,
        proxies=proxies,
    )
    resolved_dtype = resolve_dtype(dtype, resolved_device)
    resolved_text_encoder_dtype = resolve_text_encoder_dtype(
        model_dtype=resolved_dtype,
        text_encoder_dtype=text_encoder_dtype,
        execution_device=resolved_device,
    )
    warn_if_unsafe_fp16(resolved_device=resolved_device, resolved_dtype=resolved_dtype)
    load_device = resolved_device
    resolved_model_path = resolve_single_file_path(
        components.model_path,
        options=load_options,
        allow_remote_url=True,
    )

    transformer = load_transformer_native(
        model_path=resolved_model_path,
        device=load_device,
        dtype=resolved_dtype,
    )
    vae = load_vae(
        device=load_device,
        dtype=resolved_dtype,
        options=load_options,
    )
    text_encoder = load_text_encoder(
        device=load_device,
        dtype=resolved_text_encoder_dtype,
        options=load_options,
    )
    prompt_tokenizer = load_prompt_tokenizer(
        qwen_tokenizer_source=_QWEN_TOKENIZER_SOURCE,
        t5_tokenizer_source=_T5_TOKENIZER_SOURCE,
        options=load_options,
    )

    resolved_scheduler = scheduler
    if resolved_scheduler is None:
        resolved_scheduler = AnimaFlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
            use_dynamic_shifting=False,
        )
    else:
        resolved_scheduler = coerce_anima_scheduler(resolved_scheduler)

    runtime = AnimaPipeline(
        transformer=transformer,
        vae=vae,
        scheduler=resolved_scheduler,
        text_encoder=text_encoder,
        prompt_tokenizer=prompt_tokenizer,
        execution_device=resolved_device,
        model_dtype=resolved_dtype,
        text_encoder_dtype=resolved_text_encoder_dtype,
        use_module_cpu_offload=False,
    )
    return runtime

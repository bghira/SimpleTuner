from __future__ import annotations

import os
import sys
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import torch

from simpletuner.helpers.logging import get_logger

try:
    from diffusers.models.attention_dispatch import AttentionBackendName, _check_attention_backend_requirements
    from diffusers.models.attention_dispatch import attention_backend as diffusers_attention_backend
except Exception:  # pragma: no cover - diffusers is a hard dependency but we guard for older installs
    AttentionBackendName = None
    diffusers_attention_backend = None
    _check_attention_backend_requirements = None

if TYPE_CHECKING:
    from sparse_linear_attention import SparseLinearAttention

logger = get_logger("AttentionBackend")

_DIFFUSERS_BACKEND_TARGETS: Dict[str, str] = {
    "flash": "flash",
    "flash-attn": "flash",
    "flash_attn": "flash",
    "flash-attn-2": "flash_varlen",
    "flash_attn_2": "flash_varlen",
    "flash-attn-varlen": "flash_varlen",
    "flash_attn_varlen": "flash_varlen",
    "flash-varlen": "flash_varlen",
    "flash-attn-3": "_flash_3",
    "flash_attn_3": "_flash_3",
    "flash3": "_flash_3",
    "flash-attn-3-varlen": "_flash_varlen_3",
    "flash_attn_3_varlen": "_flash_varlen_3",
    "flash3-varlen": "_flash_varlen_3",
    "flash_attn3": "_flash_3",
    "flash_attn3_varlen": "_flash_varlen_3",
    "flex": "flex",
    "flex-attn": "flex",
    "cudnn": "_native_cudnn",
    "native-cudnn": "_native_cudnn",
    "native-efficient": "_native_efficient",
    "native-flash": "_native_flash",
    "native-math": "_native_math",
    "native-npu": "_native_npu",
    "native-xla": "_native_xla",
}

_DIFFUSERS_BACKEND_ALIASES: Dict[str, AttentionBackendName] = {}
if AttentionBackendName is not None:
    for alias, target in _DIFFUSERS_BACKEND_TARGETS.items():
        try:
            _DIFFUSERS_BACKEND_ALIASES[alias] = AttentionBackendName(target)
        except Exception:
            continue


class AttentionBackendMode(str, Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    TRAINING_AND_INFERENCE = "training+inference"

    @property
    def allows_training(self) -> bool:
        return self in (AttentionBackendMode.TRAINING, AttentionBackendMode.TRAINING_AND_INFERENCE)

    @property
    def allows_inference(self) -> bool:
        return self in (AttentionBackendMode.INFERENCE, AttentionBackendMode.TRAINING_AND_INFERENCE)

    @classmethod
    def from_raw(cls, raw_value: Any) -> "AttentionBackendMode":
        if isinstance(raw_value, cls):
            return raw_value
        if raw_value in (None, "", "None"):
            return cls.INFERENCE
        normalized = str(raw_value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported attention backend mode '{raw_value}'. Expected one of: "
                f"{', '.join(member.value for member in cls)}"
            ) from exc


class AttentionPhase(Enum):
    TRAIN = "train"
    EVAL = "eval"


def is_sageattention_available() -> bool:
    """Check if sageattention package is importable."""
    try:
        import sageattention  # noqa: F401

        return True
    except ImportError:
        return False


def xformers_compute_capability_error() -> Optional[str]:
    """Return an error message if xformers is unsupported on the current GPU, else None.

    xformers does not support compute capability 9.0+ (Hopper architecture).
    """
    if not torch.cuda.is_available():
        return None
    major, _ = torch.cuda.get_device_capability()
    if major >= 9:
        return (
            f"xformers is not supported on GPUs with compute capability 9.0+ "
            f"(detected {major}.x). Use a different attention mechanism such as 'diffusers'."
        )
    return None


SLAKey = Tuple[int, str, Optional[int], torch.dtype]


class AttentionBackendController:
    _active_backend: Optional[str] = None
    _active_phase: AttentionPhase = AttentionPhase.TRAIN
    _sla_cache: Dict[SLAKey, "SparseLinearAttention"] = {}
    _sla_settings: Dict[str, Any] | None = None
    _optimizer: Optional[Any] = None
    _parameter_sink: Optional[list] = None
    _sink_param_ids: set[int] = set()
    _optimizer_param_ids: set[int] = set()
    _attention_logit_consumer: Optional[Callable[[Dict[str, torch.Tensor]], None]] = None
    _sla_state_store: Dict[Tuple[int, str], Dict[str, torch.Tensor]] = {}
    _sla_state_filename: str = "sla_attention.pt"
    _diffusers_backend_context = None
    _diffusers_backend_name: Optional[str] = None

    @classmethod
    def apply(cls, config, phase: AttentionPhase) -> None:
        cls._active_phase = phase
        backend_value = getattr(config, "attention_mechanism", "diffusers")
        if not isinstance(backend_value, str):
            backend_value = "diffusers"
        backend = (backend_value or "diffusers").strip().lower()
        if not backend:
            backend = "diffusers"
        backend_alias = backend.replace("_", "-")

        if AttentionBackendName is None and backend_alias in _DIFFUSERS_BACKEND_TARGETS:
            message = (
                f"Attention backend '{backend_alias}' requires a newer diffusers release. "
                "Please upgrade diffusers to use this backend."
            )
            logger.error(message)
            raise RuntimeError(message)

        if cls._is_sageattention_backend(backend):
            cls._configure_sageattention(config, backend, phase)
            return

        if backend == "sla":
            cls._enable_sla(config, phase)
            return

        diffusers_backend = cls._resolve_diffusers_backend(backend_alias)
        if diffusers_backend is not None:
            cls._enable_diffusers_backend(backend_alias, diffusers_backend)
            return

        cls.restore_default()

    @classmethod
    def restore_default(cls) -> None:
        cls._disable_diffusers_backend()
        functional = torch.nn.functional
        if hasattr(functional, "scaled_dot_product_attention_sdpa"):
            functional.scaled_dot_product_attention = functional.scaled_dot_product_attention_sdpa
        cls._active_backend = None
        cls._active_phase = AttentionPhase.TRAIN

    @classmethod
    def _store_sdpa_reference(cls) -> None:
        functional = torch.nn.functional
        if not hasattr(functional, "scaled_dot_product_attention_sdpa"):
            setattr(functional, "scaled_dot_product_attention_sdpa", functional.scaled_dot_product_attention)

    @classmethod
    def _call_original_sdpa(cls, query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa):
        functional = torch.nn.functional
        original = getattr(functional, "scaled_dot_product_attention_sdpa", functional.scaled_dot_product_attention)
        return original(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

    @staticmethod
    def _normalize_backend_key(value: str) -> str:
        return value.replace("_", "-")

    @classmethod
    def _resolve_diffusers_backend(cls, backend: str) -> Optional[AttentionBackendName]:
        if AttentionBackendName is None or not _DIFFUSERS_BACKEND_ALIASES:
            return None
        normalized = cls._normalize_backend_key(backend)
        return _DIFFUSERS_BACKEND_ALIASES.get(normalized)

    @classmethod
    def _enable_diffusers_backend(cls, backend_key: str, backend_enum: AttentionBackendName) -> None:
        if cls._diffusers_backend_name == backend_key:
            return
        if diffusers_attention_backend is None or _check_attention_backend_requirements is None:
            message = f"Diffusers attention backend helpers are unavailable. Upgrade diffusers to at least 0.35 to use {backend_key}."
            logger.error(message)
            raise RuntimeError(message)

        try:
            _check_attention_backend_requirements(backend_enum)
        except Exception as exc:  # pragma: no cover - exercised only when backend requirements fail
            message = f"Attention backend '{backend_key}' is unavailable: {exc}"
            logger.error(message)
            raise RuntimeError(message) from exc

        cls._disable_diffusers_backend()
        try:
            context = diffusers_attention_backend(backend_enum)
            context.__enter__()
        except Exception as exc:
            message = f"Failed to enable attention backend '{backend_key}': {exc}"
            logger.error(message)
            raise RuntimeError(message) from exc

        cls._diffusers_backend_context = context
        cls._diffusers_backend_name = backend_key
        cls._active_backend = backend_key

    @classmethod
    def _disable_diffusers_backend(cls) -> None:
        if cls._diffusers_backend_context is None:
            return
        try:
            cls._diffusers_backend_context.__exit__(None, None, None)
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.debug("Failed to exit attention backend context cleanly (%s). Ignoring.", exc)
        finally:
            cls._diffusers_backend_context = None
            cls._diffusers_backend_name = None

    @staticmethod
    def _is_sageattention_backend(backend: str) -> bool:
        return backend.startswith("sageattention")

    @classmethod
    def _configure_sageattention(cls, config, backend: str, phase: AttentionPhase) -> None:
        cls._disable_diffusers_backend()
        usage_value = getattr(config, "sageattention_usage", AttentionBackendMode.INFERENCE)
        usage = AttentionBackendMode.from_raw(usage_value)
        setattr(config, "sageattention_usage", usage)

        should_enable = (phase == AttentionPhase.TRAIN and usage.allows_training) or (
            phase == AttentionPhase.EVAL and usage.allows_inference
        )

        if should_enable:
            if (
                phase == AttentionPhase.TRAIN
                and usage.allows_training
                and not getattr(config, "_sageattention_training_warned", False)
            ):
                logger.info("Using %s for training. This is an unsupported, experimental configuration.", backend)
                setattr(config, "_sageattention_training_warned", True)
            cls._enable_sageattention(backend, usage)
        else:
            cls.restore_default()

    @classmethod
    def _enable_sageattention(cls, backend: str, usage: AttentionBackendMode) -> None:
        if cls._active_backend == backend:
            return

        cls._disable_diffusers_backend()

        try:
            from sageattention import (
                sageattn,
                sageattn_qk_int8_pv_fp8_cuda,
                sageattn_qk_int8_pv_fp16_cuda,
                sageattn_qk_int8_pv_fp16_triton,
            )
        except ImportError as exc:
            message = f"Could not import SageAttention. Please install it to use --attention_mechanism={backend}."
            logger.error(message)
            logger.error(repr(exc))
            raise RuntimeError(message) from exc

        sageattn_functions = {
            "sageattention": sageattn,
            "sageattention-int8-fp16-triton": sageattn_qk_int8_pv_fp16_triton,
            "sageattention-int8-fp16-cuda": sageattn_qk_int8_pv_fp16_cuda,
            "sageattention-int8-fp8-cuda": sageattn_qk_int8_pv_fp8_cuda,
        }

        if backend not in sageattn_functions:
            message = f"Unsupported SageAttention backend '{backend}'."
            logger.error(message)
            raise ValueError(message)

        selected_impl = sageattn_functions[backend]
        cls._store_sdpa_reference()

        def wrapper(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
            enable_gqa=False,
        ):
            try:
                return selected_impl(query, key, value, is_causal=is_causal)
            except Exception as exc:
                logger.error(
                    "Could not run SageAttention with %s (%s). Falling back to PyTorch SDPA.",
                    backend,
                    exc,
                )
                return cls._call_original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

        torch.nn.functional.scaled_dot_product_attention = wrapper
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention_sage"):
            setattr(torch.nn.functional, "scaled_dot_product_attention_sage", wrapper)

        if usage.allows_training:
            logger.warning(
                "Using %s for attention calculations during training. Your attention layers will not be trained. "
                "To disable SageAttention, remove or set --attention_mechanism to a different value.",
                backend,
            )

        cls._active_backend = backend

    @classmethod
    def _enable_sla(cls, config, phase: AttentionPhase) -> None:
        if cls._active_backend == "sla":
            return

        cls._disable_diffusers_backend()

        try:
            from sparse_linear_attention import SparseLinearAttention
        except ImportError as exc:
            message = "Could not import SparseLinearAttention. Install it to use --attention_mechanism=sla."
            logger.error(message)
            logger.error(repr(exc))
            raise RuntimeError(message) from exc

        cls._store_sdpa_reference()
        defaults = {
            "topk": 0.2,
            "feature_map": "softmax",
            "blkq": 64,
            "blkk": 64,
            "tie_feature_map_qk": True,
        }

        user_config = getattr(config, "sla_config", None)
        if not isinstance(user_config, dict):
            user_config = {}

        def _get_value(key: str, attr_name: str, fallback, caster):
            candidate = user_config.get(key, getattr(config, attr_name, None))
            if candidate in (None, "", "None"):
                candidate = fallback
            try:
                return caster(candidate)
            except Exception as exc:
                raise ValueError(f"Invalid SLA setting '{key}': {candidate}") from exc

        def _to_bool(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)

        settings = {
            "topk": float(_get_value("topk", "sla_topk", defaults["topk"], float)),
            "feature_map": str(_get_value("feature_map", "sla_feature_map", defaults["feature_map"], str)),
            "blkq": int(_get_value("blkq", "sla_blkq", defaults["blkq"], int)),
            "blkk": int(_get_value("blkk", "sla_blkk", defaults["blkk"], int)),
            "tie_feature_map_qk": bool(
                _to_bool(
                    _get_value("tie_feature_map_qk", "sla_tie_feature_map_qk", defaults["tie_feature_map_qk"], lambda v: v)
                )
            ),
        }

        if cls._sla_settings != settings:
            cls._sla_cache.clear()
            cls._sink_param_ids.clear()
            cls._optimizer_param_ids.clear()
            cls._sla_state_store.clear()
            cls._sla_settings = settings

        def wrapper(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
            enable_gqa=False,
        ):
            if cls._sla_should_fallback(query, attn_mask, dropout_p, is_causal, enable_gqa):
                return cls._call_original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

            try:
                module = cls._get_sla_module(SparseLinearAttention, query)
            except Exception as exc:
                logger.error("Failed to prepare SLA module (%s). Falling back to PyTorch SDPA.", exc)
                return cls._call_original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

            try:
                return module(query, key, value)
            except Exception as exc:
                logger.error("Failed to execute SLA (%s). Falling back to PyTorch SDPA.", exc)
                return cls._call_original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

        torch.nn.functional.scaled_dot_product_attention = wrapper
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention_sla"):
            setattr(torch.nn.functional, "scaled_dot_product_attention_sla", wrapper)
        cls._active_backend = "sla"
        cls._active_phase = phase

    @classmethod
    def _sla_should_fallback(cls, query, attn_mask, dropout_p, is_causal, enable_gqa) -> bool:
        if query.device.type != "cuda":
            return True
        if query.ndim != 4:
            return True
        if attn_mask is not None:
            return True
        if dropout_p not in (0, 0.0, None):
            return True
        if is_causal:
            return True
        if enable_gqa:
            return True
        if query.dtype not in (torch.float16, torch.bfloat16):
            return True
        return False

    @classmethod
    def _get_sla_module(cls, module_cls, query: torch.Tensor):
        if cls._sla_settings is None:
            raise RuntimeError("SLA settings have not been initialised.")
        head_dim = query.shape[-1]
        device = query.device
        cache_key: SLAKey = (head_dim, device.type, getattr(device, "index", None), query.dtype)
        module = cls._sla_cache.get(cache_key)
        if module is None:
            module = module_cls(
                head_dim=head_dim,
                topk=cls._sla_settings["topk"],
                feature_map=cls._sla_settings["feature_map"],
                BLKQ=cls._sla_settings["blkq"],
                BLKK=cls._sla_settings["blkk"],
                use_bf16=query.dtype == torch.bfloat16,
                tie_feature_map_qk=cls._sla_settings.get("tie_feature_map_qk", True),
            )
            module.to(device)
            cls._force_proj_fp32(module)
            cls._sla_cache[cache_key] = module
            cls._apply_state_to_module(module, head_dim, query.dtype)
            cls._sync_parameter_consumers()
        module.train(cls._active_phase == AttentionPhase.TRAIN)
        return module

    @staticmethod
    def _force_proj_fp32(module) -> None:
        proj = getattr(module, "proj_l", None)
        if proj is None:
            return
        try:
            device = next(proj.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        module.proj_l = proj.to(device=device, dtype=torch.float32)

    @classmethod
    def attach_parameter_sink(cls, sink: list) -> None:
        cls._parameter_sink = sink
        cls._sync_parameter_consumers()

    @classmethod
    def bind_optimizer(cls, optimizer: Any) -> None:
        cls._optimizer = optimizer
        cls._sync_parameter_consumers()

    @classmethod
    def get_trainable_parameters(cls) -> list[torch.nn.Parameter]:
        if cls._active_backend != "sla" or not cls._sla_cache:
            return []
        params: list[torch.nn.Parameter] = []
        for module in cls._sla_cache.values():
            params.extend(param for param in module.parameters() if param.requires_grad)
        return params

    @classmethod
    def _sync_parameter_consumers(cls) -> None:
        params = cls.get_trainable_parameters()
        if not params:
            return

        if cls._parameter_sink is not None:
            sink_candidates = [param for param in params if id(param) not in cls._sink_param_ids]
            if sink_candidates:
                cls._parameter_sink.extend(sink_candidates)
                cls._sink_param_ids.update(id(param) for param in sink_candidates)

        if cls._optimizer is not None:
            opt_candidates = [param for param in params if id(param) not in cls._optimizer_param_ids]
            if opt_candidates:
                add_group = getattr(cls._optimizer, "add_param_group", None)
                if callable(add_group):
                    add_group({"params": opt_candidates})
                    cls._optimizer_param_ids.update(id(param) for param in opt_candidates)
                else:
                    logger.error(
                        "Optimizer %s does not support add_param_group; SLA parameters cannot be registered automatically.",
                        type(cls._optimizer).__name__,
                    )
        cls._update_state_store_from_cache()

    @classmethod
    def register_attention_logit_consumer(cls, consumer: Callable[[Dict[str, torch.Tensor]], None]) -> None:
        cls._attention_logit_consumer = consumer

    @classmethod
    def publish_attention_max_logits(cls, logits: Dict[str, torch.Tensor]) -> None:
        if cls._attention_logit_consumer is None:
            return
        try:
            cls._attention_logit_consumer(logits)
        except Exception:
            logger.exception("Failed to forward attention max logits to consumer.")

    @classmethod
    def lookup_param_name(cls, param: Optional[torch.nn.Parameter]) -> str:
        if param is None or cls._optimizer is None:
            return ""
        mapping = getattr(cls._optimizer, "_param_to_name", None)
        if not isinstance(mapping, dict):
            return ""
        return mapping.get(id(param), "")

    @classmethod
    def _store_key(cls, head_dim: int, dtype: torch.dtype) -> Tuple[int, str]:
        return head_dim, cls._dtype_token(dtype)

    @staticmethod
    def _dtype_token(dtype: torch.dtype) -> str:
        if dtype == torch.bfloat16:
            return "bf16"
        if dtype == torch.float16:
            return "fp16"
        if dtype == torch.float32:
            return "fp32"
        return str(dtype).replace("torch.", "")

    @staticmethod
    def _dtype_from_token(token: str) -> torch.dtype:
        mapping = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        return mapping.get(token, getattr(torch, token, torch.bfloat16))

    @classmethod
    def _update_state_store_from_cache(cls) -> None:
        if not cls._sla_cache:
            return
        for cache_key, module in cls._sla_cache.items():
            head_dim, _, _, dtype = cache_key
            store_key = cls._store_key(head_dim, dtype)
            cls._sla_state_store[store_key] = {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}

    @classmethod
    def _apply_state_to_module(cls, module, head_dim: int, dtype: torch.dtype) -> None:
        store_key = cls._store_key(head_dim, dtype)
        state_dict = cls._sla_state_store.get(store_key)
        if not state_dict:
            return
        device = next(module.parameters()).device
        mapped = {name: tensor.to(device=device) for name, tensor in state_dict.items()}
        try:
            module.load_state_dict(mapped, strict=False)
        except Exception as exc:
            logger.warning("Failed to load SLA state for head_dim=%s dtype=%s (%s)", head_dim, dtype, exc)

    @classmethod
    def _serialize_state(cls) -> Optional[Dict[str, Any]]:
        cls._update_state_store_from_cache()
        if not cls._sla_state_store:
            return None
        serialized: Dict[str, Dict[str, torch.Tensor]] = {}
        for (head_dim, dtype_token), tensor_dict in cls._sla_state_store.items():
            key = f"{head_dim}:{dtype_token}"
            serialized[key] = {name: tensor.cpu() for name, tensor in tensor_dict.items()}
        return {
            "settings": cls._sla_settings,
            "state": serialized,
            "version": 1,
        }

    @classmethod
    def _load_state_store(cls, payload: Dict[str, Any]) -> None:
        serialized = payload.get("state") or {}
        restored: Dict[Tuple[int, str], Dict[str, torch.Tensor]] = {}
        for key, tensor_dict in serialized.items():
            try:
                head_dim_str, dtype_token = key.split(":", 1)
                head_dim = int(head_dim_str)
            except ValueError:
                logger.warning("Invalid SLA state key '%s'; skipping.", key)
                continue
            restored[(head_dim, dtype_token)] = {name: tensor.cpu() for name, tensor in tensor_dict.items()}
        cls._sla_state_store = restored
        saved_settings = payload.get("settings")
        if saved_settings and cls._sla_settings and cls._sla_settings != saved_settings:
            logger.warning(
                "SLA runtime settings differ from checkpoint settings. Runtime=%s, Checkpoint=%s. Proceeding with runtime configuration.",
                cls._sla_settings,
                saved_settings,
            )
        elif saved_settings and not cls._sla_settings:
            cls._sla_settings = saved_settings
        cls._apply_store_to_cache()

    @classmethod
    def _apply_store_to_cache(cls) -> None:
        if not cls._sla_cache:
            return
        for cache_key, module in cls._sla_cache.items():
            head_dim, _, _, dtype = cache_key
            cls._apply_state_to_module(module, head_dim, dtype)

    @classmethod
    def on_save_checkpoint(cls, directory: str, *, is_main_process: bool = True) -> None:
        if not is_main_process:
            return
        payload = cls._serialize_state()
        if not payload:
            return
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, cls._sla_state_filename)
        try:
            torch.save(payload, path)
        except Exception as exc:
            logger.error("Failed to save SLA attention state to %s (%s)", path, exc)

    @classmethod
    def on_load_checkpoint(cls, directory: str) -> None:
        path = os.path.join(directory, cls._sla_state_filename)
        if not os.path.exists(path):
            return
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:
            logger.error("Failed to load SLA attention state from %s (%s)", path, exc)
            return
        if not isinstance(payload, dict):
            logger.error("Invalid SLA attention payload type: %s", type(payload))
            return
        cls._load_state_store(payload)


__all__ = [
    "AttentionBackendController",
    "AttentionBackendMode",
    "AttentionPhase",
]

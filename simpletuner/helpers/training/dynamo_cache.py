import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import struct
import sys
import tempfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 2
_BUNDLE_MAGIC = b"SIMPLETUNER_DYNAMO_MEGACACHE\x00\x02"
_CONFIG_SIGNATURE_FIELDS = (
    "model_family",
    "model_flavour",
    "model_type",
    "revision",
    "variant",
    "resolution",
    "resolution_type",
    "base_model_precision",
    "model_precision",
    "mixed_precision",
    "quantize_via",
    "quantization_config",
    "attention_mechanism",
    "dynamo_backend",
    "dynamo_mode",
    "dynamo_fullgraph",
    "dynamo_dynamic",
    "dynamo_use_regional_compilation",
    "dynamo_wrapper",
    "gradient_checkpointing",
    "gradient_checkpointing_backend",
    "gradient_checkpointing_interval",
    "gradient_checkpointing_segment_stride",
    "gradient_checkpointing_offload_attention",
    "gradient_checkpointing_offload_prefetch",
    "train_batch_size",
    "lora_type",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_target",
    "lora_format",
    "fsdp_enable",
    "fsdp_sharding_strategy",
    "musubi_blocks_to_swap",
    "ramtorch",
    "ramtorch_target_modules",
    "ramtorch_transformer_percent",
    "context_parallel_size",
    "tensor_parallel_size",
    "xm_training_target",
    "distillation_method",
)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, set):
        normalized_items = [_json_value(item) for item in value]
        return sorted(normalized_items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Enum):
        return _json_value(value.value)
    return str(value)


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_signature() -> dict[str, Any]:
    accelerator: dict[str, Any]
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        accelerator = {
            "type": "cuda",
            "name": torch.cuda.get_device_name(device_index),
            "capability": list(torch.cuda.get_device_capability(device_index)),
        }
    else:
        accelerator = {"type": "cpu"}

    return {
        "torch": torch.__version__,
        "triton": _package_version("triton"),
        "cuda": getattr(torch.version, "cuda", None),
        "hip": getattr(torch.version, "hip", None),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "accelerator": accelerator,
    }


def _software_signature() -> dict[str, str | None]:
    return {package: _package_version(package) for package in ("simpletuner", "diffusers", "accelerate", "peft")}


def _config_signature(config: Any) -> tuple[str, dict[str, Any]]:
    values = {field: _json_value(getattr(config, field, None)) for field in _CONFIG_SIGNATURE_FIELDS}
    values["torchinductor_cpp_wrapper"] = os.environ.get("TORCHINDUCTOR_CPP_WRAPPER")
    serialized = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest(), values


def _cache_keys(cache_info: Any) -> dict[str, set[str]]:
    artifacts = getattr(cache_info, "artifacts", None) or {}
    return {str(kind): {str(key) for key in keys} for kind, keys in artifacts.items()}


def _cache_counts(cache_keys: dict[str, set[str]]) -> dict[str, int]:
    return {kind: len(keys) for kind, keys in sorted(cache_keys.items())}


def _merge_cache_keys(*inventories: dict[str, set[str]]) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = {}
    for inventory in inventories:
        for kind, keys in inventory.items():
            merged.setdefault(kind, set()).update(keys)
    return merged


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _slug(value: Any, fallback: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() else "-" for character in str(value or ""))
    normalized = "-".join(part for part in normalized.split("-") if part)
    return normalized or fallback


def _pack_segments(segments: list[bytes]) -> bytes:
    payload = bytearray(_BUNDLE_MAGIC)
    payload.extend(struct.pack(">I", len(segments)))
    for segment in segments:
        payload.extend(struct.pack(">Q", len(segment)))
        payload.extend(segment)
    return bytes(payload)


def _unpack_segments(payload: bytes) -> list[bytes]:
    if not payload.startswith(_BUNDLE_MAGIC):
        return [payload]
    offset = len(_BUNDLE_MAGIC)
    if len(payload) < offset + 4:
        raise ValueError("Dynamo cache bundle is truncated before its segment count.")
    segment_count = struct.unpack_from(">I", payload, offset)[0]
    offset += 4
    segments: list[bytes] = []
    for _ in range(segment_count):
        if len(payload) < offset + 8:
            raise ValueError("Dynamo cache bundle is truncated before a segment length.")
        segment_length = struct.unpack_from(">Q", payload, offset)[0]
        offset += 8
        segment_end = offset + segment_length
        if segment_end > len(payload):
            raise ValueError("Dynamo cache bundle contains a truncated segment.")
        segments.append(payload[offset:segment_end])
        offset = segment_end
    if offset != len(payload):
        raise ValueError("Dynamo cache bundle contains trailing data.")
    if not segments:
        raise ValueError("Dynamo cache bundle contains no segments.")
    return segments


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


class DynamoCacheManager:
    """Load, grow, and optionally publish PyTorch Mega-Cache artifacts."""

    def __init__(self, config: Any):
        configured_path = getattr(config, "dynamo_cache_export", None)
        self.enabled = isinstance(configured_path, (str, os.PathLike)) and str(configured_path) not in (
            "",
            "None",
        )
        self.config = config
        configured_hub_repo = getattr(config, "dynamo_hub_repo_id", None)
        if isinstance(configured_hub_repo, str):
            self.hub_repo_id = configured_hub_repo.strip() or None
        else:
            self.hub_repo_id = None
        self.runtime_signature = _runtime_signature()
        self.config_signature, self.config_values = _config_signature(config)
        self.generated_filename = self._standard_filename()
        self.local_path, directory_mode = self._resolve_local_path(configured_path)
        self.manifest_path = Path(f"{self.local_path}.manifest.json") if self.local_path is not None else None
        self.hub_path = self._hub_path(str(configured_path), directory_mode) if self.enabled else None
        self.hub_manifest_path = f"{self.hub_path}.manifest.json" if self.hub_path else None
        self.known_keys: dict[str, set[str]] = {}
        self.base_segments: list[bytes] = []
        self.base_keys: dict[str, set[str]] = {}
        self.process_keys: dict[str, set[str]] = {}
        self.manifest: dict[str, Any] = {}
        self.loaded = False
        self.first_step_export_attempted = False

        if self.hub_repo_id and not self.enabled:
            raise ValueError("--dynamo_hub_repo_id requires --dynamo_cache_export to name the cache blob.")

    def _standard_filename(self) -> str:
        model = _slug(getattr(self.config, "model_family", None), "model")
        flavour = _slug(getattr(self.config, "model_flavour", None), "default")
        accelerator = _slug(self.runtime_signature.get("accelerator", {}).get("name"), "cpu")
        torch_version = _slug(self.runtime_signature.get("torch"), "torch")
        runtime_payload = json.dumps(self.runtime_signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
        runtime_digest = hashlib.sha256(runtime_payload).hexdigest()[:10]
        return (
            f"simpletuner-dynamo-{model}-{flavour}-{accelerator}-{torch_version}-"
            f"{runtime_digest}-{self.config_signature[:12]}.ptcache"
        )

    def _resolve_local_path(self, configured_path: Any) -> tuple[Path | None, bool]:
        if not self.enabled:
            return None, False
        raw_path = str(configured_path)
        expanded = Path(os.path.expanduser(raw_path))
        directory_mode = raw_path.endswith(("/", "\\")) or expanded.is_dir() or not expanded.suffix
        if directory_mode:
            expanded = expanded / self.generated_filename
        return expanded, directory_mode

    def _hub_path(self, configured_path: str, directory_mode: bool) -> str:
        expanded = Path(os.path.expanduser(configured_path))
        if expanded.is_absolute():
            return self.generated_filename if directory_mode else expanded.name
        candidate = PurePosixPath(configured_path.replace("\\", "/"))
        if ".." in candidate.parts:
            raise ValueError("--dynamo_cache_export cannot contain '..' when used as a Hub artifact path.")
        normalized = str(candidate).lstrip("./")
        if not normalized:
            raise ValueError("--dynamo_cache_export must name a file.")
        if directory_mode:
            normalized = str(PurePosixPath(normalized) / self.generated_filename)
        return normalized

    @staticmethod
    def _hub_token() -> str | None:
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    def _runtime_compatible(self, manifest: dict[str, Any]) -> bool:
        expected = manifest.get("runtime")
        return not expected or expected == self.runtime_signature

    def _read_local_source(self) -> tuple[bytes, dict[str, Any]] | None:
        if self.local_path is None or not self.local_path.is_file():
            return None
        manifest: dict[str, Any] = {}
        if self.manifest_path is not None and self.manifest_path.is_file():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return self.local_path.read_bytes(), manifest

    def _read_hub_source(self) -> tuple[bytes, dict[str, Any]] | None:
        if self.hub_repo_id is None or self.hub_path is None:
            return None
        api = HfApi(token=self._hub_token())
        if not api.file_exists(self.hub_repo_id, self.hub_path, token=self._hub_token()):
            return None
        blob_path = hf_hub_download(
            repo_id=self.hub_repo_id,
            filename=self.hub_path,
            token=self._hub_token(),
        )
        manifest: dict[str, Any] = {}
        if self.hub_manifest_path and api.file_exists(
            self.hub_repo_id,
            self.hub_manifest_path,
            token=self._hub_token(),
        ):
            downloaded_manifest = hf_hub_download(
                repo_id=self.hub_repo_id,
                filename=self.hub_manifest_path,
                token=self._hub_token(),
            )
            manifest = json.loads(Path(downloaded_manifest).read_text(encoding="utf-8"))
        return Path(blob_path).read_bytes(), manifest

    def load(self) -> bool:
        if not self.enabled:
            return False
        load_cache_artifacts = getattr(torch.compiler, "load_cache_artifacts", None)
        if load_cache_artifacts is None:
            logger.warning("This PyTorch build does not support Mega-Cache loading; continuing without a Dynamo cache.")
            return False

        sources: list[tuple[str, tuple[bytes, dict[str, Any]]]] = []
        if self.hub_repo_id is not None:
            try:
                hub_source = self._read_hub_source()
                if hub_source is not None:
                    sources.append((f"{self.hub_repo_id}/{self.hub_path}", hub_source))
            except Exception as exc:
                logger.warning("Unable to retrieve Dynamo cache from %s: %s", self.hub_repo_id, exc)
        try:
            local_source = self._read_local_source()
            if local_source is not None:
                sources.append((str(self.local_path), local_source))
        except Exception as exc:
            logger.warning("Unable to read local Dynamo cache %s: %s", self.local_path, exc)
        if not sources:
            logger.info(
                "No existing Dynamo Mega-Cache found at %s; it will be exported after compilation.",
                self.local_path,
            )
            return False

        for source_name, (payload, manifest) in sources:
            schema_version = manifest.get("schema_version") if manifest else None
            if schema_version is not None and schema_version > _SCHEMA_VERSION:
                logger.warning(
                    "Dynamo cache %s uses unsupported manifest schema %s; ignoring it.",
                    source_name,
                    schema_version,
                )
                continue
            if manifest and not self._runtime_compatible(manifest):
                logger.warning(
                    "Dynamo cache %s was created for a different PyTorch/Triton/device runtime; ignoring it. "
                    "Use a runtime-specific cache path to preserve both variants.",
                    source_name,
                )
                continue
            expected_hash = manifest.get("sha256") if manifest else None
            if expected_hash and expected_hash != _sha256(payload):
                logger.warning("Dynamo cache %s failed its SHA256 check; ignoring it.", source_name)
                continue
            if not manifest:
                logger.warning(
                    "Loading explicitly configured Dynamo cache %s without a compatibility manifest.",
                    source_name,
                )

            try:
                segments = _unpack_segments(payload)
                loaded_inventories = []
                for segment in segments:
                    cache_info = load_cache_artifacts(segment)
                    if cache_info is None:
                        raise ValueError("PyTorch returned no cache inventory for a bundle segment.")
                    loaded_inventories.append(_cache_keys(cache_info))
            except Exception as exc:
                logger.warning("PyTorch rejected Dynamo cache %s: %s", source_name, exc)
                continue
            self.base_segments = segments
            self.base_keys = _merge_cache_keys(*loaded_inventories)
            self.known_keys = _merge_cache_keys(self.base_keys)
            self.process_keys = {}
            self.manifest = manifest
            self.loaded = True
            logger.info("Loaded Dynamo Mega-Cache %s with artifacts: %s", source_name, _cache_counts(self.known_keys))
            return True
        return False

    def _build_manifest(
        self,
        payload: bytes,
        cache_keys: dict[str, set[str]],
        segment_count: int,
    ) -> dict[str, Any]:
        config_signatures = dict(self.manifest.get("config_signatures") or {})
        config_signatures[self.config_signature] = self.config_values
        return {
            "schema_version": _SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sha256": _sha256(payload),
            "size_bytes": len(payload),
            "segment_count": segment_count,
            "runtime": self.runtime_signature,
            "software": _software_signature(),
            "artifact_counts": _cache_counts(cache_keys),
            "config_signatures": config_signatures,
        }

    def _upload(self, payload: bytes, manifest_payload: bytes, reason: str) -> None:
        if self.hub_repo_id is None or self.hub_path is None or self.hub_manifest_path is None:
            return
        token = self._hub_token()
        api = HfApi(token=token)
        api.create_repo(repo_id=self.hub_repo_id, token=token, exist_ok=True, private=True)
        api.create_commit(
            repo_id=self.hub_repo_id,
            token=token,
            commit_message=f"Update SimpleTuner Dynamo cache ({reason})",
            operations=[
                CommitOperationAdd(path_in_repo=self.hub_path, path_or_fileobj=payload),
                CommitOperationAdd(path_in_repo=self.hub_manifest_path, path_or_fileobj=manifest_payload),
            ],
        )

    def export(self, reason: str) -> bool:
        if not self.enabled or self.local_path is None or self.manifest_path is None:
            return False
        save_cache_artifacts = getattr(torch.compiler, "save_cache_artifacts", None)
        if save_cache_artifacts is None:
            logger.warning("This PyTorch build does not support Mega-Cache export.")
            return False
        try:
            result = save_cache_artifacts()
        except Exception as exc:
            logger.warning("Unable to serialize the Dynamo Mega-Cache: %s", exc)
            return False
        if result is None:
            logger.info("PyTorch produced no Dynamo cache artifacts to export during %s.", reason)
            return False

        process_payload, cache_info = result
        process_keys = _cache_keys(cache_info)
        new_keys = {
            kind: keys - self.known_keys.get(kind, set())
            for kind, keys in process_keys.items()
            if keys - self.known_keys.get(kind, set())
        }
        if not new_keys:
            logger.info("Dynamo Mega-Cache has no new artifacts during %s; skipping re-export.", reason)
            return False
        cache_keys = _merge_cache_keys(self.base_keys, process_keys)
        segments = [*self.base_segments, process_payload]
        payload = _pack_segments(segments)
        manifest = self._build_manifest(payload, cache_keys, len(segments))
        manifest_payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        try:
            _atomic_write(self.local_path, payload)
            _atomic_write(self.manifest_path, manifest_payload)
        except Exception as exc:
            logger.warning("Unable to write Dynamo Mega-Cache to %s: %s", self.local_path, exc)
            return False

        self.known_keys = cache_keys
        self.process_keys = process_keys
        self.manifest = manifest
        logger.info(
            "Exported Dynamo Mega-Cache to %s with new artifacts %s.",
            self.local_path,
            _cache_counts(new_keys),
        )
        if self.hub_repo_id is not None:
            try:
                self._upload(payload, manifest_payload, reason)
                logger.info("Published Dynamo Mega-Cache to %s/%s.", self.hub_repo_id, self.hub_path)
            except Exception as exc:
                logger.warning(
                    "Unable to publish Dynamo Mega-Cache to %s: %s. The local export remains available.",
                    self.hub_repo_id,
                    exc,
                )
        return True

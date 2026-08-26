"""Audio identity transfer dataset transform plumbing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from simpletuner.helpers.data_transforms.base import DataTransformTask, register_data_transform
from simpletuner.helpers.training.multi_process import should_log

logger = logging.getLogger(__name__)
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel(logging.ERROR)


VOICE_TRANSFORM_FORMAT = "simpletuner-voice-transform"
VOICE_TRANSFORM_FORMAT_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> str:
    return repr(value)


def _stable_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=_json_default)


def _sha256_json(data: Dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json(data).encode("utf-8")).hexdigest()


def _get_arg_value(args: Any, key: str, default: Any = None) -> Any:
    if isinstance(args, dict):
        return args.get(key, default)
    return getattr(args, key, default)


class RVCTransformLogger:
    """Small local JSON logger for startup voice-transform work."""

    def __init__(self, output_dir: str, accelerator: Any = None) -> None:
        self.accelerator = accelerator
        self.enabled = accelerator is None or bool(getattr(accelerator, "is_main_process", True))
        self.log_dir = Path(output_dir) / "logs" / "rvc"
        self.events_path = self.log_dir / "training_stats.jsonl"
        self.summary_path = self.log_dir / "summary.json"

    def event(self, transform_id: str, event: str, **payload: Any) -> None:
        if not self.enabled:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": _utc_now(),
            "transform_id": transform_id,
            "event": event,
            **payload,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")

    def summary(self, transform_id: str, **payload: Any) -> None:
        if not self.enabled:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": _utc_now(),
            "transform_id": transform_id,
            **payload,
        }
        tmp_path = self.summary_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
        tmp_path.replace(self.summary_path)


@dataclass(frozen=True)
class VoiceModelArtifact:
    cache_dir: Path
    manifest_path: Path
    model_path: Path
    index_path: Optional[Path]
    manifest: Dict[str, Any]


class RVCTrainer:
    def train(
        self,
        source_backend_config: Dict[str, Any],
        transform_config: Dict[str, Any],
        cache_dir: Path,
        fingerprint: str,
        manifest_base: Dict[str, Any],
        accelerator: Any = None,
        logger: Optional[RVCTransformLogger] = None,
    ) -> VoiceModelArtifact:
        from simpletuner.helpers.rvc.simple import SimpleRVCTrainer

        artifact = SimpleRVCTrainer().train(
            source_backend_config=source_backend_config,
            transform_config=transform_config,
            cache_dir=cache_dir,
            fingerprint=fingerprint,
            manifest_base=manifest_base,
            accelerator=accelerator,
            run_logger=logger,
        )
        return VoiceModelArtifact(
            cache_dir=artifact.cache_dir,
            manifest_path=artifact.manifest_path,
            model_path=artifact.model_path,
            index_path=artifact.index_path,
            manifest=artifact.manifest,
        )


class RVCConverter:
    def convert(
        self,
        source_backend_config: Dict[str, Any],
        target_backend_config: Dict[str, Any],
        transform_config: Dict[str, Any],
        artifact: VoiceModelArtifact,
        input_paths: List[str],
        accelerator: Any = None,
        logger: Optional[RVCTransformLogger] = None,
    ) -> None:
        from simpletuner.helpers.rvc.simple import SimpleRVCArtifact, SimpleRVCConverter

        simple_artifact = SimpleRVCArtifact(
            cache_dir=artifact.cache_dir,
            manifest_path=artifact.manifest_path,
            model_path=artifact.model_path,
            index_path=artifact.index_path,
            manifest=artifact.manifest,
        )
        SimpleRVCConverter().convert(
            source_backend_config=source_backend_config,
            target_backend_config=target_backend_config,
            transform_config=transform_config,
            artifact=simple_artifact,
            input_paths=input_paths,
            accelerator=accelerator,
            run_logger=logger,
        )


class HubVoiceModelCache:
    def __init__(self, hub_model_id: str, token: Optional[str] = None, public: bool = False) -> None:
        self.hub_model_id = hub_model_id
        self.token = token
        self.public = public

    def download_if_compatible(self, cache_dir: Path, fingerprint: str) -> Optional[VoiceModelArtifact]:
        try:
            from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
            from huggingface_hub_rvc import RVCPipeline
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub and huggingface_hub_rvc are required when identity_transfer.reuse_from_hub is enabled."
            ) from exc

        try:
            pipeline = RVCPipeline.from_pretrained(
                self.hub_model_id,
                token=self.token,
            )
        except (EntryNotFoundError, RepositoryNotFoundError):
            return None

        manifest = pipeline.artifact.manifest
        if not _manifest_matches(manifest, fingerprint):
            return None

        pipeline.save_pretrained(cache_dir)
        return _voice_model_artifact_from_hub_pipeline(RVCPipeline.from_pretrained(cache_dir, local_files_only=True))

    def upload(self, artifact: VoiceModelArtifact) -> None:
        try:
            from huggingface_hub_rvc import RVCConfig, RVCPipeline
            from huggingface_hub_rvc._runtime import SimpleRVCArtifact
        except ImportError as exc:
            raise ImportError("huggingface_hub_rvc is required when identity_transfer.push_to_hub is enabled.") from exc

        simple_artifact = SimpleRVCArtifact(
            cache_dir=artifact.cache_dir,
            manifest_path=artifact.manifest_path,
            model_path=artifact.model_path,
            index_path=artifact.index_path,
            manifest=artifact.manifest,
        )
        pipeline = RVCPipeline(
            artifact=simple_artifact,
            config=RVCConfig(model_name=_voice_model_name(artifact.manifest, self.hub_model_id)),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline.save_pretrained(
                temp_dir,
                push_to_hub=True,
                repo_id=self.hub_model_id,
                token=self.token,
                private=not self.public,
            )


def _manifest_matches(manifest: Dict[str, Any], fingerprint: str) -> bool:
    return (
        manifest.get("format") == VOICE_TRANSFORM_FORMAT
        and manifest.get("format_version") == VOICE_TRANSFORM_FORMAT_VERSION
        and manifest.get("task") == "identity_transfer"
        and manifest.get("method") == "rvc"
        and manifest.get("fingerprint") == fingerprint
    )


def _voice_model_artifact_from_hub_pipeline(pipeline: Any) -> VoiceModelArtifact:
    artifact = pipeline.artifact
    return VoiceModelArtifact(
        cache_dir=artifact.cache_dir,
        manifest_path=artifact.manifest_path,
        model_path=artifact.model_path,
        index_path=artifact.index_path,
        manifest=artifact.manifest,
    )


def _voice_model_name(manifest: Dict[str, Any], hub_model_id: Optional[str] = None) -> str:
    voice_model = manifest.get("voice_model") or {}
    name = manifest.get("model_name") or voice_model.get("model_name")
    if name:
        return str(name)
    if hub_model_id:
        return hub_model_id.rstrip("/").rsplit("/", 1)[-1]
    return "RVC Voice Model"


@register_data_transform
class IdentityTransferTransform(DataTransformTask):
    TASK = "identity_transfer"
    SUPPORTED_SOURCE_DATASET_TYPES = ("audio",)
    REQUIRES_METADATA_CLONE = False

    def prepare(self, existing_backend_ids: set[str]) -> List[Dict[str, Any]]:
        source_id = self.source_backend_config.get("id")
        if not source_id:
            raise ValueError("identity_transfer requires the source backend to have an id.")

        transform = self._normalise_transform_config(existing_backend_ids)
        transform_id = transform["id"]
        output_dir = self._output_dir()
        model_cache_dir = Path(transform["model"]["cache_dir"])
        generated_dir = Path(transform["target"]["instance_data_dir"])
        generated_fingerprint = self._generated_fingerprint(transform)
        voice_model_fingerprint = self._voice_model_fingerprint(transform)
        run_logger = RVCTransformLogger(output_dir, accelerator=self.accelerator)

        target_backend_config = self._target_backend_config(transform, generated_dir)
        if self._generated_cache_matches(generated_dir, generated_fingerprint):
            run_logger.event(transform_id, "generated_cache_reused", path=str(generated_dir))
            run_logger.summary(
                transform_id, status="reused_generated_cache", generated_backend_id=target_backend_config["id"]
            )
            return [target_backend_config]

        artifact = self._resolve_voice_model(transform, voice_model_fingerprint, model_cache_dir, run_logger)
        if not self._is_main_process():
            self._wait_for_everyone()
            run_logger.summary(
                transform_id, status="waiting_for_main_process", generated_backend_id=target_backend_config["id"]
            )
            return [target_backend_config]

        input_paths = self._rank_shard(self._discover_source_audio_paths())
        run_logger.event(
            transform_id,
            "conversion_start",
            source_count=len(input_paths),
            generated_path=str(generated_dir),
            world_size=self._world_size(),
        )
        RVCConverter().convert(
            source_backend_config=self.source_backend_config,
            target_backend_config=target_backend_config,
            transform_config=transform,
            artifact=artifact,
            input_paths=input_paths,
            accelerator=self.accelerator,
            logger=run_logger,
        )
        self._write_generated_manifest(generated_dir, generated_fingerprint, transform)
        self._wait_for_everyone()
        run_logger.summary(transform_id, status="generated", generated_backend_id=target_backend_config["id"])
        return [target_backend_config]

    def _normalise_transform_config(self, existing_backend_ids: set[str]) -> Dict[str, Any]:
        source_id = self.source_backend_config["id"]
        transform = deepcopy(self.transform_config)
        transform.setdefault("task", self.TASK)
        transform.setdefault("method", "rvc")
        if transform["method"] != "rvc":
            raise ValueError("identity_transfer currently supports method='rvc' only.")

        transform_id = transform.get("id") or f"{source_id}_identity_transfer"
        if transform_id in existing_backend_ids:
            raise ValueError(f"identity_transfer generated backend id {transform_id!r} already exists.")
        transform["id"] = transform_id

        output_dir = self._output_dir()
        transform_root = Path(output_dir) / "cache" / "data_transforms" / transform_id
        model = deepcopy(transform.get("model") or {})
        model.setdefault("train_if_missing", True)
        model.setdefault("force_retrain", False)
        model.setdefault("build_index", True)
        model.setdefault("reuse_from_hub", bool(model.get("hub_model_id")))
        model.setdefault("push_to_hub", False)
        model.setdefault("public", False)
        model.setdefault("cache_dir", str(transform_root / "rvc_model"))
        model.setdefault("asset_hub_model_id", "lj1995/VoiceConversionWebUI")
        model.setdefault("model_name", _voice_model_name({}, model.get("hub_model_id") or transform_id))
        model.setdefault("sample_rate", 48000)
        model.setdefault("identity_audio_mode", "separate")
        model.setdefault("training_steps", 1000)
        model.setdefault("batch_size", 4)
        model.setdefault("learning_rate", 1e-4)
        model.setdefault("max_seconds_per_file", 180.0)
        transform["model"] = model

        conversion = deepcopy(transform.get("conversion") or {})
        conversion.setdefault("audio_mode", "separate_convert_remix")
        conversion.setdefault("separation_method", "demucs")
        conversion.setdefault("timbre_strength", 1.0)
        conversion.setdefault("retrieval_strength", 0.75)
        transform["conversion"] = conversion

        target = deepcopy(transform.get("target") or {})
        target.setdefault("id", transform_id)
        target.setdefault("type", "local")
        target.setdefault("dataset_type", "audio")
        target.setdefault("metadata_backend", "discovery")
        target.setdefault("caption_strategy", "textfile")
        target.setdefault("instance_data_dir", str(transform_root / "generated_audio"))
        transform["target"] = target
        return transform

    def _target_backend_config(self, transform: Dict[str, Any], generated_dir: Path) -> Dict[str, Any]:
        source_cfg = deepcopy(self.source_backend_config)
        target = deepcopy(transform["target"])

        target_cfg: Dict[str, Any] = {
            "id": target["id"],
            "type": target["type"],
            "dataset_type": "audio",
            "metadata_backend": target["metadata_backend"],
            "caption_strategy": target["caption_strategy"],
            "instance_data_dir": str(generated_dir),
            "generated_by": "data_transforms",
            "data_transform_task": self.TASK,
            "source_dataset_id": self.source_backend_config["id"],
            "data_transform_source_dataset_id": self.source_backend_config["id"],
            "data_transform_config": transform,
        }
        for key in (
            "audio",
            "audio_column",
            "huggingface",
            "parquet",
            "csv",
            "cache_dir_vae",
            "minimum_image_size",
            "repeats",
            "train_batch_size",
        ):
            if key in source_cfg:
                target_cfg[key] = deepcopy(source_cfg[key])
        target_cfg.update({key: value for key, value in target.items() if key not in {"id", "type", "dataset_type"}})
        target_cfg["dataset_type"] = "audio"
        target_cfg["instance_data_dir"] = str(generated_dir)
        target_audio = deepcopy(source_cfg.get("audio") or {})
        target_audio.update(deepcopy(target.get("audio") or {}))
        if target_audio:
            target_cfg["audio"] = target_audio
        return target_cfg

    def _resolve_voice_model(
        self,
        transform: Dict[str, Any],
        fingerprint: str,
        model_cache_dir: Path,
        run_logger: RVCTransformLogger,
    ) -> VoiceModelArtifact:
        transform_id = transform["id"]
        local_artifact = self._local_artifact(model_cache_dir, fingerprint)
        if local_artifact and not transform["model"].get("force_retrain", False):
            run_logger.event(transform_id, "voice_model_reused", source="local", path=str(model_cache_dir))
            return local_artifact

        model_cfg = transform["model"]
        hub_model_id = model_cfg.get("hub_model_id")
        if hub_model_id and model_cfg.get("reuse_from_hub", True) and not model_cfg.get("force_retrain", False):
            artifact = HubVoiceModelCache(hub_model_id, token=model_cfg.get("hub_token")).download_if_compatible(
                model_cache_dir,
                fingerprint,
            )
            if artifact:
                run_logger.event(transform_id, "voice_model_reused", source="hub", hub_model_id=hub_model_id)
                return artifact

        if not model_cfg.get("train_if_missing", False):
            raise ValueError(
                "identity_transfer could not find a compatible local or Hub voice artifact, and "
                "model.train_if_missing is false."
            )

        run_logger.event(transform_id, "voice_model_training_start", world_size=self._world_size())
        artifact = RVCTrainer().train(
            source_backend_config=self.source_backend_config,
            transform_config=transform,
            cache_dir=model_cache_dir,
            fingerprint=fingerprint,
            manifest_base=self._manifest(fingerprint, transform),
            accelerator=self.accelerator,
            logger=run_logger,
        )
        if model_cfg.get("push_to_hub", False):
            if not hub_model_id:
                raise ValueError("identity_transfer.model.push_to_hub requires identity_transfer.model.hub_model_id.")
            HubVoiceModelCache(
                hub_model_id,
                token=model_cfg.get("hub_token"),
                public=bool(model_cfg.get("public", False)),
            ).upload(artifact)
            run_logger.event(transform_id, "voice_model_pushed", hub_model_id=hub_model_id)
        return artifact

    def _local_artifact(self, cache_dir: Path, fingerprint: str) -> Optional[VoiceModelArtifact]:
        artifact_dir = cache_dir / "voice_transform" if (cache_dir / "voice_transform").exists() else cache_dir
        manifest_path = artifact_dir / "manifest.json"
        model_path = artifact_dir / "model.safetensors"
        if not model_path.exists():
            model_path = artifact_dir / "model.pth"
        index_path = artifact_dir / "index.index"
        if not manifest_path.exists() or not model_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not _manifest_matches(manifest, fingerprint):
            return None
        return VoiceModelArtifact(
            cache_dir=artifact_dir,
            manifest_path=manifest_path,
            model_path=model_path,
            index_path=index_path if index_path.exists() else None,
            manifest=manifest,
        )

    def _generated_cache_matches(self, generated_dir: Path, fingerprint: str) -> bool:
        manifest_path = generated_dir / ".simpletuner_identity_transfer.json"
        if not manifest_path.exists() or not generated_dir.exists():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return _manifest_matches(manifest, fingerprint)

    def _write_generated_manifest(self, generated_dir: Path, fingerprint: str, transform: Dict[str, Any]) -> None:
        generated_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._manifest(fingerprint, transform, source_dataset_id=self.source_backend_config["id"])
        (generated_dir / ".simpletuner_identity_transfer.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _manifest(
        self,
        fingerprint: str,
        transform: Dict[str, Any],
        source_dataset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        manifest = {
            "format": VOICE_TRANSFORM_FORMAT,
            "format_version": VOICE_TRANSFORM_FORMAT_VERSION,
            "task": self.TASK,
            "method": "rvc",
            "fingerprint": fingerprint,
            "transform_id": transform["id"],
            "created_by": "simpletuner",
            "created_at": _utc_now(),
        }
        if source_dataset_id is not None:
            manifest["source_dataset_id"] = source_dataset_id
        return manifest

    def _generated_fingerprint(self, transform: Dict[str, Any]) -> str:
        source_cfg = deepcopy(self.source_backend_config)
        source_cfg.pop("data_transforms", None)
        transform_for_hash = deepcopy(transform)
        model_cfg = transform_for_hash.get("model") or {}
        for transient_key in (
            "asset_hub_token",
            "cache_dir",
            "device",
            "demucs_device",
            "force_retrain",
            "hub_model_id",
            "hub_token",
            "model_name",
            "push_to_hub",
            "public",
            "reuse_from_hub",
            "train_if_missing",
        ):
            model_cfg.pop(transient_key, None)
        model_cfg["voice_model_fingerprint"] = self._voice_model_fingerprint(transform)
        transform_for_hash["model"] = model_cfg
        return _sha256_json(
            {
                "source_backend": source_cfg,
                "transform": transform_for_hash,
                "format_version": VOICE_TRANSFORM_FORMAT_VERSION,
            }
        )

    def _voice_model_fingerprint(self, transform: Dict[str, Any]) -> str:
        model_cfg = deepcopy(transform.get("model") or {})
        for transient_key in (
            "asset_hub_token",
            "cache_dir",
            "device",
            "demucs_device",
            "force_retrain",
            "hub_model_id",
            "hub_token",
            "model_name",
            "push_to_hub",
            "public",
            "reuse_from_hub",
            "train_if_missing",
        ):
            model_cfg.pop(transient_key, None)
        return _sha256_json(
            {
                "method": transform.get("method", "rvc"),
                "model": model_cfg,
                "format_version": VOICE_TRANSFORM_FORMAT_VERSION,
            }
        )

    def _fingerprint(self, transform: Dict[str, Any]) -> str:
        return self._generated_fingerprint(transform)

    def _discover_source_audio_paths(self) -> List[str]:
        source_type = self.source_backend_config.get("type")
        if source_type != "local":
            raise NotImplementedError(
                "identity_transfer input discovery is currently implemented for local audio source backends only."
            )
        instance_data_dir = self.source_backend_config.get("instance_data_dir")
        if not instance_data_dir:
            raise ValueError("identity_transfer local source backend requires instance_data_dir.")
        audio_exts = {".flac", ".wav", ".mp3", ".ogg", ".m4a", ".aac", ".opus"}
        root = Path(instance_data_dir)
        if not root.exists():
            raise FileNotFoundError(f"identity_transfer source instance_data_dir does not exist: {instance_data_dir}")
        paths = [str(path) for path in sorted(root.rglob("*")) if path.suffix.lower() in audio_exts]
        if not paths:
            raise ValueError(f"identity_transfer found no audio files under {instance_data_dir}.")
        return paths

    def _rank_shard(self, items: Iterable[str]) -> List[str]:
        rank = int(getattr(self.accelerator, "process_index", 0) or 0)
        world_size = self._world_size()
        return [item for idx, item in enumerate(items) if idx % world_size == rank]

    def _world_size(self) -> int:
        return int(getattr(self.accelerator, "num_processes", 1) or 1)

    def _is_main_process(self) -> bool:
        return self.accelerator is None or bool(getattr(self.accelerator, "is_main_process", True))

    def _wait_for_everyone(self) -> None:
        if self.accelerator is not None and hasattr(self.accelerator, "wait_for_everyone"):
            self.accelerator.wait_for_everyone()

    def _output_dir(self) -> str:
        output_dir = _get_arg_value(self.global_config, "output_dir", None)
        if output_dir:
            return str(output_dir)
        return os.path.join(os.getcwd(), ".simpletuner_output")

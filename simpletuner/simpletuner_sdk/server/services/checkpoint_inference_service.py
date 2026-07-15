"""Checkpoint inference orchestration for the WebUI."""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import status

from simpletuner.helpers.prompts import prompts as built_in_prompts
from simpletuner.helpers.training.validation import parse_validation_resolutions
from simpletuner.helpers.utils.checkpoint_manager import CheckpointManager
from simpletuner.simpletuner_sdk import process_keeper
from simpletuner.simpletuner_sdk.api_state import APIState
from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
from simpletuner.simpletuner_sdk.server.services.field_registry import field_registry
from simpletuner.simpletuner_sdk.server.services.prompt_library_service import PromptLibraryError, PromptLibraryService
from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore


class CheckpointInferenceServiceError(Exception):
    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class CheckpointInferenceService:
    FILENAME_STYLES = {"descriptive", "compact", "prompt", "content-hash"}
    MEDIA_SUFFIXES = {".png", ".mp4"}

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, dict[str, str]] = {}
        self._environment_sessions: dict[str, str] = {}

    def _get_config_store(self) -> ConfigStore:
        defaults = WebUIStateStore().load_defaults()
        if defaults.configs_dir:
            return ConfigStore(config_dir=Path(defaults.configs_dir).expanduser(), config_type="model")
        return ConfigStore(config_type="model")

    def _load_environment(self, environment: str) -> tuple[dict[str, Any], Path]:
        try:
            config, _ = self._get_config_store().load_config(environment)
        except FileNotFoundError as exc:
            raise CheckpointInferenceServiceError(
                f"Environment '{environment}' not found.", status.HTTP_404_NOT_FOUND
            ) from exc
        output_dir = config.get("--output_dir") or config.get("output_dir")
        if not output_dir:
            raise CheckpointInferenceServiceError(f"Environment '{environment}' has no output directory configured.")
        return config, Path(os.path.expanduser(str(output_dir))).resolve()

    @staticmethod
    def _config_value(config: dict[str, Any], name: str, default=None):
        return config.get(f"--{name}", config.get(name, default))

    @staticmethod
    def _config_value_with_registry_default(config: dict[str, Any], name: str) -> Any:
        value = CheckpointInferenceService._config_value(config, name)
        if value is not None:
            return value
        field = field_registry.get_field(name)
        if field is None:
            raise RuntimeError(f"Configuration field '{name}' is not registered.")
        return field.default_value

    @classmethod
    def _unsupported_multigpu_modes(cls, config: dict[str, Any]) -> list[str]:
        modes = []
        if cls._config_value(config, "validation_multigpu") == "batch-parallel":
            modes.append("batch-parallel")

        context_parallel_size = cls._config_value(config, "context_parallel_size")
        if context_parallel_size not in (None, ""):
            try:
                context_parallel_enabled = int(context_parallel_size) > 1
            except (TypeError, ValueError) as exc:
                raise CheckpointInferenceServiceError("context_parallel_size must be an integer.") from exc
            if context_parallel_enabled:
                modes.append("context-parallel")
        return modes

    @classmethod
    def _inference_resolutions(cls, config: dict[str, Any], settings: dict[str, Any]) -> list[tuple[int, int]]:
        value = settings.get("validation_resolution")
        if value is None:
            value = cls._config_value_with_registry_default(config, "validation_resolution")
        try:
            return parse_validation_resolutions(
                value,
                model_flavour=cls._config_value(config, "model_flavour"),
            )
        except (TypeError, ValueError) as exc:
            raise CheckpointInferenceServiceError(str(exc)) from exc

    @staticmethod
    def _write_session_state(path: Path, state: dict[str, Any]) -> None:
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)

    def _training_is_active(self) -> bool:
        job_id = APIState.get_state("current_job_id")
        if not job_id:
            return False
        return process_keeper.get_process_status(job_id) in {"pending", "running", "aborting"}

    def active_session(self) -> dict[str, str] | None:
        active = None
        with self._lock:
            for session_id, record in list(self._sessions.items()):
                process_status = process_keeper.get_process_status(record["job_id"])
                if process_status in {"pending", "running", "aborting"}:
                    if active is None:
                        active = record
                    continue
                self._discard_session(session_id)
        return active

    def _discard_session(self, session_id: str) -> None:
        with self._lock:
            record = self._sessions.pop(session_id, None)
            if record and self._environment_sessions.get(record["environment"]) == session_id:
                self._environment_sessions.pop(record["environment"], None)

    @classmethod
    def _is_generated_media_path(cls, relative: Path) -> bool:
        return (
            not relative.is_absolute()
            and len(relative.parts) == 3
            and not any(part.startswith(".") for part in relative.parts)
            and relative.suffix.lower() in cls.MEDIA_SUFFIXES
        )

    @staticmethod
    def _is_preview_media_path(relative: Path) -> bool:
        return (
            not relative.is_absolute()
            and len(relative.parts) == 2
            and not relative.parts[0].startswith(".")
            and relative.parts[1] == "preview.png"
        )

    @staticmethod
    def _normalise_entry(shortname: str, value: Any, source: str) -> dict[str, Any]:
        if isinstance(value, str):
            payload: dict[str, Any] = {"prompt": value}
        elif isinstance(value, dict):
            payload = dict(value)
        else:
            raise CheckpointInferenceServiceError(f"Prompt '{shortname}' has an unsupported value.")
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise CheckpointInferenceServiceError(f"Prompt '{shortname}' is empty.")
        return {
            "shortname": shortname,
            "prompt": prompt,
            "source": source,
            **{key: payload[key] for key in ("adapter_strength", "bbox_entities", "bbox_keyframes") if key in payload},
        }

    def resolve_prompts(
        self,
        *,
        config: dict[str, Any],
        use_configured_prompt: bool,
        use_builtin_library: bool,
        user_library_filename: str | None,
        custom_prompts: list[str],
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        seen_shortnames: set[str] = set()

        def append(shortname: str, value: Any, source: str) -> None:
            candidate = shortname
            suffix = 2
            while candidate in seen_shortnames:
                candidate = f"{shortname}_{suffix}"
                suffix += 1
            seen_shortnames.add(candidate)
            entries.append(self._normalise_entry(candidate, value, source))

        if use_configured_prompt:
            configured = self._config_value(config, "validation_prompt")
            if configured not in (None, "", "None"):
                append("configured", str(configured), "configured")

        if use_builtin_library:
            for shortname, prompt in built_in_prompts.items():
                append(f"builtin_{shortname}", prompt, "builtin")

        if user_library_filename:
            try:
                configured_library = self._config_value(config, "user_prompt_library")
                configured_path = Path(str(configured_library)).expanduser() if configured_library else None
                if (
                    configured_path is not None
                    and configured_path.name == Path(user_library_filename).name
                    and configured_path.is_file()
                ):
                    with configured_path.open("r", encoding="utf-8") as handle:
                        parsed_entries = PromptLibraryService.parse_entries(json.load(handle))
                    library = {"entries": PromptLibraryService.serialise_entries(parsed_entries)}
                else:
                    library = PromptLibraryService().read_library(Path(user_library_filename).name)
            except (OSError, json.JSONDecodeError, PromptLibraryError) as exc:
                raise CheckpointInferenceServiceError(str(exc)) from exc
            for shortname, prompt in library["entries"].items():
                append(f"library_{shortname}", prompt, "user-library")

        for index, prompt in enumerate(custom_prompts, start=1):
            prompt = str(prompt).strip()
            if prompt:
                append(f"custom_{index:03d}", prompt, "custom")
        return entries

    def prompt_sources(self, environment: str) -> dict[str, Any]:
        config, _ = self._load_environment(environment)
        configured_prompt = self._config_value(config, "validation_prompt")
        configured_library = self._config_value(config, "user_prompt_library")
        inference_steps = int(self._config_value_with_registry_default(config, "validation_num_inference_steps"))
        guidance_scale = float(self._config_value_with_registry_default(config, "validation_guidance"))
        validation_resolution = str(self._config_value_with_registry_default(config, "validation_resolution"))
        records = PromptLibraryService().list_libraries()
        libraries = [asdict(record) for record in records]
        configured_filename = Path(str(configured_library)).name if configured_library else None
        if configured_library and configured_filename not in {record["filename"] for record in libraries}:
            configured_path = Path(str(configured_library)).expanduser()
            if configured_path.is_file():
                try:
                    with configured_path.open("r", encoding="utf-8") as handle:
                        configured_entries = PromptLibraryService.parse_entries(json.load(handle))
                except (OSError, json.JSONDecodeError, PromptLibraryError) as exc:
                    raise CheckpointInferenceServiceError(str(exc)) from exc
                libraries.append(
                    {
                        "filename": configured_filename,
                        "display_name": configured_path.stem,
                        "prompt_count": len(configured_entries),
                    }
                )
        return {
            "configured_prompt": configured_prompt if configured_prompt not in (None, "None") else None,
            "builtin_count": len(built_in_prompts),
            "configured_user_library": configured_filename,
            "user_libraries": libraries,
            "inference_defaults": {
                "num_inference_steps": inference_steps,
                "guidance_scale": guidance_scale,
                "validation_resolution": validation_resolution,
            },
            "unsupported_multigpu_modes": self._unsupported_multigpu_modes(config),
        }

    def start(
        self,
        *,
        environment: str,
        checkpoint_names: list[str],
        use_configured_prompt: bool,
        use_builtin_library: bool,
        user_library_filename: str | None,
        custom_prompts: list[str],
        filename_style: str,
        keep_loaded: bool,
        streaming_preview: bool,
        idle_timeout_minutes: int,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        if not checkpoint_names:
            raise CheckpointInferenceServiceError("Select at least one checkpoint.")
        if keep_loaded and len(checkpoint_names) != 1:
            raise CheckpointInferenceServiceError("Keep loaded is available only for a single checkpoint.")
        if filename_style not in self.FILENAME_STYLES:
            raise CheckpointInferenceServiceError("Unsupported inference filename style.")
        if self._training_is_active():
            raise CheckpointInferenceServiceError(
                "Training currently owns the accelerator. Use Trigger Validation or wait for training to finish.",
                status.HTTP_409_CONFLICT,
            )

        config, output_dir = self._load_environment(environment)
        validation_resolutions = self._inference_resolutions(config, settings)
        manager = CheckpointManager(str(output_dir))
        for checkpoint_name in checkpoint_names:
            if Path(checkpoint_name).name != checkpoint_name:
                raise CheckpointInferenceServiceError("Checkpoint names may not contain path components.")
            valid, message = manager.validate_checkpoint(checkpoint_name)
            if not valid:
                raise CheckpointInferenceServiceError(f"{checkpoint_name} is not inference-ready: {message}")

        prompt_entries = self.resolve_prompts(
            config=config,
            use_configured_prompt=use_configured_prompt,
            use_builtin_library=use_builtin_library,
            user_library_filename=user_library_filename,
            custom_prompts=custom_prompts,
        )
        if not prompt_entries:
            raise CheckpointInferenceServiceError("Select a prompt source or enter a custom prompt.")

        with self._lock:
            active = self.active_session()
            if active:
                raise CheckpointInferenceServiceError(
                    f"Inference session {active['session_id']} is already using the accelerator.",
                    status.HTTP_409_CONFLICT,
                )
            session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:8]}"
            job_id = f"infer-{uuid.uuid4().hex[:8]}"
            session_dir = output_dir / "inference" / session_id
            session_dir.mkdir(parents=True, exist_ok=False)
            created_at = datetime.now(timezone.utc).isoformat()
            payload = {
                "session_id": session_id,
                "session_dir": str(session_dir),
                "job_id": job_id,
                "environment": environment,
                "checkpoint_names": checkpoint_names,
                "prompt_entries": prompt_entries,
                "filename_style": filename_style,
                "keep_loaded": keep_loaded,
                "streaming_preview": streaming_preview,
                "idle_timeout_seconds": idle_timeout_minutes * 60,
                "settings": settings,
                "trainer_config": config,
                "created_at": created_at,
            }
            from simpletuner.inference import run_checkpoint_inference

            process_keeper.submit_job(job_id, run_checkpoint_inference, payload)
            record = {"session_id": session_id, "job_id": job_id, "environment": environment}
            self._sessions[session_id] = record
            self._environment_sessions[environment] = session_id

        return {
            **record,
            "status": "loading",
            "prompt_count": len(prompt_entries),
            "generation_count": len(prompt_entries) * len(checkpoint_names) * len(validation_resolutions),
            "streaming_preview": streaming_preview,
        }

    def _session_path(self, environment: str, session_id: str) -> Path:
        _, output_dir = self._load_environment(environment)
        candidate = (output_dir / "inference" / session_id).resolve()
        root = (output_dir / "inference").resolve()
        if candidate.parent != root:
            raise CheckpointInferenceServiceError("Invalid inference session ID.")
        return candidate

    def status(self, environment: str, session_id: str) -> dict[str, Any]:
        session_path = self._session_path(environment, session_id)
        state_path = session_path / "session.json"
        if not state_path.exists():
            with self._lock:
                record = self._sessions.get(session_id)
            if record:
                return {**record, "status": process_keeper.get_process_status(record["job_id"])}
            raise CheckpointInferenceServiceError("Inference session not found.", status.HTTP_404_NOT_FOUND)
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        with self._lock:
            record = self._sessions.get(session_id)
        if record and state.get("status") in {
            "pending",
            "queued",
            "loading",
            "running",
            "loaded",
            "unloading",
            "cancelling",
        }:
            process_status = process_keeper.get_process_status(record["job_id"])
            if process_status in {"terminated", "failed", "crashed"}:
                state["status"] = "cancelled" if process_status == "terminated" else "failed"
                state["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._write_session_state(state_path, state)
        if state.get("status") in {"completed", "failed", "cancelled"}:
            self._discard_session(session_id)
        self._add_session_media(state, environment, session_path)
        return state

    @staticmethod
    def _media_url(environment: str, relative_media: str, *, version: int | None = None) -> str:
        query = f"environment={quote(environment, safe='')}"
        if version is not None:
            query += f"&v={version}"
        return f"/api/checkpoints/inference/media/{quote(relative_media, safe='/')}?{query}"

    def _add_session_media(self, state: dict[str, Any], environment: str, session_path: Path) -> None:
        inference_root = session_path.parent
        resolved_root = inference_root.resolve()
        outputs: list[dict[str, Any]] = []
        for sidecar in session_path.glob("*/*.*.json"):
            if not sidecar.resolve().is_relative_to(resolved_root):
                continue
            try:
                with sidecar.open("r", encoding="utf-8") as handle:
                    record = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            relative_media = sidecar.relative_to(inference_root).as_posix()[: -len(".json")]
            relative = Path(relative_media)
            media = (inference_root / relative).resolve()
            if not self._is_generated_media_path(relative) or not media.is_relative_to(resolved_root) or not media.is_file():
                continue
            record["media_path"] = relative_media
            record["media_url"] = self._media_url(environment, relative_media)
            record["streaming"] = False
            outputs.append(record)
        if outputs:
            state["latest_output"] = max(outputs, key=lambda item: item.get("created_at", ""))

        preview_path = session_path / "preview.png"
        preview_metadata_path = session_path / "preview.json"
        if not preview_path.is_file() or not preview_metadata_path.is_file():
            return
        try:
            with preview_metadata_path.open("r", encoding="utf-8") as handle:
                preview = json.load(handle)
            relative_preview = preview_path.relative_to(inference_root).as_posix()
            preview["media_path"] = relative_preview
            preview["media_url"] = self._media_url(
                environment,
                relative_preview,
                version=preview_path.stat().st_mtime_ns,
            )
            preview["streaming"] = True
            state["preview"] = preview
        except (OSError, json.JSONDecodeError):
            return

    def generate(
        self,
        *,
        environment: str,
        session_id: str,
        custom_prompts: list[str],
        filename_style: str | None,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        state = self.status(environment, session_id)
        if state.get("status") != "loaded":
            raise CheckpointInferenceServiceError(
                "The inference session is not ready for custom prompts.", status.HTTP_409_CONFLICT
            )
        with self._lock:
            record = self._sessions.get(session_id)
        if not record:
            raise CheckpointInferenceServiceError(
                "The loaded inference worker is no longer available.", status.HTTP_409_CONFLICT
            )
        entries = self.resolve_prompts(
            config={},
            use_configured_prompt=False,
            use_builtin_library=False,
            user_library_filename=None,
            custom_prompts=custom_prompts,
        )
        if not entries:
            raise CheckpointInferenceServiceError("Enter at least one custom prompt.")
        if filename_style is not None and filename_style not in self.FILENAME_STYLES:
            raise CheckpointInferenceServiceError("Unsupported inference filename style.")
        config, _ = self._load_environment(environment)
        self._inference_resolutions(config, settings)
        state_path = self._session_path(environment, session_id) / "session.json"
        state["status"] = "queued"
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_session_state(state_path, state)
        process_keeper.send_process_command(
            record["job_id"],
            "inference_generate",
            {"prompt_entries": entries, "filename_style": filename_style, "settings": settings},
        )
        return {"session_id": session_id, "status": "queued", "prompt_count": len(entries)}

    def stop(self, environment: str, session_id: str, *, cancel: bool) -> dict[str, Any]:
        with self._lock:
            record = self._sessions.get(session_id)
        if not record or record.get("environment") != environment:
            state = self.status(environment, session_id)
            return {"session_id": session_id, "status": state.get("status")}
        if cancel:
            process_keeper.terminate_process(record["job_id"])
        else:
            process_keeper.send_process_command(record["job_id"], "inference_unload")
        return {"session_id": session_id, "status": "cancelling" if cancel else "unloading"}

    def history(self, environment: str, *, page: int, page_size: int) -> dict[str, Any]:
        _, output_dir = self._load_environment(environment)
        inference_root = output_dir / "inference"
        resolved_root = inference_root.resolve()
        records: list[dict[str, Any]] = []
        if inference_root.exists():
            for sidecar in inference_root.glob("*/*/*.*.json"):
                if not sidecar.resolve().is_relative_to(resolved_root):
                    continue
                try:
                    with sidecar.open("r", encoding="utf-8") as handle:
                        record = json.load(handle)
                except (OSError, json.JSONDecodeError):
                    continue
                relative_media = sidecar.relative_to(inference_root).as_posix()[: -len(".json")]
                relative = Path(relative_media)
                media = (inference_root / relative).resolve()
                if (
                    not self._is_generated_media_path(relative)
                    or not media.is_relative_to(resolved_root)
                    or not media.is_file()
                ):
                    continue
                record["media_path"] = relative_media
                record["media_url"] = self._media_url(environment, relative_media)
                records.append(record)
        records.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        total = len(records)
        offset = (page - 1) * page_size
        return {"items": records[offset : offset + page_size], "page": page, "page_size": page_size, "total": total}

    def delete_history(self, environment: str, media_paths: list[str]) -> dict[str, Any]:
        if not media_paths:
            raise CheckpointInferenceServiceError("Select at least one inference output.")
        _, output_dir = self._load_environment(environment)
        root = (output_dir / "inference").resolve()
        targets: list[tuple[Path, Path]] = []
        seen: set[str] = set()

        for media_path in media_paths:
            if media_path in seen:
                raise CheckpointInferenceServiceError("Duplicate inference output path.")
            seen.add(media_path)

            relative = Path(media_path)
            candidate = (root / relative).resolve()
            if relative.is_absolute() or not candidate.is_relative_to(root):
                raise CheckpointInferenceServiceError("Invalid inference output path.")
            if not self._is_generated_media_path(relative):
                raise CheckpointInferenceServiceError("Invalid inference output path.")

            sidecar = candidate.with_suffix(f"{candidate.suffix}.json")
            if not candidate.is_file() or not sidecar.is_file():
                raise CheckpointInferenceServiceError(
                    f"Inference output '{media_path}' was not found.", status.HTTP_404_NOT_FOUND
                )
            targets.append((candidate, sidecar))

        try:
            for media, sidecar in targets:
                media.unlink()
                sidecar.unlink()
        except OSError as exc:
            raise CheckpointInferenceServiceError(
                f"Failed to delete inference output: {exc}", status.HTTP_500_INTERNAL_SERVER_ERROR
            ) from exc

        return {"deleted_count": len(targets), "media_paths": media_paths}

    def media_path(self, environment: str, media_path: str) -> Path:
        _, output_dir = self._load_environment(environment)
        root = (output_dir / "inference").resolve()
        relative = Path(media_path)
        candidate = (root / relative).resolve()
        is_generated = self._is_generated_media_path(relative)
        is_preview = self._is_preview_media_path(relative)
        if relative.is_absolute() or not candidate.is_relative_to(root) or not (is_generated or is_preview):
            raise CheckpointInferenceServiceError("Inference output not found.", status.HTTP_404_NOT_FOUND)
        sidecar = candidate.parent / "preview.json" if is_preview else candidate.with_suffix(f"{candidate.suffix}.json")
        if not candidate.is_file() or not sidecar.is_file():
            raise CheckpointInferenceServiceError("Inference output not found.", status.HTTP_404_NOT_FOUND)
        return candidate


CHECKPOINT_INFERENCE_SERVICE = CheckpointInferenceService()

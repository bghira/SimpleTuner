"""Checkpoint inference worker used by the WebUI."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from simpletuner.helpers.caching.text_embeds import TextEmbeddingCache
from simpletuner.helpers.data_backend.local import LocalDataBackend
from simpletuner.helpers.training.attention_backend import AttentionBackendController, AttentionPhase
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.training.trainer import Trainer
from simpletuner.helpers.training.validation import (
    ValidationAbortedException,
    _validation_text_cache_key,
    parse_validation_resolutions,
)

logger = logging.getLogger("SimpleTuner-inference")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def _prompt_slug(prompt: str, limit: int = 48) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", prompt).strip("-").lower()
    return slug[:limit].rstrip("-") or "prompt"


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


class CheckpointInferenceRuntime:
    """Own one loaded checkpoint and its validation pipeline."""

    def __init__(
        self,
        *,
        trainer_config: dict[str, Any],
        checkpoint_name: str,
        session_dir: Path,
        job_id: str,
        abort_callback,
        streaming_preview: bool = False,
    ) -> None:
        self.checkpoint_name = checkpoint_name
        self.session_dir = session_dir
        self.job_id = job_id
        self.trainer: Trainer | None = None
        self.embed_cache: TextEmbeddingCache | None = None
        self._load(trainer_config, abort_callback, streaming_preview)

    def _load(self, trainer_config: dict[str, Any], abort_callback, streaming_preview: bool) -> None:
        config = dict(trainer_config)
        config.update(
            {
                "--resume_from_checkpoint": self.checkpoint_name,
                "--validation_disable": False,
                "--validation_prompt_library": False,
                "--user_prompt_library": None,
                "--validation_prompt": None,
                "--validation_disable_unconditional": True,
                "--validation_using_datasets": False,
                "--validation_preview": streaming_preview,
                "--disable_benchmark": True,
                "--gradient_checkpointing": False,
                "--deepspeed_config": None,
                "--fsdp_enable": False,
                "--report_to": "none",
                "--push_to_hub": False,
                "__skip_config_fallback__": True,
            }
        )
        trainer = Trainer(config=config, job_id=self.job_id, exit_on_error=True)
        self.trainer = trainer
        trainer.config.should_abort = abort_callback
        trainer._external_abort_checker = abort_callback
        if streaming_preview:
            trainer.config._validation_preview_callback = self._write_preview

        trainer.init_noise_schedule()
        trainer.init_seed()
        trainer.init_preprocessing_models()
        trainer.init_precision(preprocessing_models_only=True)
        trainer.init_load_base_model()
        trainer.init_controlnet_model()
        trainer.init_tread_model()
        trainer.init_precision()
        trainer.init_freeze_models()
        trainer.init_trainable_peft_adapter()
        trainer.init_ema_model()
        trainer.init_precision(ema_only=True)
        trainer.move_models(destination="accelerator")
        trainer.init_distillation()
        trainer.init_hooks()
        trainer.prepare_model_for_inference()

        checkpoint_path = Path(trainer.config.output_dir) / self.checkpoint_name
        trainer.accelerator.load_state(str(checkpoint_path))
        AttentionBackendController.on_load_checkpoint(str(checkpoint_path))
        if trainer.distiller is not None:
            trainer.distiller.on_load_checkpoint(str(checkpoint_path))

        cache_id = f"inference-{self.job_id}"
        cache_dir = self.session_dir / ".prompt_cache" / self.checkpoint_name
        backend = LocalDataBackend(trainer.accelerator, id=cache_id)
        self.embed_cache = TextEmbeddingCache(
            id=cache_id,
            data_backend=backend,
            text_encoders=trainer.model.text_encoders,
            tokenizers=trainer.model.tokenizers,
            accelerator=trainer.accelerator,
            cache_dir=str(cache_dir),
            model_type=trainer.config.model_family,
            model=trainer.model,
        )
        StateTracker.set_default_text_embed_cache(self.embed_cache)
        self.embed_cache.discover_all_files()
        if trainer.model.VALIDATION_USES_NEGATIVE_PROMPT:
            negative_prompt = trainer.config.validation_negative_prompt or ""
            if trainer.config.model_family == "ideogram":
                self.embed_cache.encode_validation_negative_prompt(negative_prompt)
            elif trainer.model.should_precompute_validation_negative_prompt():
                self.embed_cache.compute_embeddings_for_prompts(
                    [negative_prompt],
                    is_validation=True,
                    load_from_cache=False,
                )
            else:
                self.embed_cache.encode_validation_negative_prompt(negative_prompt)
            self._flush_embed_cache()
        trainer.validation_prompt_metadata = {
            "validation_prompts": [],
            "validation_shortnames": [],
            "validation_sample_images": None,
        }
        trainer.init_validations(preserve_text_encoders=True)
        AttentionBackendController.apply(trainer.config, AttentionPhase.EVAL)
        trainer.validation.setup_pipeline("checkpoint")
        if trainer.model.pipeline is None:
            raise RuntimeError(f"Could not create an inference pipeline for {self.checkpoint_name}.")
        trainer.validation.setup_scheduler()

    def _write_preview(self, *, structured_data: dict[str, Any], images: list[Any], videos: Any) -> None:
        if not images or not isinstance(images[0], Image.Image):
            return
        preview_path = self.session_dir / "preview.png"
        temporary = preview_path.with_suffix(".png.tmp")
        try:
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            images[0].save(temporary, format="PNG")
            os.replace(temporary, preview_path)
            data = structured_data.get("data") or {}
            _write_json(
                self.session_dir / "preview.json",
                {
                    "checkpoint": self.checkpoint_name,
                    "prompt": data.get("prompt") or structured_data.get("body") or "",
                    "step": data.get("step"),
                    "step_label": data.get("step_label"),
                    "updated_at": _utc_now(),
                    "media_type": "image",
                    "filename": preview_path.name,
                },
            )
        except OSError as exc:
            logger.warning("Could not write streaming inference preview: %s", exc)

    def _flush_embed_cache(self) -> None:
        if self.embed_cache is None:
            raise RuntimeError("Inference prompt cache is not initialized.")
        self.embed_cache.process_write_batches = False
        self.embed_cache.batch_write_thread.join()

    def _prepare_prompt(self, entry: dict[str, Any]) -> None:
        if self.trainer is None or self.embed_cache is None:
            raise RuntimeError("Inference runtime is not loaded.")
        prompt = str(entry["prompt"])
        shortname = str(entry["shortname"])
        key = _validation_text_cache_key(self.trainer.config, shortname, prompt)
        prompt_record: dict[str, Any] = {"prompt": prompt, "key": key}
        metadata = entry.get("metadata")
        if isinstance(metadata, dict) and metadata:
            prompt_record["metadata"] = metadata
        self.embed_cache.compute_embeddings_for_prompts(
            [prompt_record],
            is_validation=True,
            load_from_cache=False,
        )

        entity_records = []
        for index, entity in enumerate(entry.get("bbox_entities") or []):
            entity_records.append(
                {
                    "prompt": entity["label"],
                    "key": f"__grounding_val_{shortname}__bbox_{index}",
                }
            )
        if entity_records:
            self.embed_cache.compute_embeddings_for_prompts(
                entity_records,
                return_concat=False,
                is_validation=True,
                load_from_cache=False,
            )
        self._flush_embed_cache()

    def _output_stem(self, *, prompt: str, seed: int, style: str) -> str:
        timestamp = _timestamp_slug()
        if style == "compact":
            return f"{timestamp}_seed{seed}"
        if style == "prompt":
            return f"{_prompt_slug(prompt)}_{timestamp}"
        if style == "content-hash":
            return f"pending_{timestamp}"
        return f"{timestamp}_{_prompt_slug(prompt)}_seed{seed}"

    def _save_media(
        self,
        media: Any,
        *,
        prompt: str,
        shortname: str,
        seed: int,
        style: str,
        index: int,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        output_dir = self.session_dir / self.checkpoint_name
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self._output_stem(prompt=prompt, seed=seed, style=style)}_{index + 1:02d}"

        if isinstance(media, Image.Image):
            extension = ".png"
            output_path = output_dir / f"{stem}{extension}"
            media.save(output_path, format="PNG")
            media_type = "image"
        elif isinstance(media, list) and media and all(isinstance(frame, Image.Image) for frame in media):
            from diffusers.utils.export_utils import export_to_video

            extension = ".mp4"
            output_path = output_dir / f"{stem}{extension}"
            export_to_video(media, str(output_path), fps=int(getattr(self.trainer.config, "framerate", 16)))
            media_type = "video"
        else:
            raise TypeError(f"Unsupported inference output type: {type(media).__name__}")

        digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
        if style == "content-hash":
            hashed_path = output_path.with_name(f"{digest}{extension}")
            os.replace(output_path, hashed_path)
            output_path = hashed_path

        metadata = {
            "session_id": self.session_dir.name,
            "checkpoint": self.checkpoint_name,
            "prompt": prompt,
            "shortname": shortname,
            "seed": seed,
            "created_at": _utc_now(),
            "media_type": media_type,
            "filename": output_path.name,
            "sha256": digest,
            "settings": settings,
        }
        _write_json(output_path.with_suffix(f"{output_path.suffix}.json"), metadata)
        return metadata

    def _apply_settings(self, settings: dict[str, Any]) -> int:
        if self.trainer is None:
            raise RuntimeError("Inference runtime is not loaded.")
        seed = int(settings.get("seed", getattr(self.trainer.config, "validation_seed", 42) or 42))
        self.trainer.config.validation_seed = seed
        if settings.get("num_inference_steps") is not None:
            self.trainer.config.validation_num_inference_steps = int(settings["num_inference_steps"])
        if settings.get("guidance_scale") is not None:
            self.trainer.config.validation_guidance = float(settings["guidance_scale"])
        if settings.get("validation_resolution") is not None:
            resolution_value = str(settings["validation_resolution"])
            self.trainer.config.validation_resolution = resolution_value
            self.trainer.validation.validation_resolutions = parse_validation_resolutions(
                resolution_value,
                model_flavour=self.trainer.config.model_flavour,
            )
        return seed

    def generate(
        self,
        entries: list[dict[str, Any]],
        *,
        filename_style: str,
        settings: dict[str, Any],
        progress_callback,
        abort_callback,
    ) -> list[dict[str, Any]]:
        if self.trainer is None:
            raise RuntimeError("Inference runtime is not loaded.")
        validation = self.trainer.validation
        results: list[dict[str, Any]] = []
        seed = self._apply_settings(settings)

        for entry_index, entry in enumerate(entries):
            if abort_callback():
                raise ValidationAbortedException("Inference was cancelled.")
            self._prepare_prompt(entry)
            shortname = str(entry["shortname"])
            prompt = str(entry["prompt"])
            _, checkpoint_media, _, _ = validation.validate_prompt(
                prompt,
                shortname,
                validation_type="checkpoint",
                adapter_strength=entry.get("adapter_strength"),
                cache_shortname=shortname,
                bbox_entities=entry.get("bbox_entities"),
                bbox_keyframes=entry.get("bbox_keyframes"),
            )
            prompt_media = checkpoint_media.get(shortname, [])
            if not prompt_media:
                raise RuntimeError(f"Inference produced no image or video for prompt '{shortname}'.")
            effective_settings = {
                "seed": seed,
                "num_inference_steps": self.trainer.config.validation_num_inference_steps,
                "guidance_scale": self.trainer.config.validation_guidance,
                "validation_resolution": self.trainer.config.validation_resolution,
            }
            for media_index, media in enumerate(prompt_media):
                results.append(
                    self._save_media(
                        media,
                        prompt=prompt,
                        shortname=shortname,
                        seed=seed,
                        style=filename_style,
                        index=media_index,
                        settings=effective_settings,
                    )
                )
            progress_callback(entry_index + 1, len(entries), prompt)
        return results

    def close(self) -> None:
        if self.trainer is None:
            return
        try:
            if self.trainer.validation is not None:
                self.trainer.validation.clean_pipeline()
        finally:
            if self.embed_cache is not None:
                self.embed_cache.process_write_batches = False
                if self.embed_cache.batch_write_thread.is_alive():
                    self.embed_cache.batch_write_thread.join(timeout=2)
            self.trainer.cleanup()
            self.trainer = None
            self.embed_cache = None


def run_checkpoint_inference(config) -> dict[str, Any]:
    """Process keeper entry point for checkpoint inference sessions."""
    write_json = _write_json
    payload = dict(config.__dict__)
    session_dir = Path(payload["session_dir"])
    state_path = session_dir / "session.json"
    checkpoints = list(payload["checkpoint_names"])
    entries = list(payload["prompt_entries"])
    filename_style = str(payload.get("filename_style", "descriptive"))
    settings = dict(payload.get("settings") or {})
    keep_loaded = bool(payload.get("keep_loaded")) and len(checkpoints) == 1
    streaming_preview = bool(payload.get("streaming_preview"))
    idle_timeout = max(60, int(payload.get("idle_timeout_seconds", 900)))
    completed_outputs: list[dict[str, Any]] = []

    state = {
        "session_id": session_dir.name,
        "job_id": payload["job_id"],
        "environment": payload["environment"],
        "checkpoint_names": checkpoints,
        "status": "loading",
        "created_at": payload.get("created_at") or _utc_now(),
        "updated_at": _utc_now(),
        "completed_prompts": 0,
        "total_prompts": len(entries) * len(checkpoints),
        "output_count": 0,
        "keep_loaded": keep_loaded,
        "streaming_preview": streaming_preview,
    }

    def update_state(**changes) -> None:
        state.update(changes)
        state["updated_at"] = _utc_now()
        write_json(state_path, state)

    def aborted() -> bool:
        return bool(config.should_abort())

    runtime: CheckpointInferenceRuntime | None = None
    try:
        completed_prompts = 0
        for checkpoint_name in checkpoints:
            if aborted():
                raise ValidationAbortedException("Inference was cancelled.")
            update_state(status="loading", loaded_checkpoint=checkpoint_name)
            runtime = CheckpointInferenceRuntime(
                trainer_config=payload["trainer_config"],
                checkpoint_name=checkpoint_name,
                session_dir=session_dir,
                job_id=payload["job_id"],
                abort_callback=config.should_abort,
                streaming_preview=streaming_preview,
            )
            update_state(status="running", loaded_checkpoint=checkpoint_name)

            def report(current: int, total: int, prompt: str) -> None:
                update_state(
                    status="running",
                    current_prompt=prompt,
                    completed_prompts=completed_prompts + current,
                    output_count=len(completed_outputs),
                )

            generated = runtime.generate(
                entries,
                filename_style=filename_style,
                settings=settings,
                progress_callback=report,
                abort_callback=aborted,
            )
            completed_outputs.extend(generated)
            completed_prompts += len(entries)
            update_state(completed_prompts=completed_prompts, output_count=len(completed_outputs))
            if checkpoint_name != checkpoints[-1] or not keep_loaded:
                runtime.close()
                runtime = None

        if not keep_loaded:
            update_state(status="completed", loaded_checkpoint=None, completed_at=_utc_now())
            return state

        update_state(status="loaded", current_prompt=None)
        last_activity = time.monotonic()
        while not aborted():
            command = config.consume_process_command(timeout=0.5)
            if command is None:
                if time.monotonic() - last_activity >= idle_timeout:
                    update_state(status="unloading", message="Inference session reached its idle timeout.")
                    break
                continue
            command_name = str(command.get("command") or "").lower()
            if command_name == "inference_unload":
                update_state(status="unloading")
                break
            if command_name != "inference_generate":
                continue
            command_data = command.get("data") or {}
            command_entries = list(command_data.get("prompt_entries") or [])
            if not command_entries:
                continue
            last_activity = time.monotonic()
            state["total_prompts"] += len(command_entries)
            update_state(status="running")

            base_completed = int(state["completed_prompts"])

            def report_interactive(current: int, total: int, prompt: str) -> None:
                update_state(
                    status="running",
                    current_prompt=prompt,
                    completed_prompts=base_completed + current,
                    output_count=len(completed_outputs),
                )

            generated = runtime.generate(
                command_entries,
                filename_style=str(command_data.get("filename_style") or filename_style),
                settings=dict(command_data.get("settings") or settings),
                progress_callback=report_interactive,
                abort_callback=aborted,
            )
            completed_outputs.extend(generated)
            update_state(
                status="loaded",
                current_prompt=None,
                completed_prompts=base_completed + len(command_entries),
                output_count=len(completed_outputs),
            )

        update_state(status="completed", loaded_checkpoint=None, completed_at=_utc_now())
        return state
    except ValidationAbortedException as exc:
        update_state(status="cancelled", message=str(exc), completed_at=_utc_now())
        return state
    except Exception as exc:
        logger.exception("Checkpoint inference failed")
        update_state(status="failed", message=str(exc), completed_at=_utc_now())
        raise
    finally:
        if runtime is not None:
            runtime.close()


def main() -> int:
    raise SystemExit("simpletuner-inference is managed by the SimpleTuner WebUI checkpoint inference service.")

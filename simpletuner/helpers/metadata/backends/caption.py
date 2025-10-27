from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.helpers.metadata.captions import CaptionRecord, normalize_caption_text
from simpletuner.helpers.training.multi_process import should_log
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger("CaptionMetadataBackend")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class CaptionMetadataBackend(MetadataBackend):
    """Metadata backend that indexes caption-only datasets."""

    DEFAULT_EXTENSIONS: Sequence[str] = ("txt", "json", "jsonl")

    def __init__(
        self,
        id: str,
        instance_data_dir: str,
        cache_file: str,
        metadata_file: str,
        data_backend: BaseDataBackend,
        accelerator,
        batch_size: int = 1,
        caption_extensions: Optional[Sequence[str]] = None,
        metadata_update_interval: int = 3600,
        cache_file_suffix: Optional[str] = None,
        repeats: int = 0,
        **_: object,
    ):
        self.caption_records: Dict[str, CaptionRecord] = {}
        self._ordered_ids: List[str] = []
        self.caption_extensions = tuple(
            sorted({ext.lstrip(".").lower() for ext in (caption_extensions or self.DEFAULT_EXTENSIONS)})
        )
        super().__init__(
            id=id,
            instance_data_dir=instance_data_dir,
            cache_file=cache_file,
            metadata_file=metadata_file,
            data_backend=data_backend,
            accelerator=accelerator,
            batch_size=max(int(batch_size or 1), 1),
            resolution=1,
            resolution_type="pixel",
            metadata_update_interval=metadata_update_interval,
            cache_file_suffix=cache_file_suffix,
            repeats=repeats,
        )

    # ------------------------------------------------------------------
    # Metadata persistence
    # ------------------------------------------------------------------
    def reload_cache(self, set_config: bool = True):
        self._ordered_ids = []
        cache_exists = False
        try:
            cache_exists = self.data_backend.exists(self.cache_file)
        except Exception:
            cache_exists = False

        if cache_exists:
            try:
                raw = self.data_backend.read(self.cache_file)
                cache_data = json.loads(raw)
                self._ordered_ids = list(cache_data.get("ordered_ids", []))
                if set_config and cache_data.get("config"):
                    StateTracker.set_data_backend_config(self.id, cache_data["config"])
            except Exception as exc:  # pragma: no cover - extremely defensive
                logger.warning(f"(id={self.id}) Failed to load caption cache: {exc}")
                self._ordered_ids = []

        self.aspect_ratio_bucket_indices = {"captions": list(self._ordered_ids)}

    def save_cache(self, enforce_constraints: bool = False):
        del enforce_constraints  # Caption datasets do not use bucket constraints.
        if self.read_only:
            logger.debug(f"(id={self.id}) Skipping caption cache write in read-only mode.")
            return
        payload = {
            "config": StateTracker.get_data_backend_config(self.id),
            "ordered_ids": list(self._ordered_ids),
        }
        try:
            self.data_backend.write(self.cache_file, json.dumps(payload))
        except Exception as exc:  # pragma: no cover - storage edge cases
            logger.warning(f"(id={self.id}) Unable to write caption cache: {exc}")

    def load_image_metadata(self):
        self.caption_records = {}
        self.image_metadata = {}
        self.image_metadata_loaded = False
        try:
            if not self.data_backend.exists(self.metadata_file):
                return
            raw = self.data_backend.read(self.metadata_file)
            entries = json.loads(raw) or []
        except FileNotFoundError:
            return
        except Exception as exc:  # pragma: no cover - decode guard
            logger.warning(f"(id={self.id}) Failed to load caption metadata: {exc}")
            return

        ordered_ids = []
        for item in entries:
            try:
                record = CaptionRecord.from_payload(item)
            except KeyError:
                logger.debug(f"(id={self.id}) Skipping malformed caption payload: {item}")
                continue
            self.caption_records[record.metadata_id] = record
            ordered_ids.append(record.metadata_id)
            self.image_metadata[record.metadata_id] = record.to_payload()

        self._ordered_ids = ordered_ids
        self.aspect_ratio_bucket_indices = {"captions": list(self._ordered_ids)}
        self.image_metadata_loaded = True

    def save_image_metadata(self):
        serialized = [self.caption_records[mid].to_payload() for mid in self._ordered_ids]
        self.data_backend.write(self.metadata_file, json.dumps(serialized))
        self.image_metadata_loaded = True

    # ------------------------------------------------------------------
    # Caption ingestion helpers
    # ------------------------------------------------------------------
    def ingest_from_file_cache(self, caption_files: Dict[str, bool]) -> int:
        """Populate caption records from a cache built by StateTracker.set_caption_files."""
        if not caption_files:
            logger.warning(f"(id={self.id}) No caption files discovered.")
            self.caption_records = {}
            self._ordered_ids = []
            self.image_metadata = {}
            self.save_image_metadata()
            self.save_cache()
            return 0

        created = 0
        for file_path in sorted(caption_files.keys()):
            try:
                for index, caption_text in enumerate(self._extract_captions_from_file(file_path)):
                    metadata_id = self._build_metadata_id(file_path, index)
                    if metadata_id in self.caption_records:
                        continue
                    self._store_record(metadata_id, caption_text, source_path=file_path)
                    created += 1
            except FileNotFoundError:
                logger.warning(f"(id={self.id}) Caption file not found: {file_path}")
            except Exception as exc:
                logger.warning(f"(id={self.id}) Failed to read caption file {file_path}: {exc}")

        if created or not self.data_backend.exists(self.metadata_file):
            self.save_image_metadata()
            self.save_cache()
        return created

    def _store_record(self, metadata_id: str, caption_text: str, source_path: Optional[str]) -> None:
        record = CaptionRecord(
            metadata_id=metadata_id,
            caption_text=caption_text,
            data_backend_id=self.id,
            source_path=source_path,
        )
        self.caption_records[metadata_id] = record
        self.image_metadata[metadata_id] = record.to_payload()
        self._ordered_ids.append(metadata_id)
        self.aspect_ratio_bucket_indices = {"captions": list(self._ordered_ids)}

    def _extract_captions_from_file(self, filepath: str) -> Iterator[str]:
        data = self.data_backend.read(filepath)
        if isinstance(data, bytes):
            decoded = data.decode("utf-8", errors="ignore")
        else:
            decoded = str(data)

        suffix = Path(filepath).suffix.lower()
        if suffix == ".txt":
            for entry in self._split_plaintext(decoded):
                yield entry
        elif suffix == ".jsonl":
            for line in decoded.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    try:
                        yield normalize_caption_text(line)
                    except ValueError:
                        continue
                    continue
                yield from self._extract_from_json_payload(payload)
        elif suffix == ".json":
            try:
                payload = json.loads(decoded)
            except json.JSONDecodeError:
                for entry in self._split_plaintext(decoded):
                    yield entry
                return
            yield from self._extract_from_json_payload(payload)
        else:
            for entry in self._split_plaintext(decoded):
                yield entry

    def _split_plaintext(self, contents: str) -> Iterator[str]:
        for line in contents.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield normalize_caption_text(line)
            except ValueError:
                continue

    def _extract_from_json_payload(self, payload) -> Iterator[str]:
        if isinstance(payload, str):
            try:
                yield normalize_caption_text(payload)
            except ValueError:
                return
            return
        if isinstance(payload, list):
            for item in payload:
                yield from self._extract_from_json_payload(item)
            return
        if isinstance(payload, dict):
            for key in ("caption", "captions", "text", "texts", "value"):
                if key in payload:
                    yield from self._extract_from_json_payload(payload[key])
                    return
            for value in payload.values():
                if isinstance(value, (dict, list, str)):
                    yield from self._extract_from_json_payload(value)
            return
        # Unsupported types are ignored.

    def _build_metadata_id(self, filepath: str, index: int) -> str:
        # Preserve ordering but normalise separators for remote/local parity.
        safe_path = filepath.replace("\\", "/")
        return f"{safe_path}#C{index}"

    # ------------------------------------------------------------------
    # Access helpers for datasets / samplers
    # ------------------------------------------------------------------
    def iter_records(self) -> Iterable[CaptionRecord]:
        if not self.caption_records or not self._ordered_ids:
            self.load_image_metadata()
        for metadata_id in self._ordered_ids:
            record = self.caption_records.get(metadata_id)
            if record is not None:
                yield record

    def get_record(self, metadata_id: str) -> Optional[CaptionRecord]:
        if metadata_id not in self.caption_records:
            self.load_image_metadata()
        return self.caption_records.get(metadata_id)

    def __len__(self) -> int:  # pragma: no cover - convenience wrapper
        if not self._ordered_ids:
            self.load_image_metadata()
        return len(self._ordered_ids)

    def list_metadata_ids(self) -> List[str]:
        """Return ordered metadata identifiers for sampler construction."""
        if not self._ordered_ids:
            self.load_image_metadata()
        return list(self._ordered_ids)

    @property
    def supported_extensions(self) -> Sequence[str]:
        return self.caption_extensions

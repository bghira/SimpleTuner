from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

import pandas as pd

from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.helpers.metadata.captions import CaptionRecord, normalize_caption_text
from simpletuner.helpers.training.multi_process import should_log
from simpletuner.helpers.training.state_tracker import StateTracker

# Standard named logger so it inherits global handlers.
logger = logging.getLogger("CaptionMetadataBackend")
logger.setLevel(logging._nameToLevel.get(str(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")).upper(), logging.INFO))


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
        caption_ingest_strategy: str = "discovery",
        parquet_config: Optional[Dict[str, Any]] = None,
        hf_config: Optional[Dict[str, Any]] = None,
        quality_filter: Optional[Dict[str, Any]] = None,
        **_: object,
    ):
        self.caption_records: Dict[str, CaptionRecord] = {}
        self._ordered_ids: List[str] = []
        self.caption_extensions = tuple(
            sorted({ext.lstrip(".").lower() for ext in (caption_extensions or self.DEFAULT_EXTENSIONS)})
        )
        self.caption_ingest_strategy = (caption_ingest_strategy or "discovery").lower()
        self.parquet_config = dict(parquet_config or {})
        self.hf_config = dict(hf_config or {})
        self.quality_filter = dict(quality_filter or {})
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
        self.reset_records()
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

    def reset_records(self) -> None:
        """Clear cached caption entries before rebuilding from a new source."""
        self.caption_records = {}
        self.image_metadata = {}
        self._ordered_ids = []
        self.aspect_ratio_bucket_indices = {"captions": []}

    def _store_record(
        self, metadata_id: str, caption_text: str, source_path: Optional[str], extras: Optional[Dict[str, Any]] = None
    ) -> None:
        record = CaptionRecord(
            metadata_id=metadata_id,
            caption_text=caption_text,
            data_backend_id=self.id,
            source_path=source_path,
            extras=extras or {},
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
    # Structured ingestion (parquet / huggingface)
    # ------------------------------------------------------------------
    def ingest_from_parquet_config(self) -> int:
        """Build caption records from a parquet/json/jsonl manifest."""
        if not self.parquet_config:
            raise ValueError("Parquet ingestion requested without parquet_config.")
        manifest_path = self.parquet_config.get("path")
        if not manifest_path:
            raise ValueError("Parquet ingestion requires 'path' in the parquet_config.")
        if not self.data_backend.exists(manifest_path):
            raise FileNotFoundError(f"Parquet manifest not found: {manifest_path}")

        raw_payload = self.data_backend.read(manifest_path)
        rows = self._decode_structured_rows(raw_payload, manifest_path)
        if not rows:
            logger.warning(f"(id={self.id}) Parquet manifest {manifest_path} did not yield any rows.")
            self.reset_records()
            self.save_image_metadata()
            self.save_cache()
            return 0

        self.reset_records()
        caption_columns = self._ensure_list(self.parquet_config.get("caption_column") or [])
        fallback_columns = self._ensure_list(self.parquet_config.get("fallback_caption_column") or [])
        identifier_column = self.parquet_config.get("filename_column") or self.parquet_config.get("id_column")
        identifier_includes_extension = bool(self.parquet_config.get("identifier_includes_extension", False))
        default_extension = self.parquet_config.get("default_extension", "caption")
        identifier_prefix = self.parquet_config.get("identifier_prefix") or Path(manifest_path).stem

        created = 0
        for row_index, row in enumerate(rows):
            identifier = self._parquet_identifier_from_row(
                row=row,
                row_index=row_index,
                identifier_column=identifier_column,
                includes_extension=identifier_includes_extension,
                default_extension=default_extension,
                identifier_prefix=identifier_prefix,
            )
            caption_values = self._extract_captions_from_row_payload(
                row,
                primary_columns=caption_columns,
                fallback_columns=fallback_columns,
            )
            if not caption_values:
                continue

            for caption_idx, caption_text in enumerate(caption_values):
                metadata_id = f"{identifier}#P{row_index}_{caption_idx}"
                if metadata_id in self.caption_records:
                    continue
                self._store_record(
                    metadata_id,
                    caption_text,
                    source_path=f"{manifest_path}:{row_index}",
                    extras={"row_index": row_index},
                )
                created += 1

        if created or not self.data_backend.exists(self.metadata_file):
            self.save_image_metadata()
            self.save_cache()
        return created

    def ingest_from_huggingface_dataset(self) -> int:
        """Populate caption records from a Hugging Face dataset backend."""
        dataset = getattr(self.data_backend, "dataset", None)
        if dataset is None:
            raise ValueError("Hugging Face ingestion requires a HuggingfaceDatasetsBackend data backend.")

        caption_columns = self._ensure_list(self.hf_config.get("caption_column") or ["caption"])
        fallback_columns = self._ensure_list(self.hf_config.get("fallback_caption_column") or [])
        quality_column = self.hf_config.get("quality_column", None)
        description_column = self.hf_config.get("description_column", None)
        dataset_label = (
            self.hf_config.get("repo_id")
            or self.hf_config.get("dataset_name")
            or self.hf_config.get("dataset_path")
            or "huggingface"
        )

        self.reset_records()
        created = 0
        total_items = len(dataset)
        for idx in range(total_items):
            item = dataset[idx]
            if quality_column and quality_column in item and not self._passes_quality_filter(item[quality_column]):
                continue
            caption_values = self._extract_hf_captions(
                item,
                caption_columns=caption_columns,
                fallback_columns=fallback_columns,
                description_column=description_column,
            )
            if not caption_values:
                continue
            for caption_idx, caption_text in enumerate(caption_values):
                metadata_id = f"{dataset_label}:{idx}:{caption_idx}"
                if metadata_id in self.caption_records:
                    continue
                self._store_record(
                    metadata_id=metadata_id,
                    caption_text=caption_text,
                    source_path=f"hf://{dataset_label}/{idx}",
                    extras={"dataset_index": idx},
                )
                created += 1

        if created or not self.data_backend.exists(self.metadata_file):
            self.save_image_metadata()
            self.save_cache()
        return created

    def _decode_structured_rows(self, raw_payload: Any, manifest_path: str) -> List[Dict[str, Any]]:
        if isinstance(raw_payload, bytes):
            payload_str = raw_payload.decode("utf-8", errors="ignore")
        else:
            payload_str = str(raw_payload)
            raw_payload = payload_str.encode("utf-8")
        suffix = Path(manifest_path).suffix.lower()
        if suffix == ".jsonl":
            rows: List[Dict[str, Any]] = []
            for line in payload_str.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"(id={self.id}) Skipping invalid JSONL row: {line[:48]}")
            return rows
        if suffix == ".json":
            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError as exc:
                raise ValueError(f"(id={self.id}) Failed to parse JSON manifest: {exc}") from exc
            if isinstance(payload, dict):
                return [payload]
            if isinstance(payload, list):
                return [entry for entry in payload if isinstance(entry, (dict, list))]
            raise ValueError("JSON manifest must contain a list or dict of entries.")

        if pd is None:
            raise ImportError("pandas is required to read parquet caption manifests.")
        buffer = io.BytesIO(raw_payload)
        frame = pd.read_parquet(buffer, engine=self.parquet_config.get("engine", "pyarrow"))
        return frame.to_dict(orient="records")

    def _parquet_identifier_from_row(
        self,
        row: Union[Dict[str, Any], "pd.Series"],
        row_index: int,
        identifier_column: Optional[str],
        includes_extension: bool,
        default_extension: str,
        identifier_prefix: str,
    ) -> str:
        value = None
        if identifier_column:
            value = self._get_row_value(row, identifier_column)
        if value is None:
            value = f"{identifier_prefix}_{row_index}"
        value = str(value)
        if not includes_extension:
            value = os.path.splitext(value)[0]
        if not os.path.splitext(value)[1]:
            value = f"{value}.{default_extension}"
        return value.replace("\\", "/")

    def _extract_captions_from_row_payload(
        self,
        row: Union[Dict[str, Any], "pd.Series"],
        primary_columns: List[str],
        fallback_columns: List[str],
    ) -> List[str]:
        candidates: List[str] = []
        for column in primary_columns:
            cell = self._get_row_value(row, column)
            candidates.extend(self._normalize_caption_cell(cell))
        if not candidates:
            for column in fallback_columns:
                cell = self._get_row_value(row, column)
                candidates.extend(self._normalize_caption_cell(cell))
        return candidates

    def _extract_hf_captions(
        self,
        item: Dict[str, Any],
        caption_columns: List[str],
        fallback_columns: List[str],
        description_column: Optional[str],
    ) -> List[str]:
        candidates: List[str] = []
        for column in caption_columns:
            candidates.extend(self._normalize_caption_cell(self._get_nested_value(item, column)))
        if not candidates:
            for column in fallback_columns:
                candidates.extend(self._normalize_caption_cell(self._get_nested_value(item, column)))
        if not candidates and description_column:
            candidates.extend(self._normalize_caption_cell(item.get(description_column)))
        return candidates

    def _normalize_caption_cell(self, cell: Any) -> List[str]:
        outputs: List[str] = []
        if cell is None:
            return outputs
        if isinstance(cell, (list, tuple)):
            for value in cell:
                try:
                    outputs.append(normalize_caption_text(value))
                except ValueError:
                    continue
            return outputs
        if isinstance(cell, (dict,)):
            for value in cell.values():
                try:
                    outputs.append(normalize_caption_text(value))
                except ValueError:
                    continue
            return outputs
        try:
            outputs.append(normalize_caption_text(cell))
        except ValueError:
            return []
        return outputs

    def _get_row_value(self, row: Union[Dict[str, Any], "pd.Series"], column: str) -> Any:
        if hasattr(row, "get"):
            return row.get(column)
        try:
            return row[column]
        except Exception:
            return None

    def _ensure_list(self, value: Union[str, Sequence[str], None]) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value if item]
        return [str(value)]

    def _get_nested_value(self, payload: Dict[str, Any], column: str) -> Any:
        if not column:
            return None
        current = payload
        for key in column.split("."):
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(key)
            else:
                return None
        return current

    def _passes_quality_filter(self, quality_assessment: Dict[str, Any]) -> bool:
        if not self.quality_filter or not quality_assessment:
            return True
        for key, min_value in self.quality_filter.items():
            try:
                if float(quality_assessment.get(key, 0)) < float(min_value):
                    return False
            except (TypeError, ValueError):
                return False
        return True

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

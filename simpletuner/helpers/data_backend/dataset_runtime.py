"""Dataset access shared by training and preparation consumers."""

from collections.abc import Callable, Iterable, Iterator, Sequence
from copy import deepcopy
from typing import Any

from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.helpers.prompts import CaptionNotFoundError, PromptHandler
from simpletuner.helpers.utils.pathing import normalize_data_path


class DatasetRuntime:
    def __init__(
        self,
        id: str,
        data_backend: BaseDataBackend,
        metadata_backend: MetadataBackend,
        config: dict[str, Any],
    ):
        self.id = id
        self.data_backend = data_backend
        self.metadata_backend = metadata_backend
        self.config = {
            "dataset_type": "image",
            "caption_strategy": "filename",
            "use_captions": True,
            "prepend_instance_prompt": False,
            "instance_prompt": None,
            "disable_multiline_split": False,
            **deepcopy(config),
        }

    def read_samples(self, sample_refs: Sequence[str]) -> list[dict[str, Any]]:
        """Return independent occurrence metadata in request order, including duplicates."""
        records = []
        for sample_ref in sample_refs:
            metadata = self.metadata_backend.get_metadata_by_filepath(sample_ref)
            record = deepcopy(metadata) if metadata is not None else {}
            record.setdefault("dataset_type", self.config["dataset_type"])
            record["data_backend_id"] = self.id
            record["image_path"] = sample_ref
            records.append(record)
        return records

    def resolve_caption(self, sample_ref: str) -> str | list[str] | None:
        """Resolve caption alternatives without choosing an occurrence's alternative."""
        return PromptHandler.magic_prompt(
            image_path=sample_ref,
            data_backend=self.data_backend,
            sampler_backend_id=self.id,
            metadata_backend=self.metadata_backend,
            backend_config=self.config,
            caption_strategy=self.config["caption_strategy"],
            use_captions=self.config["use_captions"],
            prepend_instance_prompt=self.config["prepend_instance_prompt"],
            instance_prompt=self.config["instance_prompt"],
            disable_multiline_split=self.config["disable_multiline_split"],
        )

    def iter_caption_records(
        self,
        sample_refs: Iterable[str],
        *,
        on_missing_caption: Callable[[str], None] | None = None,
    ) -> Iterator[tuple[str, str]]:
        """Visit the caller's sample selection without retaining captions or selecting alternatives."""
        for sample_ref in sample_refs:
            try:
                caption = self.resolve_caption(sample_ref)
            except CaptionNotFoundError:
                if on_missing_caption is None:
                    raise
                on_missing_caption(sample_ref)
                continue
            for value in PromptHandler._normalize_caption_payload(caption):
                yield value, sample_ref

    def iter_text_embed_records(
        self,
        model,
        sample_refs: Iterable[str],
        *,
        init_backend: dict[str, Any],
        on_missing_caption: Callable[[str], None] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Build artifact requests incrementally, preserving model keys and caption variants."""
        from simpletuner.helpers.models.common import TextEmbedCacheKey

        key_type = model.text_embed_cache_key()
        dataset_root = self.config.get("instance_data_dir")
        for caption, sample_ref in self.iter_caption_records(sample_refs, on_missing_caption=on_missing_caption):
            image_path = str(sample_ref)
            relative_path = normalize_data_path(image_path, dataset_root)
            metadata = {
                "image_path": image_path,
                "data_backend_id": self.id,
                "prompt": caption,
                "dataset_relative_path": relative_path,
            }
            metadata_builder = getattr(model, "text_embed_cache_metadata_for_filepath", None)
            if callable(metadata_builder):
                metadata.update(
                    metadata_builder(
                        init_backend=init_backend,
                        image_path=image_path,
                        prompt=caption,
                        data_backend_id=self.id,
                        dataset_relative_path=relative_path,
                    )
                )
            if key_type is TextEmbedCacheKey.DATASET_AND_FILENAME:
                key = f"{self.id}:{relative_path}"
            elif key_type is TextEmbedCacheKey.FILENAME:
                key = normalize_data_path(image_path, None)
            else:
                key = caption
            key_builder = getattr(model, "text_embed_cache_key_value", None)
            if callable(key_builder):
                key = key_builder(prompt=caption, default_key=key, metadata=metadata)
            yield {"prompt": caption, "key": key, "metadata": metadata}

            sample_metadata = self.metadata_backend.get_metadata_by_filepath(image_path)
            if sample_metadata is not None:
                for index, entity in enumerate(sample_metadata.get("bbox_entities") or []):
                    label = entity.get("label", "")
                    if label:
                        yield {
                            "prompt": label,
                            "key": f"{relative_path}__bbox_{index}",
                            "metadata": {"grounding_entity": True},
                        }

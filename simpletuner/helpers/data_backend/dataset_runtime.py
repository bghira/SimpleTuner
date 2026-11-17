"""Dataset access shared by training and preparation consumers."""

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.helpers.prompts import PromptHandler


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

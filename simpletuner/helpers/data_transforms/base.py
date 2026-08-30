"""Runtime expansion for generated dataset transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Type


class DataTransformTask(ABC):
    """Base class for data transforms that expand one dataset into another."""

    TASK: str = ""
    SUPPORTED_SOURCE_DATASET_TYPES: tuple[str, ...] = ()
    SOURCE_ONLY_BY_DEFAULT = False
    REQUIRES_METADATA_CLONE = False

    def __init__(
        self,
        global_config: Any,
        source_backend_config: Dict[str, Any],
        transform_config: Dict[str, Any],
        accelerator: Any = None,
    ) -> None:
        self.global_config = global_config
        self.source_backend_config = source_backend_config
        self.transform_config = deepcopy(transform_config)
        self.accelerator = accelerator

    @abstractmethod
    def prepare(self, existing_backend_ids: set[str]) -> List[Dict[str, Any]]:
        """Materialize any required artifacts and return generated backend configs."""

    def validate_source_dataset_type(self) -> None:
        dataset_type = self.source_backend_config.get("dataset_type", "image")
        if self.SUPPORTED_SOURCE_DATASET_TYPES and dataset_type not in self.SUPPORTED_SOURCE_DATASET_TYPES:
            supported = ", ".join(self.SUPPORTED_SOURCE_DATASET_TYPES)
            raise ValueError(
                f"Data transform '{self.TASK}' only supports source dataset_type values: {supported}. "
                f"Received dataset_type={dataset_type!r} for backend {self.source_backend_config.get('id')!r}."
            )


_TASK_REGISTRY: Dict[str, Type[DataTransformTask]] = {}


def register_data_transform(task_cls: Type[DataTransformTask]) -> Type[DataTransformTask]:
    if not task_cls.TASK:
        raise ValueError("Data transform task classes must define TASK.")
    _TASK_REGISTRY[task_cls.TASK] = task_cls
    return task_cls


def get_data_transform_task(task: str) -> Type[DataTransformTask]:
    try:
        return _TASK_REGISTRY[task]
    except KeyError as exc:
        known = ", ".join(sorted(_TASK_REGISTRY)) or "<none>"
        raise ValueError(f"Unknown data transform task {task!r}. Known tasks: {known}.") from exc


def _normalise_transform_list(source_backend_id: str, transform_block: Any) -> List[Dict[str, Any]]:
    if transform_block in (None, [], {}):
        return []
    if isinstance(transform_block, dict):
        return [transform_block]
    if not isinstance(transform_block, list):
        raise ValueError(f"data_transforms for backend {source_backend_id!r} must be a dict or list of dicts.")
    for transform in transform_block:
        if not isinstance(transform, dict):
            raise ValueError(f"Every data_transforms entry for backend {source_backend_id!r} must be a dict.")
    return transform_block


def process_data_transforms(
    global_config: Any,
    data_backend_config: List[Dict[str, Any]],
    accelerator: Any = None,
) -> List[Dict[str, Any]]:
    """Expand configured transforms into additional dataset backends."""
    generated_backends: List[Dict[str, Any]] = []
    existing_backend_ids = {backend.get("id") for backend in data_backend_config if backend.get("id")}

    for backend in data_backend_config:
        if backend.get("disabled", False) or backend.get("disable", False):
            continue

        transform_configs = _normalise_transform_list(backend.get("id", "<unknown>"), backend.get("data_transforms"))
        source_only = bool(backend.get("data_transform_source_only", False))
        for transform_config in transform_configs:
            task_name = transform_config.get("task")
            if not task_name:
                raise ValueError(f"data_transforms entry for backend {backend.get('id')!r} requires a 'task' value.")
            source_only = source_only or bool(transform_config.get("source_only", False))
            task_cls = get_data_transform_task(str(task_name))
            source_only = source_only or bool(getattr(task_cls, "SOURCE_ONLY_BY_DEFAULT", False))
            task = task_cls(
                global_config=global_config,
                source_backend_config=backend,
                transform_config=transform_config,
                accelerator=accelerator,
            )
            task.validate_source_dataset_type()
            new_backends = task.prepare(existing_backend_ids=existing_backend_ids)
            for new_backend in new_backends:
                backend_id = new_backend.get("id")
                if not backend_id:
                    raise ValueError(f"Data transform {task_name!r} generated a backend without an id.")
                if backend_id in existing_backend_ids:
                    raise ValueError(f"Data transform {task_name!r} generated duplicate backend id {backend_id!r}.")
                existing_backend_ids.add(backend_id)
                generated_backends.append(new_backend)
        if source_only and transform_configs:
            backend["_data_transform_source_only"] = True

    if generated_backends:
        data_backend_config.extend(generated_backends)
    return [backend for backend in data_backend_config if not backend.get("_data_transform_source_only", False)]

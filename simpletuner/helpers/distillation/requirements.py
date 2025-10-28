"""Parsing helpers for distiller data requirements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from simpletuner.helpers.data_backend.dataset_types import DatasetType, ensure_dataset_type


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _dedupe_preserve_order(values: Iterable[DatasetType]) -> Tuple[DatasetType, ...]:
    seen: list[DatasetType] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return tuple(seen)


@dataclass(frozen=True)
class DataRequirement:
    """Represents a single dependency group (must satisfy one of the listed dataset types)."""

    dataset_types: Tuple[DatasetType, ...]

    def __post_init__(self) -> None:
        if not self.dataset_types:
            raise ValueError("DataRequirement requires at least one dataset type.")

    def is_satisfied_by(self, dataset_type: DatasetType) -> bool:
        """Return True if the provided dataset type fulfils this requirement group."""
        return dataset_type in self.dataset_types

    def describe(self) -> str:
        """Human-readable description (e.g., 'image | video')."""
        return " | ".join(dataset_type.value for dataset_type in self.dataset_types)


@dataclass(frozen=True)
class DistillerRequirementProfile:
    """Normalised requirement data for a registered distiller."""

    requirements: Tuple[DataRequirement, ...] = ()
    is_data_generator: bool = False
    notes: Tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.requirements) or self.is_data_generator

    def requires_dataset_type(self, dataset_type: DatasetType) -> bool:
        """Return True if any requirement explicitly lists the provided dataset type."""
        return any(req.is_satisfied_by(dataset_type) for req in self.requirements)


EMPTY_PROFILE = DistillerRequirementProfile()


@dataclass(frozen=True)
class RequirementEvaluation:
    """Result of verifying dataset declarations against a requirement profile."""

    profile: DistillerRequirementProfile
    dataset_types: Tuple[DatasetType, ...]
    missing_requirements: Tuple[DataRequirement, ...]

    @property
    def fulfilled(self) -> bool:
        return not self.missing_requirements


def _parse_requirement_group(entry: Any) -> DataRequirement:
    if entry is None:
        raise ValueError("Requirement groups may not contain None entries.")

    if _is_sequence(entry):
        if not entry:
            raise ValueError("Requirement groups must list at least one dataset type.")
        normalized = _dedupe_preserve_order(ensure_dataset_type(item) for item in entry)
    else:
        normalized = (ensure_dataset_type(entry),)
    return DataRequirement(dataset_types=normalized)


def parse_requirement_matrix(raw: Any) -> Tuple[DataRequirement, ...]:
    """Convert user metadata into a tuple of DataRequirement objects."""
    if raw in (None, False, [], ()):
        return ()

    if not _is_sequence(raw):
        return (_parse_requirement_group(raw),)

    requirements: list[DataRequirement] = []
    for entry in raw:
        requirements.append(_parse_requirement_group(entry))
    return tuple(requirements)


def _normalize_notes(raw: Any) -> Tuple[str, ...]:
    if not raw:
        return ()
    if _is_sequence(raw):
        values = raw
    else:
        values = (raw,)

    cleaned: list[str] = []
    for item in values:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return tuple(cleaned)


def parse_distiller_requirement_profile(metadata: Mapping[str, Any] | None) -> DistillerRequirementProfile:
    """Build a DistillerRequirementProfile from registry metadata."""
    if not metadata:
        return EMPTY_PROFILE

    try:
        requirements = parse_requirement_matrix(metadata.get("data_requirements"))
    except ValueError as exc:
        raise ValueError(f"Invalid data requirement metadata: {exc}") from exc

    is_data_generator = bool(metadata.get("is_data_generator", False))
    notes = _normalize_notes(metadata.get("requirement_notes"))
    if not requirements and not is_data_generator and not notes:
        return EMPTY_PROFILE

    return DistillerRequirementProfile(
        requirements=requirements,
        is_data_generator=is_data_generator,
        notes=notes,
    )


def _entry_disabled(entry: Any) -> bool:
    return isinstance(entry, Mapping) and bool(entry.get("disabled") or entry.get("disable"))


def _iter_dataset_entries(datasets: Any) -> Iterable[Any]:
    if datasets is None:
        return []
    if isinstance(datasets, Mapping):
        return list(datasets.values())
    if isinstance(datasets, (list, tuple, set)):
        return list(datasets)
    return [datasets]


def _extract_dataset_type(entry: Any) -> Optional[DatasetType]:
    if entry is None:
        return None
    if isinstance(entry, DatasetType):
        return entry
    if isinstance(entry, str):
        try:
            return ensure_dataset_type(entry)
        except ValueError:
            return None
    if isinstance(entry, Mapping):
        dataset_type = entry.get("dataset_type")
        try:
            return ensure_dataset_type(dataset_type, default=DatasetType.IMAGE)
        except ValueError:
            return None
    return None


def collect_dataset_types(datasets: Any, *, include_disabled: bool = False) -> Tuple[DatasetType, ...]:
    """Convert heterogeneous dataset declarations into dataset type enums."""
    collected: list[DatasetType] = []
    for entry in _iter_dataset_entries(datasets):
        if not include_disabled and _entry_disabled(entry):
            continue
        dataset_type = _extract_dataset_type(entry)
        if dataset_type is not None:
            collected.append(dataset_type)
    return tuple(collected)


def evaluate_requirement_profile(
    profile: Optional[DistillerRequirementProfile],
    datasets: Any,
    *,
    include_disabled: bool = False,
) -> RequirementEvaluation:
    """Check whether datasets satisfy the requirement profile."""
    normalized_profile = profile or EMPTY_PROFILE
    dataset_types = collect_dataset_types(datasets, include_disabled=include_disabled)
    if not normalized_profile.requirements:
        missing: Tuple[DataRequirement, ...] = ()
    else:
        missing = tuple(
            requirement
            for requirement in normalized_profile.requirements
            if not any(requirement.is_satisfied_by(dataset_type) for dataset_type in dataset_types)
        )

    return RequirementEvaluation(
        profile=normalized_profile,
        dataset_types=dataset_types,
        missing_requirements=missing,
    )


def describe_requirement_groups(groups: Iterable[DataRequirement]) -> str:
    """Render requirement groups for human-readable error messages."""
    rendered = [f"[{group.describe()}]" for group in groups]
    return ", ".join(rendered)

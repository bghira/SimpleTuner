import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Pattern, Sequence

_PATH_MATCH_MODES = {"auto", "contains", "glob", "regex", "exact"}
_ROW_FILTER_KEYS = {"collection", "quality_thresholds", "min_width", "min_height"}


@dataclass(frozen=True)
class _CompiledPathPattern:
    pattern: str
    mode: str
    regex: Optional[Pattern[str]] = None

    @classmethod
    def build(cls, pattern: str, mode: str) -> "_CompiledPathPattern":
        compiled_mode = _resolve_path_pattern_mode(pattern, mode)
        compiled_pattern = (
            pattern[3:] if mode == "auto" and compiled_mode == "regex" and pattern.startswith("re:") else pattern
        )
        if compiled_mode == "regex":
            try:
                regex = re.compile(compiled_pattern)
            except re.error as exc:
                raise ValueError(f"Invalid filter_func path regex pattern: {exc}") from exc
            return cls(pattern=compiled_pattern, mode=compiled_mode, regex=regex)
        return cls(pattern=compiled_pattern, mode=compiled_mode)

    def matches(self, value: str) -> bool:
        if self.mode == "contains":
            return self.pattern in value
        if self.mode == "exact":
            return self.pattern == value
        if self.mode == "glob":
            return fnmatch.fnmatchcase(value, self.pattern)
        if self.regex is None:
            raise RuntimeError("Regex path matcher was constructed without a compiled pattern.")
        return self.regex.search(value) is not None


@dataclass(frozen=True)
class PathFilterSpec:
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    mode: str = "contains"
    include_matchers: tuple[_CompiledPathPattern, ...] = ()
    exclude_matchers: tuple[_CompiledPathPattern, ...] = ()

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "PathFilterSpec":
        flat_config_present = any(key in config for key in ("path_include", "path_exclude", "path_match"))
        path_config = config.get("path")
        if path_config is None and not flat_config_present:
            return cls()
        if path_config is not None and not isinstance(path_config, Mapping):
            raise ValueError("filter_func.path must be an object.")
        path_config = path_config or {}

        mode_default = "auto" if flat_config_present else "contains"
        mode = str(config.get("path_match", path_config.get("mode", mode_default))).lower()
        if mode not in _PATH_MATCH_MODES:
            raise ValueError("filter_func.path_match must be 'auto', 'contains', 'glob', 'regex', or 'exact'.")

        include = _normalise_patterns(config.get("path_include", path_config.get("include")))
        exclude = _normalise_patterns(config.get("path_exclude", path_config.get("exclude")))
        return cls(
            include=include,
            exclude=exclude,
            mode=mode,
            include_matchers=tuple(_CompiledPathPattern.build(pattern, mode) for pattern in include),
            exclude_matchers=tuple(_CompiledPathPattern.build(pattern, mode) for pattern in exclude),
        )

    def active(self) -> bool:
        return bool(self.include or self.exclude)

    def matches(self, path: Any) -> bool:
        if not self.active():
            return True

        candidates = _path_candidates(path)
        if self.include_matchers and not any(
            self._matches_any(candidate, self.include_matchers) for candidate in candidates
        ):
            return False
        if self.exclude_matchers and any(self._matches_any(candidate, self.exclude_matchers) for candidate in candidates):
            return False
        return True

    def to_webshart_kwargs(self) -> dict[str, Any]:
        return {
            "path_include": list(self.include),
            "path_exclude": list(self.exclude),
            "path_filter_mode": self.mode,
        }

    def _matches_any(self, value: str, patterns: Sequence[_CompiledPathPattern]) -> bool:
        return any(pattern.matches(value) for pattern in patterns)


class DatasetFilter:
    def __init__(self, config: Optional[Mapping[str, Any]]) -> None:
        if config is None:
            config = {}
        if not isinstance(config, Mapping):
            raise ValueError("filter_func must be an object.")
        self.config = dict(config)
        self.path = PathFilterSpec.from_config(self.config)
        self.collection_values = None
        if "collection" in self.config:
            self.collection_values = frozenset(_normalise_item_values(self.config["collection"]))

    def active(self) -> bool:
        return self.path_active() or any(key in self.config for key in _ROW_FILTER_KEYS)

    def path_active(self) -> bool:
        return self.path.active()

    def matches_path(self, path: Any) -> bool:
        return self.path.matches(path)

    def matches_item(self, item: Mapping[str, Any], path: Any = None) -> bool:
        if self.collection_values is not None:
            collection_values = _normalise_item_values(item.get("collection"))
            if not any(value in self.collection_values for value in collection_values):
                return False

        if "quality_thresholds" in self.config:
            quality = item.get(self.config.get("quality_column", "quality_assessment"), {})
            if not quality:
                return False
            for metric, threshold in self.config["quality_thresholds"].items():
                if quality.get(metric, 0) < threshold:
                    return False

        if "min_width" in self.config and item.get("width", 0) < self.config["min_width"]:
            return False
        if "min_height" in self.config and item.get("height", 0) < self.config["min_height"]:
            return False

        if self.path_active():
            path_values = [path] if path is not None else []
            path_values.extend(item.get(key) for key in ("file_name", "filename", "path") if item.get(key) is not None)
            if not path_values:
                return False
            if not any(self.path.matches(candidate) for candidate in path_values):
                return False

        return True


def resolve_filter_config(backend_config: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    filter_config = backend_config.get("filter_func")
    if filter_config is not None:
        return filter_config

    hf_config = backend_config.get("huggingface", {})
    if isinstance(hf_config, Mapping):
        return hf_config.get("filter_func")
    return None


def build_dataset_filter(backend_config: Mapping[str, Any]) -> Optional[DatasetFilter]:
    filter_config = resolve_filter_config(backend_config)
    if filter_config is None:
        return None
    dataset_filter = DatasetFilter(filter_config)
    return dataset_filter if dataset_filter.active() else None


def filter_file_list(file_list: Sequence[Any], dataset_filter: Optional[DatasetFilter]) -> list[Any]:
    if dataset_filter is None or not dataset_filter.path_active():
        return list(file_list)
    return [path for path in file_list if dataset_filter.matches_path(path)]


def _normalise_patterns(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence):
        values = value
    else:
        raise ValueError("filter_func path include/exclude values must be strings or lists of strings.")
    return tuple(str(item) for item in values if str(item))


def _resolve_path_pattern_mode(pattern: str, mode: str) -> str:
    if mode != "auto":
        return mode
    if pattern.startswith("re:"):
        return "regex"
    if any(character in pattern for character in "*?"):
        return "glob"
    return "contains"


def _normalise_item_values(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _path_candidates(path: Any) -> tuple[str, ...]:
    value = os.fspath(path) if isinstance(path, Path) else str(path)
    basename = os.path.basename(value)
    return (value, basename) if basename != value else (value,)

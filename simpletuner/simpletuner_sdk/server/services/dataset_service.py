"""Dataset-related helpers shared across routes and services."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from ..utils.paths import get_config_directory, get_simpletuner_root, resolve_config_path
from .config_store import ConfigStore
from .dataset_plan import DatasetPlanStore
from .webui_state import WebUIStateStore


def _get_config_store(config_type: str = "model") -> ConfigStore:
    """Get a ConfigStore instance that respects user's configured configs_dir."""
    try:
        defaults = WebUIStateStore().load_defaults()
        if defaults.configs_dir:
            expanded_dir = Path(defaults.configs_dir).expanduser()
            return ConfigStore(config_dir=expanded_dir, config_type=config_type)
    except Exception:
        pass
    return ConfigStore(config_type=config_type)


def _format_dataset_path(path: Path) -> str:
    """Return a dataset path string relative to the project root when possible."""
    resolved = path.expanduser().resolve(strict=False)
    try:
        return str(resolved.relative_to(get_simpletuner_root()))
    except ValueError:
        return str(resolved)


def _normalise_dataset_path(
    resolved: Path,
    configs_dir: Optional[str] = None,
    extra_roots: Optional[Iterable[Path]] = None,
) -> str:
    """Return a normalised dataset path preferring relative paths within known config roots."""

    candidate_roots: List[Path] = []

    def _add_candidate(raw: Optional[Path | str]) -> None:
        if not raw:
            return
        try:
            candidate = Path(raw).expanduser().resolve(strict=False)
        except Exception:
            candidate = Path(raw).expanduser()
        if candidate not in candidate_roots:
            candidate_roots.append(candidate)

    _add_candidate(configs_dir)

    if extra_roots:
        for root in extra_roots:
            _add_candidate(root)

    for root in sorted(candidate_roots, key=lambda p: len(str(p)), reverse=True):
        try:
            relative = resolved.relative_to(root)
            posix = relative.as_posix()
            if posix and posix != ".":
                return posix
            return resolved.name
        except ValueError:
            continue

    return _format_dataset_path(resolved)


def normalize_dataset_config_value(
    value: Optional[str],
    configs_dir: Optional[str] = None,
) -> Optional[str]:
    """Return a canonical dataset config string that matches selector option values."""

    if not value:
        return value

    extra_roots: List[Path] = []
    try:
        dataset_plan_root = DatasetPlanStore().path.parent
        extra_roots.append(dataset_plan_root)
    except Exception:
        pass
    try:
        extra_roots.append(Path(get_config_directory()).expanduser())
    except Exception:
        pass
    try:
        extra_roots.append(Path(get_simpletuner_root()).expanduser())
    except Exception:
        pass

    try:
        resolved = resolve_config_path(value, config_dir=configs_dir, check_cwd_first=True)
    except Exception:
        resolved = None

    if resolved:
        return _normalise_dataset_path(Path(resolved), configs_dir=configs_dir, extra_roots=extra_roots)

    if configs_dir:
        try:
            config_basename = Path(configs_dir).expanduser().name
            parts = Path(value).parts
            if parts and parts[0] == config_basename and len(parts) > 1:
                trimmed = Path(*parts[1:])
                try:
                    resolved_trimmed = resolve_config_path(trimmed, config_dir=configs_dir, check_cwd_first=True)
                except Exception:
                    resolved_trimmed = None
                if resolved_trimmed:
                    return _normalise_dataset_path(Path(resolved_trimmed), configs_dir=configs_dir, extra_roots=extra_roots)
                return trimmed.as_posix()
        except Exception:
            pass

    return value


def build_data_backend_choices() -> List[Dict[str, str]]:
    """Collect available dataset configuration candidates for selection widgets."""

    config_roots: Set[Path] = set()
    options: Dict[str, Dict[str, str]] = {}
    option_priorities: Dict[str, int] = {}
    workspace_config_root: Optional[Path] = None

    # Detect workspace config root FIRST before setting default paths
    try:
        defaults = WebUIStateStore().load_defaults()
        if defaults.configs_dir:
            base = Path(defaults.configs_dir).expanduser()
            try:
                workspace_config_root = base.resolve(strict=False)
            except Exception:
                workspace_config_root = base
    except Exception:
        pass

    try:
        simpletuner_root = Path(get_simpletuner_root()).expanduser().resolve(strict=False)
    except Exception:
        simpletuner_root = None

    # Only use project config root if no workspace is configured
    project_config_root: Optional[Path] = None
    if not workspace_config_root and simpletuner_root is not None:
        candidate_project_root = simpletuner_root.parent / "config"
        try:
            resolved_candidate = candidate_project_root.expanduser().resolve(strict=False)
            if resolved_candidate.exists():
                project_config_root = resolved_candidate
        except Exception:
            pass

    # Only use SimpleTuner's default config directory if no workspace is configured
    default_config_root: Optional[Path] = None
    if not workspace_config_root:
        try:
            default_config_root = Path(get_config_directory()).expanduser().resolve(strict=False)
        except Exception:
            default_config_root = None

    def _is_under(path: Path, root: Optional[Path]) -> bool:
        if not root:
            return False
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _compute_priority(resolved: Path) -> int:
        if workspace_config_root and _is_under(resolved, workspace_config_root):
            return 3
        if project_config_root and _is_under(resolved, project_config_root):
            return 2
        if default_config_root and _is_under(resolved, default_config_root):
            return 1
        return 2

    def _add_config_root(raw_path: Optional[Path]) -> None:
        if not raw_path:
            return
        try:
            candidate = Path(raw_path).expanduser().resolve(strict=False)
        except Exception:
            candidate = Path(raw_path).expanduser()
        config_roots.add(candidate)

    def _relative_to_config_roots(path: Path) -> str:
        for root in sorted(config_roots, key=lambda p: len(str(p)), reverse=True):
            try:
                return str(path.relative_to(root))
            except ValueError:
                continue
        try:
            default_root = Path(get_config_directory()).expanduser().resolve(strict=False)
            return str(path.relative_to(default_root))
        except Exception:
            pass
        try:
            project_root = Path(get_simpletuner_root()).expanduser().resolve(strict=False)
            return str(path.relative_to(project_root))
        except Exception:
            pass
        return path.name

    def _extract_environment_name(path: Path) -> Optional[str]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                metadata = data.get("_metadata")
                if isinstance(metadata, dict):
                    name = metadata.get("name")
                    if isinstance(name, str) and name.strip():
                        return name.strip()
        except Exception:
            return None
        return None

    def _register_path(
        path: Path,
        env_name: Optional[str] = None,
    ) -> None:
        if not path:
            return
        resolved = path.expanduser().resolve(strict=False)
        value = _normalise_dataset_path(
            resolved,
            configs_dir=str(workspace_config_root) if workspace_config_root else None,
            extra_roots=config_roots,
        )

        rel_path = _relative_to_config_roots(resolved)
        rel_path_path = Path(rel_path)
        parent_segment = rel_path_path.parent if rel_path_path.parent != Path(".") else None

        environment = env_name or _extract_environment_name(resolved) or resolved.stem
        if not env_name and parent_segment:
            environment = parent_segment.name

        display_key = f"{(environment or '').strip().lower()}|{rel_path.lower()}"
        priority = _compute_priority(resolved)
        existing_priority = option_priorities.get(display_key)

        if existing_priority is not None and priority <= existing_priority:
            return

        options[display_key] = {
            "value": value,
            "environment": environment,
            "path": rel_path,
        }
        option_priorities[display_key] = priority

    candidate_dirs: Set[Path] = set()

    def _add_candidate_dir(raw_path: Optional[Path]) -> None:
        if not raw_path:
            return
        try:
            candidate = Path(raw_path).expanduser()
        except Exception:
            return

        try:
            resolved_candidate = candidate.resolve(strict=False)
        except Exception:
            resolved_candidate = candidate

        candidate_dirs.add(resolved_candidate)
        _add_config_root(resolved_candidate)

    # Include user-configured workspace directory (already detected at top)
    if workspace_config_root:
        _add_candidate_dir(workspace_config_root)

    # Include ConfigStore-managed dataloader configs and directory
    try:
        dataloader_store = _get_config_store(config_type="dataloader")
        # Only add dataloader store config dir if it matches workspace or no workspace is configured
        dataloader_config_path = Path(dataloader_store.config_dir).resolve()
        if (
            not workspace_config_root
            or dataloader_config_path == workspace_config_root
            or dataloader_config_path.parent == workspace_config_root
        ):
            _add_candidate_dir(dataloader_config_path)
        for metadata in dataloader_store.list_configs():
            name = metadata.get("name")
            if not name:
                continue
            try:
                path = dataloader_store._get_config_path(name)
            except Exception:
                continue
            if not path or not path.exists():
                continue

            _register_path(path, env_name=str(name))
    except Exception:
        pass

    # Include dataset plan store path (global plan)
    try:
        dataset_plan_store = DatasetPlanStore()
        dataset_plan_parent = dataset_plan_store.path.parent.resolve()
        # Only add dataset plan parent if it's the workspace or no workspace is configured
        if (
            not workspace_config_root
            or dataset_plan_parent == workspace_config_root
            or dataset_plan_parent.parent == workspace_config_root
        ):
            _add_candidate_dir(dataset_plan_parent)
        if dataset_plan_store.path.exists():
            _register_path(dataset_plan_store.path)
    except Exception:
        pass

    # Include paths referenced by active configuration
    try:
        model_store = _get_config_store()
        # Only add model store config dir if it's the workspace config root or if no workspace is configured
        model_config_path = Path(model_store.config_dir).resolve()
        if not workspace_config_root or model_config_path == workspace_config_root:
            _add_candidate_dir(model_config_path)
        active_name = model_store.get_active_config()
        if active_name:
            config, _ = model_store.load_config(active_name)
            backend_path = config.get("--data_backend_config") or config.get("data_backend_config")
            if backend_path:
                resolved = resolve_config_path(backend_path, config_dir=model_store.config_dir, check_cwd_first=True)
                if resolved and resolved.exists():
                    _register_path(resolved)
                    # Only add parent dir if it's not already under workspace config root
                    # (workspace will be scanned recursively anyway)
                    parent_dir = resolved.parent.resolve()
                    should_add_parent = True
                    if workspace_config_root:
                        try:
                            parent_dir.relative_to(workspace_config_root)
                            # Parent is under workspace, don't add separately
                            should_add_parent = False
                        except ValueError:
                            # Parent is not under workspace, add it
                            pass
                    if should_add_parent:
                        _add_candidate_dir(parent_dir)
    except Exception:
        pass

    # Include the default config directory under SimpleTuner installation
    # ONLY if no workspace config root is configured (to avoid showing example configs from source tree)
    if not workspace_config_root:
        try:
            _add_candidate_dir(Path(get_config_directory()))
        except Exception:
            pass

    # Discover dataset configs within the candidate directories
    # Only search recursively in known config directories, not CWD
    cwd = Path.cwd().resolve()

    # Determine what are "safe" config directories (not CWD or its random subdirs)
    safe_config_dirs = set()
    if workspace_config_root:
        safe_config_dirs.add(workspace_config_root.resolve())
    if project_config_root:
        safe_config_dirs.add(project_config_root.resolve())
    if default_config_root:
        safe_config_dirs.add(default_config_root.resolve())

    for directory in sorted(candidate_dirs):
        if not directory or not directory.exists():
            continue
        try:
            resolved_dir = directory.resolve()

            # Check if this directory is under one of the safe config roots
            is_safe = False
            for safe_root in safe_config_dirs:
                try:
                    resolved_dir.relative_to(safe_root)
                    is_safe = True
                    break
                except ValueError:
                    continue

            if is_safe:
                # Recursive search in safe config directories
                for path in directory.glob("**/multidatabackend*.json"):
                    _register_path(path)
            else:
                # For non-safe directories (like CWD or random paths from active config),
                # only do shallow search to avoid picking up test/example files
                # Direct files only
                for path in directory.glob("multidatabackend*.json"):
                    _register_path(path)
                # One level of subdirectories
                try:
                    for subdir in directory.iterdir():
                        if subdir.is_dir():
                            for path in subdir.glob("multidatabackend*.json"):
                                _register_path(path)
                except (PermissionError, OSError):
                    pass
        except Exception:
            continue

    if not options:
        return []

    entries = list(options.values())
    max_env_len = max(len(entry["environment"]) for entry in entries)

    for entry in entries:
        entry["label"] = f"{entry['environment']:<{max_env_len}} | {entry['path']}"

    entries.sort(key=lambda item: item["environment"].lower())
    return entries

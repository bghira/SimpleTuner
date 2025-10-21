"""Path utilities for SimpleTuner server."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


def get_simpletuner_root() -> Path:
    """Get the root directory of SimpleTuner installation.

    Returns:
        Path to SimpleTuner root directory (package dir for installed, project root for source)
    """
    import simpletuner

    # Get the path to the simpletuner package directory
    simpletuner_package = Path(simpletuner.__file__).parent

    # Check if we're running from a source checkout by looking for pyproject.toml
    # one level up (this means we're in a development environment)
    potential_project_root = simpletuner_package.parent
    if (potential_project_root / "pyproject.toml").exists():
        # Running from source - return project root, not package dir
        return potential_project_root

    # Installed as package - return package directory
    return simpletuner_package


def get_config_directory() -> Path:
    """Get the default configuration directory.

    Returns:
        Path to the config directory relative to SimpleTuner root
    """
    env_override = os.environ.get("SIMPLETUNER_CONFIG_DIR")
    if env_override:
        return Path(env_override).expanduser()

    candidate_roots = []
    if Path("/workspace").exists():
        candidate_roots.append(Path("/workspace/simpletuner"))
    if Path("/notebooks").exists():
        candidate_roots.append(Path("/notebooks/simpletuner"))
    candidate_roots.append(Path.home() / ".simpletuner")

    for root in candidate_roots:
        candidate = root / "config"
        if candidate.exists():
            return candidate

    if candidate_roots:
        preferred = candidate_roots[0] / "config"
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred

    # Fall back to project/package config directory
    default_dir = get_simpletuner_root() / "config"
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir


def get_template_directory() -> Path:
    """Get the template directory.

    Returns:
        Path to the templates directory (always in package, not project root)
    """
    import simpletuner

    # Templates are always in the package directory, not project root
    simpletuner_package = Path(simpletuner.__file__).parent
    return simpletuner_package / "templates"


def get_static_directory() -> Path:
    """Get the static files directory.

    Returns:
        Path to the static directory (always in package, not project root)
    """
    import simpletuner

    # Static files are always in the package directory, not project root
    simpletuner_package = Path(simpletuner.__file__).parent
    return simpletuner_package / "static"


def resolve_config_path(
    path: Union[str, Path],
    config_dir: Optional[Union[str, Path]] = None,
    check_cwd_first: bool = True,
) -> Optional[Path]:
    """Resolve a configuration file path using multiple resolution strategies.

    For absolute paths:
    - Expands user paths (~/)
    - Returns the path as-is if it exists

    For relative paths, checks in order:
    1. Relative to the provided config_dir (if supplied)
    2. Relative to current working directory (if check_cwd_first is True)
    3. Relative to SimpleTuner's default config directory
    4. Relative to SimpleTuner's root directory (for paths like 'config/...')

    Args:
        path: The path to resolve (can be relative or absolute)
        config_dir: Optional custom config directory to check
        check_cwd_first: Whether to check CWD first for relative paths

    Returns:
        Resolved Path object if file exists, None otherwise
    """
    path_str = str(path)

    # Expand user path if present
    expanded_path = os.path.expanduser(path_str)

    # If it's an absolute path, return it if it exists
    if os.path.isabs(expanded_path):
        abs_path = Path(expanded_path)
        return abs_path if abs_path.exists() else None

    # For relative paths, try multiple resolution strategies
    paths_to_check = []

    # 1. Check relative to provided config directory (user preference wins)
    if config_dir:
        config_path = Path(os.path.expanduser(str(config_dir)))
        paths_to_check.append(config_path / expanded_path)

    # 2. Check relative to CWD (only when explicitly allowed)
    if check_cwd_first:
        paths_to_check.append(Path.cwd() / expanded_path)

    # 3. Check relative to SimpleTuner's default config directory
    default_config = get_config_directory()
    paths_to_check.append(default_config / expanded_path)

    # 4. Check relative to SimpleTuner root (for paths like 'config/examples/...')
    simpletuner_root = get_simpletuner_root()
    paths_to_check.append(simpletuner_root / expanded_path)

    # Return the first existing path
    for check_path in paths_to_check:
        if check_path.exists():
            return check_path.resolve()

    # Handle legacy paths that redundantly include the config directory name,
    # e.g. config/deepfloyd/multidatabackend.json when the configs_dir already
    # points at .../config. In that scenario drop the leading segment and retry.
    if config_dir and not expanded_path.startswith(os.sep):
        try:
            parts = Path(expanded_path).parts
        except Exception:
            parts = ()

        if parts:
            config_root = Path(os.path.expanduser(str(config_dir)))
            config_basename = config_root.name.lower()
            leading_variants = {config_basename, "config", "configs"}
            if config_basename.endswith("s") and len(config_basename) > 1:
                leading_variants.add(config_basename[:-1])

            first_part = parts[0].lower()
            if first_part in leading_variants and len(parts) > 1:
                trimmed_path = Path(os.path.join(*parts[1:]))
                alt_candidate = config_root / trimmed_path
                if alt_candidate.exists():
                    return alt_candidate.resolve()

    return None

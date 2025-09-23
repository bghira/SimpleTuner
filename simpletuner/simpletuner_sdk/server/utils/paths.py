"""Path utilities for SimpleTuner server."""

from __future__ import annotations

from pathlib import Path


def get_simpletuner_root() -> Path:
    """Get the root directory of SimpleTuner installation.

    Returns:
        Path to SimpleTuner root directory
    """
    import simpletuner

    # Get the path to the simpletuner package
    simpletuner_package = Path(simpletuner.__file__).parent

    # Go up one level to get the project root
    return simpletuner_package.parent


def get_config_directory() -> Path:
    """Get the default configuration directory.

    Returns:
        Path to the config directory relative to SimpleTuner root
    """
    return get_simpletuner_root() / "config"


def get_template_directory() -> Path:
    """Get the template directory.

    Returns:
        Path to the templates directory relative to SimpleTuner root
    """
    return get_simpletuner_root() / "templates"


def get_static_directory() -> Path:
    """Get the static files directory.

    Returns:
        Path to the static directory relative to SimpleTuner root
    """
    return get_simpletuner_root() / "static"
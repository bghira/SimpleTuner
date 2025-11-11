"""Lightweight compatibility stubs used across unit tests."""

import importlib.machinery
import logging
import sys
import types

_DEF_SPEC = importlib.machinery.ModuleSpec


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__spec__ = _DEF_SPEC(name, loader=None)
        sys.modules[name] = module
    return module


def install_peft_stub():
    """Ensure the real peft dependency is importable during tests."""
    try:
        import peft  # type: ignore  # noqa: F401
        import peft.tuners  # type: ignore  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "PEFT must be installed for the test suite. Activate the project's virtualenv before running tests."
        ) from exc


def install_diffusers_stub():
    """Ensure the real diffusers dependency is importable during tests."""
    try:
        import diffusers  # type: ignore  # noqa: F401
        from diffusers.configuration_utils import ConfigMixin  # type: ignore

        if not hasattr(ConfigMixin, "register_to_config"):
            raise AttributeError
    except (ModuleNotFoundError, AttributeError) as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "Diffusers must be installed for the test suite. Activate the project's virtualenv before running tests."
        ) from exc


def ensure_test_stubs_installed() -> None:
    install_peft_stub()
    install_diffusers_stub()


ensure_test_stubs_installed()

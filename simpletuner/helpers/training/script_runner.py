import logging
import os
import shlex
import string
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.utils.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)

# Shared executor so hooks don't block training loops
_EXECUTOR: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="external_script")
    return _EXECUTOR


def build_script_command(template: str, value_resolver: Callable[[str], str | None]) -> list[str]:
    """
    Expand placeholders in a script template and split it into a command list.

    Args:
        template: A string containing optional `{placeholder}` tokens.
        value_resolver: Callable that returns a string value for a placeholder name, or None for empty.

    Raises:
        ValueError: If the template is empty, contains unknown placeholders, or resolves to no command tokens.
    """
    if template in (None, "", "None"):
        raise ValueError("Script template must be a non-empty string.")

    formatter = string.Formatter()
    placeholders = {field_name for _, field_name, _, _ in formatter.parse(template) if field_name}
    values: dict[str, str | None] = {}
    for name in placeholders:
        try:
            resolved = value_resolver(name)
        except KeyError as exc:
            raise ValueError(f"Unknown placeholder '{name}' in script template.") from exc
        values[name] = "" if resolved is None else str(resolved)

    expanded = os.path.expandvars(os.path.expanduser(template.format(**values)))
    command = shlex.split(expanded)
    if not command:
        raise ValueError("Script template resolved to an empty command.")
    return command


def submit_script(command: list[str]) -> None:
    """Run the provided command asynchronously; logs but does not raise failures."""

    def _task():
        try:
            subprocess.run(command, check=True)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("External script failed (%s): %s", command, exc)

    _get_executor().submit(_task)


def resolve_latest_checkpoint_path(output_dir: str | None) -> str:
    if not output_dir:
        raise ValueError("Cannot resolve checkpoint path without output_dir.")
    checkpoint_manager = CheckpointManager(output_dir)
    latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
    if latest_checkpoint is None:
        raise ValueError("Script requires {local_checkpoint_path}, but no checkpoints exist in output_dir.")
    checkpoint_path = os.path.join(output_dir, latest_checkpoint)
    if not os.path.isdir(checkpoint_path):
        raise ValueError(f"Resolved checkpoint path '{checkpoint_path}', but it does not exist.")
    return checkpoint_path


def run_hook_script(
    script_template: str,
    *,
    config,
    local_path: str | None = None,
    remote_path: str | None = None,
    global_step: int | None = None,
) -> None:
    """Format and submit an external hook script with shared placeholders."""
    if script_template in (None, "", "None"):
        return
    output_dir = getattr(config, "output_dir", None)

    def _resolver(name: str):
        if name == "local_checkpoint_path":
            if local_path:
                return local_path
            return resolve_latest_checkpoint_path(output_dir)
        if name == "remote_checkpoint_path":
            return remote_path or ""
        if name == "global_step":
            step_value = global_step if global_step is not None else StateTracker.get_global_step()
            return "" if step_value is None else str(step_value)
        if name == "tracker_run_name":
            return getattr(config, "tracker_run_name", "") or ""
        if name == "tracker_project_name":
            return getattr(config, "tracker_project_name", "") or ""
        if name == "model_family":
            model_family = getattr(config, "model_family", None) or StateTracker.get_model_family()
            return "" if model_family is None else str(model_family)
        if name == "huggingface_path":
            return getattr(config, "hub_model_id", "") or ""
        if name == "model_type":
            return getattr(config, "model_type", "") or ""
        if name == "lora_type":
            return getattr(config, "lora_type", "") or ""
        if name.startswith("validation_"):
            return getattr(config, name, "") or ""
        raise KeyError(name)

    try:
        command = build_script_command(script_template, _resolver)
    except ValueError as exc:
        logger.error("Failed to format external script command: %s", exc)
        return
    submit_script(command)

import contextlib
import contextvars
from collections.abc import Callable, Iterator
from typing import Any

import torch


def _coerce_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def dynamo_config_patches(config: Any) -> dict[str, Any]:
    patches: dict[str, Any] = {}
    if _coerce_flag(getattr(config, "dynamo_dynamic", None)):
        patches["capture_dynamic_output_shape_ops"] = True
        patches["force_parameter_static_shapes"] = False
    return patches


@contextlib.contextmanager
def dynamo_config_context(config: Any) -> Iterator[None]:
    patches = dynamo_config_patches(config)
    if not patches:
        yield
        return

    with torch._dynamo.config.patch(patches):
        yield


def run_with_dynamo_config(config: Any, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    with dynamo_config_context(config):
        context = contextvars.copy_context()
        return context.run(fn, *args, **kwargs)

import os
from typing import Any, Callable

import torch


def _dynamo_backend_enabled() -> bool:
    backend = os.environ.get("TRAINING_DYNAMO_BACKEND", "")
    return backend.strip().lower() not in {"", "no", "none", "disabled"}


def checkpoint(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    use_reentrant = kwargs.get("use_reentrant")
    if use_reentrant is False and _dynamo_backend_enabled():
        kwargs.setdefault("determinism_check", "none")
        if torch.compiler.is_compiling():
            checkpoint_eager = torch.compiler.disable(torch.utils.checkpoint.checkpoint)
            return checkpoint_eager(function, *args, **kwargs)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)

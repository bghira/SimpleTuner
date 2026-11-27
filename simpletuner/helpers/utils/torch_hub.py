from pathlib import Path

import torch
from filelock import FileLock


def load_torch_hub_model(repo_or_dir, model: str, *args, **kwargs):
    """Load a GitHub entrypoint, reusing Torch Hub's cached default branch across ranks."""
    if kwargs.get("source", "github").lower() != "github":
        return torch.hub.load(repo_or_dir, model, *args, **kwargs)
    repo, separator, _ = repo_or_dir.partition(":")
    owner, name = repo.split("/")
    hub_dir = Path(torch.hub.get_dir())
    hub_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(hub_dir / f".simpletuner-{name}.lock"):
        if not separator and not kwargs.get("force_reload", False):
            # These are the two default refs Torch Hub resolves, in its preference order.
            for ref in ("main", "master"):
                if (hub_dir / f"{owner}_{name}_{ref}" / "hubconf.py").is_file():
                    repo_or_dir = f"{repo}:{ref}"
                    break
        return torch.hub.load(repo_or_dir, model, *args, **kwargs)

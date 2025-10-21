import os
from typing import Dict, Iterable, Optional

import torch

try:
    from diffusers.hooks import apply_group_offloading
    from diffusers.hooks.group_offloading import _is_group_offload_enabled

    _DIFFUSERS_GROUP_OFFLOAD_AVAILABLE = True
except ImportError:  # pragma: no cover - handled by runtime checks
    apply_group_offloading = None  # type: ignore[assignment]
    _is_group_offload_enabled = None  # type: ignore[assignment]
    _DIFFUSERS_GROUP_OFFLOAD_AVAILABLE = False


def enable_group_offload_on_components(
    components: Dict[str, torch.nn.Module],
    *,
    device: torch.device,
    offload_type: str = "block_level",
    number_blocks_per_group: Optional[int] = 1,
    use_stream: bool = False,
    record_stream: bool = False,
    low_cpu_mem_usage: bool = False,
    non_blocking: bool = False,
    offload_to_disk_path: Optional[str] = None,
    exclude: Optional[Iterable[str]] = None,
    required_import_error_message: str = "Group offloading requires diffusers>=0.33.0",
) -> None:
    """
    Apply diffusers group offloading to a set of pipeline components.

    Parameters
    ----------
    components:
        Dictionary of pipeline components (module name -> instance).
    device:
        Target device for on-loading modules (typically the accelerator device).
    offload_type:
        "block_level" (default) or "leaf_level".
    number_blocks_per_group:
        Number of blocks per group when using block-level offloading.
    use_stream:
        Whether to use CUDA streams for asynchronous transfers.
    record_stream / low_cpu_mem_usage / non_blocking:
        Additional flags routed to diffusers group offloading helpers.
    offload_to_disk_path:
        Optional directory to spill parameters to disk.
    exclude:
        Optional iterable of component names to skip (defaults to ["vae", "vqvae"]).
    required_import_error_message:
        Custom message if diffusers does not expose group offloading utilities.
    """

    if not _DIFFUSERS_GROUP_OFFLOAD_AVAILABLE:
        raise ImportError(required_import_error_message)

    onload_device = torch.device(device)
    offload_device = torch.device("cpu")

    if offload_to_disk_path:
        os.makedirs(offload_to_disk_path, exist_ok=True)

    excluded_names = set(exclude or [])
    if "vae" not in excluded_names:
        excluded_names.add("vae")
    if "vqvae" not in excluded_names:
        excluded_names.add("vqvae")

    for name, module in components.items():
        if name in excluded_names:
            continue

        if module is None or not isinstance(module, torch.nn.Module):
            continue

        if _is_group_offload_enabled(module):  # type: ignore[operator]
            continue

        kwargs = {
            "offload_type": offload_type,
            "use_stream": use_stream,
            "record_stream": record_stream,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "non_blocking": non_blocking,
            "offload_to_disk_path": offload_to_disk_path,
        }

        if offload_type == "block_level" and number_blocks_per_group is not None:
            kwargs["num_blocks_per_group"] = number_blocks_per_group

        if hasattr(module, "enable_group_offload"):
            module.enable_group_offload(  # type: ignore[call-arg]
                onload_device=onload_device,
                offload_device=offload_device,
                **kwargs,
            )
        else:
            apply_group_offloading(  # type: ignore[misc]
                module=module,
                onload_device=onload_device,
                offload_device=offload_device,
                **kwargs,
            )

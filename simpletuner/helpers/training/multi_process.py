import os
from typing import Iterable, List, Optional, TypeVar

import torch.distributed as dist


def _get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def _get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def rank_info() -> str:
    try:
        return f"(Rank: {_get_rank()}) "
    except Exception:
        return ""


def should_log() -> bool:
    return _get_rank() == 0


T = TypeVar("T")


def broadcast_object_from_main(obj: T, *, src: int = 0) -> T:
    """
    Broadcast an arbitrary Python object from the source rank to every process.
    """
    if not dist.is_available() or not dist.is_initialized() or _get_world_size() == 1:
        return obj
    payload: List[Optional[T]]
    if _get_rank() == src:
        payload = [obj]
    else:
        payload = [None]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]  # type: ignore[return-value]


def split_across_processes(accelerator, values: Iterable[T], *, apply_padding: bool = False) -> list[T]:
    """
    Evenly split an iterable across accelerator processes, mirroring the text embed cache strategy.
    """
    if accelerator is None or getattr(accelerator, "num_processes", 1) <= 1:
        return list(values)
    sequence = list(values)
    if not sequence:
        return []
    with accelerator.split_between_processes(sequence, apply_padding=apply_padding) as shard:
        if shard is None:
            return []
        return list(shard)


def gather_across_processes(obj: T) -> list[T]:
    """
    Gather a picklable object from every process and return the non-null payloads.
    """
    if not dist.is_available() or not dist.is_initialized() or _get_world_size() == 1:
        return [obj]
    gathered: List[Optional[T]] = [None for _ in range(_get_world_size())]
    dist.all_gather_object(gathered, obj)
    return [item for item in gathered if item is not None]

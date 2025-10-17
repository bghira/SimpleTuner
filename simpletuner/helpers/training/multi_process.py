import os

import torch.distributed as dist


def _get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return int(os.environ.get("RANK", 0))


def rank_info():
    try:
        return f"(Rank: {_get_rank()}) "
    except:
        return ""


def should_log():
    return _get_rank() == 0

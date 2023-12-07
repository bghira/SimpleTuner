import accelerate
import torch.distributed as dist


def _get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def rank_info():
    try:
        return f"(Rank: {_get_rank()}) "
    except:
        return ""

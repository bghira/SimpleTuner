import torch


def rank_info():
    if not torch.is_distributed():
        return None
    try:
        return f"(Rank: {torch.distributed.get_rank()}) "
    except:
        return "(Rank info unavailable) "

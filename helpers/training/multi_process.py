import torch


def rank_info():
    try:
        if not torch.is_distributed():
            return None
        return f"(Rank: {torch.distributed.get_rank()}) "
    except:
        return "(Rank info unavailable) "

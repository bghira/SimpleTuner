import accelerate, torch


def rank_info(acc: accelerate.accelerator.Accelerator):
    try:
        if not acc.use_distributed_training:
            return None
        return f"(Rank: {torch.distributed.get_rank()}) "
    except:
        return "(Rank info unavailable) "

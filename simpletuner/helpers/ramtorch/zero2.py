import torch
import torch.distributed as dist

from .utils import register_ramtorch_hook


def setup_grad_sharding_hooks(rank_param_groups, current_rank):
    """
    Setup backward hooks for gradient sharding using the same parameter
    assignment structure as ZeRO-1.

    Args:
        rank_param_groups: Dict from create_zero_param_groups()
        current_rank: Current process rank
    """
    # Create a mapping from parameter to its owner rank
    param_to_owner = {}

    for owner_rank, param_groups in rank_param_groups.items():
        for group in param_groups:
            for param in group["params"]:
                param_to_owner[param] = owner_rank

    def create_pytorch_zero_2_hook(param):

        # this grad hook will fire after grad is computed
        # and it will immediately reduce the grad towards the owner rank
        def hook(grad):
            owner_rank = param_to_owner[param]
            # reduce towards the owner rank
            # non ramtorch params will use default dist reduce stream
            dist.reduce(grad, dst=owner_rank, op=dist.ReduceOp.SUM, async_op=True)

            if current_rank == owner_rank:
                # owner rank keeps the reduced gradient
                return grad
            else:
                # boot grad on non-owner ranks
                return None

        return hook

    def create_ramtorch_zero_2_hook(param):

        # this grad hook will fire after grad is computed
        # and it will immediately reduce the grad towards the owner rank
        def hook(grad):
            owner_rank = param_to_owner[param]
            # reduce towards the owner rank
            # ramtorch uses internal stream to manage data transfer so async op is false here
            dist.reduce(grad, dst=owner_rank, op=dist.ReduceOp.SUM, async_op=False)

            if current_rank == owner_rank:
                # owner rank keeps the reduced gradient
                return grad
            else:
                # boot grad on non-owner ranks
                return None

        return hook

    handles = []
    # register hooks for all pre sharded parameters
    for owner_rank, param_groups in rank_param_groups.items():
        for group in param_groups:
            for param in group["params"]:
                # if the
                if hasattr(param, "is_ramtorch") and param.is_ramtorch:
                    handle = register_ramtorch_hook(
                        param,
                        create_ramtorch_zero_2_hook(param),
                        "_ramtorch_zero_2_hooks",
                    )
                else:
                    handle = param.register_hook(create_pytorch_zero_2_hook(param))

                handles.append(handle)

    return handles

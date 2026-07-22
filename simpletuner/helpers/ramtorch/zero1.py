import torch
import torch.distributed as dist


def create_zero_param_groups(param_groups, world_size):
    """
    Create parameter groups for ZeRO-1 optimizer sharding and generate a map
    of parameter owners for broadcasting.

    Args:
        param_groups: A list of parameter groups in the standard PyTorch format.
            Example:
            ```python
            [{'params': model.parameters(), 'lr': 0.001}]
            ```
        world_size: Total number of ranks.

    Returns:
        dict: similar to param_groups but chunked into ranks

    Example of input format:
    ```python
    param_groups = [
        {'params': norm_params, 'lr': 1e-3, 'weight_decay': 0.0},
        {'params': other_params, 'lr': 1e-3, 'weight_decay': 0.01}
    ]
    ```

    Example of output format, world_size=2:
    ```python
    # technically you can just construct this format
    # by yourself with your own sharding method
    rank_param_groups = {
        0: [
            {'params': norm_params_shard_0, 'lr': 1e-3, 'weight_decay': 0.0},
            {'params': other_params_shard_0, 'lr': 1e-3, 'weight_decay': 0.01}
        ],
        1: [
            {'params': norm_params_shard_1, 'lr': 1e-3, 'weight_decay': 0.0},
            {'params': other_params_shard_1, 'lr': 1e-3, 'weight_decay': 0.01}
        ],
    }
    # optimize the shard
    optim = AdamW(rank_param_groups[rank])
    ```
    """
    rank_param_groups = {rank: [] for rank in range(world_size)}

    for group_idx, group in enumerate(param_groups):
        # empty groups for each rank for (lr, weight_decay, whatever)
        group_config = {k: v for k, v in group.items() if k != "params"}

        for rank in range(world_size):
            rank_param_groups[rank].append({"params": [], **group_config})

        # round-robin the param reference
        current_rank = 0
        for param in group["params"]:
            # put it to current_rank
            rank_param_groups[current_rank][group_idx]["params"].append(param)

            current_rank = (current_rank + 1) % world_size

    return rank_param_groups


def broadcast_zero_params(rank_param_groups, async_op=True, include_ramtorch=False):
    """
    Broadcast parameters from owner ranks to the rest of the ranks after grad sync and optim step.

    Args:
        rank_param_groups: Dict mapping rank -> list of param groups for that rank
        async_op (bool): If True, performs non-blocking broadcasts and waits for
                         them before returning. Defaults to True.
        include_ramtorch (bool): If True, also broadcast RamTorch CPU parameters. Defaults to False.
    """
    work_handles = []
    with torch.no_grad():
        for owner_rank, param_groups in rank_param_groups.items():
            for group in param_groups:
                for param in group["params"]:
                    if getattr(param, "is_ramtorch", False) and not include_ramtorch:
                        continue
                    work_handle = dist.broadcast(param.data, src=owner_rank, async_op=async_op)
                    if async_op:
                        work_handles.append(work_handle)

        if not work_handles:
            return
        for handle in work_handles:
            handle.wait()

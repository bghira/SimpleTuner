from diffusers.utils.torch_utils import is_compiled_module


def _is_fsdp_module(model) -> bool:
    return model is not None and model.__class__.__name__ == "FullyShardedDataParallel"


def _unwrap_execution_wrappers(model):
    while model is not None and not _is_fsdp_module(model):
        wrapped = None
        if is_compiled_module(model):
            wrapped = model._orig_mod
        elif hasattr(model, "module"):
            wrapped = model.module

        if wrapped is None or wrapped is model:
            break
        model = wrapped
    return model


def unwrap_model(accelerator, model, keep_fp32_wrapper: bool = True):
    if accelerator is not None:
        try:
            model = accelerator.unwrap_model(model, keep_fp32_wrapper=keep_fp32_wrapper)
        except TypeError:
            model = accelerator.unwrap_model(model)
    return _unwrap_execution_wrappers(model)


def gather_dict_of_tensors_shapes(tensors: dict) -> dict:
    if "prompt_embeds" in tensors and isinstance(tensors["prompt_embeds"], list):
        # some models like HiDream return a list of batched tensors..
        return {k: [x.shape for x in v] for k, v in tensors.items()}
    else:
        return {k: v.shape if v is not None else None for k, v in tensors.items()}


def move_dict_of_tensors_to_device(tensors: dict, device) -> dict:
    """
    Move a dictionary of tensors to a specified device, including dictionaries of nested tensors in lists (HiDream outputs).

    Args:
        tensors (dict): Dictionary of tensors to move.
        device (torch.device): The device to move the tensors to.

    Returns:
        dict: Dictionary of tensors moved to the specified device.
    """
    if "prompt_embeds" in tensors and isinstance(tensors["prompt_embeds"], list):
        return {k: [x.to(device) for x in v] for k, v in tensors.items()}
    else:
        return {k: v.to(device) if v is not None else None for k, v in tensors.items()}

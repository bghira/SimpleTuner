from diffusers.utils.torch_utils import is_compiled_module


def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


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

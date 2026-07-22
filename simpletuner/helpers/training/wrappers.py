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
    def _shape(value):
        if isinstance(value, dict):
            return {k: _shape(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_shape(item) for item in value]
        if hasattr(value, "shape"):
            return value.shape
        return None

    if "prompt_embeds" in tensors and isinstance(tensors["prompt_embeds"], list):
        # some models like HiDream return a list of batched tensors..
        return {k: [x.shape for x in v] for k, v in tensors.items()}
    return {k: _shape(v) for k, v in tensors.items()}


def move_dict_of_tensors_to_device(tensors: dict, device) -> dict:
    """
    Move a dictionary of tensors to a specified device, including dictionaries of nested tensors in lists (HiDream outputs).

    Args:
        tensors (dict): Dictionary of tensors to move.
        device (torch.device): The device to move the tensors to.

    Returns:
        dict: Dictionary of tensors moved to the specified device.
    """

    def _move(value):
        if isinstance(value, dict):
            return {k: _move(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_move(item) for item in value]
        if hasattr(value, "to"):
            return value.to(device)
        return value

    return {k: _move(v) for k, v in tensors.items()}

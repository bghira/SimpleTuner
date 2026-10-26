from functools import partial, update_wrapper

import torch

from simpletuner.helpers.logging import get_logger

logger = get_logger("VAETiling")


def _call_with_tiling(vae, operation, *args, **kwargs):
    original = getattr(vae, f"_simpletuner_original_{operation}")
    try:
        return original(*args, **kwargs)
    except torch.cuda.OutOfMemoryError as error:
        cuda_input = any(isinstance(value, torch.Tensor) and value.is_cuda for value in (*args, *kwargs.values()))
        cuda_model = getattr(getattr(vae, "device", None), "type", None) == "cuda"
        if not (cuda_input or cuda_model) or getattr(vae, "use_tiling", False) or vae._simpletuner_oom_tiling_enabled:
            raise
        message = str(error)

    # Leave the exception handler before retrying so its traceback releases failed intermediates.
    logger.warning("CUDA OOM during VAE %s; enabling tiling and retrying once. %s", operation, message)
    vae.enable_tiling()
    vae._simpletuner_oom_tiling_enabled = True
    torch.cuda.empty_cache()
    return original(*args, **kwargs)


def _encode(vae, *args, **kwargs):
    return _call_with_tiling(vae, "encode", *args, **kwargs)


def _decode(vae, *args, **kwargs):
    return _call_with_tiling(vae, "decode", *args, **kwargs)


def enable_automatic_vae_tiling(vae):
    if getattr(vae, "_simpletuner_auto_tiling_installed", False) or not callable(getattr(vae, "enable_tiling", None)):
        return vae
    vae._simpletuner_oom_tiling_enabled = False
    for name, wrapper in (("encode", _encode), ("decode", _decode)):
        original = getattr(vae, name, None)
        if callable(original):
            setattr(vae, f"_simpletuner_original_{name}", original)
            setattr(vae, name, update_wrapper(partial(wrapper, vae), original))
    vae._simpletuner_auto_tiling_installed = True
    return vae

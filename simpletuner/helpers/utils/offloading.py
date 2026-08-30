def unpack_offload_state(offload_state):
    """Normalize the value returned by Diffusers' offload-state helper."""

    if isinstance(offload_state, tuple):
        padded = list(offload_state) + [False] * (2 - len(offload_state))
        return bool(padded[0]), bool(padded[1])

    return bool(offload_state), False


def restore_offload_state(_pipeline, is_model_cpu_offload, is_sequential_cpu_offload):
    """Re-apply the pipeline's previous CPU offload hooks."""

    if _pipeline is None:
        return

    if is_model_cpu_offload and hasattr(_pipeline, "enable_model_cpu_offload"):
        _pipeline.enable_model_cpu_offload()
    elif is_sequential_cpu_offload and hasattr(_pipeline, "enable_sequential_cpu_offload"):
        _pipeline.enable_sequential_cpu_offload()

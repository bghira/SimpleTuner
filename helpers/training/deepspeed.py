import accelerate
from accelerate.state import AcceleratorState


def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = (
        AcceleratorState().deepspeed_plugin
        if accelerate.state.is_initialized()
        else None
    )
    if deepspeed_plugin is None:
        return []

    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

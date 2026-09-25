from contextlib import contextmanager

import torch
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import AuxiliaryTrainingWrapper


@contextmanager
def base_model_generation_context(model):
    """Run the frozen base model and restore adapter and training state on exit."""
    component = model.get_trained_component()
    modules = [(module, module.training) for module in component.modules()]
    gradients = [(parameter, parameter.requires_grad) for parameter in component.parameters()]
    adapters = [
        (module, module.disable_adapters)
        for module, _ in modules
        if isinstance(module, (BaseTunerLayer, AuxiliaryTrainingWrapper))
    ]
    lycoris = getattr(model.accelerator, "_lycoris_wrapped_network", None)
    multiplier = lycoris.multiplier if lycoris is not None else None
    try:
        for module, _ in adapters:
            module.enable_adapters(False)
        if lycoris is not None:
            lycoris.set_multiplier(0.0)
        component.eval()
        with torch.no_grad():
            yield
    finally:
        for module, disabled in adapters:
            module.enable_adapters(not disabled)
        if lycoris is not None:
            lycoris.set_multiplier(multiplier)
        for parameter, requires_grad in gradients:
            parameter.requires_grad_(requires_grad)
        for module, training in modules:
            module.training = training

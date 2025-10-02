"""Register field definitions segmented by domain."""

from typing import TYPE_CHECKING, Callable, Iterable

from . import (
    advanced,
    data,
    logging_fields,
    loss,
    lora,
    memory,
    model,
    optimizer,
    training,
    validation,
)

if TYPE_CHECKING:
    from ..registry import FieldRegistry


_REGISTRARS: Iterable[Callable[["FieldRegistry"], None]] = (
    model.register_model_fields,
    training.register_training_fields,
    lora.register_lora_fields,
    data.register_data_fields,
    validation.register_validation_fields,
    advanced.register_advanced_fields,
    loss.register_loss_fields,
    optimizer.register_optimizer_fields,
    memory.register_memory_fields,
    logging_fields.register_logging_fields,
)


def register_all_sections(registry: "FieldRegistry") -> None:
    """Invoke all registrar functions in the expected order."""

    for registrar in _REGISTRARS:
        registrar(registry)

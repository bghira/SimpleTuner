import logging

logger = logging.getLogger(__name__)

_PATCHED_FROM_FLOAT_ATTR = "_simpletuner_patched_from_float"


def _detect_fake_mode(tensor) -> object | None:
    try:
        from torch._guards import detect_fake_mode
    except ImportError:
        return None
    return detect_fake_mode(tensor)


def apply_sdnq_workarounds() -> None:
    try:
        from sdnq.training.tensor import SDNQTensor
    except ImportError:
        return

    if getattr(SDNQTensor, _PATCHED_FROM_FLOAT_ATTR, False):
        return

    original_from_float = SDNQTensor.from_float

    def from_float_dequantizing_existing_sdnq(weight, *args, **kwargs):
        if isinstance(weight, SDNQTensor) and _detect_fake_mode(weight) is None:
            weight = weight.dequantize()
        return original_from_float(weight, *args, **kwargs)

    SDNQTensor.from_float = staticmethod(from_float_dequantizing_existing_sdnq)
    setattr(SDNQTensor, _PATCHED_FROM_FLOAT_ATTR, True)
    logger.debug("Patched SDNQTensor.from_float to dequantize existing SDNQTensor inputs before requantizing.")

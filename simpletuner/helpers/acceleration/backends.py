"""AccelerationBackend enum defining available memory optimization strategies."""

from enum import Enum, auto


class AccelerationBackend(Enum):
    """
    Memory optimization/acceleration backends.

    Models declare UNSUPPORTED_BACKENDS (not supported ones) since most support all.
    """

    GRADIENT_CHECKPOINTING = auto()
    RAMTORCH = auto()
    MUSUBI_BLOCK_SWAP = auto()
    FEED_FORWARD_CHUNKING = auto()
    GROUP_OFFLOAD = auto()
    DEEPSPEED_ZERO_1 = auto()
    DEEPSPEED_ZERO_2 = auto()
    DEEPSPEED_ZERO_3 = auto()
    FSDP2 = auto()
    SDNQ = auto()  # SD.Next Quantization - works on AMD, Apple, NVIDIA
    TORCHAO = auto()  # TorchAO quantization - NVIDIA only
    QUANTO = auto()  # Quanto quantization - works on AMD, Apple, NVIDIA
    BITSANDBYTES = auto()  # BitsAndBytes quantization - NVIDIA only

# This file was adapted from Tencent's HunyuanVideo 1.5 code (Tencent Hunyuan Community License).
# It is now distributed under the AGPL-3.0-or-later for SimpleTuner contributors.

from dataclasses import dataclass
from typing import Optional


@dataclass
class InferState:
    enable_sageattn: bool = False  # whether to use SageAttention
    sage_blocks_range: Optional[range] = None  # block range to use SageAttention
    enable_torch_compile: bool = False  # whether to use torch compile


__infer_state = None


def parse_range(value):
    if "-" in value:
        start, end = map(int, value.split("-"))
        return list(range(start, end + 1))
    return [int(x) for x in value.split(",")]


def initialize_infer_state(args):
    global __infer_state
    sage_blocks_range = parse_range(args.sage_blocks_range)
    use_sageattn = getattr(args, "use_sageattn", False)
    __infer_state = InferState(
        enable_sageattn=use_sageattn,
        sage_blocks_range=sage_blocks_range,
        enable_torch_compile=args.enable_torch_compile,
    )
    return __infer_state


def get_infer_state():
    return __infer_state

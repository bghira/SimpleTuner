"""Reuse a compiled host runner without changing the selected GPU schedule."""

from dataclasses import asdict
from functools import lru_cache


@lru_cache(maxsize=32)
def native_runner(schedule, forward_threads=128):
    from simpletuner.helpers.training.kohaku_fa_cute.native import extension

    options = asdict(schedule)
    if forward_threads != 128:
        options["forward_threads"] = forward_threads
    return extension().AttentionRunner("warp", options)

"""Wrapper for tqdm that can be disabled via environment variable."""

import os
from typing import Any, Iterable, Optional

# Check if tqdm should be disabled (for testing)
DISABLE_TQDM = os.environ.get("SIMPLETUNER_DISABLE_TQDM", "").lower() in ("true", "1", "yes")


class FakeTqdm:
    """Fake tqdm that does nothing when tqdm is disabled."""

    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
        self.n = 0
        self.total = kwargs.get("total", 0)

    def __iter__(self):
        if self.iterable is not None:
            for item in self.iterable:
                self.n += 1
                yield item

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, n=1):
        self.n += n

    def set_description(self, desc=None, refresh=True):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def refresh(self):
        pass

    def close(self):
        pass


if DISABLE_TQDM:
    tqdm = FakeTqdm
else:
    from tqdm import tqdm as _real_tqdm

    tqdm = _real_tqdm

"""
Lightweight trainingsample stub for unit tests.

Provides callable placeholders that raise if invoked so tests can run
without the compiled trainingsample dependency.
"""


def __getattr__(name):
    def _missing(*args, **kwargs):
        raise RuntimeError(f"trainingsample stub called for {name}")

    return _missing

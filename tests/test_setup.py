"""
Global test setup for SimpleTuner unittest tests.
Import this at the top of test files to configure logging.
"""

import atexit
import logging
import os
import sys
import warnings

# suppress logging from test dependencies
os.environ["SIMPLETUNER_LOG_LEVEL"] = "ERROR"
os.environ["ACCELERATE_LOG_LEVEL"] = "ERROR"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"
os.environ["SIMPLETUNER_DISABLE_COLORS"] = "1"
os.environ["CLICOLOR"] = "0"
os.environ["FORCE_COLOR"] = "0"

logging.disable(logging.WARNING)
noisy_loggers = [
    "simpletuner",
    "SimpleTuner",
    "SimpleTunerSDK",
    "ProcessKeeper",
    "SubprocessWrapper",
    "EventRoutes",
    "ModelRoutes",
    "ArgsParser",
    "torch",
    "torch.distributed",
    "torch.distributed.elastic",
    "accelerate",
    "transformers",
    "datasets",
    "diffusers",
    "PIL",
    "urllib3",
    "asyncio",
    "multiprocess",
    "fastapi",
    "uvicorn",
]

for logger_name in noisy_loggers:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
    logging.getLogger(logger_name).propagate = False

# suppress common test warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")

# prevent tqdm semaphore leaks in tests
try:
    import tqdm

    class _NullLock:
        def acquire(self, *args, **kwargs):
            return True

        def release(self, *args, **kwargs):
            return True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    tqdm.tqdm.set_lock(_NullLock())
except Exception:
    pass

# track multiprocessing leaks that hang the interpreter
try:
    from multiprocessing import resource_tracker

    _original_register = resource_tracker.register
    _original_unregister = resource_tracker.unregister
    _tracked_resources = set()
    _resource_stacks = {}

    def _tracking_register(name, rtype):
        _tracked_resources.add((name, rtype))
        try:
            import traceback

            _resource_stacks[(name, rtype)] = traceback.format_stack(limit=10)
        except Exception:
            _resource_stacks[(name, rtype)] = ["<stack unavailable>"]
        return _original_register(name, rtype)

    def _tracking_unregister(name, rtype):
        _tracked_resources.discard((name, rtype))
        _resource_stacks.pop((name, rtype), None)
        return _original_unregister(name, rtype)

    def _report_resource_leaks():
        if _tracked_resources:
            sys.stderr.write(f"\n[ResourceTracker] Leaked resources detected: {_tracked_resources}\n")
            for key, stack in _resource_stacks.items():
                sys.stderr.write(f"Resource {key} allocated at:\n{''.join(stack)}\n")
            sys.stderr.flush()

    resource_tracker.register = _tracking_register
    resource_tracker.unregister = _tracking_unregister
    atexit.register(_report_resource_leaks)
except Exception:
    pass

## Custom Accelerate Trackers

This directory is reserved for user-supplied trackers that plug into Hugging Face Accelerate. Place one module here (for example `my_tracker.py`) and point the trainer at it with:

```
--report_to=custom-tracker --custom_tracker=my_tracker
```

SimpleTuner expectations:
- The filename (without `.py`) must match the `--custom_tracker` value and be a valid Python identifier.
- Each module must define **exactly one** subclass of `accelerate.tracking.GeneralTracker`.
- That tracker is instantiated as `TrackerClass(run_name, logging_dir=...)`. If your tracker needs a log directory, set `requires_logging_directory = True` on the class; otherwise it will be passed only when your `__init__` accepts it.
- The class must provide the `name` string attribute, `requires_logging_directory` boolean attribute, and a `tracker` property plus the `store_init_configuration` and `log` methods described in the Accelerate docs.

Minimal skeleton:

```python
from accelerate.state import PartialState
from accelerate.tracking import GeneralTracker


class MyTracker(GeneralTracker):
    name = "my_tracker"
    requires_logging_directory = False

    def __init__(self, run_name: str, logging_dir: str | None = None):
        self.run_name = run_name
        self._records = []

    @property
    def tracker(self):
        return self._records

    def store_init_configuration(self, values: dict):
        if PartialState().is_main_process:
            self._records.append({"config": values})

    def log(self, values: dict, step: int | None = None):
        if PartialState().is_main_process:
            self._records.append({"step": step, **values})
```

When the module is present, SimpleTuner will import it directly from `simpletuner/custom-trackers`, instantiate it with the current `tracker_run_name` (or defaults), and pass it to the Accelerator `log_with` parameter.

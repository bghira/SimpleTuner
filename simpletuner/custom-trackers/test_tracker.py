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
        # Mock behavior for testing
        self._records.append({"config": values})

    def log(self, values: dict, step: int | None = None):
        # Mock behavior for testing
        self._records.append({"step": step, **values})

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import simpletuner

PROBE = """
import json, runpy, sys
runpy.run_path(sys.argv[1], run_name="training_entry_probe")
import simpletuner.helpers.training.multi_process as multi_process
with open(sys.argv[2], "w") as report:
    json.dump({"distributed_helpers": multi_process.dist is not None}, report)
"""


class TestTrainingEntryEnvironment(unittest.TestCase):
    def test_a_trainer_launched_by_the_cli_keeps_its_distributed_helpers(self):
        # Importing simpletuner.cli puts SIMPLETUNER_SKIP_TORCH=1 in the environment a trainer inherits.
        train_py = Path(simpletuner.__file__).with_name("train.py")
        env = {**os.environ, "SIMPLETUNER_SKIP_TORCH": "1"}
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory, "report.json")
            result = subprocess.run(
                [sys.executable, "-c", PROBE, str(train_py), str(report)],
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            self.assertEqual(json.loads(report.read_text()), {"distributed_helpers": True})

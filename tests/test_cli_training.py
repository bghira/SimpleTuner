import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from simpletuner.cli.train import run_training


class TestTrainingLaunch(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Path(self.temp.name) / "accelerate.yaml"
        self.config.write_text("{}")
        for target, kwargs in (
            ("simpletuner.helpers.configuration.loader.load_env_variables", {}),
            ("simpletuner.cli.train.list_examples", {"return_value": ["example"]}),
            ("simpletuner.cli.train.setup_environment_from_example", {"return_value": {"ENV": "examples/example"}}),
            ("simpletuner.cli.train.find_accelerate_config", {"return_value": None}),
            ("simpletuner.cli.train.subprocess.Popen", {"return_value": Mock(wait=Mock(return_value=0))}),
        ):
            patcher = patch(target, **kwargs)
            mocked = patcher.start()
            self.addCleanup(patcher.stop)
            if target.endswith("Popen"):
                self.popen = mocked
        env_patch = patch.dict(os.environ, {}, clear=True)
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def launch(self, *args):
        self.assertEqual(run_training(example="example", extra_args=list(args)), 0)
        return self.popen.call_args.args[0]

    def test_example_preserves_explicit_accelerate_overrides(self):
        command = self.launch("--num_processes=2", "--num_machines", "3", "--dynamo_backend=inductor")
        self.assertIn("--num_processes=2", command)
        self.assertIn("--num_machines=3", command)
        self.assertIn("--dynamo_backend=inductor", command)

    def test_example_preserves_accelerate_config_and_extra_args(self):
        command = self.launch("--accelerate_config", str(self.config), "--accelerate_extra_args=--cpu")
        self.assertIn(f"--config_file={self.config}", command)
        self.assertIn("--cpu", command)
        self.assertLess(command.index("--cpu"), next(i for i, arg in enumerate(command) if arg.endswith("/train.py")))

    def test_explicit_launch_overrides_take_precedence_over_config_file(self):
        command = self.launch("--accelerate_config", str(self.config), "--num_processes=2")
        self.assertIn(f"--config_file={self.config}", command)
        self.assertIn("--num_processes=2", command)

    def test_similarly_prefixed_arguments_are_forwarded_to_trainer(self):
        command = self.launch("--num_processes_extra=7")
        self.assertIn("--num_processes_extra=7", command)
        self.assertIn("--num_processes=1", command)

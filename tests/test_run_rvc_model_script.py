import importlib.util
import tempfile
import unittest
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_rvc_model.py"
    spec = importlib.util.spec_from_file_location("run_rvc_model", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestRunRVCModelScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = _load_script_module()

    def test_dry_run_builds_identity_transfer_backend_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            identity_dir = root / "identity"
            generated_dir = root / "generated"
            output_dir = root / "output"
            source_dir.mkdir()
            identity_dir.mkdir()

            args = self.script.build_parser().parse_args(
                [
                    "--source-dir",
                    str(source_dir),
                    "--identity-dir",
                    str(identity_dir),
                    "--generated-dir",
                    str(generated_dir),
                    "--output-dir",
                    str(output_dir),
                    "--training-steps",
                    "123",
                    "--batch-size",
                    "1",
                    "--model-name",
                    "Test Voice",
                    "--public",
                    "--device",
                    "cpu",
                    "--dry-run",
                ]
            )

            config = self.script.run(args)

        source_backend = config[0]
        transform = source_backend["data_transforms"][0]
        self.assertEqual(source_backend["dataset_type"], "audio")
        self.assertEqual(transform["task"], "identity_transfer")
        self.assertEqual(transform["method"], "rvc")
        self.assertEqual(transform["model"]["identity_audio_mode"], "separate")
        self.assertEqual(transform["model"]["separation_method"], "demucs")
        self.assertNotIn("identity_stem_debug_dir", transform["model"])
        self.assertEqual(transform["model"]["model_name"], "Test Voice")
        self.assertTrue(transform["model"]["public"])
        self.assertEqual(transform["conversion"]["audio_mode"], "separate_convert_remix")
        self.assertEqual(transform["model"]["training_steps"], 123)
        self.assertEqual(transform["model"]["batch_size"], 1)
        self.assertEqual(transform["model"]["device"], "cpu")
        self.assertEqual(transform["target"]["instance_data_dir"], str(generated_dir))

    def test_build_config_accepts_identity_stem_debug_dir(self):
        args = self.script.build_parser().parse_args(
            [
                "--source-dir",
                "source",
                "--identity-dir",
                "identity",
                "--generated-dir",
                "generated",
                "--output-dir",
                "output",
                "--identity-stem-debug-dir",
                "debug-stems",
            ]
        )

        config = self.script.build_data_backend_config(args)

        transform = config[0]["data_transforms"][0]
        self.assertEqual(transform["model"]["identity_stem_debug_dir"], "debug-stems")


if __name__ == "__main__":
    unittest.main()

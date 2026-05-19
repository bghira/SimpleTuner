import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


def import_cog_module():
    os.environ["SIMPLETUNER_SKIP_TORCH"] = "1"

    loader_module = types.ModuleType("simpletuner.helpers.configuration.loader")
    loader_module.load_config = lambda *_, **__: {}

    trainer_module = types.ModuleType("simpletuner.helpers.training.trainer")
    trainer_module.run_trainer_job = lambda config: dict(config)

    sys.modules["simpletuner.helpers.configuration.loader"] = loader_module
    sys.modules["simpletuner.helpers.training.trainer"] = trainer_module
    sys.modules.pop("simpletuner.cog", None)
    return importlib.import_module("simpletuner.cog")


class TestCogNsfwOverrides(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cog = import_cog_module()

    def test_force_nsfw_check_replaces_user_values(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runner = self.cog.SimpleTunerCogRunner(
                dataset_root=root / "datasets",
                output_root=root / "output",
                config_root=root / "config",
            )
            config = {
                "enable_nsfw_check": False,
                "--nsfw_check_models": "example/bypass",
                "nsfw_check_min_votes": 1,
                "--nsfw_check_backend_types": "aws",
                "nsfw_check_sample_types": "video",
            }

            runner._force_nsfw_check(config)

        self.assertNotIn("enable_nsfw_check", config)
        self.assertNotIn("nsfw_check_min_votes", config)
        self.assertEqual(config["--enable_nsfw_check"], True)
        self.assertEqual(config["--nsfw_check_min_votes"], 2)
        self.assertEqual(config["--nsfw_check_backend_types"], "all")
        self.assertEqual(config["--nsfw_check_sample_types"], "image,conditioning")
        self.assertIn("Falconsai/nsfw_image_detection:threshold=0.5", config["--nsfw_check_models"])
        self.assertIn("AdamCodd/vit-base-nsfw-detector:threshold=0.5", config["--nsfw_check_models"])
        self.assertIn("hoangtrung1801/nsfw-vit-model:threshold=0.5", config["--nsfw_check_models"])
        self.assertNotIn("Marqo/", config["--nsfw_check_models"])

    def test_model_specs_prefer_baked_classifier_dirs(self):
        with TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "hf_falconsai"
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_text("", encoding="utf-8")

            with patch.dict("os.environ", {"NSFW_CLASSIFIER_MODEL_DIR": temp_dir}):
                specs = self.cog.build_cog_nsfw_model_specs_csv()

        self.assertIn(f"{model_dir}:threshold=0.5", specs)
        self.assertIn("AdamCodd/vit-base-nsfw-detector:threshold=0.5", specs)
        self.assertIn("hoangtrung1801/nsfw-vit-model:threshold=0.5", specs)

    def test_run_applies_nsfw_overrides_after_user_config(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runner = self.cog.SimpleTunerCogRunner(
                dataset_root=root / "datasets",
                output_root=root / "output",
                config_root=root / "config",
            )

            result = runner.run(
                base_config_dict={"--model_type": "lora", "--enable_nsfw_check": False},
                dataloader_config_dict=[
                    {
                        "id": "remote",
                        "type": "aws",
                        "dataset_type": "image",
                        "aws_data_prefix": "s3://bucket/images",
                    }
                ],
                config_overrides={"--nsfw_check_min_votes": 1},
                job_id="test-job",
            )

        training_config = result["training_result"]
        self.assertEqual(training_config["--enable_nsfw_check"], True)
        self.assertEqual(training_config["--nsfw_check_min_votes"], 2)
        self.assertEqual(training_config["--nsfw_check_backend_types"], "all")
        self.assertEqual(training_config["--nsfw_check_sample_types"], "image,conditioning")


if __name__ == "__main__":
    unittest.main()

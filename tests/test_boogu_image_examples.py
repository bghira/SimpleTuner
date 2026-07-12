import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simpletuner.cli.examples import copy_example, find_referenced_files

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "simpletuner" / "examples"


class BooguImageExampleTests(unittest.TestCase):
    def _load_example(self, name: str) -> dict:
        with (EXAMPLES_DIR / name / "config.json").open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_peft_lora_example_uses_boogu_base(self):
        config = self._load_example("boogu-image-v0.1.peft-lora")

        self.assertEqual(config["model_family"], "boogu_image")
        self.assertEqual(config["model_flavour"], "v0.1-base")
        self.assertEqual(config["model_type"], "lora")
        self.assertEqual(config["lora_type"], "standard")
        self.assertEqual(config["base_model_precision"], "no_change")

    def test_lycoris_lokr_example_uses_boogu_base(self):
        config = self._load_example("boogu-image-v0.1.lycoris-lokr")
        with (EXAMPLES_DIR / "boogu-image-v0.1.lycoris-lokr" / "lycoris_config.json").open(
            "r",
            encoding="utf-8",
        ) as handle:
            lycoris_config = json.load(handle)

        self.assertEqual(config["model_family"], "boogu_image")
        self.assertEqual(config["model_flavour"], "v0.1-base")
        self.assertEqual(config["model_type"], "lora")
        self.assertEqual(config["lora_type"], "lycoris")
        self.assertEqual(config["base_model_precision"], "no_change")
        self.assertEqual(config["lycoris_config"], "config/examples/boogu-image-v0.1.lycoris-lokr/lycoris_config.json")
        self.assertEqual(lycoris_config["algo"], "lokr")
        self.assertEqual(lycoris_config["apply_preset"]["target_module"], ["Attention"])
        self.assertEqual(lycoris_config["apply_preset"]["exclude_name"], ["ref_image_*"])

    def test_copy_example_preserves_boogu_lycoris_config(self):
        config_path = EXAMPLES_DIR / "boogu-image-v0.1.lycoris-lokr" / "config.json"
        self.assertIn(
            "boogu-image-v0.1.lycoris-lokr/lycoris_config.json",
            find_referenced_files(config_path),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("simpletuner.cli.examples.get_examples_dir", return_value=EXAMPLES_DIR):
                self.assertTrue(copy_example("boogu-image-v0.1.lycoris-lokr", tmpdir))

            copied_dir = Path(tmpdir) / "boogu-image-v0.1.lycoris-lokr"
            with (copied_dir / "config.json").open("r", encoding="utf-8") as handle:
                copied_config = json.load(handle)
            with (copied_dir / "lycoris_config.json").open("r", encoding="utf-8") as handle:
                copied_lycoris = json.load(handle)

            self.assertEqual(copied_config["lycoris_config"], "lycoris_config.json")
            self.assertEqual(copied_lycoris["apply_preset"]["target_module"], ["Attention"])
            self.assertEqual(copied_lycoris["apply_preset"]["exclude_name"], ["ref_image_*"])


if __name__ == "__main__":
    unittest.main()

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simpletuner.cli.common import _candidate_config_paths
from simpletuner.cli.examples import copy_example, find_referenced_files
from simpletuner.cli.train import get_examples_dir, list_examples, setup_environment_from_example
from simpletuner.helpers.configuration.examples import discover_examples


class ExampleCatalogTests(unittest.TestCase):
    def test_environment_resolution_keeps_dots_in_nested_recipe_names(self):
        from simpletuner.helpers.configuration.json_file import _candidate_json_paths

        name = "anima/base-v1.0/default/anyflow-stage1.peft-lora"
        expected = get_examples_dir() / name / "config.json"
        self.assertIn(expected, _candidate_config_paths(f"examples/{name}", "json", None))
        with patch.dict("os.environ", {}, clear=True):
            self.assertIn(expected, _candidate_json_paths(f"examples/{name}"))

    def test_discovery_uses_relative_names_and_excludes_shared_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("flux/kontext/24g/peft-lora", "flux/kontext/24g/lycoris-lokr", "_shared/ignored", ".local/ignored"):
                bundle = root / name
                bundle.mkdir(parents=True)
                (bundle / "config.json").write_text("{}")
            (root / "flux/kontext/24g/peft-lora/config.env").write_text("")
            (root / "prompts.json").write_text("{}")
            (root / "empty-family").mkdir()
            with patch("simpletuner.cli.train.get_examples_dir", return_value=root):
                self.assertEqual(list_examples(), ["flux/kontext/24g/lycoris-lokr", "flux/kontext/24g/peft-lora"])
                env = setup_environment_from_example("flux/kontext/24g/peft-lora")
            self.assertEqual(env["CONFIG_BACKEND"], "json")
            self.assertEqual(env["ENV"], "examples/flux/kontext/24g/peft-lora")
            self.assertEqual(env["CONFIG_PATH"], str(root / "flux/kontext/24g/peft-lora/config"))

    def test_bundled_catalog_layout_and_shared_assets(self):
        root = get_examples_dir()
        catalog = discover_examples(root)
        self.assertGreater(len(catalog), 100)
        for name, path in catalog.items():
            with self.subTest(example=name):
                family, flavour, vram, recipe = name.split("/")
                config = json.loads(path.read_text())
                self.assertEqual(config.get("model_family", config.get("--model_family")), family)
                self.assertEqual(config.get("model_flavour", config.get("--model_flavour")) or "default", flavour)
                self.assertRegex(vram, r"^(default|(?:\d+x)?\d+g)$")
                for reference in find_referenced_files(path):
                    if reference.startswith("_shared/"):
                        self.assertTrue((root / reference).is_file(), reference)

    def test_copy_nested_example_keeps_shared_and_local_assets(self):
        name = "flux3_image/klein/24g/peft-lora"
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(copy_example(name, directory))
            bundle = Path(directory) / name
            config = json.loads((bundle / "config.json").read_text())
            self.assertEqual(config["data_backend_config"], "dataloader.json")
            self.assertEqual(config["user_prompt_library"], "flux3_image-prompts.json")
            self.assertTrue((bundle / config["data_backend_config"]).is_file())
            self.assertTrue((bundle / config["user_prompt_library"]).is_file())
            self.assertFalse(copy_example(name, directory))

    def test_copy_rejects_paths_outside_the_catalog(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(copy_example("../configuration", directory))
            with self.assertRaises(ValueError):
                setup_environment_from_example("../configuration")

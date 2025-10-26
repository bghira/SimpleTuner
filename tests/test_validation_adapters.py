import os
import tempfile
import unittest

from simpletuner.helpers.training.validation_adapters import ValidationAdapterRun, build_validation_adapter_runs


class ValidationAdapterBuilderTests(unittest.TestCase):
    def test_no_inputs_returns_empty(self):
        runs = build_validation_adapter_runs(None, None)
        self.assertEqual([], runs)

    def test_remote_path_uses_default_weight(self):
        runs = build_validation_adapter_runs("foo/bar:baz.safetensors", None)
        self.assertEqual(1, len(runs))
        run = runs[0]
        self.assertIsInstance(run, ValidationAdapterRun)
        self.assertEqual("bar_baz", run.slug)
        self.assertEqual(1, len(run.adapters))
        adapter = run.adapters[0]
        self.assertFalse(adapter.is_local)
        self.assertEqual("foo/bar", adapter.repo_id)
        self.assertEqual("baz.safetensors", adapter.weight_name)
        self.assertEqual(1.0, adapter.scale)

    def test_local_path_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "my_adapter.safetensors")
            with open(file_path, "wb") as handle:
                handle.write(b"data")
            runs = build_validation_adapter_runs(file_path, None)
            adapter = runs[0].adapters[0]
            self.assertTrue(adapter.is_local)
            self.assertEqual(os.path.abspath(file_path), adapter.path)
            self.assertIsNone(adapter.weight_name)

    def test_config_supports_multiple_runs(self):
        config = [
            "org/base_adapter",
            {
                "label": "combo",
                "adapters": [
                    {"path": "repo/char", "scale": 0.5},
                    {"path": "repo/style:style.safetensors"},
                ],
            },
        ]
        runs = build_validation_adapter_runs(None, config)
        self.assertEqual(2, len(runs))

        first = runs[0]
        self.assertEqual("base_adapter", first.slug)
        self.assertEqual("org/base_adapter", first.adapters[0].repo_id)

        second = runs[1]
        self.assertEqual("combo", second.label)
        self.assertEqual(2, len(second.adapters))
        scales = [adapter.scale for adapter in second.adapters]
        self.assertEqual([0.5, 1.0], scales)
        self.assertEqual("repo/style", second.adapters[1].repo_id)
        self.assertEqual("style.safetensors", second.adapters[1].weight_name)


if __name__ == "__main__":
    unittest.main()

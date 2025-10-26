import os
import tempfile
import unittest

from simpletuner.helpers.training.validation_adapters import ValidationAdapterRun, build_validation_adapter_runs


class ValidationAdapterBuilderTests(unittest.TestCase):
    def test_no_inputs_returns_base_run(self):
        runs = build_validation_adapter_runs(None, None)
        self.assertEqual(1, len(runs))
        self.assertTrue(runs[0].is_base)

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
        self.assertEqual(1.0, adapter.strength)

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
        self.assertEqual(3, len(runs))

        first = runs[0]
        self.assertTrue(first.is_base)

        second = runs[1]
        self.assertEqual("base_adapter", second.slug)
        self.assertEqual("org/base_adapter", second.adapters[0].repo_id)

        third = runs[2]
        self.assertEqual("combo", third.label)
        self.assertEqual(2, len(third.adapters))
        strengths = [adapter.strength for adapter in third.adapters]
        self.assertEqual([0.5, 1.0], strengths)
        self.assertEqual("repo/style", third.adapters[1].repo_id)
        self.assertEqual("style.safetensors", third.adapters[1].weight_name)

    def test_adapter_mode_comparison_includes_base(self):
        runs = build_validation_adapter_runs(
            "foo/bar", None, adapter_mode="comparison", adapter_strength=0.8, adapter_name="sample"
        )
        self.assertEqual(2, len(runs))
        self.assertTrue(runs[0].is_base)
        second = runs[1]
        self.assertEqual("sample", second.slug)
        self.assertEqual("sample", second.adapters[0].adapter_name)
        self.assertEqual(0.8, second.adapters[0].strength)

    def test_adapter_mode_none_skips_loading(self):
        runs = build_validation_adapter_runs("foo/bar", None, adapter_mode="none")
        self.assertEqual(1, len(runs))
        self.assertTrue(runs[0].is_base)

    def test_config_entry_with_strength_and_name(self):
        config = [
            {
                "label": "custom",
                "path": "repo/hero",
                "adapter_name": "hero_adapter",
                "strength": 1.25,
            }
        ]
        runs = build_validation_adapter_runs(None, config)
        self.assertEqual(2, len(runs))
        run = runs[1]
        self.assertEqual("custom", run.label)
        adapter = run.adapters[0]
        self.assertEqual("hero_adapter", adapter.adapter_name)
        self.assertAlmostEqual(1.25, adapter.strength)


if __name__ == "__main__":
    unittest.main()

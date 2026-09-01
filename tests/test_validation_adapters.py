import os
import tempfile
import unittest

from simpletuner.helpers.training.validation import Validation
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
        self.assertEqual(2, len(runs))

        first = runs[0]
        self.assertFalse(first.is_base)
        self.assertEqual("base_adapter", first.slug)
        self.assertEqual("org/base_adapter", first.adapters[0].repo_id)

        second = runs[1]
        self.assertEqual("combo", second.label)
        self.assertEqual(2, len(second.adapters))
        strengths = [adapter.strength for adapter in second.adapters]
        self.assertEqual([0.5, 1.0], strengths)
        self.assertEqual("repo/style", second.adapters[1].repo_id)
        self.assertEqual("style.safetensors", second.adapters[1].weight_name)

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

    def test_adapter_mode_comparison_with_config_includes_base(self):
        runs = build_validation_adapter_runs(
            None,
            [{"label": "custom", "path": "repo/hero"}],
            adapter_mode="comparison",
        )
        self.assertEqual(2, len(runs))
        self.assertTrue(runs[0].is_base)
        self.assertEqual("custom", runs[1].slug)

    def test_adapter_mode_adapter_only_with_config_excludes_base(self):
        runs = build_validation_adapter_runs(
            None,
            [{"label": "custom", "path": "repo/hero"}],
            adapter_mode="adapter_only",
        )
        self.assertEqual(1, len(runs))
        self.assertFalse(runs[0].is_base)
        self.assertEqual("custom", runs[0].slug)

    def test_adapter_mode_none_skips_loading(self):
        runs = build_validation_adapter_runs("foo/bar", None, adapter_mode="none")
        self.assertEqual(1, len(runs))
        self.assertTrue(runs[0].is_base)

    def test_adapter_mode_none_skips_config_loading(self):
        runs = build_validation_adapter_runs(
            None,
            [{"label": "custom", "path": "repo/hero"}],
            adapter_mode="none",
        )
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
        self.assertEqual(1, len(runs))
        run = runs[0]
        self.assertEqual("custom", run.label)
        adapter = run.adapters[0]
        self.assertEqual("hero_adapter", adapter.adapter_name)
        self.assertAlmostEqual(1.25, adapter.strength)

    def test_config_entry_supports_run_level_target_stage(self):
        config = [
            {
                "label": "refiner",
                "target_stage": "Two",
                "adapters": [
                    {"path": "repo/detail", "scale": 0.7},
                    {"path": "repo/color", "target_stage": "one"},
                ],
            }
        ]
        runs = build_validation_adapter_runs(None, config)

        run = runs[0]
        self.assertEqual("repo/detail", run.adapters[0].repo_id)
        self.assertEqual("two", run.adapters[0].target_stage)
        self.assertEqual("one", run.adapters[1].target_stage)


class _AdapterTarget:
    def __init__(self):
        self.set_calls = []
        self.delete_calls = []

    def set_adapters(self, names, scales):
        self.set_calls.append((names, scales))

    def delete_adapters(self, names):
        self.delete_calls.append(names)


class _TrackingAdapterTarget:
    def __init__(self):
        self.peft_config = {}
        self.active_adapters = []
        self.set_calls = []
        self.delete_calls = []

    def set_adapters(self, names, scales):
        names = [names] if isinstance(names, str) else list(names)
        missing = set(names) - set(self.peft_config)
        if missing:
            present = set(self.peft_config)
            raise ValueError(f"Adapter name(s) {missing} not in the list of present adapters: {present}.")
        self.active_adapters = names
        self.set_calls.append((names, scales))

    def delete_adapters(self, names):
        names = [names] if isinstance(names, str) else list(names)
        for name in names:
            self.peft_config.pop(name, None)
        self.active_adapters = [name for name in self.active_adapters if name not in names]
        self.delete_calls.append(names)


class _Pipeline:
    def __init__(self):
        self.load_calls = []
        self.set_calls = []
        self.delete_calls = []
        self.transformer = _AdapterTarget()
        self.transformer_2 = _AdapterTarget()
        self.components = {
            "transformer": self.transformer,
            "transformer_2": self.transformer_2,
        }

    def load_lora_weights(self, location, **kwargs):
        self.load_calls.append((location, kwargs))

    def set_adapters(self, names, scales):
        self.set_calls.append((names, scales))

    def delete_adapters(self, names):
        self.delete_calls.append(names)


class _ChildActivePipeline:
    def __init__(self):
        self.load_calls = []
        self.transformer = _TrackingAdapterTarget()
        self.components = {"transformer": self.transformer}

    def load_lora_weights(self, location, **kwargs):
        adapter_name = kwargs["adapter_name"]
        self.load_calls.append((location, kwargs))
        self.transformer.peft_config[adapter_name] = object()
        self.transformer.active_adapters = [adapter_name]

    def get_active_adapters(self):
        return list(self.transformer.active_adapters)

    def set_adapters(self, names, scales):
        self.transformer.set_adapters(names, scales)

    def delete_adapters(self, names):
        self.transformer.delete_adapters(names)


class _StageModel:
    NAME = "test-model"

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def supports_multistage_validation(self):
        return True

    def validation_adapter_stage_aliases(self):
        return {
            "stage1": {"stage1", "one"},
            "stage2": {"stage2", "two"},
        }

    def validation_adapter_load_kwargs(self, target_stage):
        return {}

    def validation_adapter_component(self, target_stage):
        return None


class _WanStageModel(_StageModel):
    NAME = "wan"

    def validation_adapter_stage_aliases(self):
        return {"high": {"high"}, "low": {"low"}}

    def validation_adapter_load_kwargs(self, target_stage):
        return {"load_into_transformer_2": target_stage == "low"}

    def validation_adapter_component(self, target_stage):
        return "transformer_2" if target_stage == "low" else "transformer"


class ValidationAdapterStageLoadingTests(unittest.TestCase):
    def _validator(self, model):
        validator = Validation.__new__(Validation)
        validator.model = model
        validator._active_validation_adapter_run = None
        return validator

    def test_targeted_adapter_loads_only_for_matching_stage(self):
        base_pipeline = _Pipeline()
        stage2_pipeline = _Pipeline()
        validator = self._validator(_StageModel(base_pipeline))
        run = build_validation_adapter_runs(
            None,
            [{"label": "refiner", "path": "repo/refiner", "target_stage": "two", "strength": 0.4}],
        )[0]

        with validator._temporary_validation_adapters(run):
            self.assertEqual([], base_pipeline.load_calls)
            with validator._temporary_validation_stage_adapters(base_pipeline, "stage1"):
                self.assertEqual([], base_pipeline.load_calls)
            with validator._temporary_validation_stage_adapters(stage2_pipeline, "stage2"):
                self.assertEqual(1, len(stage2_pipeline.load_calls))
                self.assertEqual("repo/refiner", stage2_pipeline.load_calls[0][0])
                self.assertEqual("refiner_two", stage2_pipeline.load_calls[0][1]["adapter_name"])
                self.assertEqual(("refiner_two", 0.4), stage2_pipeline.set_calls[0])

        self.assertEqual("refiner_two", stage2_pipeline.delete_calls[0])

    def test_wan_low_stage_uses_transformer_2_component(self):
        pipeline = _Pipeline()
        validator = self._validator(_WanStageModel(pipeline))
        run = build_validation_adapter_runs(
            None,
            [{"label": "low-noise", "path": "repo/low", "target_stage": "low", "strength": 0.6}],
        )[0]

        with validator._temporary_validation_adapters(run):
            with validator._temporary_validation_stage_adapters(pipeline, ("high", "low")):
                self.assertEqual(1, len(pipeline.load_calls))
                self.assertTrue(pipeline.load_calls[0][1]["load_into_transformer_2"])
                self.assertEqual([], pipeline.set_calls)
                self.assertEqual([("low_noise_low", 0.6)], pipeline.transformer_2.set_calls)

        self.assertEqual(["low_noise_low"], pipeline.transformer_2.delete_calls)

    def test_stage_adapter_preserves_global_adapter_on_component_target(self):
        pipeline = _Pipeline()
        validator = self._validator(_WanStageModel(pipeline))
        run = build_validation_adapter_runs(
            None,
            [
                {
                    "label": "combo",
                    "adapters": [
                        {"path": "repo/global", "strength": 0.25},
                        {"path": "repo/low", "target_stage": "low", "strength": 0.6},
                    ],
                }
            ],
        )[0]

        with validator._temporary_validation_adapters(run):
            self.assertEqual(("combo", 0.25), pipeline.set_calls[0])
            with validator._temporary_validation_stage_adapters(pipeline, "low"):
                self.assertEqual(
                    [(["combo", "combo_low"], [0.25, 0.6])],
                    pipeline.transformer_2.set_calls,
                )
            self.assertEqual(
                [(["combo", "combo_low"], [0.25, 0.6]), ("combo", 0.25)],
                pipeline.transformer_2.set_calls,
            )

        self.assertEqual("combo", pipeline.delete_calls[0])
        self.assertEqual(["combo_low"], pipeline.transformer_2.delete_calls)

    def test_validation_adapter_restore_ignores_loaded_temporary_adapter(self):
        pipeline = _ChildActivePipeline()
        validator = self._validator(_StageModel(pipeline))
        run = build_validation_adapter_runs(
            None,
            [{"label": "Krea2 Turbo adapter", "path": "repo/krea2", "adapter_name": "krea2_turbo_adapter"}],
        )[0]

        with validator._temporary_validation_adapters(run):
            self.assertEqual("repo/krea2", pipeline.load_calls[0][0])
            self.assertEqual("krea2_turbo_adapter", pipeline.load_calls[0][1]["adapter_name"])
            self.assertEqual([(["krea2_turbo_adapter"], 1.0)], pipeline.transformer.set_calls)

        self.assertEqual([["krea2_turbo_adapter"]], pipeline.transformer.delete_calls)
        self.assertEqual([], pipeline.transformer.active_adapters)

    def test_validation_adapter_restore_preserves_existing_active_adapter(self):
        pipeline = _ChildActivePipeline()
        pipeline.transformer.peft_config["base"] = object()
        pipeline.transformer.active_adapters = ["base"]
        validator = self._validator(_StageModel(pipeline))
        run = build_validation_adapter_runs(
            None,
            [{"label": "Krea2 Turbo adapter", "path": "repo/krea2", "adapter_name": "krea2_turbo_adapter"}],
        )[0]

        with validator._temporary_validation_adapters(run):
            self.assertEqual(
                [(["base", "krea2_turbo_adapter"], [1.0, 1.0])],
                pipeline.transformer.set_calls,
            )

        self.assertEqual(
            [(["base", "krea2_turbo_adapter"], [1.0, 1.0]), (["base"], 1.0)],
            pipeline.transformer.set_calls,
        )
        self.assertEqual(["base"], pipeline.transformer.active_adapters)


if __name__ == "__main__":
    unittest.main()

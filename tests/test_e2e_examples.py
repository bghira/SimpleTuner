#!/usr/bin/env python3
"""
SimpleTuner End-to-End Example Tests

These tests load all example configurations from simpletuner/examples/*/config.json
and run them through the Trainer with minimal steps to ensure they work end-to-end.

Tests are only run when SIMPLETUNER_E2E_TEST environment variable is set to "1" or "TRUE".
"""

import json
import os
import shutil
import unittest
from pathlib import Path
from typing import Dict, List

# Check if E2E tests should run
SHOULD_RUN_E2E_TESTS = os.environ.get("SIMPLETUNER_E2E_TEST", "0").upper() in ["1", "TRUE"]


def get_examples_dir() -> Path:
    """Get the path to the examples directory."""
    # Find simpletuner package directory
    import simpletuner

    simpletuner_dir = simpletuner._get_package_dir()
    return simpletuner_dir / "examples"


def discover_examples() -> List[Path]:
    """Discover all example configurations."""
    examples_dir = get_examples_dir()
    if not examples_dir.exists():
        return []

    examples = []
    for item in examples_dir.iterdir():
        if item.is_dir() and (item / "config.json").exists():
            examples.append(item)

    return sorted(examples)


def load_example_config(example_path: Path) -> Dict:
    """Load and modify config.json for testing."""
    config_path = example_path / "config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    # Override settings for fast e2e testing
    config["max_train_steps"] = 10
    config["validation_steps"] = 5
    config["checkpoint_step_interval"] = 5

    # Ensure output goes to a test directory
    original_output = config.get("output_dir", "output")
    config["output_dir"] = f"test_outputs/e2e/{example_path.name}"

    # Disable unnecessary features for faster testing
    config.setdefault("use_ema", False)
    config.setdefault("push_to_hub", False)
    config.setdefault("push_checkpoints_to_hub", False)

    # Ensure we have minimal batch size for testing
    config.setdefault("train_batch_size", 1)

    # Fix paths that reference config/examples/ to point to simpletuner/examples/
    # These files are in simpletuner/examples/ not config/examples/
    examples_dir = get_examples_dir()
    path_fields = [
        "data_backend_config",
        "validation_prompt_library",
        "controlnet_config",
        "reference_config",
        "lycoris_config",
    ]

    for field in path_fields:
        if field in config and config[field]:
            # Replace config/examples/ with the actual examples directory path
            if isinstance(config[field], str) and "config/examples/" in config[field]:
                filename = Path(config[field]).name
                config[field] = str(examples_dir / filename)

    return config


@unittest.skipUnless(SHOULD_RUN_E2E_TESTS, "End-to-end tests only run when SIMPLETUNER_E2E_TEST=1 or =TRUE")
class TestE2EExamples(unittest.TestCase):
    """Test all example configurations end-to-end."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Set multiprocessing start method like train.py
        try:
            import multiprocessing

            multiprocessing.set_start_method("fork")
        except Exception:
            pass  # May already be set

        # Clean test outputs directory
        test_output_dir = Path("test_outputs/e2e")
        if test_output_dir.exists():
            shutil.rmtree(test_output_dir)
        test_output_dir.mkdir(parents=True, exist_ok=True)

        # Clean cache directory for fresh runs
        cache_dir = Path("cache")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    def _run_example(self, example_path: Path):
        """Run a single example configuration."""
        from simpletuner.helpers.training.attention_backend import AttentionBackendController, AttentionPhase
        from simpletuner.helpers.training.trainer import Trainer

        # Load and modify config
        config = load_example_config(example_path)

        # Create trainer and run
        try:
            trainer = Trainer(
                config=config,
                exit_on_error=True,
            )

            # Full initialization sequence from train.py
            trainer.configure_webhook()
            trainer.init_noise_schedule()
            trainer.init_seed()
            trainer.init_huggingface_hub()
            trainer.init_preprocessing_models()
            trainer.init_precision(preprocessing_models_only=True)
            trainer.init_data_backend()
            trainer.init_unload_text_encoder()
            trainer.init_unload_vae()
            trainer.init_load_base_model()
            trainer.init_controlnet_model()
            trainer.init_tread_model()
            trainer.init_precision()
            trainer.init_freeze_models()
            trainer.init_trainable_peft_adapter()
            trainer.init_ema_model()
            trainer.init_precision(ema_only=True)
            trainer.move_models(destination="accelerator")
            trainer.init_distillation()
            trainer.init_validations()
            AttentionBackendController.apply(trainer.config, AttentionPhase.EVAL)
            trainer.init_benchmark_base_model()
            AttentionBackendController.apply(trainer.config, AttentionPhase.TRAIN)
            trainer.resume_and_prepare()
            trainer.init_trackers()

            # Run training
            trainer.train()

        except Exception as e:
            import traceback

            self.fail(f"Example {example_path.name} failed: {e}\n{traceback.format_exc()}")

    # Dynamically generate test methods for each example
    # This happens at module load time


# Dynamically generate test methods for each example
def _generate_test_methods():
    """Generate a test method for each discovered example."""
    examples = discover_examples()

    for example_path in examples:
        example_name = example_path.name

        # Create a test method for this example
        def test_method(self, example_path=example_path):
            """Test that the example configuration runs successfully."""
            self._run_example(example_path)

        # Set the method name and docstring
        test_method.__name__ = f"test_example_{example_name.replace('.', '_').replace('-', '_')}"
        test_method.__doc__ = f"Test example: {example_name}"

        # Add the method to the test class
        setattr(TestE2EExamples, test_method.__name__, test_method)


# Generate test methods when module is loaded
_generate_test_methods()


if __name__ == "__main__":
    # Check if environment variable is set
    if os.environ.get("SIMPLETUNER_E2E_TEST", "0").upper() not in ["1", "TRUE"]:
        print("Skipping end-to-end tests. Set SIMPLETUNER_E2E_TEST=1 or =TRUE to run.")
        exit(0)

    unittest.main()

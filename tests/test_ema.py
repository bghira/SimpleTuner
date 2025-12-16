import os
import tempfile
import unittest

import torch

from simpletuner.helpers.training.ema import EMAModel


class TestEMAModel(unittest.TestCase):
    def setUp(self):
        # Set up a simple model and its parameters
        self.model = torch.nn.Linear(10, 5)  # Simple linear model
        self.args = type(
            "Args",
            (),
            {"ema_update_interval": None, "ema_device": "cpu", "ema_cpu_only": True},
        )
        self.accelerator = None  # For simplicity, assuming no accelerator in tests
        self.ema_model = EMAModel(
            args=self.args,
            accelerator=self.accelerator,
            parameters=self.model.parameters(),
            decay=0.999,
            min_decay=0.999,  # Force decay to be 0.999
            update_after_step=-1,  # Ensure decay is used from step 1
            use_ema_warmup=False,  # Disable EMA warmup
            foreach=False,
        )

    def test_ema_initialization(self):
        """Test that the EMA model initializes correctly."""
        self.assertEqual(len(self.ema_model.shadow_params), len(list(self.model.parameters())))
        for shadow_param, model_param in zip(self.ema_model.shadow_params, self.model.parameters()):
            self.assertTrue(torch.equal(shadow_param, model_param))

    def test_ema_step(self):
        """Test that the EMA model updates correctly after a step."""
        # Perform a model parameter update
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        dummy_input = torch.randn(1, 10)  # Adjust to match input size
        dummy_output = self.model(dummy_input)
        loss = dummy_output.sum()  # A dummy loss function
        loss.backward()
        optimizer.step()

        # Save a copy of the model parameters after the update but before the EMA update.
        model_params = [p.clone() for p in self.model.parameters()]
        # Save a copy of the shadow parameters before the EMA update.
        shadow_params_before = [p.clone() for p in self.ema_model.shadow_params]

        # Perform an EMA update
        self.ema_model.step(self.model.parameters(), global_step=1)
        decay = self.ema_model.cur_decay_value  # This should be 0.999

        # Verify that the decay used is as expected
        self.assertAlmostEqual(decay, 0.999, places=6, msg="Decay value is not as expected.")

        # Verify shadow parameters have changed
        for shadow_param, shadow_param_before in zip(self.ema_model.shadow_params, shadow_params_before):
            self.assertFalse(
                torch.equal(shadow_param, shadow_param_before),
                "Shadow parameters did not update correctly.",
            )

        # Compute and check expected shadow parameter values
        for shadow_param, shadow_param_before, model_param in zip(
            self.ema_model.shadow_params, shadow_params_before, self.model.parameters()
        ):
            expected_shadow = decay * shadow_param_before + (1 - decay) * model_param
            self.assertTrue(
                torch.allclose(shadow_param, expected_shadow, atol=1e-6),
                f"Shadow parameter does not match expected value.",
            )

    def test_save_and_load_state_dict(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, "ema_model_state.pth")

            # Save the state
            self.ema_model.save_state_dict(temp_path)

            # Create a new EMA model and load the state
            new_ema_model = EMAModel(
                args=self.args,
                accelerator=self.accelerator,
                parameters=self.model.parameters(),
                decay=0.999,
            )
            new_ema_model.load_state_dict(temp_path)

            # Check that the new EMA model's shadow parameters match the saved state
            for shadow_param, new_shadow_param in zip(self.ema_model.shadow_params, new_ema_model.shadow_params):
                self.assertTrue(torch.equal(shadow_param, new_shadow_param))

    def test_copy_to_handles_reordered_parameters(self):
        class MixedParamModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight_a = torch.nn.Parameter(torch.zeros(2, 2))
                self.weight_b = torch.nn.Parameter(torch.zeros(1, 1, 1, 3))

        model = MixedParamModule()
        args = type(
            "Args",
            (),
            {"ema_update_interval": None, "ema_device": "cpu", "ema_cpu_only": True},
        )
        ema_model = EMAModel(
            args=args,
            accelerator=None,
            parameters=model.parameters(),
            decay=0.5,
            foreach=True,
            update_after_step=-1,
        )

        with torch.no_grad():
            ema_model.shadow_params[0].fill_(3.0)
            ema_model.shadow_params[1].fill_(7.0)

        ema_model.copy_to(reversed(list(model.parameters())))

        self.assertTrue(torch.allclose(model.weight_a, torch.full_like(model.weight_a, 3.0)))
        self.assertTrue(torch.allclose(model.weight_b, torch.full_like(model.weight_b, 7.0)))

    def test_copy_to_subset_of_parameters(self):
        class SmallModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight_a = torch.nn.Parameter(torch.zeros(2, 2))
                self.weight_b = torch.nn.Parameter(torch.zeros(1, 1, 1, 3))

        model = SmallModule()
        args = type(
            "Args",
            (),
            {"ema_update_interval": None, "ema_device": "cpu", "ema_cpu_only": True},
        )
        ema_model = EMAModel(
            args=args,
            accelerator=None,
            parameters=model.parameters(),
            decay=0.5,
            foreach=False,
            update_after_step=-1,
        )

        with torch.no_grad():
            ema_model.shadow_params[0].fill_(5.0)
            ema_model.shadow_params[1].fill_(9.0)

        # Only copy the second parameter (subset of tracked params).
        ema_model.copy_to([model.weight_b])

        self.assertTrue(torch.allclose(model.weight_a, torch.zeros_like(model.weight_a)))
        self.assertTrue(torch.allclose(model.weight_b, torch.full_like(model.weight_b, 9.0)))

    def test_copy_to_ignores_untracked_superset_parameters(self):
        class PartialModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight_a = torch.nn.Parameter(torch.zeros(2, 2))
                self.weight_b = torch.nn.Parameter(torch.zeros(1, 1, 1, 3))

        model = PartialModule()
        args = type(
            "Args",
            (),
            {"ema_update_interval": None, "ema_device": "cpu", "ema_cpu_only": True},
        )

        # Track only one parameter in EMA (simulating LoRA-only or frozen base weights).
        ema_model = EMAModel(
            args=args,
            accelerator=None,
            parameters=[model.weight_a],
            decay=0.5,
            foreach=False,
            update_after_step=-1,
        )

        with torch.no_grad():
            ema_model.shadow_params[0].fill_(4.0)

        # Pass the full parameter list (tracked + untracked); only tracked params should update.
        ema_model.copy_to(model.parameters())

        self.assertTrue(torch.allclose(model.weight_a, torch.full_like(model.weight_a, 4.0)))
        self.assertTrue(torch.allclose(model.weight_b, torch.zeros_like(model.weight_b)))

    def test_store_and_restore_with_reordered_parameters(self):
        class SmallModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight_a = torch.nn.Parameter(torch.ones(2, 2))
                self.weight_b = torch.nn.Parameter(torch.ones(1, 1, 1, 3))

        model = SmallModule()
        original_params = [param.clone() for param in model.parameters()]
        args = type(
            "Args",
            (),
            {"ema_update_interval": None, "ema_device": "cpu", "ema_cpu_only": True},
        )
        ema_model = EMAModel(
            args=args,
            accelerator=None,
            parameters=model.parameters(),
            decay=0.5,
            foreach=False,
            update_after_step=-1,
        )

        ema_model.store(model.parameters())
        with torch.no_grad():
            for param in model.parameters():
                param.add_(5.0)

        ema_model.restore(reversed(list(model.parameters())))

        for param, expected in zip(model.parameters(), original_params):
            self.assertTrue(torch.equal(param, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA for mixed-device EMA handling.")
    def test_mixed_device_parameters_disable_foreach(self):
        mixed_model = torch.nn.Module()
        mixed_model.register_parameter("cpu_weight", torch.nn.Parameter(torch.ones(2, 2, device="cpu")))
        mixed_model.register_parameter("cuda_weight", torch.nn.Parameter(torch.ones(2, 2, device="cuda")))

        args = type(
            "Args",
            (),
            {"ema_update_interval": None, "ema_device": "cpu", "ema_cpu_only": True},
        )
        ema_model = EMAModel(
            args=args,
            accelerator=None,
            parameters=mixed_model.parameters(),
            decay=0.5,
            foreach=True,
            update_after_step=-1,
        )
        ema_model.to(device="cpu")

        # Ensure the EMA step does not raise due to mixed CPU/CUDA parameters.
        ema_model.step(mixed_model.parameters(), global_step=2)

        self.assertIsNotNone(ema_model.cur_decay_value)
        self.assertTrue(all(param.device.type == "cpu" for param in ema_model.shadow_params))


if __name__ == "__main__":
    unittest.main()

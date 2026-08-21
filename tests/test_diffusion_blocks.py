import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from simpletuner.helpers.configuration.cmd_args import parse_cmdline_args
from simpletuner.helpers.models.common import ModelFoundation, ModelTypes
from simpletuner.helpers.training.diffusion_blocks import (
    DiffusionBlocksConfig,
    DiffusionBlocksController,
    discover_block_paths,
    normalize_model_timesteps,
)
from simpletuner.helpers.training.trainer import Trainer


class _ToyTransformer(nn.Module):
    def __init__(self, depth=6):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(depth)])

    def forward(self, hidden_states, timestep):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class _IndexedToyTransformer(nn.Module):
    def __init__(self, depth=6):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([nn.Identity() for _ in range(depth)])
        self.visited_indices = []

    def forward(self, hidden_states, timestep):
        self.visited_indices = []
        for index, block in enumerate(self.transformer_blocks):
            self.visited_indices.append(index)
            hidden_states = block(hidden_states)
        return hidden_states


class _IndexedSliceToyTransformer(nn.Module):
    def __init__(self, depth=6):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([nn.Identity() for _ in range(depth)])
        self.visited_blocks = []

    def forward(self, hidden_states, timestep):
        self.visited_blocks = []
        for index, _block in enumerate(self.transformer_blocks):
            segment = self.transformer_blocks[index : index + 1]
            self.visited_blocks.extend(segment)
            hidden_states = segment[0](hidden_states)
        return hidden_states


class _TwoStageTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(6)])
        self.single_transformer_blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.adapter = nn.Module()
        self.adapter.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])

    def forward(self, hidden_states, timestep):
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        for block in self.single_transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class _AmbiguousTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a = nn.Module()
        self.branch_a.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
        self.branch_b = nn.Module()
        self.branch_b.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])

    def forward(self, hidden_states, timestep):
        return hidden_states


class _TextVisualTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_transformer_blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
        self.visual_transformer_blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    def forward(self, hidden_states, timestep):
        for block in self.text_transformer_blocks:
            hidden_states = block(hidden_states)
        for block in self.visual_transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class DiffusionBlocksConfigTests(unittest.TestCase):
    def test_config_requires_positive_layers_per_block(self):
        with self.assertRaisesRegex(ValueError, "layers_per_block"):
            DiffusionBlocksConfig.from_dict({})

    def test_config_normalizes_optional_fields(self):
        config = DiffusionBlocksConfig.from_dict(
            {
                "layers_per_block": 2,
                "blocks_to_train": [0, 2],
                "overlap": 0.1,
                "block_paths": ["transformer_blocks"],
                "timestep_boundaries": [0.0, 0.2, 0.6, 1.0],
            }
        )
        self.assertEqual(config.blocks_to_train, (0, 2))
        self.assertEqual(config.block_paths, ("transformer_blocks",))
        self.assertEqual(config.timestep_boundaries, (0.0, 0.2, 0.6, 1.0))

    def test_cli_parses_json_configuration(self):
        args = parse_cmdline_args(
            input_args=[
                "--model_family=anima",
                "--output_dir=/tmp/output",
                "--model_type=lora",
                "--optimizer=adamw_bf16",
                "--data_backend_config=/tmp/config.json",
                '--diffusion_blocks_config={"layers_per_block": 4}',
            ],
            exit_on_error=True,
        )
        self.assertEqual(args.diffusion_blocks_config, {"layers_per_block": 4})


class DiffusionBlocksControllerTests(unittest.TestCase):
    def test_discovers_primary_and_secondary_transformer_stages(self):
        model = _TwoStageTransformer()
        self.assertEqual(discover_block_paths(model), ["transformer_blocks", "single_transformer_blocks"])

    def test_discovers_text_and_visual_stages_in_execution_order(self):
        self.assertEqual(
            discover_block_paths(_TextVisualTransformer()),
            ["text_transformer_blocks", "visual_transformer_blocks"],
        )

    def test_ambiguous_generic_layer_lists_require_explicit_paths(self):
        model = _AmbiguousTransformer()
        self.assertEqual(discover_block_paths(model), [])
        with self.assertRaisesRegex(ValueError, "block_paths explicitly"):
            DiffusionBlocksController(model, DiffusionBlocksConfig(layers_per_block=1))

    def test_iteration_uses_only_active_slice_without_changing_state_keys(self):
        model = _ToyTransformer()
        original_keys = tuple(model.state_dict())
        expected_blocks = list(model.transformer_blocks[2:4])
        controller = DiffusionBlocksController(model, DiffusionBlocksConfig(layers_per_block=2))

        controller.activate(1)

        self.assertEqual(list(model.transformer_blocks), expected_blocks)
        self.assertEqual(tuple(model.state_dict()), original_keys)
        controller.activate(None)
        self.assertEqual(len(list(model.transformer_blocks)), 6)

    def test_physical_enumerate_indices_address_physical_segment_slices(self):
        model = _IndexedSliceToyTransformer()
        expected_blocks = list(model.transformer_blocks[2:4])
        controller = DiffusionBlocksController(model, DiffusionBlocksConfig(layers_per_block=2))
        controller.set_training_block(1)

        model(torch.ones(1, 4), torch.tensor([0.5]))

        self.assertEqual(model.visited_blocks, expected_blocks)

    def test_partitions_multiple_stages_into_same_number_of_groups(self):
        model = _TwoStageTransformer()
        controller = DiffusionBlocksController(model, DiffusionBlocksConfig(layers_per_block=2))

        self.assertEqual(controller.num_blocks, 5)
        self.assertEqual(
            controller.layer_slices(1),
            {"transformer_blocks": (2, 4), "single_transformer_blocks": (0, 0)},
        )
        self.assertEqual(
            controller.layer_slices(3),
            {"transformer_blocks": (0, 0), "single_transformer_blocks": (0, 2)},
        )

    def test_high_noise_activates_early_network_group(self):
        controller = DiffusionBlocksController(_ToyTransformer(), DiffusionBlocksConfig(layers_per_block=2))

        self.assertEqual(controller.block_for_sigmas(torch.tensor([0.95, 0.8])), 0)
        self.assertEqual(controller.block_for_sigmas(torch.tensor([0.5])), 1)
        self.assertEqual(controller.block_for_sigmas(torch.tensor([0.1])), 2)

    def test_mixed_noise_groups_are_rejected(self):
        controller = DiffusionBlocksController(_ToyTransformer(), DiffusionBlocksConfig(layers_per_block=2))
        with self.assertRaisesRegex(ValueError, "same noise-range block"):
            controller.block_for_sigmas(torch.tensor([0.9, 0.1]))

    def test_eval_forward_selects_block_from_timestep(self):
        model = _ToyTransformer()
        controller = DiffusionBlocksController(model, DiffusionBlocksConfig(layers_per_block=2))
        calls = [0] * 6
        handles = []
        for index, block in enumerate(model.transformer_blocks):
            handles.append(block.register_forward_hook(lambda *_args, idx=index: calls.__setitem__(idx, calls[idx] + 1)))
        try:
            model.eval()(torch.ones(1, 4), torch.tensor([0.9]))
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(controller.active_block, 0)
        self.assertEqual(calls, [1, 1, 0, 0, 0, 0])

    def test_training_block_overrides_forward_timestep_mapping(self):
        model = _ToyTransformer()
        controller = DiffusionBlocksController(model, DiffusionBlocksConfig(layers_per_block=2))
        controller.set_training_block(2)

        model.train()(torch.ones(1, 4), torch.tensor([0.9]))

        self.assertEqual(controller.active_block, 2)

    def test_deepcopy_uses_its_own_controller(self):
        model = _ToyTransformer()
        controller = DiffusionBlocksController(model, DiffusionBlocksConfig(layers_per_block=2))
        copied_model = copy.deepcopy(model)

        copied_model.eval()(torch.ones(1, 4), torch.tensor([0.1]))

        self.assertEqual(copied_model._diffusion_blocks_controller.active_block, 2)
        self.assertIsNone(controller.active_block)
        self.assertIs(copied_model._diffusion_blocks_controller.model, copied_model)

    def test_training_overlap_extends_both_sides(self):
        controller = DiffusionBlocksController(
            _ToyTransformer(),
            DiffusionBlocksConfig(layers_per_block=2, overlap=0.1),
        )
        low, high = controller.sigma_range(1, include_overlap=True)
        self.assertAlmostEqual(low, 0.3)
        self.assertAlmostEqual(high, 0.7)

    def test_active_iteration_preserves_indices_and_clips_tread_routes_in_global_coordinates(self):
        model = _IndexedToyTransformer()
        model._tread_routes = [{"start_layer_idx": 1, "end_layer_idx": 4, "selection_ratio": 0.5}]
        controller = DiffusionBlocksController(model, DiffusionBlocksConfig(layers_per_block=2))

        controller.activate(1)
        model(torch.ones(1, 4), torch.tensor([0.5]))

        self.assertEqual(model.visited_indices, [2, 3])
        self.assertEqual(
            model._tread_routes,
            [{"start_layer_idx": 2, "end_layer_idx": 3, "selection_ratio": 0.5}],
        )
        controller.activate(2)
        self.assertEqual(
            model._tread_routes,
            [{"start_layer_idx": 4, "end_layer_idx": 4, "selection_ratio": 0.5}],
        )
        controller.activate(None)
        self.assertEqual(
            model._tread_routes,
            [{"start_layer_idx": 1, "end_layer_idx": 4, "selection_ratio": 0.5}],
        )

    def test_negative_tread_route_indices_are_clipped_against_full_model_depth(self):
        model = _IndexedToyTransformer()
        model._tread_routes = [{"start_layer_idx": 3, "end_layer_idx": -1, "selection_ratio": 0.5}]
        controller = DiffusionBlocksController(model, DiffusionBlocksConfig(layers_per_block=2))

        controller.activate(2)

        self.assertEqual(
            model._tread_routes,
            [{"start_layer_idx": 4, "end_layer_idx": 5, "selection_ratio": 0.5}],
        )

    def test_restricted_training_blocks_are_validated(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 2"):
            DiffusionBlocksController(
                _ToyTransformer(),
                DiffusionBlocksConfig(layers_per_block=2, blocks_to_train=(3,)),
            )

    def test_restricted_training_blocks_freeze_other_layer_groups(self):
        model = _ToyTransformer()
        controller = DiffusionBlocksController(
            model,
            DiffusionBlocksConfig(layers_per_block=2, blocks_to_train=(1,)),
        )

        frozen_parameters = controller.freeze_unselected_blocks()

        self.assertGreater(frozen_parameters, 0)
        self.assertFalse(
            any(parameter.requires_grad for block in model.transformer_blocks[:2] for parameter in block.parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad for block in model.transformer_blocks[2:4] for parameter in block.parameters())
        )
        self.assertFalse(
            any(parameter.requires_grad for block in model.transformer_blocks[4:] for parameter in block.parameters())
        )

    def test_explicit_boundaries_must_match_derived_group_count(self):
        with self.assertRaisesRegex(ValueError, r"num_blocks \+ 1"):
            DiffusionBlocksController(
                _ToyTransformer(),
                DiffusionBlocksConfig(layers_per_block=2, timestep_boundaries=(0.0, 0.5, 1.0)),
            )


class DiffusionBlocksTimestepTests(unittest.TestCase):
    def test_normalizes_scheduler_timesteps(self):
        normalized = normalize_model_timesteps(torch.tensor([1000.0, 500.0, 0.0]))
        torch.testing.assert_close(normalized, torch.tensor([1.0, 0.5, 0.0]))


class DiffusionBlocksSamplingTests(unittest.TestCase):
    def test_flow_batch_rejection_sampling_keeps_one_noise_group(self):
        controller = DiffusionBlocksController(_ToyTransformer(), DiffusionBlocksConfig(layers_per_block=2))
        samples = iter(
            [
                (torch.tensor([0.1, 0.9]), torch.tensor([100.0, 900.0])),
                (torch.tensor([0.8, 0.7]), torch.tensor([800.0, 700.0])),
            ]
        )
        foundation = SimpleNamespace(
            diffusion_blocks_controller=controller,
            sample_flow_sigmas=lambda **_kwargs: next(samples),
            _initialize_diffusion_blocks_flow_boundaries=lambda _batch: None,
        )
        batch = {"latents": torch.zeros(2, 1)}

        with patch.object(controller, "choose_training_block", return_value=0):
            sigmas, timesteps = ModelFoundation._sample_diffusion_blocks_flow_batch(foundation, batch=batch, state={})

        torch.testing.assert_close(sigmas, torch.tensor([0.9, 0.8]))
        torch.testing.assert_close(timesteps, torch.tensor([900.0, 800.0]))
        self.assertEqual(batch["diffusion_blocks_block_index"], 0)
        self.assertEqual(controller.training_block, 0)

    def test_discrete_sampling_masks_other_noise_groups(self):
        controller = DiffusionBlocksController(_ToyTransformer(), DiffusionBlocksConfig(layers_per_block=2))
        foundation = SimpleNamespace(
            diffusion_blocks_controller=controller,
            config=SimpleNamespace(diffusion_blocks_config={"layers_per_block": 2}),
        )
        weights = torch.ones(10)

        with patch.object(controller, "choose_training_block", return_value=2):
            timesteps = ModelFoundation._sample_diffusion_blocks_discrete_timesteps(foundation, weights, 64)

        self.assertTrue(torch.all(timesteps <= 3))
        self.assertEqual(controller.training_block, 2)
        self.assertEqual(
            foundation.config.diffusion_blocks_config["timestep_boundaries"],
            list(controller.boundaries),
        )

    def test_preserves_flow_sigma_timesteps(self):
        normalized = normalize_model_timesteps(torch.tensor([1.0, 0.5, 0.0]))
        torch.testing.assert_close(normalized, torch.tensor([1.0, 0.5, 0.0]))


class DiffusionBlocksDistributedTests(unittest.TestCase):
    def test_ddp_enables_unused_parameter_detection(self):
        trainer = Trainer.__new__(Trainer)
        trainer.config = SimpleNamespace(
            diffusion_blocks_config={"layers_per_block": 2},
            find_unused_parameters=None,
        )
        self.assertTrue(trainer._resolve_ddp_find_unused_parameters())

    def test_ddp_rejects_explicit_unused_parameter_disable(self):
        trainer = Trainer.__new__(Trainer)
        trainer.config = SimpleNamespace(
            diffusion_blocks_config={"layers_per_block": 2},
            find_unused_parameters=False,
        )
        with self.assertRaisesRegex(ValueError, "find_unused_parameters=true"):
            trainer._resolve_ddp_find_unused_parameters()


class DiffusionBlocksCompatibilityTests(unittest.TestCase):
    def test_crepa_is_rejected_before_controller_installation(self):
        foundation = SimpleNamespace(
            config=SimpleNamespace(
                diffusion_blocks_config={"layers_per_block": 2},
                crepa_enabled=True,
                controlnet=False,
                musubi_blocks_to_swap=0,
                twinflow_enabled=False,
                scheduled_sampling_max_step_offset=0,
                layersync_enabled=False,
            ),
            MODEL_TYPE=ModelTypes.TRANSFORMER,
            uses_noise_schedule=lambda: True,
        )

        with self.assertRaisesRegex(ValueError, "CREPA fixed-layer capture"):
            ModelFoundation.diffusion_blocks_init(foundation)


if __name__ == "__main__":
    unittest.main()

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from simpletuner.helpers.training.optimizers.adamw_bfloat16 import AdamWBF16


class AdamWBF16MuPTests(unittest.TestCase):
    def parameter(self, dimensions, width):
        param = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16))
        param.infshape = Mock()
        param.infshape.ninf.return_value = dimensions
        param.infshape.width_mult.return_value = width
        return param

    def test_scales_only_matrix_like_weights_and_preserves_group_options(self):
        fixed, vector, matrix, wider = [self.parameter(n, w) for n, w in [(0, 1), (1, 4), (2, 2), (2, 4)]]
        optimizer = AdamWBF16(
            [{"params": [("fixed", fixed), ("matrix", matrix), ("vector", vector)], "initial_lr": 0.04}],
            lr=0.02,
            weight_decay=0.1,
            mup=True,
        )
        optimizer.add_param_group({"params": [("wider", wider)], "lr": 0.008, "weight_decay": 0})
        groups = optimizer.param_groups
        self.assertEqual([g["param_names"] for g in groups], [["fixed", "vector"], ["matrix"], ["wider"]])
        self.assertEqual([g["lr"] for g in groups], [0.02, 0.01, 0.002])
        self.assertEqual([g["weight_decay"] for g in groups], [0.1, 0.2, 0])
        self.assertEqual([g["initial_lr"] for g in groups[:2]], [0.04, 0.02])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5**step)
        optimizer.step()
        scheduler.step()
        self.assertEqual(scheduler.get_last_lr(), [0.02, 0.01, 0.001])

    def test_updates_match_manually_scaled_bf16_through_resume(self):
        matrix = self.parameter(2, 4)
        manual = torch.nn.Parameter(matrix.detach().clone())
        optimizer = AdamWBF16([matrix], lr=0.008, weight_decay=0.25, mup=True)
        reference = AdamWBF16([manual], lr=0.002, weight_decay=1.0)
        for step in range(5):
            if step == 3:
                saved = copy.deepcopy(optimizer.state_dict())
                matrix = self.parameter(2, 4)
                matrix.data.copy_(manual)
                optimizer = AdamWBF16([matrix], lr=0.008, weight_decay=0.25, mup=True)
                optimizer.load_state_dict(saved)
            for param, opt in [(matrix, optimizer), (manual, reference)]:
                param.grad = torch.full_like(param, step + 1)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(step)
                    opt.step()
            self.assertTrue(torch.equal(matrix, manual))
            state = optimizer.state[matrix]
            for key in ("exp_avg", "exp_avg_sq", "shift"):
                self.assertEqual(state[key].dtype, torch.bfloat16)
                self.assertTrue(torch.equal(state[key], reference.state[manual][key]))
            self.assertEqual(state["accumulated_decay"], reference.state[manual]["accumulated_decay"])
        self.assertFalse(torch.equal(matrix, torch.ones_like(matrix)))

    def test_registered_optimizers_agree_on_width_scaling(self):
        from simpletuner.helpers.training.optimizer_param import optimizer_parameters

        params = [self.parameter(n, w) for n, w in [(2, 2), (2, 4), (1, 2)]]
        rates = []
        for name, config in [("mup-adamw", None), ("adamw_bf16", "mup=true")]:
            args = SimpleNamespace(
                optimizer=name,
                optimizer_config=config,
                optimizer_beta1=None,
                optimizer_beta2=None,
                optimizer_release_gradients=False,
            )
            optimizer_cls, details = optimizer_parameters(name, args)
            optimizer = optimizer_cls(params, lr=0.01, **details["default_settings"])
            rates.append({id(p): (g["lr"], g["weight_decay"]) for g in optimizer.param_groups for p in g["params"]})
        self.assertEqual(rates[0], rates[1])

    def test_mup_requires_valid_base_shapes_only_when_enabled(self):
        plain = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        AdamWBF16([plain])
        optimizer = AdamWBF16([self.parameter(1, 2)], mup=True)
        for param, error in [(plain, "set_base_shapes"), (self.parameter(3, 2), "two infinite dimensions")]:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                optimizer.add_param_group({"params": [param]})
            self.assertEqual(len(optimizer.param_groups), 1)

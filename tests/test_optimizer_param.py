import unittest
from types import SimpleNamespace

from simpletuner.helpers.training.optimizer_param import convert_arg_to_parameters


class OptimizerParamTests(unittest.TestCase):
    def test_optimizer_config_preserves_generic_beta_overrides(self):
        args = SimpleNamespace(
            optimizer_config="weight_decay=0.0,eps=1e-8",
            optimizer_beta1=0.9,
            optimizer_beta2=0.95,
        )

        self.assertEqual(
            convert_arg_to_parameters(args),
            {
                "weight_decay": 0.0,
                "eps": 1e-8,
                "betas": (0.9, 0.95),
            },
        )


if __name__ == "__main__":
    unittest.main()

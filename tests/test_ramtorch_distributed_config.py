import os
import unittest
from unittest.mock import patch

from simpletuner.helpers.training.trainer import Trainer


class RamTorchDistributedConfigTests(unittest.TestCase):
    def test_shared_parameters_default_to_enabled(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(Trainer._ramtorch_shared_parameters_enabled())

    def test_shared_parameters_can_be_disabled(self):
        for value in ("0", "false", "NO", "off"):
            with (
                self.subTest(value=value),
                patch.dict(
                    os.environ,
                    {"SIMPLETUNER_RAMTORCH_SHARED_PARAMETERS": value},
                    clear=True,
                ),
            ):
                self.assertFalse(Trainer._ramtorch_shared_parameters_enabled())


if __name__ == "__main__":
    unittest.main()

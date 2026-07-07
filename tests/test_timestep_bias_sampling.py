"""Unit tests for per-dataset flow-matching ``timestep_sampling_offset`` sampling."""

import unittest
from unittest.mock import patch

from simpletuner.helpers.models.common import ModelFoundation


class TestTimestepBias(unittest.TestCase):
    """``ModelFoundation._get_dataset_timestep_sampling_offset`` resolves the per-dataset bias."""

    @patch("simpletuner.helpers.models.common.StateTracker.get_data_backend_config")
    def test_reads_per_dataset_bias(self, mock_get_config):
        mock_get_config.return_value = {"timestep_sampling_offset": -0.5}
        bias = ModelFoundation._get_dataset_timestep_sampling_offset(object(), {"data_backend_id": "ds1"})
        mock_get_config.assert_called_once_with("ds1")
        self.assertEqual(bias, -0.5)

    @patch("simpletuner.helpers.models.common.StateTracker.get_data_backend_config")
    def test_defaults_to_zero_when_unset(self, mock_get_config):
        mock_get_config.return_value = {}
        bias = ModelFoundation._get_dataset_timestep_sampling_offset(object(), {"data_backend_id": "ds1"})
        self.assertEqual(bias, 0.0)

    @patch("simpletuner.helpers.models.common.StateTracker.get_data_backend_config")
    def test_missing_backend_id_is_unbiased(self, mock_get_config):
        mock_get_config.return_value = {}
        bias = ModelFoundation._get_dataset_timestep_sampling_offset(object(), {})
        mock_get_config.assert_called_once_with(None)
        self.assertEqual(bias, 0.0)


if __name__ == "__main__":
    unittest.main()

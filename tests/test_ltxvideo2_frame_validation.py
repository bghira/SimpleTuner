import unittest
from unittest.mock import patch

from simpletuner.helpers.data_backend.factory import init_backend_config


class TestLTXVideo2FrameValidation(unittest.TestCase):
    def test_ltxvideo2_auto_adjusts_invalid_frame_count(self):
        """Invalid frame counts should be auto-adjusted with a warning, not raise an error."""
        backend = {
            "id": "ltx2-video",
            "dataset_type": "video",
            "video": {
                "num_frames": 48,
                "min_frames": 48,
            },
        }
        args = {
            "model_family": "ltxvideo2",
            "model_flavour": "dev",
            "framerate": 25,
        }

        with patch("simpletuner.helpers.data_backend.factory.warning_log") as mock_warning:
            result = init_backend_config(backend, args, accelerator=None)

            # Should auto-adjust 48 -> 41 (nearest valid: (48-1)//8*8+1 = 41)
            self.assertEqual(result["config"]["video"]["min_frames"], 41)
            self.assertEqual(result["config"]["video"]["num_frames"], 41)

            # Should emit warnings for both adjustments
            self.assertGreaterEqual(mock_warning.call_count, 2)
            warning_messages = [str(call) for call in mock_warning.call_args_list]
            self.assertTrue(
                any("min_frames" in msg and "48" in msg and "41" in msg for msg in warning_messages),
                f"Expected warning about min_frames adjustment, got: {warning_messages}",
            )
            self.assertTrue(
                any("num_frames" in msg and "48" in msg and "41" in msg for msg in warning_messages),
                f"Expected warning about num_frames adjustment, got: {warning_messages}",
            )

    def test_ltxvideo2_valid_frame_count_unchanged(self):
        """Valid frame counts should pass through unchanged."""
        backend = {
            "id": "ltx2-video",
            "dataset_type": "video",
            "video": {
                "num_frames": 49,
                "min_frames": 49,
            },
        }
        args = {
            "model_family": "ltxvideo2",
            "model_flavour": "dev",
            "framerate": 25,
        }

        with patch("simpletuner.helpers.data_backend.factory.warning_log") as mock_warning:
            result = init_backend_config(backend, args, accelerator=None)

            # Should remain unchanged
            self.assertEqual(result["config"]["video"]["min_frames"], 49)
            self.assertEqual(result["config"]["video"]["num_frames"], 49)

            # Check that no frame adjustment warnings were emitted
            # (other warnings may still be emitted for is_i2v etc.)
            warning_messages = [str(call) for call in mock_warning.call_args_list]
            frame_adjustment_warnings = [
                msg for msg in warning_messages if "Adjusted video->min_frames" in msg or "Adjusted video->num_frames" in msg
            ]
            self.assertEqual(len(frame_adjustment_warnings), 0)


if __name__ == "__main__":
    unittest.main()

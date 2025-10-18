import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from simpletuner.helpers.utils import offloading as offloading_utils


class GroupOffloadHelperTests(unittest.TestCase):
    def test_enable_group_offload_skips_excluded_components(self):
        linear = torch.nn.Linear(8, 8)
        components = {"transformer": linear, "vae": torch.nn.Linear(8, 8)}

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch.object(offloading_utils, "_DIFFUSERS_GROUP_OFFLOAD_AVAILABLE", True),
            patch.object(offloading_utils, "_is_group_offload_enabled", return_value=False),
            patch.object(offloading_utils, "apply_group_offloading") as mock_apply,
        ):
            offloading_utils.enable_group_offload_on_components(
                components,
                device=torch.device("cpu"),
                number_blocks_per_group=2,
                use_stream=False,
                offload_to_disk_path=tmp_dir,
            )

            self.assertEqual(mock_apply.call_count, 1)
            _, kwargs = mock_apply.call_args
            self.assertEqual(kwargs["module"], linear)
            self.assertEqual(kwargs["offload_to_disk_path"], tmp_dir)
            self.assertEqual(kwargs["num_blocks_per_group"], 2)
            self.assertTrue(os.path.isdir(tmp_dir))

    def test_enable_group_offload_requires_diffusers(self):
        with patch.object(offloading_utils, "_DIFFUSERS_GROUP_OFFLOAD_AVAILABLE", False):
            with self.assertRaises(ImportError):
                offloading_utils.enable_group_offload_on_components(
                    {"transformer": torch.nn.Linear(4, 4)}, device=torch.device("cpu")
                )


if __name__ == "__main__":
    unittest.main()

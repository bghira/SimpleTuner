import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from simpletuner.helpers.models.common import ModelFoundation, ModelTypes
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

    def test_group_offload_streams_disabled_with_checkpointing(self):
        class DummyModel(ModelFoundation):
            MODEL_TYPE = ModelTypes.TRANSFORMER

            def __init__(self):
                # Skip parent init to avoid pulling in unrelated setup.
                self.config = SimpleNamespace(
                    enable_group_offload=True,
                    gradient_checkpointing=True,
                    group_offload_use_stream=True,
                )
                self.accelerator = SimpleNamespace(device=torch.device("cuda"))
                self.model = torch.nn.Linear(2, 2)
                self._group_offload_configured = False

            def unwrap_model(self, model=None):
                return model or self.model

            # Abstract methods not exercised in this test suite.
            def model_predict(self, prepared_batch, custom_timesteps: list = None):
                return prepared_batch

            def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
                return prompts

            def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor, prompt: str):
                return {"prompt_embeds": text_embedding}

            def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor, prompt: str):
                return {"negative_prompt_embeds": text_embedding}

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("simpletuner.helpers.models.common.enable_group_offload_on_components") as mock_enable,
        ):
            dummy = DummyModel()
            dummy.configure_group_offload()

        self.assertFalse(dummy.config.group_offload_use_stream)
        _, kwargs = mock_enable.call_args
        self.assertFalse(kwargs["use_stream"])


if __name__ == "__main__":
    unittest.main()

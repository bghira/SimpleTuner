import unittest
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.ltxvideo2 import pipeline_ltx2, pipeline_ltx2_image2video


def _build_comfy_state_dict():
    return {
        "diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(1, 1),
        "diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight": torch.zeros(1, 1),
        "diffusion_model.transformer_blocks.0.attn.to_q.alpha": torch.tensor(8.0),
    }


class TestLTX2LoraFormat(unittest.TestCase):
    def _run_pipeline_test(self, pipeline_module, pipeline_class_name):
        pipeline_cls = getattr(pipeline_module, pipeline_class_name)
        pipe = pipeline_cls.__new__(pipeline_cls)
        pipe.transformer_name = "transformer"
        pipe.transformer = MagicMock()

        state_dict = _build_comfy_state_dict()

        with (
            patch.object(pipe, "lora_state_dict", return_value=(state_dict, {"some": "meta"})),
            patch.object(pipeline_module, "USE_PEFT_BACKEND", True),
            patch.object(pipeline_module, "is_peft_version", return_value=True),
        ):
            pipe.load_lora_weights("dummy", lora_format="comfyui")

        args, kwargs = pipe.transformer.load_lora_adapter.call_args
        passed_state = args[0]
        self.assertTrue(all(key.startswith("transformer.") for key in passed_state))
        self.assertFalse(any(key.endswith(".alpha") for key in passed_state))
        self.assertIsNone(kwargs.get("metadata"))
        self.assertIsNotNone(kwargs.get("network_alphas"))

    def test_text2video_pipeline_loads_comfyui_lora(self):
        self._run_pipeline_test(pipeline_ltx2, "LTX2Pipeline")

    def test_image2video_pipeline_loads_comfyui_lora(self):
        self._run_pipeline_test(pipeline_ltx2_image2video, "LTX2ImageToVideoPipeline")


if __name__ == "__main__":
    unittest.main()

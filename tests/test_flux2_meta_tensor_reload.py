import types
import unittest

import torch

from simpletuner.helpers.models.flux2.model import Flux2


class Flux2MetaTensorReloadTestCase(unittest.TestCase):
    def test_load_model_retries_when_meta_tensors_detected(self):
        flux2 = Flux2.__new__(Flux2)

        class DummyAccelerator:
            device = torch.device("cpu")

            def unwrap_model(self, model):
                return model

        class DummyModelLoader:
            calls = []

            @classmethod
            def from_pretrained(cls, model_path, subfolder=None, **kwargs):
                cls.calls.append({"model_path": model_path, "subfolder": subfolder, **kwargs})
                module = torch.nn.Linear(2, 2)
                if len(cls.calls) == 1:
                    module.to("meta")
                return module

        flux2.accelerator = DummyAccelerator()
        flux2.MODEL_CLASS = DummyModelLoader
        flux2.model = None
        flux2.controlnet = None
        flux2.vae = None
        flux2.text_encoders = None
        flux2.tokenizers = None
        flux2.layersync_regularizer = None
        flux2.pipeline_quantization_active = False

        flux2.config = types.SimpleNamespace(
            revision=None,
            variant=None,
            weight_dtype=torch.float32,
            model_family="flux2",
            model_flavour="dev",
            pretrained_model_name_or_path="dummy-model",
            pretrained_transformer_model_name_or_path=None,
            pretrained_unet_model_name_or_path=None,
            pretrained_transformer_subfolder="transformer",
            pretrained_unet_subfolder="unet",
            pretrained_vae_model_name_or_path="dummy-model",
            vae_path="dummy-model",
            base_model_precision="no_change",
            quantization_config=None,
            quantize_via="accelerator",
            controlnet=False,
            control=False,
            ramtorch=False,
            enable_chunked_feed_forward=False,
            feed_forward_chunk_size=None,
            feed_forward_chunk_dim=None,
            gradient_checkpointing_interval=None,
            gradient_checkpointing=False,
            fuse_qkv_projections=False,
            musubi_blocks_to_swap=0,
            musubi_block_swap_device="cpu",
            layersync_enabled=False,
            use_fsdp=False,
        )

        flux2.load_model(move_to_device=False)

        self.assertEqual(len(DummyModelLoader.calls), 2)
        self.assertIn("low_cpu_mem_usage", DummyModelLoader.calls[1])
        self.assertFalse(DummyModelLoader.calls[1]["low_cpu_mem_usage"])
        self.assertTrue(isinstance(flux2.model, torch.nn.Module))
        self.assertFalse(any(param.device.type == "meta" for param in flux2.model.parameters()))


if __name__ == "__main__":
    unittest.main()

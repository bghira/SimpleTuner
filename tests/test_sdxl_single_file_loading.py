import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from simpletuner.helpers.models.common import PipelineTypes
from simpletuner.helpers.models.sdxl.model import SDXL


class DummyModule:
    def __init__(self, name, *, scaling_factor=None):
        self.name = name
        self.device = "cpu"
        self.dtype = torch.float32
        self.to_calls = []
        self.config = SimpleNamespace(scaling_factor=scaling_factor)

    def to(self, device, dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.to_calls.append((device, dtype))
        return self


class DummyPipeline:
    calls = []
    pipeline = None

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_single_file(cls, checkpoint_path, **kwargs):
        cls.calls.append((checkpoint_path, kwargs))
        return cls.pipeline


class TestSDXLSingleFileLoading(unittest.TestCase):
    def _make_model(self, *, vae_path=None):
        model = SDXL.__new__(SDXL)
        checkpoint_path = "/tmp/illustrious-xl-v2.safetensors"
        model.config = SimpleNamespace(
            model_family="sdxl",
            pretrained_model_name_or_path=checkpoint_path,
            pretrained_vae_model_name_or_path=vae_path or checkpoint_path,
            vae_path=vae_path or checkpoint_path,
            revision=None,
            variant=None,
            weight_dtype=torch.bfloat16,
            controlnet=False,
            delete_model_after_load=False,
            vae_enable_tiling=False,
            vae_enable_slicing=False,
            crepa_drop_vae_encoder=False,
            vae_dtype="bf16",
            local_files_only=True,
        )
        model.accelerator = SimpleNamespace(device="cpu")
        model.pipelines = {}
        model._single_file_component_cache = None
        model._validation_preview_decoder = None
        model._validation_preview_decoder_failed = False
        model._ramtorch_vae_requested = lambda: False
        model._ramtorch_text_encoders_requested = lambda: False
        model.post_vae_load_setup = lambda: None
        model.PIPELINE_CLASSES = {PipelineTypes.TEXT2IMG: DummyPipeline}
        return model

    def setUp(self):
        DummyPipeline.calls = []
        DummyPipeline.pipeline = SimpleNamespace(
            vae=DummyModule("pipeline_vae", scaling_factor=0.13025),
            text_encoder=DummyModule("text_encoder"),
            text_encoder_2=DummyModule("text_encoder_2"),
            tokenizer=object(),
            tokenizer_2=object(),
        )

    def test_single_file_loads_vae_and_text_components_from_pipeline(self):
        model = self._make_model()

        with (
            patch("simpletuner.helpers.models.sdxl.model.AutoencoderKL.from_pretrained") as mock_vae_loader,
            patch("simpletuner.helpers.models.sdxl.model.CLIPTokenizer.from_pretrained") as mock_tokenizer_loader,
            patch("simpletuner.helpers.models.sdxl.model.CLIPTextModel.from_pretrained") as mock_te1_loader,
            patch("simpletuner.helpers.models.sdxl.model.CLIPTextModelWithProjection.from_pretrained") as mock_te2_loader,
        ):
            model.load_vae(move_to_device=False)
            model.load_text_encoder(move_to_device=False)

        self.assertIs(model.vae, DummyPipeline.pipeline.vae)
        self.assertEqual(model.AUTOENCODER_SCALING_FACTOR, 0.13025)
        self.assertEqual(model.text_encoders, [DummyPipeline.pipeline.text_encoder, DummyPipeline.pipeline.text_encoder_2])
        self.assertEqual(model.tokenizers, [DummyPipeline.pipeline.tokenizer, DummyPipeline.pipeline.tokenizer_2])
        self.assertEqual(len(DummyPipeline.calls), 1)
        self.assertEqual(DummyPipeline.calls[0][0], "/tmp/illustrious-xl-v2.safetensors")
        self.assertEqual(DummyPipeline.calls[0][1]["config"], "stabilityai/stable-diffusion-xl-base-1.0")
        self.assertTrue(DummyPipeline.calls[0][1]["local_files_only"])
        mock_vae_loader.assert_not_called()
        mock_tokenizer_loader.assert_not_called()
        mock_te1_loader.assert_not_called()
        mock_te2_loader.assert_not_called()

    def test_explicit_external_vae_override_skips_single_file_pipeline_vae(self):
        model = self._make_model(vae_path="madebyollin/sdxl-vae-fp16-fix")
        external_vae = DummyModule("external_vae", scaling_factor=0.5)

        with patch(
            "simpletuner.helpers.models.sdxl.model.AutoencoderKL.from_pretrained",
            return_value=external_vae,
        ) as mock_vae_loader:
            model.load_vae(move_to_device=False)

        self.assertIs(model.vae, external_vae)
        self.assertEqual(model.AUTOENCODER_SCALING_FACTOR, 0.5)
        self.assertEqual(DummyPipeline.calls, [])
        mock_vae_loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()

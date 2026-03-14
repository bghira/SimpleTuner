import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.training.crepa import CrepaMode


class TestAnimaModel(unittest.TestCase):
    def test_model_import(self):
        from simpletuner.helpers.models.anima.model import Anima

        self.assertIsNotNone(Anima)
        self.assertEqual(Anima.NAME, "Anima")
        self.assertEqual(Anima.DEFAULT_MODEL_FLAVOUR, "preview")

    def test_transformer_import(self):
        from simpletuner.helpers.models.anima.transformer import AnimaTransformerModel

        self.assertIsNotNone(AnimaTransformerModel)
        self.assertTrue(hasattr(AnimaTransformerModel, "from_pretrained"))
        self.assertTrue(hasattr(AnimaTransformerModel, "from_single_file"))

    def test_pipeline_import(self):
        from simpletuner.helpers.models.anima.pipeline import AnimaPipeline

        self.assertIsNotNone(AnimaPipeline)
        self.assertTrue(hasattr(AnimaPipeline, "from_pretrained"))
        self.assertTrue(hasattr(AnimaPipeline, "from_single_file"))

    def test_vendored_headers_present(self):
        anima_dir = Path(__file__).parent.parent / "simpletuner/helpers/models/anima"
        for relative_path in ("transformer.py", "pipeline.py", "loading.py", "scheduler.py"):
            contents = (anima_dir / relative_path).read_text()
            self.assertTrue(contents.startswith("# Vendored from diffusers-anima: "))

    def test_metadata_contains_anima(self):
        metadata_path = Path(__file__).parent.parent / "simpletuner/helpers/models/model_metadata.json"
        with open(metadata_path) as handle:
            metadata = json.load(handle)

        self.assertIn("anima", metadata)
        self.assertEqual(metadata["anima"]["class_name"], "Anima")
        self.assertEqual(metadata["anima"]["module_path"], "simpletuner.helpers.models.anima.model")

    def test_tokenizer_loader_ignores_proxies(self):
        from simpletuner.helpers.models.anima.loading import AnimaLoaderOptions, load_tokenizer_from_source

        options = AnimaLoaderOptions(local_files_only=True, proxies={"https": "http://proxy"})
        with patch("simpletuner.helpers.models.anima.loading.AutoTokenizer.from_pretrained") as mock_from_pretrained:
            load_tokenizer_from_source("repo::tokenizer", options=options)

        self.assertNotIn("proxies", mock_from_pretrained.call_args.kwargs)

    def test_lora_loader_ignores_proxies(self):
        from simpletuner.helpers.models.anima.lora_pipeline import _fetch_anima_lora_state_dict

        with (
            patch("simpletuner.helpers.models.anima.lora_pipeline.hf_hub_download") as mock_download,
            patch("simpletuner.helpers.models.anima.lora_pipeline.torch.load") as mock_torch_load,
        ):
            mock_download.return_value = "weights.bin"
            mock_torch_load.return_value = {"transformer.block.lora_A.weight": object()}
            _fetch_anima_lora_state_dict(
                "repo-id",
                weight_name="weights.bin",
                use_safetensors=False,
                local_files_only=True,
                cache_dir=None,
                force_download=False,
                proxies={"https": "http://proxy"},
                token=None,
                revision=None,
                subfolder=None,
                allow_pickle=True,
            )

        self.assertNotIn("proxies", mock_download.call_args.kwargs)

    def test_transformer_validation_guards(self):
        from simpletuner.helpers.models.anima.transformer import _AdapterAttention, _RotaryEmbedding

        with self.assertRaisesRegex(ValueError, "even"):
            _RotaryEmbedding(63)

        with self.assertRaisesRegex(ValueError, "divisible"):
            _AdapterAttention(query_dim=1025, context_dim=1024, heads=16)

    def test_model_supports_crepa_self_flow_and_image_mode(self):
        from simpletuner.helpers.models.anima.model import Anima

        model = Anima.__new__(Anima)
        self.assertTrue(model.supports_crepa_self_flow())
        self.assertEqual(model.crepa_mode, CrepaMode.IMAGE)

    def test_prepare_crepa_self_flow_batch_builds_tokenwise_student_and_teacher_views(self):
        from simpletuner.helpers.models.anima.model import Anima

        model = Anima.__new__(Anima)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(weight_dtype=torch.float32, crepa_self_flow_mask_ratio=0.5)
        model.model = MagicMock(config=SimpleNamespace(patch_size=(1, 2, 2)))
        model.unwrap_model = lambda model=None, wrapped=None: model if model is not None else wrapped
        model.sample_flow_sigmas = MagicMock(return_value=(torch.tensor([0.8]), torch.tensor([800.0])))

        batch = {
            "latents": torch.zeros(1, 16, 1, 4, 4, dtype=torch.float32),
            "input_noise": torch.ones(1, 16, 1, 4, 4, dtype=torch.float32),
            "sigmas": torch.tensor([0.2], dtype=torch.float32),
            "timesteps": torch.tensor([200.0], dtype=torch.float32),
        }
        fake_mask_rand = torch.tensor([[[[0.2, 0.7], [0.9, 0.1]]]], dtype=torch.float32)

        with patch("torch.rand", return_value=fake_mask_rand):
            result = model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(result["timesteps"].shape, (1, 4))
        self.assertEqual(result["sigmas"].shape, (1, 1, 1, 4, 4))
        self.assertEqual(result["crepa_teacher_timesteps"].shape, (1,))
        unique_timesteps = torch.unique(result["timesteps"].view(-1)).cpu()
        torch.testing.assert_close(unique_timesteps, torch.tensor([200.0, 800.0], dtype=torch.float32))
        self.assertTrue(torch.equal(result["crepa_self_flow_mask"], fake_mask_rand < 0.5))

    def test_model_predict_preserves_tokenwise_timesteps_and_capture_override(self):
        from simpletuner.helpers.models.anima.model import Anima

        model = Anima.__new__(Anima)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(weight_dtype=torch.float32)
        captured = torch.randn(1, 4, 8)

        def _forward(hidden_states, timestep, encoder_hidden_states, **kwargs):
            kwargs["hidden_states_buffer"]["layer_7"] = captured
            self.assertTrue(torch.equal(timestep, torch.tensor([[100.0, 900.0, 100.0, 900.0]], dtype=torch.float32)))
            return (torch.randn(1, 16, 1, 4, 4),)

        model.model = MagicMock(side_effect=_forward, config=SimpleNamespace(patch_size=(1, 2, 2)))
        model._new_hidden_state_buffer = MagicMock(return_value={})
        model.unwrap_model = lambda model=None, wrapped=None: model if model is not None else wrapped
        model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=2)

        prepared_batch = {
            "noisy_latents": torch.randn(1, 16, 1, 4, 4),
            "timesteps": torch.tensor([[100.0, 900.0, 100.0, 900.0]], dtype=torch.float32),
            "encoder_hidden_states": torch.randn(1, 3, 8),
            "crepa_capture_block_index": 7,
            "t5xxl_ids": None,
            "t5xxl_weights": None,
        }

        result = model.model_predict(prepared_batch)

        self.assertIs(result["crepa_hidden_states"], captured)
        self.assertIs(result["hidden_states_buffer"], model._new_hidden_state_buffer.return_value)


if __name__ == "__main__":
    unittest.main()

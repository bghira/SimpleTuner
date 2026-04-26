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
        self.assertEqual(Anima.DEFAULT_MODEL_FLAVOUR, "preview-3")

    def test_model_flavours_use_converted_diffusers_repos(self):
        from simpletuner.helpers.models.anima.model import Anima

        self.assertEqual(
            Anima.HUGGINGFACE_PATHS["preview-3"],
            "CalamitousFelicitousness/Anima-Preview-3-sdnext-diffusers",
        )
        self.assertEqual(
            Anima.HUGGINGFACE_PATHS["preview-2"],
            "CalamitousFelicitousness/Anima-Preview-2-sdnext-diffusers",
        )
        self.assertEqual(
            Anima.HUGGINGFACE_PATHS["preview"],
            "CalamitousFelicitousness/Anima-sdnext-diffusers",
        )

    def test_diffusers_layout_switches_component_sources(self):
        from simpletuner.helpers.models.anima.model import Anima
        from simpletuner.helpers.models.common import ImageModelFoundation

        model = Anima.__new__(Anima)
        model.config = SimpleNamespace(
            pretrained_model_name_or_path="CalamitousFelicitousness/Anima-Preview-3-sdnext-diffusers",
            model_flavour="preview-3",
        )

        self.assertTrue(model._uses_diffusers_repo_layout())
        self.assertEqual(
            model._prompt_tokenizer_sources(),
            (
                "CalamitousFelicitousness/Anima-Preview-3-sdnext-diffusers::tokenizer",
                "CalamitousFelicitousness/Anima-Preview-3-sdnext-diffusers::t5_tokenizer",
            ),
        )

        with patch.object(ImageModelFoundation, "load_model", return_value="loaded") as mock_load_model:
            self.assertEqual(model.load_model(move_to_device=False), "loaded")

        self.assertEqual(model.MODEL_SUBFOLDER, "transformer")
        mock_load_model.assert_called_once_with(move_to_device=False)

    def test_diffusers_layout_loads_text_encoder_and_vae_from_standard_subfolders(self):
        from simpletuner.helpers.models.anima.model import Anima

        model = Anima.__new__(Anima)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(
            pretrained_model_name_or_path="CalamitousFelicitousness/Anima-Preview-2-sdnext-diffusers",
            pretrained_text_encoder_model_name_or_path=None,
            model_flavour="preview-2",
            revision=None,
            text_encoder_revision=None,
            weight_dtype=torch.float32,
            local_files_only=True,
            cache_dir=None,
            force_download=False,
            token=False,
        )
        model.prompt_tokenizer = object()
        model.text_encoders = None
        text_encoder = MagicMock()
        text_encoder.eval.return_value = text_encoder
        text_encoder.requires_grad_.return_value = text_encoder
        vae = MagicMock(config=SimpleNamespace(scaling_factor=0.18215))
        vae.eval.return_value = vae
        vae.requires_grad_.return_value = vae
        fake_autoencoder = SimpleNamespace(from_pretrained=MagicMock(return_value=vae))

        with (
            patch(
                "simpletuner.helpers.models.anima.model.Qwen3Model.from_pretrained", return_value=text_encoder
            ) as mock_text_encoder,
            patch.object(Anima, "AUTOENCODER_CLASS", fake_autoencoder),
        ):
            model.load_text_encoder(move_to_device=False)
            model.load_vae(move_to_device=False)

        mock_text_encoder.assert_called_once()
        self.assertEqual(mock_text_encoder.call_args.kwargs["subfolder"], "text_encoder")
        self.assertNotIn("token", mock_text_encoder.call_args.kwargs)
        fake_autoencoder.from_pretrained.assert_called_once()
        self.assertEqual(fake_autoencoder.from_pretrained.call_args.kwargs["subfolder"], "vae")
        self.assertNotIn("token", fake_autoencoder.from_pretrained.call_args.kwargs)
        self.assertIs(model.text_encoder, text_encoder)
        self.assertIs(model.vae, vae)

    def test_text_encoder_override_uses_resolved_path_layout(self):
        from simpletuner.helpers.models.anima.model import Anima

        model = Anima.__new__(Anima)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(
            pretrained_model_name_or_path="CalamitousFelicitousness/Anima-Preview-3-sdnext-diffusers",
            pretrained_text_encoder_model_name_or_path="circlestone-labs/Anima",
            model_flavour="preview-3",
            revision=None,
            text_encoder_revision=None,
            weight_dtype=torch.float32,
            local_files_only=True,
            cache_dir=None,
            force_download=False,
            token=None,
        )
        model.prompt_tokenizer = object()
        model.text_encoders = None
        text_encoder = MagicMock()

        with (
            patch("simpletuner.helpers.models.anima.model.Qwen3Model.from_pretrained") as mock_from_pretrained,
            patch(
                "simpletuner.helpers.models.anima.model._resolve_weight_path",
                return_value="weights.safetensors",
            ) as mock_resolve,
            patch("simpletuner.helpers.models.anima.model.load_text_encoder_single_file", return_value=text_encoder),
        ):
            model.load_text_encoder(move_to_device=False)

        mock_from_pretrained.assert_not_called()
        mock_resolve.assert_called_once_with(
            "circlestone-labs/Anima",
            filename="model.safetensors",
            subfolder="split_files/text_encoders",
            revision=None,
        )
        self.assertIs(model.text_encoder, text_encoder)

    def test_model_flavour_does_not_override_explicit_legacy_layout(self):
        from simpletuner.helpers.models.anima.model import Anima

        model = Anima.__new__(Anima)
        model.config = SimpleNamespace(
            pretrained_model_name_or_path="circlestone-labs/Anima",
            model_flavour="preview-3",
        )

        self.assertFalse(model._uses_diffusers_repo_layout())

    def test_transformer_subfolder_falls_back_to_single_file_when_diffusers_load_fails(self):
        from simpletuner.helpers.models.anima.transformer import AnimaTransformerModel

        sentinel = object()
        with (
            patch(
                "simpletuner.helpers.models.anima.transformer.ModelMixin.from_pretrained",
                side_effect=OSError("missing transformer/config.json"),
            ) as mock_from_pretrained,
            patch.object(AnimaTransformerModel, "from_single_file", return_value=sentinel) as mock_from_single_file,
        ):
            result = AnimaTransformerModel.from_pretrained("circlestone-labs/Anima", subfolder="transformer")

        self.assertIs(result, sentinel)
        mock_from_pretrained.assert_called_once()
        mock_from_single_file.assert_called_once()
        self.assertEqual(mock_from_single_file.call_args.args[0], "circlestone-labs/Anima")
        self.assertEqual(mock_from_single_file.call_args.kwargs["subfolder"], "transformer")

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

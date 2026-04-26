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

    def test_pipeline_load_without_base_model_uses_cached_embed_components_only(self):
        from simpletuner.helpers.models.anima.model import Anima
        from simpletuner.helpers.models.common import PipelineTypes

        class DummyPipeline:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        model = Anima.__new__(Anima)
        model.model = object()
        model.vae = None
        model.text_encoders = None
        model.prompt_tokenizer = None
        model.pipelines = {}
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(weight_dtype=torch.float32)
        model.PIPELINE_CLASSES = {PipelineTypes.TEXT2IMG: DummyPipeline}
        model.unwrap_model = lambda model=None, wrapped=None: model if model is not None else wrapped
        model.get_vae = lambda: model.vae
        model.load_model = MagicMock()

        model.load_vae = MagicMock(side_effect=lambda move_to_device=True: setattr(model, "vae", object()))
        model.load_text_encoder = MagicMock()
        model.load_text_tokenizer = MagicMock()

        pipeline = model._load_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)

        model.load_model.assert_not_called()
        model.load_vae.assert_called_once_with(move_to_device=True)
        model.load_text_encoder.assert_not_called()
        model.load_text_tokenizer.assert_not_called()
        self.assertIs(pipeline.transformer, model.model)
        self.assertIs(pipeline.vae, model.vae)
        self.assertIsNone(pipeline.text_encoder)
        self.assertIsNone(pipeline.prompt_tokenizer)

    def test_transformer_import(self):
        from simpletuner.helpers.models.anima.transformer import AnimaTransformerModel

        self.assertIsNotNone(AnimaTransformerModel)
        self.assertTrue(hasattr(AnimaTransformerModel, "from_pretrained"))
        self.assertTrue(hasattr(AnimaTransformerModel, "from_single_file"))

    def test_transformer_forward_captures_hidden_states(self):
        from simpletuner.helpers.models.anima.transformer import AnimaTransformerModel

        model = AnimaTransformerModel(
            in_channels=2,
            out_channels=2,
            num_attention_heads=2,
            attention_head_dim=4,
            num_layers=1,
            mlp_ratio=2.0,
            text_embed_dim=8,
            adaln_lora_dim=8,
            max_size=(2, 4, 4),
            patch_size=(1, 2, 2),
            adapter_dim=8,
            adapter_layers=1,
            adapter_heads=2,
        )
        hidden_states = torch.randn(1, 2, 1, 4, 4)
        encoder_hidden_states = torch.randn(1, 3, 8)
        hidden_states_buffer = {}

        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                timestep=torch.tensor([100.0]),
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
                hidden_states_buffer=hidden_states_buffer,
            )

        self.assertEqual(output[0].shape, hidden_states.shape)
        self.assertIn("layer_0", hidden_states_buffer)
        self.assertEqual(hidden_states_buffer["layer_0"].shape, (1, 4, 8))

    def test_transformer_gradient_checkpointing_controls_core(self):
        from simpletuner.helpers.models.anima.transformer import AnimaTransformerModel

        model = AnimaTransformerModel(
            in_channels=2,
            out_channels=2,
            num_attention_heads=2,
            attention_head_dim=4,
            num_layers=1,
            mlp_ratio=2.0,
            text_embed_dim=8,
            adaln_lora_dim=8,
            max_size=(2, 4, 4),
            patch_size=(1, 2, 2),
            adapter_dim=8,
            adapter_layers=1,
            adapter_heads=2,
        )

        self.assertFalse(model.core.gradient_checkpointing)
        model.enable_gradient_checkpointing()
        self.assertTrue(model.core.gradient_checkpointing)
        model.disable_gradient_checkpointing()
        self.assertFalse(model.core.gradient_checkpointing)

    def test_diffusers_transformer_loads_sibling_llm_adapter(self):
        from tempfile import TemporaryDirectory

        from safetensors.torch import save_file

        from simpletuner.helpers.models.anima.transformer import AnimaTransformerModel

        source = AnimaTransformerModel(
            in_channels=2,
            out_channels=2,
            num_attention_heads=2,
            attention_head_dim=4,
            num_layers=1,
            mlp_ratio=2.0,
            text_embed_dim=8,
            adaln_lora_dim=8,
            max_size=(2, 4, 4),
            patch_size=(1, 2, 2),
            adapter_dim=8,
            adapter_layers=1,
            adapter_heads=2,
        )

        with TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            transformer_path = repo_path / "transformer"
            adapter_dir = repo_path / "llm_adapter"
            adapter_dir.mkdir()
            source.core.save_pretrained(str(transformer_path), safe_serialization=True)
            with open(adapter_dir / "config.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "source_dim": 8,
                        "target_dim": 8,
                        "model_dim": 8,
                        "num_layers": 1,
                        "num_heads": 2,
                        "vocab_size": 32128,
                    },
                    handle,
                )
            adapter_path = adapter_dir / "diffusion_pytorch_model.safetensors"
            save_file(source.llm_adapter.state_dict(), str(adapter_path))

            loaded = AnimaTransformerModel.from_pretrained(
                str(repo_path),
                subfolder="transformer",
                local_files_only=True,
                token=False,
            )

        for name, parameter in loaded.core.state_dict().items():
            torch.testing.assert_close(parameter, source.core.state_dict()[name])
        for name, parameter in loaded.llm_adapter.state_dict().items():
            torch.testing.assert_close(parameter, source.llm_adapter.state_dict()[name])

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

    def test_lora_state_dict_preserves_diffusers_peft_checkpoint_keys(self):
        from simpletuner.helpers.models.anima.lora_pipeline import AnimaLoraLoaderMixin

        down = torch.randn(4, 8)
        up = torch.randn(8, 4)
        state_dict = {
            "transformer.core.transformer_blocks.0.attn1.to_k.lora.down.weight": down,
            "transformer.core.transformer_blocks.0.attn1.to_k.lora.up.weight": up,
        }

        loaded = AnimaLoraLoaderMixin.lora_state_dict(state_dict)

        self.assertIs(loaded["transformer.core.transformer_blocks.0.attn1.to_k.lora.down.weight"], down)
        self.assertIs(loaded["transformer.core.transformer_blocks.0.attn1.to_k.lora.up.weight"], up)

    def test_lora_converter_accepts_peft_down_up_suffixes(self):
        from simpletuner.helpers.models.anima.lora_pipeline import _convert_non_diffusers_anima_lora_to_diffusers

        down = torch.randn(4, 8)
        up = torch.randn(8, 4)
        converted = _convert_non_diffusers_anima_lora_to_diffusers(
            {
                "blocks.0.self_attn.k_proj.lora.down.weight": down,
                "blocks.0.self_attn.k_proj.lora.up.weight": up,
            }
        )

        self.assertIs(converted["transformer.core.transformer_blocks.0.attn1.to_k.lora_A.weight"], down)
        self.assertIs(converted["transformer.core.transformer_blocks.0.attn1.to_k.lora_B.weight"], up)

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
        torch.testing.assert_close(unique_timesteps, torch.tensor([0.2, 0.8], dtype=torch.float32))
        self.assertTrue(torch.equal(result["crepa_self_flow_mask"], fake_mask_rand < 0.5))

    def test_model_predict_preserves_tokenwise_timesteps_and_capture_override(self):
        from simpletuner.helpers.models.anima.model import Anima

        model = Anima.__new__(Anima)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(weight_dtype=torch.float32)
        captured = torch.randn(1, 4, 8)

        def _forward(hidden_states, timestep, encoder_hidden_states, **kwargs):
            kwargs["hidden_states_buffer"]["layer_7"] = captured
            torch.testing.assert_close(timestep, torch.tensor([[0.1, 0.9, 0.1, 0.9]], dtype=torch.float32))
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
        self.assertEqual(result["model_prediction"].shape, (1, 16, 1, 4, 4))

    def test_sample_flow_sigmas_returns_sigma_space_model_timesteps(self):
        from simpletuner.helpers.models.anima.model import Anima
        from simpletuner.helpers.models.common import ImageModelFoundation

        model = Anima.__new__(Anima)
        model.noise_schedule = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        with patch.object(
            ImageModelFoundation,
            "sample_flow_sigmas",
            return_value=(torch.tensor([0.25, 0.75]), torch.tensor([250.0, 750.0])),
        ):
            sigmas, timesteps = model.sample_flow_sigmas(batch={}, state={})

        torch.testing.assert_close(sigmas, torch.tensor([0.25, 0.75]))
        torch.testing.assert_close(timesteps, torch.tensor([0.25, 0.75]))

    def test_model_predict_preserves_frame_axis_to_match_flow_target(self):
        from simpletuner.helpers.models.anima.model import Anima

        model = Anima.__new__(Anima)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(weight_dtype=torch.float32)
        prediction = torch.randn(1, 16, 1, 4, 4)
        model.model = MagicMock(return_value=(prediction,), config=SimpleNamespace(patch_size=(1, 2, 2)))
        model._new_hidden_state_buffer = MagicMock(return_value={})
        model.unwrap_model = lambda model=None, wrapped=None: model if model is not None else wrapped
        model.crepa_regularizer = None

        prepared_batch = {
            "latents": torch.randn(1, 16, 1, 4, 4),
            "noise": torch.randn(1, 16, 1, 4, 4),
            "noisy_latents": torch.randn(1, 16, 1, 4, 4),
            "timesteps": torch.tensor([500.0], dtype=torch.float32),
            "encoder_hidden_states": torch.randn(1, 3, 8),
            "t5xxl_ids": None,
            "t5xxl_weights": None,
        }

        result = model.model_predict(prepared_batch)
        target = model.get_prediction_target(prepared_batch)

        self.assertEqual(result["model_prediction"].shape, target.shape)
        self.assertEqual((result["model_prediction"] - target).shape, target.shape)

    def test_expand_sigmas_matches_anima_latent_rank(self):
        from simpletuner.helpers.models.anima.model import Anima

        model = Anima.__new__(Anima)
        batch = {
            "latents": torch.zeros(2, 16, 1, 4, 4),
            "sigmas": torch.tensor([0.25, 0.75], dtype=torch.float32),
        }

        result = model.expand_sigmas(batch)

        self.assertEqual(result["sigmas"].shape, (2, 1, 1, 1, 1))

    def test_collate_prompt_embeds_preserves_anima_adapter_inputs(self):
        from simpletuner.helpers.models.anima.model import Anima

        model = Anima.__new__(Anima)
        text_encoder_output = [
            {
                "prompt_embeds": torch.ones(1, 3, 8),
                "t5xxl_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                "t5xxl_weights": torch.tensor([1.0, 0.5, 1.0]),
            },
            {
                "prompt_embeds": torch.zeros(1, 3, 8),
                "t5xxl_ids": torch.tensor([4, 5, 6], dtype=torch.long),
                "t5xxl_weights": torch.tensor([0.25, 1.0, 0.75]),
            },
        ]

        collated = model.collate_prompt_embeds(text_encoder_output)

        self.assertEqual(collated["prompt_embeds"].shape, (2, 3, 8))
        torch.testing.assert_close(collated["t5xxl_ids"], torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long))
        torch.testing.assert_close(
            collated["t5xxl_weights"],
            torch.tensor([[[1.0], [0.5], [1.0]], [[0.25], [1.0], [0.75]]]),
        )

    def test_collate_prompt_embeds_pads_variable_length_anima_adapter_inputs(self):
        from simpletuner.helpers.models.anima.model import Anima

        model = Anima.__new__(Anima)
        text_encoder_output = [
            {
                "prompt_embeds": torch.ones(1, 2, 8),
                "t5xxl_ids": torch.tensor([[1, 2]], dtype=torch.int32),
                "t5xxl_weights": torch.tensor([[[1.0], [0.5]]]),
            },
            {
                "prompt_embeds": torch.zeros(1, 3, 8),
                "t5xxl_ids": torch.tensor([[4, 5, 6]], dtype=torch.int32),
                "t5xxl_weights": torch.tensor([[[0.25], [1.0], [0.75]]]),
            },
        ]

        collated = model.collate_prompt_embeds(text_encoder_output)

        self.assertEqual(collated["prompt_embeds"].shape, (2, 3, 8))
        self.assertEqual(collated["t5xxl_ids"].shape, (2, 3))
        self.assertEqual(collated["t5xxl_weights"].shape, (2, 3, 1))
        torch.testing.assert_close(collated["t5xxl_ids"][0], torch.tensor([1, 2, 0], dtype=torch.int32))
        torch.testing.assert_close(collated["t5xxl_weights"][0], torch.tensor([[1.0], [0.5], [0.0]]))

    def test_convert_text_embed_for_pipeline_passes_raw_anima_adapter_inputs(self):
        from simpletuner.helpers.models.anima.model import Anima

        model = Anima.__new__(Anima)
        prompt_embeds = torch.randn(1, 3, 8)
        t5xxl_ids = torch.tensor([[1, 2, 3]], dtype=torch.int32)
        t5xxl_weights = torch.ones(1, 3, 1)

        converted = model.convert_text_embed_for_pipeline(
            {
                "prompt_embeds": prompt_embeds,
                "t5xxl_ids": t5xxl_ids,
                "t5xxl_weights": t5xxl_weights,
            }
        )
        negative = model.convert_negative_text_embed_for_pipeline(
            {
                "prompt_embeds": prompt_embeds,
                "t5xxl_ids": t5xxl_ids,
                "t5xxl_weights": t5xxl_weights,
            }
        )

        self.assertIs(converted["prompt_embeds"], prompt_embeds)
        self.assertIs(converted["prompt_t5xxl_ids"], t5xxl_ids)
        self.assertIs(converted["prompt_t5xxl_weights"], t5xxl_weights)
        self.assertIs(negative["negative_prompt_embeds"], prompt_embeds)
        self.assertIs(negative["negative_prompt_t5xxl_ids"], t5xxl_ids)
        self.assertIs(negative["negative_prompt_t5xxl_weights"], t5xxl_weights)

    def test_pipeline_accepts_raw_cached_anima_adapter_inputs(self):
        from simpletuner.helpers.models.anima.pipeline import AnimaPipeline

        pipe = AnimaPipeline.__new__(AnimaPipeline)
        pipe.vae_scale_factor = 8
        pipe.patch_size = 2
        prompt_embeds = torch.randn(1, 3, 8)
        negative_prompt_embeds = torch.randn(1, 3, 8)
        t5xxl_ids = torch.tensor([[1, 2, 3]], dtype=torch.int32)
        t5xxl_weights = torch.ones(1, 3, 1)

        pipe.check_inputs(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_t5xxl_ids=t5xxl_ids,
            prompt_t5xxl_weights=t5xxl_weights,
            negative_prompt_t5xxl_ids=t5xxl_ids,
            negative_prompt_t5xxl_weights=t5xxl_weights,
            image=None,
            mask_image=None,
            strength=1.0,
            width=512,
            height=512,
            num_inference_steps=1,
            num_images_per_prompt=1,
            generator=None,
            sampler="euler_a_rf",
            sigma_schedule="beta",
            cfg_batch_mode="split",
            output_type="latent",
            callback_on_step_end_tensor_inputs=["latents"],
        )

        with self.assertRaisesRegex(ValueError, "raw cached prompt embeds require"):
            pipe.check_inputs(
                prompt=None,
                negative_prompt=None,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_t5xxl_ids=t5xxl_ids,
                prompt_t5xxl_weights=None,
                negative_prompt_t5xxl_ids=t5xxl_ids,
                negative_prompt_t5xxl_weights=t5xxl_weights,
                image=None,
                mask_image=None,
                strength=1.0,
                width=512,
                height=512,
                num_inference_steps=1,
                num_images_per_prompt=1,
                generator=None,
                sampler="euler_a_rf",
                sigma_schedule="beta",
                cfg_batch_mode="split",
                output_type="latent",
                callback_on_step_end_tensor_inputs=["latents"],
            )


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.extract_cosmos3_generator import _write_generator_component
from scripts.extract_cosmos3_reasoner import _write_component
from simpletuner.helpers.models.common import PipelineTypes, PredictionTypes, TextEmbedCacheKey
from simpletuner.helpers.models.cosmos3.audio_tokenizer import Cosmos3AVAEAudioTokenizer
from simpletuner.helpers.models.cosmos3.model import Cosmos3Image
from simpletuner.helpers.models.cosmos3.pipeline import Cosmos3OmniPipeline
from simpletuner.helpers.models.cosmos3.reasoner import (
    COSMOS3_GENERATOR_COMPONENTS,
    COSMOS3_REASONER_COMPONENTS,
    Cosmos3Reasoner,
    Cosmos3ReasonerConfig,
)
from simpletuner.helpers.models.cosmos3.transformer import Cosmos3OmniTransformer, Cosmos3ReasonerMemoryState


class TestableCosmos3Image(Cosmos3Image):
    """Lightweight test double that skips heavy base initialization paths."""

    def __init__(self):
        self.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.config = SimpleNamespace(
            weight_dtype=torch.float32,
            framerate=24.0,
            revision=None,
            model_flavour="nano",
            cosmos3_reasoner_component="test/cosmos3-reasoner",
            cosmos3_generator_component="test/cosmos3-generator",
        )
        self.model = None
        self.reasoner = FakeCosmos3Reasoner()

    def prepare_batch_conditions(self, batch: dict, state: dict) -> dict:
        return batch

    def _load_reasoner(self):
        return self.reasoner


class FakeCosmos3Adapter:
    vae_scale_factor_spatial = 16

    def __init__(self):
        self.tokenize_calls = 0

    def tokenize_prompt(self, **kwargs):
        self.tokenize_calls += 1
        return [101, 102], []

    def _prepare_text_segment(self, input_ids, device):
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "text_indexes": torch.arange(len(input_ids), dtype=torch.long, device=device),
            "und_len": len(input_ids),
            "text_mrope_ids": torch.zeros(3, len(input_ids), dtype=torch.long, device=device),
            "vision_start_temporal_offset": len(input_ids),
        }

    def _prepare_vision_segment(
        self,
        input_vision_tokens,
        has_image_condition,
        mrope_offset,
        vision_fps,
        curr,
        device,
        condition_frame_indexes=None,
    ):
        _, _, latent_t, _, _ = input_vision_tokens.shape
        frame_token_stride = 1
        condition_frame_indexes = condition_frame_indexes or ([0] if has_image_condition else [])
        cond_frames = {idx for idx in condition_frame_indexes if 0 <= idx < latent_t}
        noisy_frame_indexes = torch.tensor(
            [idx for idx in range(latent_t) if idx not in cond_frames],
            dtype=torch.long,
            device=device,
        )
        mse_loss_indexes = []
        for frame_idx in noisy_frame_indexes.tolist():
            frame_start = curr + frame_idx * frame_token_stride
            mse_loss_indexes.extend(range(frame_start, frame_start + frame_token_stride))
        num_vision_tokens = latent_t * frame_token_stride
        return {
            "vision_token_shapes": [(latent_t, 1, 1)],
            "vision_sequence_indexes": torch.arange(curr, curr + num_vision_tokens, dtype=torch.long, device=device),
            "vision_mse_loss_indexes": torch.tensor(mse_loss_indexes, dtype=torch.long, device=device),
            "vision_noisy_frame_indexes": [noisy_frame_indexes],
            "vision_mrope_ids": torch.zeros(3, num_vision_tokens, dtype=torch.long, device=device),
            "num_vision_tokens": num_vision_tokens,
            "num_noisy_vision_tokens": len(mse_loss_indexes),
        }

    def _prepare_sound_segment(self, input_sound_tokens, mrope_offset, sound_fps, curr, device):
        sound_len = input_sound_tokens.shape[1]
        return {
            "sound_token_shapes": [(sound_len, 1, 1)],
            "sound_sequence_indexes": torch.arange(curr, curr + sound_len, dtype=torch.long, device=device),
            "sound_mse_loss_indexes": torch.arange(curr, curr + sound_len, dtype=torch.long, device=device),
            "sound_noisy_frame_indexes": [torch.arange(sound_len, dtype=torch.long, device=device)],
            "sound_mrope_ids": torch.zeros(3, sound_len, dtype=torch.long, device=device),
            "sound_len": sound_len,
        }


class FakeCosmos3Transformer:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return [torch.zeros_like(kwargs["vision_tokens"][0])], [torch.zeros(64, 5)], None


class FakeCosmos3Reasoner:
    def __init__(self):
        self.calls = []
        self.requires_grad_calls = []

    def __call__(self, *, input_ids, position_ids):
        self.calls.append({"input_ids": input_ids, "position_ids": position_ids})
        return Cosmos3ReasonerMemoryState(layer_kv=[{"k": torch.zeros(2, 1, 4), "v": torch.zeros(2, 1, 4)}])

    def requires_grad_(self, requires_grad):
        self.requires_grad_calls.append(requires_grad)
        return self

    def eval(self):
        return self


class FakePipelineModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float32
        self.config = SimpleNamespace(
            latents_mean=[0.0] * 16,
            latents_std=[1.0] * 16,
            scale_factor_spatial=16,
            scale_factor_temporal=4,
        )


class FakeTextTokenizer:
    eos_token_id = 2

    def __init__(self):
        self.texts = []

    def convert_tokens_to_ids(self, token):
        return 1

    def apply_chat_template(
        self,
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        add_vision_id=False,
        return_dict=True,
    ):
        self.texts.append(conversations[-1]["content"])
        return SimpleNamespace(input_ids=[len(self.texts)])


class FakeScheduler:
    pass


class Cosmos3ModelTests(unittest.TestCase):
    def _make_pipeline(self):
        tokenizer = FakeTextTokenizer()
        pipeline = Cosmos3OmniPipeline(
            transformer=FakePipelineModule(),
            text_tokenizer=tokenizer,
            vae=FakePipelineModule(),
            scheduler=FakeScheduler(),
        )
        return pipeline, tokenizer

    def test_model_attributes(self):
        self.assertEqual(Cosmos3Image.DEFAULT_MODEL_FLAVOUR, "nano")
        self.assertEqual(
            Cosmos3Image.HUGGINGFACE_PATHS["nano"],
            "nvidia/Cosmos3-Nano",
        )
        self.assertEqual(
            Cosmos3Image.HUGGINGFACE_PATHS["super"],
            "nvidia/Cosmos3-Super",
        )
        self.assertEqual(
            Cosmos3Image.HUGGINGFACE_PATHS["super-i2v"],
            "nvidia/Cosmos3-Super-Image2Video",
        )
        self.assertEqual(
            Cosmos3Image.HUGGINGFACE_PATHS["super-t2i"],
            "nvidia/Cosmos3-Super-Text2Image",
        )
        self.assertEqual(Cosmos3Image.PREDICTION_TYPE, PredictionTypes.FLOW_MATCHING)
        self.assertEqual(Cosmos3Image.TEXT_ENCODER_CONFIGURATION, {})
        self.assertIs(Cosmos3Image.MODEL_CLASS, Cosmos3OmniTransformer)
        self.assertIs(Cosmos3Image.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG], Cosmos3OmniPipeline)
        self.assertEqual(Cosmos3OmniTransformer.__module__, "simpletuner.helpers.models.cosmos3.transformer")
        self.assertEqual(Cosmos3OmniPipeline.__module__, "simpletuner.helpers.models.cosmos3.pipeline")
        self.assertEqual(Cosmos3AVAEAudioTokenizer.__module__, "simpletuner.helpers.models.cosmos3.audio_tokenizer")
        self.assertEqual(Cosmos3Image.MODEL_LICENSE, "openmdw-1.1")

    def test_uses_text_embedding_cache_for_reasoner_payloads(self):
        model = TestableCosmos3Image()
        self.assertTrue(model.uses_text_embeddings_cache())
        self.assertFalse(model.use_text_cache_dropout_sentinel())
        self.assertEqual(model.text_embed_cache_key(), TextEmbedCacheKey.DATASET_AND_FILENAME)

    def test_lora_defaults_target_generation_path(self):
        model = TestableCosmos3Image()

        self.assertEqual(
            model.get_lora_target_layers(),
            ["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"],
        )
        self.assertNotIn("to_q", model.get_lora_target_layers())
        self.assertIn("to_q", model.DEFAULT_LORA_EXCLUDE_TARGETS)
        self.assertIn("mlp", model.DEFAULT_LORA_EXCLUDE_TARGETS)

    def test_freeze_reasoning_layers_keeps_generation_path_trainable(self):
        model = TestableCosmos3Image()
        transformer = Cosmos3OmniTransformer(
            hidden_size=8,
            intermediate_size=16,
            head_dim=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_hidden_layers=1,
            latent_channel=2,
            patch_latent_dim=8,
            vocab_size=32,
        )

        frozen = model.freeze_reasoning_layers(transformer)

        self.assertIn("embed_tokens", frozen)
        self.assertFalse(transformer.layers[0].self_attn.to_q.weight.requires_grad)
        self.assertFalse(transformer.layers[0].mlp.down_proj.weight.requires_grad)
        self.assertTrue(transformer.layers[0].self_attn.add_q_proj.weight.requires_grad)
        self.assertTrue(transformer.layers[0].mlp_moe_gen.down_proj.weight.requires_grad)

    def test_cosmos3_text_cache_metadata_and_collation(self):
        model = TestableCosmos3Image()
        latent = torch.zeros(16, 3, 4, 5)
        metadata = model.text_embed_cache_metadata_for_sample(
            example={},
            latent=latent,
            prompt="a robot drummer",
            data_backend_id="train",
            dataset_relative_path="clip.mp4",
        )
        key = model.text_embed_cache_key_value(
            prompt="a robot drummer",
            default_key="train:clip.mp4",
            metadata=metadata,
        )

        self.assertEqual(metadata["cosmos3_num_frames"], 3)
        self.assertEqual(metadata["cosmos3_height"], 64)
        self.assertEqual(metadata["cosmos3_width"], 80)
        self.assertIn(metadata["cosmos3_cache_signature"], key)
        collated = model.collate_prompt_embeds(
            [
                {"cosmos3_reasoner_cache": {"cache_signature": "one"}},
                {"cosmos3_reasoner_cache": {"cache_signature": "two"}},
            ]
        )
        self.assertEqual([entry["cache_signature"] for entry in collated["cosmos3_reasoner_cache"]], ["one", "two"])

    def test_predict_single_sample_uses_cached_reasoner_segment(self):
        model = TestableCosmos3Image()
        model.model = FakeCosmos3Transformer()
        adapter = FakeCosmos3Adapter()
        prompt = "a robot playing piano"
        signature = model._cosmos3_text_cache_signature(
            prompt=prompt,
            metadata={
                "cosmos3_num_frames": 1,
                "cosmos3_height": 32,
                "cosmos3_width": 32,
                "cosmos3_fps": 24.0,
                "cosmos3_reasoner_component": model._reasoner_component_id(),
            },
        )
        reasoner_cache = {
            "cache_signature": signature,
            "input_ids": torch.tensor([101, 102]),
            "text_indexes": torch.tensor([0, 1]),
            "und_len": torch.tensor(2),
            "text_mrope_ids": torch.zeros(3, 2, dtype=torch.long),
            "vision_start_temporal_offset": torch.tensor(2.0),
            "reasoner_memory_state": {"layer_kv": [{"k": torch.zeros(2, 1, 4), "v": torch.zeros(2, 1, 4)}]},
        }

        model._predict_single_sample(
            adapter=adapter,
            prompt=prompt,
            vision_tokens=torch.zeros(1, 16, 1, 2, 2),
            vision_timestep=torch.tensor(250.0),
            fps=24.0,
            reasoner_cache=reasoner_cache,
        )

        self.assertEqual(adapter.tokenize_calls, 0)
        self.assertIsInstance(model.model.kwargs["reasoner_memory_state"], Cosmos3ReasonerMemoryState)
        self.assertEqual(model.reasoner.calls, [])

    def test_predict_single_sample_builds_reasoner_segment_on_cache_miss(self):
        model = TestableCosmos3Image()
        model.model = FakeCosmos3Transformer()
        adapter = FakeCosmos3Adapter()

        model._predict_single_sample(
            adapter=adapter,
            prompt="a robot playing piano",
            vision_tokens=torch.zeros(1, 16, 1, 2, 2),
            vision_timestep=torch.tensor(250.0),
            fps=24.0,
        )

        self.assertEqual(adapter.tokenize_calls, 1)
        self.assertEqual(len(model.reasoner.calls), 1)
        self.assertIsInstance(model.model.kwargs["reasoner_memory_state"], Cosmos3ReasonerMemoryState)

    def test_model_metadata_contains_cosmos3(self):
        metadata = json.loads(resources.files("simpletuner.helpers.models").joinpath("model_metadata.json").read_text())

        self.assertIn("cosmos3", metadata)
        self.assertEqual(metadata["cosmos3"]["class_name"], "Cosmos3Image")
        self.assertEqual(
            metadata["cosmos3"]["module_path"],
            "simpletuner.helpers.models.cosmos3.model",
        )
        self.assertEqual(metadata["cosmos3"]["flavour_choices"], ["edge", "nano", "super", "super-i2v", "super-t2i"])
        self.assertEqual(metadata["cosmos3"]["prediction_type"], "flow_matching")

    def test_reasoner_component_auto_resolves_by_flavour(self):
        model = TestableCosmos3Image()
        model.config.cosmos3_reasoner_component = "auto"
        for flavour, component_id in COSMOS3_REASONER_COMPONENTS.items():
            model.config.model_flavour = flavour
            self.assertEqual(model._reasoner_component_id(), component_id)

        model.config.cosmos3_reasoner_component = "local/component"
        self.assertEqual(model._reasoner_component_id(), "local/component")

    def test_generator_component_auto_resolves_by_flavour(self):
        model = TestableCosmos3Image()
        model.config.cosmos3_generator_component = "auto"
        for flavour, component_id in COSMOS3_GENERATOR_COMPONENTS.items():
            model.config.model_flavour = flavour
            self.assertEqual(model._generator_component_id(), component_id)

        model.config.cosmos3_generator_component = "local/generator"
        self.assertEqual(model._generator_component_id(), "local/generator")

    def test_super_i2v_flavour_requires_conditioning_latents(self):
        model = TestableCosmos3Image()
        model.config.model_flavour = "super-i2v"

        self.assertTrue(model.requires_conditioning_dataset())
        self.assertTrue(model.requires_conditioning_latents())
        self.assertTrue(model.requires_conditioning_validation_inputs())
        self.assertTrue(model.requires_validation_i2v_samples())

    def test_pipeline_has_no_cosmos_specific_safety_checker(self):
        pipeline, _ = self._make_pipeline()

        self.assertNotIn("safety_checker", pipeline.components)
        self.assertNotIn("safety_checker", Cosmos3OmniPipeline._optional_components)
        self.assertNotIn("enable_safety_checker", pipeline.config)

    def test_tokenize_prompt_wraps_video_positive_prompt_as_cosmos_json(self):
        pipeline, tokenizer = self._make_pipeline()

        pipeline.tokenize_prompt(
            "a robot drummer plays a synchronized rhythm",
            negative_prompt="blurry",
            num_frames=49,
            height=432,
            width=768,
            fps=24,
            use_system_prompt=False,
        )

        positive = json.loads(tokenizer.texts[0])
        self.assertEqual(positive["temporal_caption"], "a robot drummer plays a synchronized rhythm")
        self.assertEqual(positive["audio_description"], "a robot drummer plays a synchronized rhythm")
        self.assertEqual(positive["resolution"], {"H": 432, "W": 768})
        self.assertEqual(positive["aspect_ratio"], "16,9")
        self.assertEqual(positive["duration"], "2s")
        self.assertEqual(positive["fps"], 24)
        self.assertEqual(positive["actions"][0]["time"], "0:00-0:02")
        self.assertTrue(tokenizer.texts[1].startswith("blurry. The video is not"))

    def test_tokenize_prompt_preserves_existing_json_with_current_metadata(self):
        pipeline, tokenizer = self._make_pipeline()
        prompt = json.dumps(
            {
                "subjects": [{"description": "custom subject"}],
                "temporal_caption": "custom temporal caption",
                "duration": "9s",
                "fps": 12,
                "resolution": {"H": 1, "W": 1},
            }
        )

        pipeline.tokenize_prompt(
            prompt,
            negative_prompt="",
            num_frames=49,
            height=432,
            width=768,
            fps=24,
            use_system_prompt=False,
        )

        positive = json.loads(tokenizer.texts[0])
        self.assertEqual(positive["subjects"], [{"description": "custom subject"}])
        self.assertEqual(positive["temporal_caption"], "custom temporal caption")
        self.assertEqual(positive["duration"], "2s")
        self.assertEqual(positive["fps"], 24)
        self.assertEqual(positive["resolution"], {"H": 432, "W": 768})
        self.assertEqual(positive["aspect_ratio"], "16,9")

    def test_tokenize_prompt_keeps_action_json_separate(self):
        pipeline, tokenizer = self._make_pipeline()

        pipeline.tokenize_prompt(
            "move the arm forward",
            negative_prompt="",
            num_frames=49,
            height=432,
            width=768,
            fps=24,
            use_system_prompt=False,
            action_mode="policy",
            action_view_point="ego_view",
        )

        positive = json.loads(tokenizer.texts[0])
        self.assertIn("actions", positive)
        self.assertIn("cinematography", positive)
        self.assertNotIn("temporal_caption", positive)

    def test_advertises_audio_latent_cache(self):
        model = TestableCosmos3Image()
        self.assertTrue(model.supports_audio_inputs())
        self.assertTrue(model.uses_audio_latents())

    def test_pre_vae_transform_distinguishes_audio_and_video(self):
        model = TestableCosmos3Image()
        audio = torch.zeros(2, 1, 16000)
        image = torch.zeros(2, 3, 64, 64)

        self.assertIs(model.pre_vae_encode_transform_sample(audio), audio)
        self.assertEqual(model.pre_vae_encode_transform_sample(image).shape, (2, 3, 1, 64, 64))

    def test_prepare_batch_carries_audio_latents(self):
        torch.manual_seed(1)
        model = TestableCosmos3Image()
        batch = {
            "latent_batch": torch.zeros(2, 16, 1, 4, 4),
            "audio_latent_batch": torch.zeros(2, 64, 10),
            "audio_latent_mask": torch.tensor([1.0, 0.0]),
        }

        prepared = model.prepare_batch(batch, state={})

        self.assertEqual(prepared["noisy_latents"].shape, (2, 16, 1, 4, 4))
        self.assertEqual(prepared["audio_noisy_latents"].shape, (2, 64, 10))
        self.assertEqual(prepared["audio_timesteps"].shape, (2,))
        self.assertTrue(torch.equal(prepared["audio_latent_mask"], torch.tensor([1.0, 0.0])))

    def test_super_i2v_prepare_batch_anchors_first_frame_conditioning(self):
        torch.manual_seed(2)
        model = TestableCosmos3Image()
        model.config.model_flavour = "super-i2v"
        conditioning = torch.ones(2, 16, 1, 4, 4)
        batch = {
            "latent_batch": torch.zeros(2, 16, 3, 4, 4),
            "conditioning_latents": conditioning,
        }

        prepared = model.prepare_batch(batch, state={})

        self.assertTrue(torch.equal(prepared["noisy_latents"][:, :, 0], conditioning[:, :, 0]))
        self.assertEqual(prepared["vision_loss_mask"].shape, (2, 1, 3, 4, 4))
        self.assertTrue(torch.equal(prepared["vision_loss_mask"][:, :, 0], torch.zeros(2, 1, 4, 4)))
        self.assertTrue(torch.equal(prepared["vision_loss_mask"][:, :, 1:], torch.ones(2, 1, 2, 4, 4)))

    def test_super_i2v_prepare_batch_requires_conditioning_latents(self):
        model = TestableCosmos3Image()
        model.config.model_flavour = "super-i2v"

        with self.assertRaisesRegex(ValueError, "requires conditioning_latents"):
            model.prepare_batch({"latent_batch": torch.zeros(1, 16, 3, 4, 4)}, state={})

    def test_predict_single_sample_packs_sound_tokens(self):
        model = TestableCosmos3Image()
        model.model = FakeCosmos3Transformer()
        vision_tokens = torch.zeros(1, 16, 1, 2, 2)
        sound_tokens = torch.zeros(64, 5)

        model._predict_single_sample(
            adapter=FakeCosmos3Adapter(),
            prompt="a robot playing piano",
            vision_tokens=vision_tokens,
            vision_timestep=torch.tensor(250.0),
            fps=24.0,
            sound_tokens=sound_tokens,
            sound_timestep=torch.tensor(500.0),
        )

        kwargs = model.model.kwargs
        self.assertEqual(kwargs["sequence_length"], 8)
        self.assertEqual(kwargs["position_ids"].shape, (3, 8))
        self.assertEqual(kwargs["vision_timesteps"].tolist(), [250.0])
        self.assertEqual(kwargs["sound_timesteps"].tolist(), [500.0] * 5)
        self.assertIs(kwargs["sound_tokens"][0], sound_tokens)

    def test_predict_single_sample_marks_i2v_condition_frame_clean(self):
        model = TestableCosmos3Image()
        model.model = FakeCosmos3Transformer()

        model._predict_single_sample(
            adapter=FakeCosmos3Adapter(),
            prompt="a robot starts from a reference image",
            vision_tokens=torch.zeros(1, 16, 3, 2, 2),
            vision_timestep=torch.tensor(250.0),
            fps=16.0,
            condition_frame_indexes=[0],
        )

        kwargs = model.model.kwargs
        self.assertEqual(kwargs["vision_noisy_frame_indexes"][0].tolist(), [1, 2])
        self.assertEqual(kwargs["vision_timesteps"].tolist(), [250.0, 250.0])

    def test_super_i2v_loss_ignores_conditioned_first_frame(self):
        model = TestableCosmos3Image()
        prepared_batch = {
            "latents": torch.ones(1, 16, 3, 2, 2),
            "noise": torch.zeros(1, 16, 3, 2, 2),
            "vision_loss_mask": torch.tensor(
                [[[[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]]]
            ),
        }
        model_output = {"model_prediction": torch.zeros(1, 16, 3, 2, 2)}

        self.assertEqual(model.loss(prepared_batch, model_output).item(), 1.0)

    def test_extract_cosmos3_reasoner_component_filters_and_scrubs_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            transformer_dir = root / "source" / "transformer"
            transformer_dir.mkdir(parents=True)
            with open(transformer_dir / "config.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "hidden_size": 4,
                        "intermediate_size": 8,
                        "head_dim": 2,
                        "num_attention_heads": 2,
                        "num_key_value_heads": 1,
                        "num_hidden_layers": 1,
                        "vocab_size": 2048,
                        "rope_axes_dim": [1, 1, 0],
                        "hidden_act": "relu2",
                        "qk_norm_for_text": False,
                        "use_und_k_norm_for_gen": True,
                    },
                    handle,
                )
            save_file(
                {
                    "embed_tokens.weight": torch.zeros(2048, 4),
                    "layers.0.input_layernorm.weight": torch.ones(4),
                    "layers.0.post_attention_layernorm.weight": torch.ones(4),
                    "layers.0.self_attn.to_q.weight": torch.zeros(4, 4),
                    "layers.0.self_attn.k_norm_und_for_gen.weight": torch.ones(2),
                    "layers.0.self_attn.add_q_proj.weight": torch.ones(4, 4),
                    "layers.0.mlp.up_proj.weight": torch.zeros(8, 4),
                    "norm.weight": torch.ones(4),
                },
                transformer_dir / "diffusion_pytorch_model.safetensors",
                metadata={"source": "/Users/kash/should-not-copy"},
            )

            output_dir = root / "out"
            keys = _write_component(
                source_repo=str(root / "source"),
                source_model_id="nvidia/Cosmos3-Nano",
                source_revision="abc123",
                revision=None,
                output_dir=output_dir,
                dtype=torch.bfloat16,
                index_filename="transformer/missing.index.json",
                weights_filename="transformer/diffusion_pytorch_model.safetensors",
                config_filename="transformer/config.json",
                max_shard_size="1KB",
            )

            self.assertIn("layers.0.self_attn.to_q.weight", keys)
            self.assertIn("layers.0.self_attn.k_norm_und_for_gen.weight", keys)
            self.assertNotIn("layers.0.self_attn.add_q_proj.weight", keys)
            config_text = (output_dir / "config.json").read_text(encoding="utf-8")
            self.assertNotIn("/Users/", config_text)
            index_path = output_dir / "model.safetensors.index.json"
            self.assertTrue(index_path.is_file())
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertIn("layers.0.self_attn.to_q.weight", index["weight_map"])
            shard_path = output_dir / index["weight_map"]["layers.0.self_attn.to_q.weight"]
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
                self.assertEqual(metadata["simpletuner_component"], "cosmos3_reasoner")
                self.assertNotIn("/Users/", json.dumps(metadata))
            state_dict = Cosmos3Reasoner._load_component_state_dict(str(output_dir))
            self.assertIn("layers.0.self_attn.to_q.weight", state_dict)

    def test_extract_cosmos3_generator_component_filters_and_scrubs_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            transformer_dir = root / "source" / "transformer"
            transformer_dir.mkdir(parents=True)
            with open(transformer_dir / "config.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "hidden_size": 4,
                        "intermediate_size": 8,
                        "head_dim": 2,
                        "num_attention_heads": 2,
                        "num_key_value_heads": 1,
                        "num_hidden_layers": 1,
                        "vocab_size": 2048,
                        "rope_axes_dim": [1, 1, 0],
                        "action_dim": 64,
                        "action_gen": True,
                        "num_embodiment_domains": 32,
                        "sound_dim": None,
                        "sound_gen": False,
                        "hidden_act": "relu2",
                        "qk_norm_for_text": False,
                        "use_und_k_norm_for_gen": True,
                        "patch_latent_dim": 4,
                    },
                    handle,
                )
            save_file(
                {
                    "embed_tokens.weight": torch.zeros(2048, 4),
                    "lm_head.weight": torch.zeros(2048, 4),
                    "layers.0.input_layernorm.weight": torch.ones(4),
                    "layers.0.self_attn.to_q.weight": torch.zeros(4, 4),
                    "layers.0.self_attn.add_q_proj.weight": torch.ones(4, 4),
                    "layers.0.self_attn.add_k_proj.weight": torch.ones(2, 4),
                    "layers.0.self_attn.add_v_proj.weight": torch.ones(2, 4),
                    "layers.0.self_attn.to_add_out.weight": torch.ones(4, 4),
                    "layers.0.self_attn.norm_added_q.weight": torch.ones(2),
                    "layers.0.self_attn.norm_added_k.weight": torch.ones(2),
                    "layers.0.self_attn.k_norm_und_for_gen.weight": torch.ones(2),
                    "layers.0.input_layernorm_moe_gen.weight": torch.ones(4),
                    "layers.0.post_attention_layernorm_moe_gen.weight": torch.ones(4),
                    "layers.0.mlp_moe_gen.up_proj.weight": torch.zeros(8, 4),
                    "layers.0.mlp_moe_gen.down_proj.weight": torch.zeros(4, 8),
                    "norm.weight": torch.ones(4),
                    "norm_moe_gen.weight": torch.ones(4),
                    "proj_in.weight": torch.zeros(4, 4),
                    "proj_out.weight": torch.zeros(4, 4),
                    "time_embedder.linear_1.weight": torch.zeros(4, 256),
                },
                transformer_dir / "diffusion_pytorch_model.safetensors",
                metadata={"source": "/Users/kash/should-not-copy"},
            )

            output_dir = root / "out"
            keys = _write_generator_component(
                source_repo=str(root / "source"),
                source_model_id="nvidia/Cosmos3-Nano",
                source_revision="abc123",
                revision=None,
                output_dir=output_dir,
                dtype=torch.bfloat16,
                index_filename="transformer/missing.index.json",
                weights_filename="transformer/diffusion_pytorch_model.safetensors",
                config_filename="transformer/config.json",
                max_shard_size="1KB",
            )

            self.assertIn("layers.0.self_attn.add_q_proj.weight", keys)
            self.assertIn("layers.0.self_attn.k_norm_und_for_gen.weight", keys)
            self.assertIn("proj_in.weight", keys)
            self.assertNotIn("layers.0.self_attn.to_q.weight", keys)
            self.assertNotIn("embed_tokens.weight", keys)
            self.assertNotIn("lm_head.weight", keys)
            config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["component"], "cosmos3_generator")
            self.assertFalse(config["load_reasoning_layers"])
            self.assertTrue(config["action_gen"])
            self.assertFalse(config["sound_gen"])
            self.assertEqual(config["hidden_act"], "relu2")
            self.assertFalse(config["qk_norm_for_text"])
            self.assertTrue(config["use_und_k_norm_for_gen"])
            self.assertNotIn("/Users/", json.dumps(config))
            index = json.loads((output_dir / "diffusion_pytorch_model.safetensors.index.json").read_text(encoding="utf-8"))
            self.assertIn("layers.0.self_attn.add_q_proj.weight", index["weight_map"])
            shard_path = output_dir / index["weight_map"]["layers.0.self_attn.add_q_proj.weight"]
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
                self.assertEqual(metadata["simpletuner_component"], "cosmos3_generator")
                self.assertNotIn("/Users/", json.dumps(metadata))

    def test_generator_only_transformer_does_not_allocate_reasoner_layers(self):
        transformer = Cosmos3OmniTransformer(
            hidden_size=8,
            intermediate_size=16,
            head_dim=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_hidden_layers=1,
            latent_channel=2,
            patch_latent_dim=8,
            vocab_size=32,
            load_reasoning_layers=False,
        )

        self.assertFalse(hasattr(transformer, "embed_tokens"))
        self.assertFalse(hasattr(transformer, "lm_head"))
        self.assertFalse(hasattr(transformer, "norm"))
        self.assertFalse(hasattr(transformer.layers[0], "mlp"))
        self.assertFalse(hasattr(transformer.layers[0], "input_layernorm"))
        self.assertFalse(hasattr(transformer.layers[0].self_attn, "to_q"))
        self.assertTrue(hasattr(transformer.layers[0].self_attn, "add_q_proj"))
        self.assertTrue(hasattr(transformer.layers[0], "mlp_moe_gen"))

        with self.assertRaisesRegex(ValueError, "requires `reasoner_memory_state`"):
            transformer(
                input_ids=torch.tensor([1, 2], dtype=torch.long),
                text_indexes=torch.tensor([0, 1], dtype=torch.long),
                position_ids=torch.zeros(3, 3, dtype=torch.long),
                und_len=2,
                sequence_length=3,
                vision_tokens=[torch.zeros(1, 2, 1, 2, 2)],
                vision_token_shapes=[(1, 1, 1)],
                vision_sequence_indexes=torch.tensor([2], dtype=torch.long),
                vision_mse_loss_indexes=torch.tensor([2], dtype=torch.long),
                vision_timesteps=torch.tensor([250.0]),
                vision_noisy_frame_indexes=[torch.tensor([0], dtype=torch.long)],
                return_dict=False,
            )

    def test_generator_only_transformer_from_pretrained_keeps_reasoner_layers_unallocated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transformer = Cosmos3OmniTransformer(
                hidden_size=8,
                intermediate_size=16,
                head_dim=4,
                num_attention_heads=2,
                num_key_value_heads=1,
                num_hidden_layers=1,
                latent_channel=2,
                patch_latent_dim=8,
                vocab_size=32,
                load_reasoning_layers=False,
            )
            transformer.save_pretrained(tmpdir, safe_serialization=True)

            loaded = Cosmos3OmniTransformer.from_pretrained(tmpdir)

        self.assertFalse(loaded.config.load_reasoning_layers)
        self.assertFalse(hasattr(loaded, "embed_tokens"))
        self.assertFalse(hasattr(loaded.layers[0].self_attn, "to_q"))
        self.assertTrue(hasattr(loaded.layers[0].self_attn, "add_q_proj"))

    def test_reasoner_outputs_generation_key_norm_cache_when_edge_uses_it(self):
        torch.manual_seed(5)
        reasoner = Cosmos3Reasoner(
            Cosmos3ReasonerConfig(
                hidden_size=8,
                intermediate_size=16,
                head_dim=4,
                num_attention_heads=2,
                num_key_value_heads=1,
                num_hidden_layers=1,
                vocab_size=32,
                rope_axes_dim=[2, 1, 1],
                hidden_act="relu2",
                qk_norm_for_text=False,
                use_und_k_norm_for_gen=True,
            )
        )

        memory_state = reasoner(
            input_ids=torch.tensor([1, 2], dtype=torch.long),
            position_ids=torch.zeros(3, 2, dtype=torch.long),
        )

        layer_kv = memory_state.layer_kv[0]
        self.assertIn("k_for_gen", layer_kv)
        self.assertEqual(layer_kv["k_for_gen"].shape, layer_kv["k"].shape)

    def test_transformer_cached_reasoner_replay_matches_joint_forward(self):
        configs = [
            {},
            {"hidden_act": "relu2", "qk_norm_for_text": False, "use_und_k_norm_for_gen": True, "rms_norm_eps": 1e-5},
        ]
        for idx, config_kwargs in enumerate(configs):
            with self.subTest(config_kwargs=config_kwargs):
                torch.manual_seed(7 + idx)
                transformer = Cosmos3OmniTransformer(
                    hidden_size=8,
                    intermediate_size=16,
                    head_dim=4,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    num_hidden_layers=2,
                    latent_channel=2,
                    patch_latent_dim=8,
                    vocab_size=32,
                    rope_axes_dim=[2, 1, 1],
                    **config_kwargs,
                )
                transformer.eval()
                input_ids = torch.tensor([1, 2], dtype=torch.long)
                text_indexes = torch.tensor([0, 1], dtype=torch.long)
                und_len = 2
                vision_tokens = [torch.randn(1, 2, 1, 2, 2)]
                vision_token_shapes = [(1, 1, 1)]
                vision_sequence_indexes = torch.tensor([2], dtype=torch.long)
                vision_mse_loss_indexes = torch.tensor([2], dtype=torch.long)
                vision_timesteps = torch.tensor([250.0])
                vision_noisy_frame_indexes = [torch.tensor([0], dtype=torch.long)]
                position_ids = torch.zeros(3, 3, dtype=torch.long)

                with torch.no_grad():
                    joint = transformer(
                        input_ids=input_ids,
                        text_indexes=text_indexes,
                        position_ids=position_ids,
                        und_len=und_len,
                        sequence_length=3,
                        vision_tokens=vision_tokens,
                        vision_token_shapes=vision_token_shapes,
                        vision_sequence_indexes=vision_sequence_indexes,
                        vision_mse_loss_indexes=vision_mse_loss_indexes,
                        vision_timesteps=vision_timesteps,
                        vision_noisy_frame_indexes=vision_noisy_frame_indexes,
                        return_dict=False,
                    )[0][0]

                    packed_text = transformer.embed_tokens(input_ids)
                    cos, sin = transformer.rotary_emb(
                        position_ids=position_ids.unsqueeze(1),
                        device=packed_text.device,
                        dtype=packed_text.dtype,
                    )
                    cos = cos.squeeze(0)
                    sin = sin.squeeze(0)
                    und_seq = packed_text
                    layer_kv = []
                    for layer in transformer.layers:
                        und_seq, kv = layer.forward_und_only(und_seq, (cos[:und_len], sin[:und_len]))
                        layer_kv.append(kv)
                    if config_kwargs.get("use_und_k_norm_for_gen"):
                        self.assertIn("k_for_gen", layer_kv[0])
                    split = transformer(
                        input_ids=input_ids,
                        text_indexes=text_indexes,
                        position_ids=position_ids,
                        und_len=und_len,
                        sequence_length=3,
                        vision_tokens=vision_tokens,
                        vision_token_shapes=vision_token_shapes,
                        vision_sequence_indexes=vision_sequence_indexes,
                        vision_mse_loss_indexes=vision_mse_loss_indexes,
                        vision_timesteps=vision_timesteps,
                        vision_noisy_frame_indexes=vision_noisy_frame_indexes,
                        reasoner_memory_state=Cosmos3ReasonerMemoryState(layer_kv=layer_kv),
                        return_dict=False,
                    )[0][0]

                self.assertTrue(torch.allclose(joint, split, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()

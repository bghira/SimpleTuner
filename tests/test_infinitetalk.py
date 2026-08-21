import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file


class InfiniteTalkAudioTests(unittest.TestCase):
    def test_waveform_alignment_matches_video_cache_duration(self):
        from simpletuner.helpers.models.infinitetalk.audio import align_waveform_to_video_frames

        long_waveform = torch.arange(20, dtype=torch.float32).unsqueeze(0)
        short_waveform = torch.arange(4, dtype=torch.float32).unsqueeze(0)

        trimmed = align_waveform_to_video_frames(long_waveform, sample_rate=10, num_frames=25)
        padded = align_waveform_to_video_frames(short_waveform, sample_rate=10, num_frames=25)

        self.assertEqual(trimmed.shape, (1, 10))
        self.assertTrue(torch.equal(trimmed, long_waveform[:, :10]))
        self.assertEqual(padded.shape, (1, 10))
        self.assertTrue(torch.equal(padded[:, :4], short_waveform))
        self.assertEqual(torch.count_nonzero(padded[:, 4:]).item(), 0)

    def test_window_audio_embeddings_matches_official_geometry(self):
        from simpletuner.helpers.models.infinitetalk.audio import window_audio_embeddings

        embeddings = torch.arange(7 * 12 * 4, dtype=torch.float32).reshape(1, 7, 12, 4)
        windowed = window_audio_embeddings(embeddings, window_size=5)

        self.assertEqual(windowed.shape, (1, 7, 5, 12, 4))
        self.assertTrue(torch.equal(windowed[:, 0, 0], embeddings[:, 0]))
        self.assertTrue(torch.equal(windowed[:, 0, 2], embeddings[:, 0]))
        self.assertTrue(torch.equal(windowed[:, 0, 4], embeddings[:, 2]))
        self.assertTrue(torch.equal(windowed[:, -1, 0], embeddings[:, -3]))
        self.assertTrue(torch.equal(windowed[:, -1, -1], embeddings[:, -1]))

    def test_audio_projector_produces_one_context_per_latent_frame(self):
        from simpletuner.helpers.models.infinitetalk.transformer import InfiniteTalkAudioProjector

        projector = InfiniteTalkAudioProjector(
            audio_window=5,
            vae_scale=4,
            audio_layers=2,
            audio_dim=3,
            intermediate_dim=8,
            output_dim=6,
            context_tokens=4,
        )
        audio = torch.randn(2, 9, 5, 2, 3)
        result = projector(audio)

        self.assertEqual(result.shape, (2 * 3, 4, 6))


class InfiniteTalkTransformerTests(unittest.TestCase):
    def test_audio_attention_is_frame_local(self):
        from simpletuner.helpers.models.infinitetalk.transformer import InfiniteTalkAudioAttention

        attention = InfiniteTalkAudioAttention(dim=8, encoder_hidden_states_dim=6, num_heads=2)
        hidden_states = torch.randn(2, 12, 8)
        audio_hidden_states = torch.randn(2 * 3, 4, 6)

        output = attention(hidden_states, audio_hidden_states, num_frames=3)

        self.assertEqual(output.shape, hidden_states.shape)

    def test_audio_attention_requires_matching_frame_count(self):
        from simpletuner.helpers.models.infinitetalk.transformer import InfiniteTalkAudioAttention

        attention = InfiniteTalkAudioAttention(dim=8, encoder_hidden_states_dim=6, num_heads=2)
        with self.assertRaisesRegex(ValueError, "audio frame count"):
            attention(torch.randn(1, 12, 8), torch.randn(2, 4, 6), num_frames=3)

    def test_transformer_forwards_audio_through_checkpointing(self):
        from simpletuner.helpers.models.infinitetalk.transformer import InfiniteTalkTransformer3DModel

        model = InfiniteTalkTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=4,
            in_channels=4,
            out_channels=4,
            text_dim=6,
            freq_dim=8,
            ffn_dim=16,
            num_layers=2,
            cross_attn_norm=True,
            image_dim=None,
            audio_layers=2,
            audio_dim=3,
            audio_output_dim=6,
            audio_intermediate_dim=8,
            audio_context_tokens=4,
        ).train()
        model.enable_gradient_checkpointing()

        with model.cache_context("cond"):
            pass

        output = model(
            hidden_states=torch.randn(1, 4, 3, 4, 4),
            timestep=torch.tensor([500]),
            encoder_hidden_states=torch.randn(1, 5, 6),
            audio_hidden_states=torch.randn(1, 9, 5, 2, 3),
            return_dict=False,
        )[0]

        self.assertEqual(output.shape, (1, 4, 3, 4, 4))

    def test_from_pretrained_materializes_audio_delta_before_return(self):
        from simpletuner.helpers.models.infinitetalk.transformer import InfiniteTalkTransformer3DModel
        from simpletuner.helpers.models.wan.transformer import WanTransformer3DModel

        config = dict(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=4,
            in_channels=4,
            out_channels=4,
            text_dim=6,
            freq_dim=8,
            ffn_dim=16,
            num_layers=1,
            cross_attn_norm=True,
            image_dim=None,
        )
        audio_config = dict(
            audio_layers=2,
            audio_dim=3,
            audio_output_dim=6,
            audio_intermediate_dim=8,
            audio_context_tokens=4,
        )
        with tempfile.TemporaryDirectory() as tempdir:
            WanTransformer3DModel(**config).save_pretrained(tempdir)
            template = InfiniteTalkTransformer3DModel(**config, **audio_config)
            delta = {
                name: torch.ones_like(value, device="cpu")
                for name, value in template.state_dict().items()
                if name in template._expected_delta_keys()
            }
            delta_path = str(Path(tempdir) / "delta.safetensors")
            save_file(delta, delta_path)

            loaded = InfiniteTalkTransformer3DModel.from_pretrained(
                tempdir,
                infinitetalk_delta_path=delta_path,
                **audio_config,
            )

        self.assertFalse(any(parameter.device.type == "meta" for parameter in loaded.parameters()))
        self.assertTrue(torch.equal(loaded.audio_proj.proj1.weight, torch.ones_like(loaded.audio_proj.proj1.weight)))


class InfiniteTalkModelTests(unittest.TestCase):
    def test_model_metadata_registration(self):
        metadata_path = Path(__file__).parents[1] / "simpletuner/helpers/models/model_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(metadata["infinitetalk"]["class_name"], "InfiniteTalk")
        self.assertEqual(metadata["infinitetalk"]["flavour_choices"], ["single-14b-480p"])

    def test_model_requires_audio_pairs(self):
        from simpletuner.helpers.models.infinitetalk.model import InfiniteTalk

        model = InfiniteTalk.__new__(InfiniteTalk)
        model.config = SimpleNamespace(model_flavour="single-14b-480p")
        self.assertTrue(model.requires_s2v_datasets())
        self.assertTrue(model.supports_audio_inputs())
        self.assertTrue(model.requires_conditioning_validation_inputs())

    def test_model_preserves_wan_text_pipeline(self):
        from simpletuner.helpers.models.common import PipelineTypes
        from simpletuner.helpers.models.infinitetalk.model import InfiniteTalk
        from simpletuner.helpers.models.wan.model import Wan

        self.assertIs(InfiniteTalk.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG], Wan.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG])

    def test_validation_conditioning_preserves_wan_pipeline_defaults(self):
        from simpletuner.helpers.models.infinitetalk.model import InfiniteTalk

        model = InfiniteTalk.__new__(InfiniteTalk)
        model.config = SimpleNamespace(model_flavour="single-14b-480p", validation_num_video_frames=49)
        conditioning = {"audio_path": "speech.wav", "image": "first-frame"}

        result = model.update_pipeline_call_kwargs(
            {
                "_s2v_conditioning": conditioning,
                "_validation_prompt_text": "prompt bookkeeping",
                "_validation_negative_prompt_text": "negative bookkeeping",
                "num_images_per_prompt": 2,
            }
        )

        self.assertEqual(result["audio"], "speech.wav")
        self.assertEqual(result["image"], "first-frame")
        self.assertEqual(result["num_frames"], 49)
        self.assertEqual(result["output_type"], "pil")
        self.assertNotIn("_validation_prompt_text", result)
        self.assertNotIn("_validation_negative_prompt_text", result)
        self.assertNotIn("num_images_per_prompt", result)
        self.assertEqual(result["num_videos_per_prompt"], 2)

    def test_config_accepts_cli_string_framerate(self):
        from simpletuner.helpers.models.infinitetalk.model import InfiniteTalk

        model = InfiniteTalk.__new__(InfiniteTalk)
        model.config = SimpleNamespace(
            aspect_bucket_alignment=32,
            base_model_precision="no_change",
            context_parallel_size=1,
            framerate="25",
            model_flavour="single-14b-480p",
            prediction_type=None,
            tokenizer_max_length=None,
            tread_config=None,
            validation_disable_unconditional=True,
            validation_num_inference_steps=40,
        )

        model.check_user_config()

        self.assertEqual(model.config.framerate, 25)

    def test_delta_key_set_must_be_exact(self):
        from simpletuner.helpers.models.infinitetalk.transformer import InfiniteTalkTransformer3DModel

        model = InfiniteTalkTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=4,
            in_channels=4,
            out_channels=4,
            text_dim=6,
            freq_dim=8,
            ffn_dim=16,
            num_layers=1,
            image_dim=None,
            audio_layers=2,
            audio_dim=3,
            audio_output_dim=6,
            audio_intermediate_dim=8,
            audio_context_tokens=4,
        )
        with self.assertRaisesRegex(ValueError, "missing required tensors"):
            with mock.patch("simpletuner.helpers.models.infinitetalk.transformer.safe_open") as opened:
                opened.return_value.__enter__.return_value.keys.return_value = ["audio_proj.proj1.weight"]
                model.load_audio_conditioning_weights("incomplete.safetensors")


if __name__ == "__main__":
    unittest.main()

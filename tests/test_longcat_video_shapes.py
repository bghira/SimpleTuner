import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, UMT5Config, UMT5EncoderModel

from simpletuner.helpers.caching.text_embeds import TextEmbeddingCache
from simpletuner.helpers.models.common import ModelFoundation
from simpletuner.helpers.models.longcat_video.model import LongCatVideo
from simpletuner.helpers.models.longcat_video.pipeline import LongCatVideoPipeline
from simpletuner.helpers.models.longcat_video.transformer import LongCatVideoTransformer3DModel


class TestLongCatVideoTextConditioning(unittest.TestCase):
    def test_uses_checkpoint_umt5_for_prompts_and_cfg(self):
        component = LongCatVideo.TEXT_ENCODER_CONFIGURATION["text_encoder"]
        self.assertIs(component["model"], UMT5EncoderModel)
        self.assertEqual(component["subfolder"], "text_encoder")
        self.assertNotIn("path", component)
        encoder = UMT5EncoderModel(
            UMT5Config(vocab_size=8, d_model=16, d_ff=32, d_kv=4, num_heads=4, num_layers=1, dropout_rate=0.0)
        ).eval()
        backend = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "cat": 2, "&": 3, "dog": 4}, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]")
        pipe = LongCatVideoPipeline(transformer=None, vae=None, scheduler=None, text_encoder=encoder, tokenizer=tokenizer)

        positive, mask, negative, neg_mask = pipe.encode_prompt(
            [" cat  &amp;   dog ", "dog"],
            torch.device("cpu"),
            num_videos_per_prompt=2,
            max_sequence_length=5,
            do_classifier_free_guidance=True,
            negative_prompt="cat",
        )
        for texts, embeds, masks in ((["cat & dog", "dog"], positive, mask), (["cat", "cat"], negative, neg_mask)):
            tokens = tokenizer(texts, max_length=5, padding="max_length", truncation=True, return_tensors="pt")
            expected = encoder(**{key: tokens[key] for key in ("input_ids", "attention_mask")}).last_hidden_state
            torch.testing.assert_close(embeds[:, 0], expected.repeat_interleave(2, dim=0))
            torch.testing.assert_close(masks[:, 0, 0], tokens.attention_mask.repeat_interleave(2, dim=0))

    def test_caption_and_dropout_cache_keys_change_for_native_encoder(self):
        model = LongCatVideo.__new__(LongCatVideo)
        cache = TextEmbeddingCache.__new__(TextEmbeddingCache)
        cache.model = model
        cache.key_type = model.text_embed_cache_key()
        cache.model_type = "longcat_video"
        cache._requires_path_based_keys = False
        for caption in ("cat", ""):
            with patch.object(model, "text_embed_cache_namespace", return_value=None):
                legacy_key = cache.create_hash(caption)
            self.assertNotEqual(legacy_key, cache.create_hash(caption))


class TestLongCatVideoTrainingLatents(unittest.TestCase):
    def test_normalizes_existing_cached_latents_before_noise_sampling(self):
        model = LongCatVideo.__new__(LongCatVideo)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.vae = SimpleNamespace(
            config=SimpleNamespace(z_dim=2, latents_mean=[-1.0, 2.0], latents_std=[2.0, 4.0]),
        )
        raw = torch.tensor([3.0, 5.0, 6.0, 10.0]).view(1, 2, 2, 1, 1)
        cached = model.scale_vae_latents_for_cache(raw, model.vae)
        model.post_vae_load_setup()
        model.vae = None
        batch = {"latent_batch": cached, "conditioning_latent_count": 1}

        with patch.object(ModelFoundation, "prepare_batch", autospec=True, side_effect=lambda model, batch, state: batch):
            prepared = model.prepare_batch(batch, {})

        torch.testing.assert_close(prepared["latent_batch"], torch.tensor([2.0, 3.0, 1.0, 2.0]).view_as(raw))
        torch.testing.assert_close(batch["latent_batch"], raw)
        self.assertEqual(prepared["conditioning_latent_count"], 1)


class TestLongCatVideoTransformerShapes(unittest.TestCase):
    def test_forward_preserves_5d_shape(self):
        transformer = LongCatVideoTransformer3DModel(
            in_channels=16,
            out_channels=16,
            hidden_size=64,
            depth=1,
            num_heads=8,
            caption_channels=64,
            mlp_ratio=2,
            adaln_tembed_dim=32,
            frequency_embedding_size=16,
            patch_size=(1, 2, 2),
            enable_flashattn3=False,
            enable_flashattn2=False,
            enable_xformers=False,
            enable_bsa=False,
            cp_split_hw=None,
            text_tokens_zero_pad=False,
        )
        transformer.gradient_checkpointing = False

        latents = torch.randn(1, 16, 1, 4, 4)
        encoder_hidden_states = torch.randn(1, 4, 64)
        encoder_attention_mask = torch.ones(1, 4)
        timestep = torch.randint(0, 1000, (1,))

        with torch.no_grad():
            out = transformer(
                latents,
                timestep,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

        self.assertEqual(out.shape, latents.shape)

    def test_forward_handles_packed_text_batch(self):
        transformer = LongCatVideoTransformer3DModel(
            in_channels=16,
            out_channels=16,
            hidden_size=64,
            depth=1,
            num_heads=8,
            caption_channels=64,
            mlp_ratio=2,
            adaln_tembed_dim=32,
            frequency_embedding_size=16,
            patch_size=(1, 2, 2),
            enable_flashattn3=False,
            enable_flashattn2=False,
            enable_xformers=False,
            enable_bsa=False,
            cp_split_hw=None,
            text_tokens_zero_pad=False,
        )
        transformer.gradient_checkpointing = False

        latents = torch.randn(2, 16, 1, 4, 4)
        encoder_hidden_states = torch.randn(2, 4, 64)
        encoder_attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.float32)
        timestep = torch.randint(0, 1000, (2,))

        with torch.no_grad():
            out = transformer(
                latents,
                timestep,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

        self.assertEqual(out.shape, latents.shape)

    def test_zero_gates_preserve_residual_and_input_gradient(self):
        from simpletuner.helpers.models.longcat_video.transformer import LongCatSingleStreamBlock

        block = LongCatSingleStreamBlock(hidden_size=32, num_heads=4, mlp_ratio=2, adaln_tembed_dim=16)
        with torch.no_grad():
            block.adaLN_modulation[1].weight.zero_()
            block.adaLN_modulation[1].bias.zero_()
        x = (5.0 + torch.randn(1, 6, 32)).requires_grad_()
        result = block(x, None, torch.randn(1, 2, 16), None, (2, 1, 3), skip_crs_attn=True)
        torch.testing.assert_close(result, x, rtol=0, atol=0)
        grad = torch.randn_like(x)
        torch.testing.assert_close(torch.autograd.grad(result, x, grad)[0], grad, rtol=0, atol=0)

    def test_forward_accepts_framewise_timesteps(self):
        transformer = LongCatVideoTransformer3DModel(
            in_channels=16,
            out_channels=16,
            hidden_size=64,
            depth=1,
            num_heads=8,
            caption_channels=64,
            mlp_ratio=2,
            adaln_tembed_dim=32,
            frequency_embedding_size=16,
            patch_size=(1, 2, 2),
            enable_flashattn3=False,
            enable_flashattn2=False,
            enable_xformers=False,
            enable_bsa=False,
            cp_split_hw=None,
            text_tokens_zero_pad=False,
        )
        transformer.gradient_checkpointing = False

        latents = torch.randn(1, 16, 2, 4, 4)
        encoder_hidden_states = torch.randn(1, 4, 64)
        timestep = torch.tensor([[100.0, 900.0]], dtype=torch.float32)

        with torch.no_grad():
            out = transformer(
                latents,
                timestep,
                encoder_hidden_states,
                return_dict=False,
            )[0]

        self.assertEqual(out.shape, latents.shape)

    def test_forward_accepts_tokenwise_timesteps(self):
        transformer = LongCatVideoTransformer3DModel(
            in_channels=16,
            out_channels=16,
            hidden_size=64,
            depth=1,
            num_heads=8,
            caption_channels=64,
            mlp_ratio=2,
            adaln_tembed_dim=32,
            frequency_embedding_size=16,
            patch_size=(1, 2, 2),
            enable_flashattn3=False,
            enable_flashattn2=False,
            enable_xformers=False,
            enable_bsa=False,
            cp_split_hw=None,
            text_tokens_zero_pad=False,
        )
        transformer.gradient_checkpointing = False

        latents = torch.randn(1, 16, 2, 4, 4)
        encoder_hidden_states = torch.randn(1, 4, 64)
        timestep = torch.tensor([[100.0, 900.0, 100.0, 900.0, 900.0, 100.0, 900.0, 100.0]], dtype=torch.float32)

        with torch.no_grad():
            out = transformer(
                latents,
                timestep,
                encoder_hidden_states,
                return_dict=False,
            )[0]

        self.assertEqual(out.shape, latents.shape)

    def test_forward_rejects_wrong_framewise_timestep_length(self):
        transformer = LongCatVideoTransformer3DModel(
            in_channels=16,
            out_channels=16,
            hidden_size=64,
            depth=1,
            num_heads=8,
            caption_channels=64,
            mlp_ratio=2,
            adaln_tembed_dim=32,
            frequency_embedding_size=16,
            patch_size=(1, 2, 2),
            enable_flashattn3=False,
            enable_flashattn2=False,
            enable_xformers=False,
            enable_bsa=False,
            cp_split_hw=None,
            text_tokens_zero_pad=False,
        )
        transformer.gradient_checkpointing = False

        latents = torch.randn(1, 16, 2, 4, 4)
        encoder_hidden_states = torch.randn(1, 4, 64)
        timestep = torch.tensor([[100.0, 900.0, 100.0]], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "expected framewise or tokenwise timesteps with sequence length 2 or 8"):
            transformer(
                latents,
                timestep,
                encoder_hidden_states,
                return_dict=False,
            )

    def test_forward_rejects_wrong_tokenwise_timestep_length(self):
        transformer = LongCatVideoTransformer3DModel(
            in_channels=16,
            out_channels=16,
            hidden_size=64,
            depth=1,
            num_heads=8,
            caption_channels=64,
            mlp_ratio=2,
            adaln_tembed_dim=32,
            frequency_embedding_size=16,
            patch_size=(1, 2, 2),
            enable_flashattn3=False,
            enable_flashattn2=False,
            enable_xformers=False,
            enable_bsa=False,
            cp_split_hw=None,
            text_tokens_zero_pad=False,
        )
        transformer.gradient_checkpointing = False

        latents = torch.randn(1, 16, 2, 4, 4)
        encoder_hidden_states = torch.randn(1, 4, 64)
        timestep = torch.tensor([[100.0, 900.0, 100.0]], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "expected framewise or tokenwise timesteps with sequence length 2 or 8"):
            transformer(
                latents,
                timestep,
                encoder_hidden_states,
                return_dict=False,
            )


if __name__ == "__main__":
    unittest.main()

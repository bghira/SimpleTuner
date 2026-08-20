import unittest
from types import SimpleNamespace

import torch

from simpletuner.helpers.models.minimaxmusic import reference_adapter as adapter


class MiniMaxMusic3ReferenceAdapterTests(unittest.TestCase):
    def test_frame_latent_starts_uses_dav_rate(self):
        starts = adapter.frame_latent_starts(128)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 441)
        self.assertTrue(all(right > left for left, right in zip(starts, starts[1:])))

    def test_pool_matrix_averages_each_frame_span(self):
        pool = adapter.build_pool_matrix([10, 13, 17])
        self.assertEqual(tuple(pool.shape), (2, 7))
        torch.testing.assert_close(pool.sum(dim=1), torch.ones(2))
        self.assertEqual(torch.count_nonzero(pool[0]).item(), 3)
        self.assertEqual(torch.count_nonzero(pool[1]).item(), 4)

    def test_config_rejects_unknown_fields(self):
        with self.assertRaisesRegex(ValueError, "Unknown RVQ encoder configuration fields"):
            adapter.RVQEncoderConfig.from_dict({"not_a_field": 1})

    def test_replay_requires_five_semantic_candidates_per_frame(self):
        with self.assertRaisesRegex(ValueError, "semantic_candidates must have shape"):
            adapter.replay_codes_diffusers(
                None,
                torch.zeros((1, 8), dtype=torch.long),
                torch.zeros((1, 4), dtype=torch.long),
                prompt="instrumental",
                lyrics="[instrumental]",
            )

    def test_replay_rejects_reference_interval_outside_supported_range(self):
        with self.assertRaisesRegex(ValueError, "reference_interval must be between 1 and 10"):
            adapter.replay_codes_diffusers(
                None,
                torch.zeros((1, 8), dtype=torch.long),
                torch.zeros((1, 5), dtype=torch.long),
                prompt="instrumental",
                lyrics="[instrumental]",
                reference_interval=11,
            )

    def test_sampled_depth_codes_preserve_semantic_code(self):
        class SemanticEmbedding(torch.nn.Module):
            def forward(self, indices):
                return torch.nn.functional.one_hot(indices.remainder(4), num_classes=4).float()

        class DepthDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Identity()
                self.audio_embeddings = torch.nn.Embedding(7 * 1024, 4)
                self.audio_heads = torch.nn.ModuleList(torch.nn.Linear(4, 1024) for _ in range(7))

            def forward(self, hidden_states):
                return hidden_states

        language_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=SemanticEmbedding()))
        semantic_codes = torch.tensor([7, 7])
        codes, hidden = adapter._sample_official_depth_codes(
            language_model,
            DepthDecoder(),
            torch.randn(2, 4),
            semantic_codes,
            torch.Generator(device="cpu").manual_seed(1),
        )
        self.assertEqual(tuple(codes.shape), (2, 8))
        self.assertEqual(tuple(hidden.shape), (1, 28))
        torch.testing.assert_close(codes[:, 0], semantic_codes)

    def test_independent_encoder_shapes(self):
        config = adapter.RVQEncoderConfig(
            latent_channels=4,
            codebook_vocab_sizes=(11, 7, 7),
            d_model=8,
            num_layers=1,
            num_heads=2,
            ff_mult=2,
            dropout=0.0,
            max_position_embeddings=4,
            conv_dilations=(1,),
            mup=True,
            mup_attention_multiplier=2.0,
        )
        model = adapter.MiniMaxMusicRVQEncoder(config).eval()
        pool = adapter.build_pool_matrix([0, 2, 5, 7, 10]).unsqueeze(0)
        with torch.inference_mode():
            logits = model(torch.randn(1, 10, 4), pool)
        self.assertEqual([tuple(value.shape) for value in logits], [(1, 4, 11), (1, 4, 7), (1, 4, 7)])

    def test_autoregressive_depth_encoder_shapes(self):
        config = adapter.RVQEncoderConfig(
            latent_channels=4,
            codebook_vocab_sizes=(11, 7, 7),
            d_model=8,
            num_layers=1,
            num_heads=2,
            ff_mult=2,
            dropout=0.0,
            max_position_embeddings=4,
            conv_dilations=(1,),
            mup=True,
            mup_attention_multiplier=2.0,
            depth_decoder=True,
            depth_decoder_dim=8,
            depth_decoder_layers=1,
            depth_decoder_heads=2,
            depth_decoder_ff_mult=2,
            depth_decoder_dropout=0.0,
        )
        model = adapter.MiniMaxMusicRVQEncoder(config).eval()
        pool = adapter.build_pool_matrix([0, 2, 5, 7, 10]).unsqueeze(0)
        with torch.inference_mode():
            logits = model(torch.randn(1, 10, 4), pool)
        self.assertEqual([tuple(value.shape) for value in logits], [(1, 4, 11), (1, 4, 7), (1, 4, 7)])

    def test_predict_codes_returns_semantic_candidates(self):
        class FakeDAV(torch.nn.Module):
            def forward(self, waveform):
                return torch.zeros((waveform.shape[0], 4, 100), device=waveform.device)

        config = adapter.RVQEncoderConfig(
            latent_channels=4,
            codebook_vocab_sizes=(11, 7, 7, 7, 7, 7, 7, 7),
            d_model=8,
            num_layers=1,
            num_heads=2,
            ff_mult=2,
            dropout=0.0,
            max_position_embeddings=4,
            conv_dilations=(1,),
            mup=True,
            mup_attention_multiplier=2.0,
        )
        reference_adapter = adapter.MiniMaxMusic3ReferenceAdapter(
            FakeDAV(),
            adapter.MiniMaxMusicRVQEncoder(config),
        )
        waveform = torch.zeros((1, adapter.SAMPLE_RATE))
        codes, candidates = reference_adapter.predict_codes_with_semantic_candidates(
            waveform,
            adapter.SAMPLE_RATE,
            semantic_top_k=3,
            device="cpu",
        )
        self.assertEqual(tuple(codes.shape), (adapter.FRAME_RATE, 8))
        self.assertEqual(tuple(candidates.shape), (adapter.FRAME_RATE, 3))
        torch.testing.assert_close(codes[:, 0], candidates[:, 0])
        with self.assertRaisesRegex(ValueError, "semantic_top_k must be between"):
            reference_adapter.predict_codes_with_semantic_candidates(
                waveform,
                adapter.SAMPLE_RATE,
                semantic_top_k=0,
                device="cpu",
            )

    def test_predict_codes_accepts_precompute_offload_control(self):
        class FakeDAV(torch.nn.Module):
            def forward(self, waveform):
                return torch.zeros((waveform.shape[0], 4, 100), device=waveform.device)

        config = adapter.RVQEncoderConfig(
            latent_channels=4,
            codebook_vocab_sizes=(11, 7, 7, 7, 7, 7, 7, 7),
            d_model=8,
            num_layers=1,
            num_heads=2,
            ff_mult=2,
            dropout=0.0,
            max_position_embeddings=4,
            conv_dilations=(1,),
            mup=True,
            mup_attention_multiplier=2.0,
        )
        reference_adapter = adapter.MiniMaxMusic3ReferenceAdapter(
            FakeDAV(),
            adapter.MiniMaxMusicRVQEncoder(config),
        )
        codes = reference_adapter.predict_codes(
            torch.zeros((1, adapter.SAMPLE_RATE)),
            adapter.SAMPLE_RATE,
            device="cpu",
            offload_after=False,
        )
        self.assertEqual(tuple(codes.shape), (adapter.FRAME_RATE, 8))


if __name__ == "__main__":
    unittest.main()

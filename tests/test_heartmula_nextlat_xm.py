import unittest

import torch

from simpletuner.helpers.models.heartmula import modeling_heartmula
from simpletuner.helpers.models.heartmula.configuration_heartmula import HeartMuLaConfig
from simpletuner.helpers.models.heartmula.model import HeartMuLa
from simpletuner.helpers.models.heartmula.modeling_heartmula import HeartMuLaModel
from simpletuner.helpers.training.explorative_modeling import ExplorativeModelingConfig
from simpletuner.helpers.utils.hidden_state_buffer import HiddenStateBuffer


class HeartMuLaNextLatXmTests(unittest.TestCase):
    def setUp(self):
        self._original_flavors = dict(modeling_heartmula._LLAMA_FLAVORS)
        modeling_heartmula._LLAMA_FLAVORS["llama-test"] = {
            "num_layers": 2,
            "num_heads": 2,
            "num_kv_heads": 2,
            "embed_dim": 8,
            "max_seq_len": 128,
            "intermediate_dim": 16,
        }

    def tearDown(self):
        modeling_heartmula._LLAMA_FLAVORS.clear()
        modeling_heartmula._LLAMA_FLAVORS.update(self._original_flavors)

    def _tiny_model(self):
        config = HeartMuLaConfig(
            backbone_flavor="llama-test",
            decoder_flavor="llama-test",
            text_vocab_size=16,
            audio_vocab_size=5,
            audio_num_codebooks=2,
        )
        return HeartMuLaModel(config)

    def test_forward_captures_backbone_layer_hidden_states(self):
        model = self._tiny_model()
        tokens = torch.zeros(1, 4, 3, dtype=torch.long)
        tokens_mask = torch.ones_like(tokens, dtype=torch.bool)
        buffer = HiddenStateBuffer(capture_layers={1})

        model(tokens=tokens, tokens_mask=tokens_mask, hidden_states_buffer=buffer)

        self.assertEqual(tuple(buffer["layer_1"].shape), (1, 4, 8))

    def test_forward_applies_configured_xm_route_embeddings(self):
        model = self._tiny_model()
        model.configure_xm_route_embeddings(candidate_count=2)
        tokens = torch.zeros(2, 4, 3, dtype=torch.long)
        tokens_mask = torch.ones_like(tokens, dtype=torch.bool)
        route_mask = torch.ones(2, 4, dtype=torch.bool)

        output = model(
            tokens=tokens,
            tokens_mask=tokens_mask,
            route_candidate_ids=torch.tensor([0, 1]),
            route_mask=route_mask,
        )

        self.assertEqual(tuple(output["codebook0_logits"].shape[:2]), (2, 3))
        self.assertTrue(model.xm_route_embeddings.weight.requires_grad)

    def test_xm_route_loss_selects_one_candidate_for_full_sample(self):
        heartmula = HeartMuLa.__new__(HeartMuLa)
        heartmula.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=2,
            training_target="route",
            selection_scope="sample",
            block_size=0,
        )

        tokens = torch.zeros(1, 5, 3, dtype=torch.long)
        tokens[:, 1:, 0] = 0
        tokens[:, 1:, 1] = 1
        prepared_batch = {
            "tokens": tokens,
            "audio_frame_mask": torch.ones(1, 5, dtype=torch.bool),
        }

        logits0 = torch.full((2, 4, 5), -5.0)
        logits_rest = torch.full((2, 4, 1, 5), -5.0)
        logits0[0, :2, 0] = 5.0
        logits_rest[0, :2, 0, 1] = 5.0
        logits0[0, 2:, 2] = 5.0
        logits_rest[0, 2:, 0, 2] = 5.0
        logits0[1, :, 0] = 4.0
        logits_rest[1, :, 0, 1] = 4.0
        model_output = {
            "codebook0_logits": logits0,
            "codebook_logits": logits_rest,
            "hidden_states": torch.randn(2, 5, 8),
            "hidden_states_buffer": {"layer_1": torch.randn(2, 5, 8)},
            "xm_candidate_count": 2,
        }

        loss = HeartMuLa.loss(heartmula, prepared_batch, model_output)

        self.assertLess(loss.item(), 0.01)
        self.assertTrue(torch.equal(model_output["xm_winner_indices"], torch.tensor([1])))
        self.assertEqual(tuple(model_output["hidden_states"].shape), (1, 5, 8))
        self.assertEqual(tuple(model_output["hidden_states_buffer"]["layer_1"].shape), (1, 5, 8))


if __name__ == "__main__":
    unittest.main()

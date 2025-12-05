import math
import unittest

import torch

from simpletuner.helpers.training import qk_clip_logging
from simpletuner.helpers.training.attention_backend import AttentionBackendController
from simpletuner.helpers.training.optimizers.muon import MuonClip
from simpletuner.helpers.training.trainer import Trainer


def _reset_attention_backend_state() -> None:
    # Best-effort cleanup to avoid leaking state across tests.
    AttentionBackendController._attention_logit_consumer = None
    AttentionBackendController._optimizer = None
    AttentionBackendController._param_to_name = {} if hasattr(AttentionBackendController, "_param_to_name") else {}


class MuonClipQKClipTests(unittest.TestCase):
    def setUp(self):
        _reset_attention_backend_state()

    def tearDown(self):
        _reset_attention_backend_state()

    def test_publish_attention_max_logits_invokes_consumer(self):
        q_param = torch.nn.Parameter(torch.randn(4, 4))
        k_param = torch.nn.Parameter(torch.randn(4, 4))
        opt = MuonClip([q_param, k_param], lr=1e-3)
        opt.register_attention_params({"layer.to_q": q_param, "layer.to_k": k_param})
        AttentionBackendController.bind_optimizer(opt)

        received = {}

        def _consumer(payload):
            received.update(payload)

        AttentionBackendController.register_attention_logit_consumer(_consumer)

        query = torch.randn(1, 2, 3, 4)  # [batch, heads, seq, dim]
        key = torch.randn(1, 2, 3, 4)
        qk_clip_logging.publish_attention_max_logits(query, key, None, q_param, k_param)

        self.assertIn("layer.to_q", received)
        self.assertIn("layer.to_k", received)
        self.assertTrue(torch.equal(received["layer.to_q"], received["layer.to_k"]))
        self.assertEqual(received["layer.to_q"].shape[0], query.shape[1])

    def test_state_dict_preserves_param_names(self):
        p1 = torch.nn.Parameter(torch.randn(2, 2))
        p2 = torch.nn.Parameter(torch.randn(2, 2))
        opt = MuonClip([p1, p2], lr=1e-3)
        opt.register_attention_params({"a.to_q": p1, "b.to_k": p2})

        state = opt.state_dict()

        new_opt = MuonClip(
            [torch.nn.Parameter(p1.clone().detach()), torch.nn.Parameter(p2.clone().detach())],
            lr=1e-3,
        )
        new_opt.load_state_dict(state)

        self.assertEqual(new_opt._param_to_name.get(id(new_opt.param_groups[0]["params"][0])), "a.to_q")
        self.assertEqual(new_opt._param_to_name.get(id(new_opt.param_groups[0]["params"][1])), "b.to_k")

    def test_record_attention_max_logits_accumulates_max(self):
        trainer = Trainer.__new__(Trainer)
        trainer._attention_max_logits = None

        logits_a = torch.tensor([1.0, 2.0])
        logits_b = torch.tensor([2.5, 1.5])

        Trainer.record_attention_max_logits(trainer, {"param": logits_a})
        Trainer.record_attention_max_logits(trainer, {"param": logits_b})

        stored = trainer._attention_max_logits["param"]
        self.assertTrue(torch.equal(stored, torch.tensor([2.5, 2.0])))

    def test_chunked_max_logits_matches_full(self):
        torch.manual_seed(0)
        q_param = torch.nn.Parameter(torch.randn(4, 4))
        k_param = torch.nn.Parameter(torch.randn(4, 4))
        opt = MuonClip([q_param, k_param], lr=1e-3)
        opt.register_attention_params({"layer.to_q": q_param, "layer.to_k": k_param})
        AttentionBackendController.bind_optimizer(opt)

        received = {}
        AttentionBackendController.register_attention_logit_consumer(received.update)

        query = torch.randn(1, 2, 3, 4)
        key = torch.randn(1, 2, 1100, 4)
        mask = torch.ones(1, 1, 1, 1100, dtype=torch.bool)
        mask[..., :5] = False  # mask out a few tokens

        qk_clip_logging.publish_attention_max_logits(query, key, mask, q_param, k_param)

        expected = torch.matmul(query.float(), key.float().transpose(-2, -1))
        expected.div_(math.sqrt(query.shape[-1]))
        expanded_mask = mask.expand_as(expected)
        expected = expected.masked_fill(~expanded_mask, float("-inf"))
        expected = torch.nan_to_num(expected, nan=float("-inf"), posinf=float("inf"), neginf=float("-inf"))
        expected = expected.amax(dim=(-1, -2)).amax(dim=0)

        self.assertTrue(torch.allclose(received["layer.to_q"], expected))
        self.assertTrue(torch.allclose(received["layer.to_k"], expected))


if __name__ == "__main__":
    unittest.main()

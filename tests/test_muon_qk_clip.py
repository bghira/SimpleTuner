import unittest

import torch

from simpletuner.helpers.training import qk_clip_logging
from simpletuner.helpers.training.attention_backend import AttentionBackendController
from simpletuner.helpers.training.optimizers.muon import MuonClip


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

        new_opt = MuonClip([p1.clone(), p2.clone()], lr=1e-3)
        new_opt.load_state_dict(state)

        self.assertEqual(new_opt._param_to_name.get(id(new_opt.param_groups[0]["params"][0])), "a.to_q")
        self.assertEqual(new_opt._param_to_name.get(id(new_opt.param_groups[0]["params"][1])), "b.to_k")


if __name__ == "__main__":
    unittest.main()

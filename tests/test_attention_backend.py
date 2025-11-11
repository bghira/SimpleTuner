import os
import shutil
import tempfile
import unittest

import torch

from simpletuner.helpers.training import attention_backend as attention_backend_module
from simpletuner.helpers.training.attention_backend import AttentionBackendController, AttentionPhase


class TestAttentionBackendPersistence(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._reset_controller()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        self._reset_controller()

    def _reset_controller(self):
        AttentionBackendController._active_backend = None
        AttentionBackendController._active_phase = AttentionPhase.TRAIN
        AttentionBackendController._sla_cache = {}
        AttentionBackendController._sla_settings = None
        AttentionBackendController._optimizer = None
        AttentionBackendController._parameter_sink = None
        AttentionBackendController._sink_param_ids = set()
        AttentionBackendController._optimizer_param_ids = set()
        AttentionBackendController._sla_state_store = {}
        AttentionBackendController._diffusers_backend_context = None
        AttentionBackendController._diffusers_backend_name = None

    def test_save_checkpoint_no_state(self):
        AttentionBackendController._active_backend = "sla"
        AttentionBackendController.on_save_checkpoint(self.tmpdir, is_main_process=True)
        self.assertFalse(os.path.exists(os.path.join(self.tmpdir, "sla_attention.pt")))

    def test_round_trip_serialization(self):
        sample_state = {"proj_l.weight": torch.ones(1, 1), "proj_l.bias": torch.zeros(1)}
        AttentionBackendController._sla_state_store = {(128, "bf16"): sample_state}
        AttentionBackendController._sla_settings = {
            "topk": 0.2,
            "feature_map": "softmax",
            "blkq": 64,
            "blkk": 64,
        }
        AttentionBackendController._active_backend = "sla"

        AttentionBackendController.on_save_checkpoint(self.tmpdir, is_main_process=True)
        payload_path = os.path.join(self.tmpdir, "sla_attention.pt")
        self.assertTrue(os.path.exists(payload_path))

        AttentionBackendController._sla_state_store = {}
        AttentionBackendController._sla_settings = None
        AttentionBackendController.on_load_checkpoint(self.tmpdir)

        self.assertIn((128, "bf16"), AttentionBackendController._sla_state_store)
        restored = AttentionBackendController._sla_state_store[(128, "bf16")]
        self.assertTrue(torch.equal(restored["proj_l.weight"], sample_state["proj_l.weight"]))
        self.assertTrue(torch.equal(restored["proj_l.bias"], sample_state["proj_l.bias"]))
        self.assertEqual(
            AttentionBackendController._sla_settings,
            {
                "topk": 0.2,
                "feature_map": "softmax",
                "blkq": 64,
                "blkk": 64,
            },
        )

    def test_force_proj_fp32(self):
        class DummyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj_l = torch.nn.Linear(4, 4, dtype=torch.bfloat16)

        module = DummyModule()
        AttentionBackendController._force_proj_fp32(module)
        self.assertEqual(module.proj_l.weight.dtype, torch.float32)
        self.assertEqual(module.proj_l.bias.dtype, torch.float32)

    def test_enable_diffusers_backend_context(self):
        if not attention_backend_module._DIFFUSERS_BACKEND_ALIASES:
            self.skipTest("Diffusers attention backend helpers unavailable in this environment.")

        config = type("Config", (object,), {"attention_mechanism": "native-math"})()
        AttentionBackendController.apply(config, AttentionPhase.TRAIN)
        self.assertEqual(AttentionBackendController._diffusers_backend_name, "native-math")
        AttentionBackendController.restore_default()
        self.assertIsNone(AttentionBackendController._diffusers_backend_name)

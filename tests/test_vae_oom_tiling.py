import copy
import inspect
import pickle
import unittest
import weakref
from unittest.mock import patch

import torch

from simpletuner.helpers.models.vae_tiling import enable_automatic_vae_tiling


class TinyVAE(torch.nn.Module):
    def __init__(self, device="cuda", error=None):
        super().__init__()
        self.device = torch.device(device)
        self.use_tiling = False
        self.error = error
        self.calls = []
        self.tiling_calls = 0
        self.failed_tensor = None

    def enable_tiling(self):
        self.use_tiling = True
        self.tiling_calls += 1

    def encode(self, sample, return_dict=True):
        self.calls.append(("encode", sample, return_dict))
        if self.error:
            raise self.error
        if not self.use_tiling:
            temporary = torch.ones(1)
            self.failed_tensor = weakref.ref(temporary)
            raise torch.cuda.OutOfMemoryError("test CUDA allocation failed")
        return sample * 2 if return_dict else (sample * 2,)

    def decode(self, sample, return_dict=True):
        self.calls.append(("decode", sample, return_dict))
        if self.error:
            raise self.error
        if not self.use_tiling:
            raise torch.cuda.OutOfMemoryError("test CUDA allocation failed")
        return sample / 2 if return_dict else (sample / 2,)


class VAEOOMTilingTests(unittest.TestCase):
    def test_wrappers_preserve_method_signatures(self):
        vae = TinyVAE()
        signatures = {name: inspect.signature(getattr(vae, name)) for name in ["encode", "decode"]}

        enable_automatic_vae_tiling(vae)

        for name, signature in signatures.items():
            self.assertEqual(inspect.signature(getattr(vae, name)), signature)
            self.assertEqual(getattr(vae, name).__name__, name)

    @patch("torch.cuda.empty_cache")
    def test_encode_retry_preserves_arguments_and_enables_decode(self, empty_cache):
        vae = enable_automatic_vae_tiling(TinyVAE())
        sample = torch.ones(2, requires_grad=True)

        result = vae.encode(sample, return_dict=False)
        result[0].sum().backward()

        torch.testing.assert_close(result[0], sample * 2)
        torch.testing.assert_close(sample.grad, torch.full_like(sample, 2))
        self.assertEqual(len(vae.calls), 2)
        self.assertTrue(all(call[1] is sample and call[2] is False for call in vae.calls))
        self.assertEqual(vae.tiling_calls, 1)
        self.assertIsNone(vae.failed_tensor())
        empty_cache.assert_called_once()
        torch.testing.assert_close(vae.decode(result[0]), sample)
        self.assertEqual(vae.tiling_calls, 1)

    @patch("torch.cuda.empty_cache")
    def test_decode_triggers_retry(self, empty_cache):
        vae = enable_automatic_vae_tiling(TinyVAE())
        sample = torch.ones(2)
        torch.testing.assert_close(vae.decode(sample, return_dict=False)[0], sample / 2)
        self.assertEqual(len(vae.calls), 2)
        self.assertEqual(vae.tiling_calls, 1)
        empty_cache.assert_called_once()

    def test_non_cuda_oom_is_not_retried(self):
        vae = enable_automatic_vae_tiling(TinyVAE(device="cpu"))
        with self.assertRaises(torch.OutOfMemoryError):
            vae.encode(torch.ones(1))
        self.assertEqual(len(vae.calls), 1)
        self.assertEqual(vae.tiling_calls, 0)

    def test_other_errors_are_not_retried(self):
        error = RuntimeError("encoder shape mismatch")
        vae = enable_automatic_vae_tiling(TinyVAE(error=error))
        with self.assertRaises(RuntimeError) as caught:
            vae.encode(torch.ones(1))
        self.assertIs(caught.exception, error)
        self.assertEqual(vae.tiling_calls, 0)

    @patch("torch.cuda.empty_cache")
    def test_repeated_oom_stops_after_one_retry(self, empty_cache):
        error = torch.cuda.OutOfMemoryError("still out of memory")
        vae = enable_automatic_vae_tiling(TinyVAE(error=error))
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            vae.encode(torch.ones(1))
        self.assertEqual(len(vae.calls), 2)
        self.assertEqual(vae.tiling_calls, 1)
        empty_cache.assert_called_once()
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            vae.decode(torch.ones(1))
        self.assertEqual(len(vae.calls), 3)
        self.assertEqual(vae.tiling_calls, 1)

    def test_already_tiled_oom_is_not_retried(self):
        vae = TinyVAE(error=torch.cuda.OutOfMemoryError("tiled OOM"))
        vae.enable_tiling()
        enable_automatic_vae_tiling(vae)
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            vae.encode(torch.ones(1))
        self.assertEqual(len(vae.calls), 1)
        self.assertEqual(vae.tiling_calls, 1)

    def test_unsupported_vae_is_unchanged(self):
        vae = TinyVAE()
        vae.enable_tiling = None
        original = vae.encode
        self.assertIs(enable_automatic_vae_tiling(vae), vae)
        self.assertEqual(vae.encode, original)

    @patch("torch.cuda.empty_cache")
    def test_repeated_install_and_deepcopy(self, empty_cache):
        original = enable_automatic_vae_tiling(TinyVAE())
        self.assertIs(enable_automatic_vae_tiling(original), original)
        duplicate = copy.deepcopy(original)
        duplicate.encode(torch.ones(1))
        self.assertTrue(duplicate.use_tiling)
        self.assertFalse(original.use_tiling)
        self.assertEqual(len(original.calls), 0)
        self.assertEqual(len(duplicate.calls), 2)

    @patch("torch.cuda.empty_cache")
    def test_pickle_round_trip_preserves_recovery(self, empty_cache):
        original = enable_automatic_vae_tiling(TinyVAE())
        restored = pickle.loads(pickle.dumps(original))

        torch.testing.assert_close(restored.encode(torch.ones(1)), torch.full((1,), 2.0))

        self.assertTrue(restored.use_tiling)
        self.assertFalse(original.use_tiling)
        self.assertEqual(len(restored.calls), 2)
        self.assertEqual(len(original.calls), 0)

    def test_successful_call_compiles_and_preserves_gradients(self):
        vae = TinyVAE(device="cpu")
        vae.enable_tiling()
        enable_automatic_vae_tiling(vae)
        decode = torch.compile(vae.decode, backend="eager", fullgraph=True)
        sample = torch.ones(2, requires_grad=True)
        decode(sample).sum().backward()
        torch.testing.assert_close(sample.grad, torch.full_like(sample, 0.5))

    @patch("torch.cuda.empty_cache")
    def test_foundation_getter_installs_on_existing_vae(self, empty_cache):
        from simpletuner.helpers.models.krea2.model import Krea2

        foundation = object.__new__(Krea2)
        foundation.AUTOENCODER_CLASS = TinyVAE
        foundation.vae = TinyVAE()
        vae = foundation.get_vae()
        torch.testing.assert_close(vae.encode(torch.ones(1)), torch.full((1,), 2.0))
        self.assertTrue(vae.use_tiling)

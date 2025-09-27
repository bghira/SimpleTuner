"""Common helpers for pipeline unit tests."""

import importlib
import inspect
import types
from dataclasses import dataclass
from typing import Callable, Dict
from unittest import mock

import torch
from PIL import Image

from tests.test_helpers import SimpleTunerTestCase

_PIPELINE_MODULE_CACHE: Dict[str, types.ModuleType] = {}


def get_pipeline_module(name: str) -> types.ModuleType:
    if name not in _PIPELINE_MODULE_CACHE:
        _PIPELINE_MODULE_CACHE[name] = importlib.import_module(f"simpletuner.helpers.models.{name}.pipeline")
    return _PIPELINE_MODULE_CACHE[name]


class PipelineTestCase(SimpleTunerTestCase):
    module_name: str = ""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not cls.module_name:
            raise ValueError("module_name must be set on PipelineTestCase subclasses")
        cls.pipeline_module = get_pipeline_module(cls.module_name)


class DummyTimestepsScheduler:
    def __init__(self):
        self.called_with = None
        self.timesteps = None

    def set_timesteps(self, timesteps=None, device=None, **kwargs):
        self.called_with = {
            "timesteps": timesteps,
            "device": device,
            "kwargs": kwargs,
        }
        self.timesteps = torch.tensor(timesteps, dtype=torch.float32)


class DummySigmasScheduler:
    def __init__(self):
        self.called_with = None
        self.timesteps = None

    def set_timesteps(self, *, sigmas=None, device=None, **kwargs):
        self.called_with = {
            "sigmas": sigmas,
            "device": device,
            "kwargs": kwargs,
        }
        self.timesteps = torch.tensor(sigmas, dtype=torch.float32)


class DummyDefaultScheduler:
    def __init__(self):
        self.called_with = None
        self.timesteps = None

    def set_timesteps(self, num_inference_steps, device=None, **kwargs):
        self.called_with = {
            "num_inference_steps": num_inference_steps,
            "device": device,
            "kwargs": kwargs,
        }
        self.timesteps = torch.arange(num_inference_steps, dtype=torch.float32)


class NoTimestepsScheduler:
    def set_timesteps(self, num_inference_steps, device=None, **kwargs):
        raise AttributeError("set_timesteps does not accept custom timesteps")


class NoSigmasScheduler:
    def set_timesteps(self, num_inference_steps, device=None, **kwargs):
        raise AttributeError("set_timesteps does not accept custom sigmas")


class DummyLatentDist:
    def __init__(self):
        self.sample_called_with = None
        self.mode_called = False

    def sample(self, generator):
        self.sample_called_with = generator
        return "sample_result"

    def mode(self):
        self.mode_called = True
        return "mode_result"


OPTIMIZED_SCALE_EXPECTATIONS: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "auraflow": lambda positives, negatives: torch.clamp(
        torch.sum(positives * negatives, dim=1, keepdim=True)
        / (torch.norm(positives, dim=1, keepdim=True) * torch.norm(negatives, dim=1, keepdim=True) + 1e-8),
        0,
        1,
    ),
    "sd3": lambda positives, negatives: torch.sum(positives * negatives, dim=1, keepdim=True)
    / (torch.sum(negatives**2, dim=1, keepdim=True) + 1e-8),
}


class RetrieveTimestepsMixin:
    def _get_retrieve_timesteps(self):
        func = getattr(self.pipeline_module, "retrieve_timesteps", None)
        if func is None:
            self.skipTest("retrieve_timesteps not implemented")
        return func

    def test_custom_timesteps(self):
        func = self._get_retrieve_timesteps()
        scheduler = DummyTimestepsScheduler()
        custom = [5, 3, 1]
        timesteps, count = func(
            scheduler,
            num_inference_steps=len(custom),
            device="cpu",
            timesteps=custom,
        )
        self.assertEqual(count, len(custom))
        self.assertTrue(torch.equal(timesteps, scheduler.timesteps))
        self.assertEqual(scheduler.called_with["timesteps"], custom)

    def test_custom_sigmas(self):
        func = self._get_retrieve_timesteps()
        scheduler = DummySigmasScheduler()
        sigmas = [0.1, 0.2, 0.3]
        timesteps, count = func(scheduler, sigmas=sigmas, device="cpu")
        self.assertEqual(count, len(sigmas))
        self.assertTrue(torch.equal(timesteps, scheduler.timesteps))
        self.assertEqual(scheduler.called_with["sigmas"], sigmas)

    def test_default_timesteps(self):
        func = self._get_retrieve_timesteps()
        scheduler = DummyDefaultScheduler()
        timesteps, count = func(scheduler, num_inference_steps=4, device="cpu")
        self.assertEqual(count, 4)
        expected = torch.arange(4, dtype=torch.float32)
        self.assertTrue(torch.equal(timesteps, expected))
        self.assertTrue(torch.equal(timesteps, scheduler.timesteps))

    def test_conflicting_inputs_raise(self):
        func = self._get_retrieve_timesteps()
        scheduler = DummyDefaultScheduler()
        with self.assertRaises(ValueError):
            func(scheduler, timesteps=[0], sigmas=[0.1])

    def test_missing_timesteps_support_raises(self):
        func = self._get_retrieve_timesteps()
        scheduler = NoTimestepsScheduler()
        with self.assertRaises(ValueError):
            func(scheduler, timesteps=[0, 1])

    def test_missing_sigmas_support_raises(self):
        func = self._get_retrieve_timesteps()
        scheduler = NoSigmasScheduler()
        with self.assertRaises(ValueError):
            func(scheduler, sigmas=[0.1, 0.2])


class OptimizedScaleMixin:
    def _get_optimized_scale(self):
        func = getattr(self.pipeline_module, "optimized_scale", None)
        if func is None:
            self.skipTest("optimized_scale not implemented")
        return func

    def test_matches_reference_implementation(self):
        func = self._get_optimized_scale()
        reference = OPTIMIZED_SCALE_EXPECTATIONS.get(self.module_name)
        if reference is None:
            self.skipTest("no optimized_scale expectation for this module")
        positives = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        negatives = torch.tensor([[0.5, 0.5], [1.0, 1.0]], dtype=torch.float32)
        result = func(positives.clone(), negatives.clone())
        expected = reference(positives.clone(), negatives.clone())
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_negative_similarity_matches_reference(self):
        func = self._get_optimized_scale()
        reference = OPTIMIZED_SCALE_EXPECTATIONS.get(self.module_name)
        if reference is None:
            self.skipTest("no optimized_scale expectation for this module")
        positives = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        negatives = torch.tensor([[-1.0, 0.0]], dtype=torch.float32)
        result = func(positives.clone(), negatives.clone())
        expected = reference(positives.clone(), negatives.clone())
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))


class CalculateShiftMixin:
    def _get_calculate_shift(self):
        func = getattr(self.pipeline_module, "calculate_shift", None)
        if func is None:
            self.skipTest("calculate_shift not implemented")
        return func

    def test_calculate_shift_edges_and_interpolation(self):
        func = self._get_calculate_shift()
        signature = inspect.signature(func)
        base_seq_len = signature.parameters["base_seq_len"].default
        max_seq_len = signature.parameters["max_seq_len"].default
        base_shift = signature.parameters["base_shift"].default
        max_shift = signature.parameters["max_shift"].default
        mid = (base_seq_len + max_seq_len) / 2
        slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        intercept = base_shift - slope * base_seq_len
        self.assertAlmostEqual(func(base_seq_len), base_shift, places=6)
        self.assertAlmostEqual(func(max_seq_len), max_shift, places=6)
        self.assertAlmostEqual(func(mid), mid * slope + intercept, places=6)


class RetrieveLatentsMixin:
    def _get_retrieve_latents(self):
        func = getattr(self.pipeline_module, "retrieve_latents", None)
        if func is None:
            self.skipTest("retrieve_latents not implemented")
        return func

    def test_sample_mode_uses_latent_dist_sample(self):
        func = self._get_retrieve_latents()
        dist = DummyLatentDist()
        generator = object()
        encoder_output = types.SimpleNamespace(latent_dist=dist)
        result = func(encoder_output, generator=generator, sample_mode="sample")
        self.assertEqual(result, "sample_result")
        self.assertIs(dist.sample_called_with, generator)

    def test_argmax_mode_uses_latent_dist_mode(self):
        func = self._get_retrieve_latents()
        dist = DummyLatentDist()
        encoder_output = types.SimpleNamespace(latent_dist=dist)
        result = func(encoder_output, sample_mode="argmax")
        self.assertEqual(result, "mode_result")
        self.assertTrue(dist.mode_called)

    def test_fallback_to_latents_attribute(self):
        func = self._get_retrieve_latents()
        latents = torch.ones(2, 2)
        encoder_output = types.SimpleNamespace(latents=latents)
        result = func(encoder_output)
        self.assertTrue(torch.equal(result, latents))

    def test_missing_latent_information_raises(self):
        func = self._get_retrieve_latents()
        encoder_output = types.SimpleNamespace()
        with self.assertRaises(AttributeError):
            func(encoder_output)


class RescaleNoiseCfgMixin:
    def _get_rescale_noise_cfg(self):
        func = getattr(self.pipeline_module, "rescale_noise_cfg", None)
        if func is None:
            self.skipTest("rescale_noise_cfg not implemented")
        return func

    def test_guidance_rescale_matches_expected(self):
        func = self._get_rescale_noise_cfg()
        noise_cfg = torch.tensor([[2.0, 6.0]], dtype=torch.float32)
        noise_pred_text = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
        guidance = 0.5
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        expected = guidance * noise_pred_rescaled + (1 - guidance) * noise_cfg
        result = func(noise_cfg.clone(), noise_pred_text.clone(), guidance_rescale=guidance)
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_zero_guidance_returns_original_noise(self):
        func = self._get_rescale_noise_cfg()
        noise_cfg = torch.randn(1, 3, 4, 4)
        noise_pred_text = torch.randn(1, 3, 4, 4)
        result = func(noise_cfg.clone(), noise_pred_text.clone(), guidance_rescale=0.0)
        self.assertTrue(torch.equal(result, noise_cfg))


class PreprocessMixin:
    def _get_preprocess(self):
        func = getattr(self.pipeline_module, "preprocess", None)
        if func is None:
            self.skipTest("preprocess not implemented")
        return func

    def test_tensor_input_returns_same_tensor(self):
        func = self._get_preprocess()
        tensor = torch.randn(1, 3, 2, 2)
        with mock.patch(f"{self.pipeline_module.__name__}.deprecate") as mock_deprecate:
            result = func(tensor)
        self.assertIs(result, tensor)
        mock_deprecate.assert_called_once()

    def test_image_input_normalization(self):
        func = self._get_preprocess()
        image = Image.new("RGB", (10, 10), color=(128, 64, 32))
        with mock.patch(f"{self.pipeline_module.__name__}.deprecate") as mock_deprecate:
            result = func(image)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1, 3, 8, 8))
        self.assertTrue(result.max() <= 1.0)
        self.assertTrue(result.min() >= -1.0)
        mock_deprecate.assert_called_once()

    def test_tensor_list_concatenation(self):
        func = self._get_preprocess()
        tensors = [torch.ones(1, 3, 4, 4), torch.zeros(1, 3, 4, 4)]
        with mock.patch(f"{self.pipeline_module.__name__}.deprecate") as mock_deprecate:
            result = func(tensors)
        self.assertTrue(torch.equal(result, torch.cat(tensors, dim=0)))
        mock_deprecate.assert_called_once()


class WanPromptCleaningMixin:
    def _get_prompt_funcs(self):
        basic = getattr(self.pipeline_module, "basic_clean", None)
        whitespace = getattr(self.pipeline_module, "whitespace_clean", None)
        prompt = getattr(self.pipeline_module, "prompt_clean", None)
        if not all((basic, whitespace, prompt)):
            self.skipTest("prompt cleaning helpers not implemented")
        return basic, whitespace, prompt

    def test_prompt_cleaning_sequence(self):
        basic, whitespace, prompt = self._get_prompt_funcs()
        raw_text = "  Hello&nbsp;&amp;amp; World\n"
        cleaned_basic = basic(raw_text)
        cleaned_whitespace = whitespace("  Hello\n\nWorld  ")
        cleaned_prompt = prompt(raw_text + "\n\n")
        self.assertEqual(cleaned_basic, "Hello\u00a0& World")
        self.assertEqual(cleaned_whitespace, "Hello World")
        self.assertEqual(cleaned_prompt, "Hello & World")


class ImportOnlyMixin:
    def test_module_imported(self):
        self.assertIsNotNone(self.pipeline_module)

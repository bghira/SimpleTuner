import unittest
from unittest.mock import patch

import torch

from simpletuner.helpers.distillation.anyflow.distiller import AnyFlowDistiller
from tests.helpers.distillation.test_anyflow_distiller import _FlowModel, _prepared_batch, _SigmaPredictionFlowModel


class AnyFlowDropoutGuidanceTests(unittest.TestCase):
    def test_meanflow_scales_derivative_only_for_conditioned_samples(self):
        model = _SigmaPredictionFlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "fuse_guidance_scale": 3.0},
        )
        batch = _prepared_batch()
        batch["text_conditioning_mask"] = torch.tensor([True, False])
        target = distiller._meanflow_target(
            prepared_batch=batch,
            model=model,
            t_sigmas=torch.full((2,), 0.7),
            r_sigmas=torch.full((2,), 0.2),
            base_target=batch["noise"] - batch["latents"],
        )

        expected = torch.tensor([5.0 / 6.0, 0.5]).view(2, 1, 1, 1).expand_as(target)
        torch.testing.assert_close(target, expected, atol=1e-5, rtol=0)

    def test_meanflow_preserves_exact_unconditional_gaussian_flow_with_fused_guidance(self):
        model = _FlowModel()
        bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        t_sigmas = torch.full((2,), 0.7, dtype=torch.float64)
        r_sigmas = torch.full((2,), 0.2, dtype=torch.float64)
        variance = (1 - t_sigmas).square() + t_sigmas.square()
        batch = _prepared_batch()
        batch["latents"] = ((1 - t_sigmas) / variance).view(2, 1, 1, 1)
        batch["noise"] = (t_sigmas / variance).view(2, 1, 1, 1)
        batch["timesteps"] = t_sigmas * 1000
        batch["flowmap_r_timesteps"] = r_sigmas * 1000
        batch["text_conditioning_mask"] = torch.zeros(2, dtype=torch.bool)

        def predict(prepared):
            t = (prepared["timesteps"] / 1000).view(2, 1, 1, 1)
            r = (prepared["flowmap_r_timesteps"] / 1000).view(2, 1, 1, 1)
            if separate_unconditional and prepared.get("is_unconditional_pass"):
                velocity = (2 * t - 1) / ((1 - t).square() + t.square())
                return {"model_prediction": prepared["noisy_latents"] * velocity}
            start_std = ((1 - t).square() + t.square()).sqrt()
            end_std = ((1 - r).square() + r.square()).sqrt()
            return {"model_prediction": prepared["noisy_latents"] * (1 - end_std / start_std) / (t - r) + bias}

        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={
                "model_type": "lora",
                "fuse_guidance_scale": 3.0,
                "central_difference_epsilon": 1e-5,
                "meanflow_weight_type": "uniform",
                "meanflow_adaptive_weighting": False,
            },
        )
        distiller._set_batch_sigma_path(batch, t_sigmas)
        for separate_unconditional in (False, True):
            with self.subTest(separate_unconditional=separate_unconditional):
                bias.grad = None
                guidance_mask = torch.full((2,), separate_unconditional, dtype=torch.bool)
                with (
                    patch.object(model, "flow_matching_guidance_mask", return_value=guidance_mask),
                    patch.object(model, "model_predict", side_effect=predict),
                ):
                    batch["target"] = distiller._meanflow_target(
                        prepared_batch=batch,
                        model=model,
                        t_sigmas=t_sigmas,
                        r_sigmas=r_sigmas,
                        base_target=batch["noise"] - batch["latents"],
                    )
                    prediction = model.model_predict(batch)
                    loss = distiller._meanflow_loss(batch, prediction)
                loss.backward()

                expected = prediction["model_prediction"].detach()
                if separate_unconditional:
                    expected = (expected + 2 * (batch["noise"] - batch["latents"])) / 3
                torch.testing.assert_close(batch["target"], expected, atol=1e-8, rtol=0)
                self.assertLess(float(loss.detach()), 1e-12)
                self.assertLess(float(bias.grad.abs()), 1e-8)

    def test_meanflow_fused_guidance_requires_text_conditioning_mask(self):
        model = _SigmaPredictionFlowModel()
        distiller = AnyFlowDistiller(teacher_model=model, noise_scheduler=None, config={"model_type": "lora"})
        for invalid in (None, torch.ones(2), torch.ones(1, dtype=torch.bool)):
            with self.subTest(mask=invalid):
                batch = _prepared_batch()
                batch["text_conditioning_mask"] = invalid
                with self.assertRaisesRegex(ValueError, "per-sample boolean text_conditioning_mask"):
                    distiller._meanflow_target(
                        prepared_batch=batch,
                        model=model,
                        t_sigmas=torch.full((2,), 0.7),
                        r_sigmas=torch.full((2,), 0.2),
                        base_target=batch["noise"] - batch["latents"],
                    )

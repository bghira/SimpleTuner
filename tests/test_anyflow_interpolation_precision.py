import unittest
from unittest.mock import patch

import torch

from simpletuner.helpers.distillation.anyflow.distiller import AnyFlowDistiller
from tests.helpers.distillation.test_anyflow_distiller import _FlowModel, _prepared_batch, _SigmaPredictionFlowModel


class AnyFlowInterpolationPrecisionTests(unittest.TestCase):
    def test_meanflow_central_difference_preserves_low_precision_latent_perturbations(self):
        model = _SigmaPredictionFlowModel()
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for epsilon in (0.005, 0.001, 0.0001):
                with self.subTest(dtype=dtype, epsilon=epsilon):
                    distiller = AnyFlowDistiller(
                        teacher_model=model,
                        noise_scheduler=None,
                        config={
                            "model_type": "lora",
                            "diffusion_ratio": 0.0,
                            "consistency_ratio": 1.0,
                            "central_difference_epsilon": epsilon,
                            "fuse_guidance_scale": 1.0,
                        },
                    )
                    batch = _prepared_batch()
                    batch["latents"] = torch.ones_like(batch["latents"], dtype=dtype)
                    batch["noise"] = 2 * batch["latents"]
                    target = distiller._meanflow_target(
                        prepared_batch=batch,
                        model=model,
                        t_sigmas=torch.full((2,), 0.7),
                        r_sigmas=torch.full((2,), 0.2),
                        base_target=batch["noise"] - batch["latents"],
                    )

                    torch.testing.assert_close(target, torch.full_like(target, 0.5), atol=1e-6, rtol=0)
                    self.assertEqual(batch["latents"].dtype, dtype)
                    self.assertEqual(batch["noise"].dtype, dtype)

    def test_meanflow_preserves_exact_velocity_solution_with_bf16_latents(self):
        model = _FlowModel()
        bias = torch.nn.Parameter(torch.tensor(0.0))

        def predict(batch):
            sigma = (batch["timesteps"].float() / 1000).view(2, 1, 1, 1)
            return {"model_prediction": (batch["noisy_latents"].float() - 1) / sigma + bias}

        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={
                "model_type": "lora",
                "diffusion_ratio": 0.0,
                "consistency_ratio": 1.0,
                "fuse_guidance_scale": 1.0,
                "meanflow_weight_type": "uniform",
                "meanflow_adaptive_weighting": False,
            },
        )
        batch = _prepared_batch()
        batch["latents"] = torch.ones_like(batch["latents"], dtype=torch.bfloat16)
        batch["noise"] = 2 * batch["latents"]
        t_sigmas = torch.full((2,), 0.7)
        distiller._set_batch_sigma_path(batch, t_sigmas)
        with patch.object(model, "model_predict", side_effect=predict):
            batch["target"] = distiller._meanflow_target(
                prepared_batch=batch,
                model=model,
                t_sigmas=t_sigmas,
                r_sigmas=torch.full((2,), 0.2),
                base_target=batch["noise"] - batch["latents"],
            )
            loss = distiller._meanflow_loss(batch, model.model_predict(batch))
        loss.backward()

        torch.testing.assert_close(batch["target"], torch.ones_like(batch["target"]))
        self.assertLess(float(loss.detach()), 1e-10)
        self.assertLess(float(bias.grad.abs()), 1e-5)

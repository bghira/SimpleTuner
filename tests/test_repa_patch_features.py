import unittest

import torch

from simpletuner.helpers.training.crepa import CrepaRegularizer


class QwenRepaTests(unittest.TestCase):
    def test_dino_uses_patch_features_not_cls_embedding(self):
        class Dino(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

            def forward(self, images):
                raise AssertionError("Default DINO forward returns only the CLS embedding")

            def forward_features(self, images):
                return {"x_norm_patchtokens": self.weight * torch.ones(images.shape[0], 4, 8)}

        reg = CrepaRegularizer.__new__(CrepaRegularizer)
        reg.encoder_name = "dinov2_vitg14"
        reg.encoder = Dino()
        features = reg._forward_encoder(torch.zeros(2, 3, 28, 28))
        self.assertEqual(features.shape, (2, 4, 8))
        self.assertFalse(features.requires_grad)

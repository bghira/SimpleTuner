import unittest
from types import SimpleNamespace
from unittest.mock import patch

from simpletuner.helpers.configuration import cmd_args


class ConfigureTF32Tests(unittest.TestCase):
    def test_new_precision_api_is_used_when_available(self):
        matmul_backend = SimpleNamespace(fp32_precision="ieee")
        cudnn_conv_backend = SimpleNamespace(fp32_precision="ieee")
        cudnn_rnn_backend = SimpleNamespace(fp32_precision="ieee")
        cudnn_backend = SimpleNamespace(fp32_precision="ieee", conv=cudnn_conv_backend, rnn=cudnn_rnn_backend)
        backends = SimpleNamespace(fp32_precision="ieee", cuda=SimpleNamespace(matmul=matmul_backend), cudnn=cudnn_backend)
        torch_stub = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
            backends=backends,
        )

        with patch.object(cmd_args, "torch", torch_stub):
            cmd_args._configure_tf32(disable_tf32=False)
            self.assertEqual(backends.fp32_precision, "tf32")
            self.assertEqual(matmul_backend.fp32_precision, "tf32")
            self.assertEqual(cudnn_backend.fp32_precision, "tf32")
            self.assertEqual(cudnn_conv_backend.fp32_precision, "tf32")
            self.assertEqual(cudnn_rnn_backend.fp32_precision, "tf32")

            cmd_args._configure_tf32(disable_tf32=True)
            self.assertEqual(backends.fp32_precision, "ieee")
            self.assertEqual(matmul_backend.fp32_precision, "ieee")
            self.assertEqual(cudnn_backend.fp32_precision, "ieee")
            self.assertEqual(cudnn_conv_backend.fp32_precision, "ieee")
            self.assertEqual(cudnn_rnn_backend.fp32_precision, "ieee")

    def test_legacy_allow_tf32_is_used_when_new_api_missing(self):
        matmul_backend = SimpleNamespace(allow_tf32=False)
        cudnn_backend = SimpleNamespace(allow_tf32=False)
        backends = SimpleNamespace(cuda=SimpleNamespace(matmul=matmul_backend), cudnn=cudnn_backend)
        torch_stub = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
            backends=backends,
        )

        with patch.object(cmd_args, "torch", torch_stub):
            cmd_args._configure_tf32(disable_tf32=False)
            self.assertTrue(matmul_backend.allow_tf32)
            self.assertTrue(cudnn_backend.allow_tf32)

            cmd_args._configure_tf32(disable_tf32=True)
            self.assertFalse(matmul_backend.allow_tf32)
            self.assertFalse(cudnn_backend.allow_tf32)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

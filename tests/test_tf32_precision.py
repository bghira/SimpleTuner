import os
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


class ConfigureROCmEnvTests(unittest.TestCase):
    def setUp(self):
        self._saved_env = {}

    def tearDown(self):
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _stash_env(self, key: str) -> None:
        if key not in self._saved_env:
            self._saved_env[key] = os.environ.get(key)

    def _clear_env(self, key: str) -> None:
        self._stash_env(key)
        os.environ.pop(key, None)

    def test_mi300_defaults_enable_tunableop_and_tf32_env(self):
        self._clear_env("PYTORCH_TUNABLEOP_ENABLED")
        self._clear_env("HIPBLASLT_ALLOW_TF32")

        torch_stub = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                get_device_properties=lambda *_: SimpleNamespace(name="AMD Instinct MI300X", gcnArchName="gfx942"),
            ),
            version=SimpleNamespace(hip="6.2"),
        )

        with patch.object(cmd_args, "torch", torch_stub):
            cmd_args._configure_rocm_environment()

        self.assertEqual(os.environ.get("PYTORCH_TUNABLEOP_ENABLED"), "1")
        self.assertEqual(os.environ.get("HIPBLASLT_ALLOW_TF32"), "1")

    def test_user_overrides_are_respected(self):
        self._clear_env("PYTORCH_TUNABLEOP_ENABLED")
        self._clear_env("HIPBLASLT_ALLOW_TF32")
        os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "0"
        os.environ["HIPBLASLT_ALLOW_TF32"] = "0"

        torch_stub = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                get_device_properties=lambda *_: SimpleNamespace(name="AMD Instinct MI300X", gcnArchName="gfx942"),
            ),
            version=SimpleNamespace(hip="6.2"),
        )

        with patch.object(cmd_args, "torch", torch_stub):
            cmd_args._configure_rocm_environment()

        self.assertEqual(os.environ.get("PYTORCH_TUNABLEOP_ENABLED"), "0")
        self.assertEqual(os.environ.get("HIPBLASLT_ALLOW_TF32"), "0")

    def test_non_mi300_rocm_only_sets_tunableop(self):
        self._clear_env("PYTORCH_TUNABLEOP_ENABLED")
        self._clear_env("HIPBLASLT_ALLOW_TF32")

        torch_stub = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                get_device_properties=lambda *_: SimpleNamespace(name="AMD Instinct MI210", gcnArchName="gfx90a"),
            ),
            version=SimpleNamespace(hip="6.0"),
        )

        with patch.object(cmd_args, "torch", torch_stub):
            cmd_args._configure_rocm_environment()

        self.assertEqual(os.environ.get("PYTORCH_TUNABLEOP_ENABLED"), "1")
        self.assertIsNone(os.environ.get("HIPBLASLT_ALLOW_TF32"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

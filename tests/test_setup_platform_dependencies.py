import runpy
import unittest
from pathlib import Path

import setuptools


def load_setup_kwargs() -> dict:
    captured: dict = {}
    original_setup = setuptools.setup

    def fake_setup(**kwargs):
        captured.update(kwargs)

    setuptools.setup = fake_setup
    try:
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "setup.py"))
    finally:
        setuptools.setup = original_setup

    return captured


class SetupPlatformDependencyTests(unittest.TestCase):
    def test_base_dependencies_require_trainingsample_0_3_range(self):
        install_requires = load_setup_kwargs()["install_requires"]

        self.assertIn("trainingsample>=0.3.0,<0.4.0", install_requires)
        self.assertNotIn("trainingsample>=0.2.18", install_requires)

    def test_platform_extras_use_latest_torchao_with_expected_torch_floor(self):
        extras_require = load_setup_kwargs()["extras_require"]

        for extra_name in ("cuda", "cuda13", "rocm", "cpu"):
            with self.subTest(extra=extra_name):
                dependencies = extras_require[extra_name]
                self.assertIn("torch>=2.11.0", dependencies)
                self.assertIn("torchvision>=0.26.0", dependencies)
                self.assertIn("torchaudio>=2.11.0", dependencies)
                self.assertIn("torchao>=0.17.0,<0.18.0", dependencies)

        apple_dependencies = extras_require["apple"]
        self.assertIn("torch>=2.13.0", apple_dependencies)
        self.assertIn("torchvision>=0.28.0", apple_dependencies)
        self.assertIn("torchaudio>=2.11.0", apple_dependencies)
        self.assertIn("torchao>=0.17.0,<0.18.0", apple_dependencies)

    def test_cuda_nightly_extras_use_latest_torchao_range(self):
        extras_require = load_setup_kwargs()["extras_require"]

        for extra_name in ("cuda-nightly", "cuda13-nightly"):
            with self.subTest(extra=extra_name):
                self.assertIn("torchao>=0.17.0,<0.18.0", extras_require[extra_name])

    def test_cuda13_runtime_pins_are_limited_to_transformerengine_extras(self):
        extras_require = load_setup_kwargs()["extras_require"]
        expected = {
            "nvidia-cublas>=13.3.0.5",
            "nvidia-cuda-nvrtc>=13.3.33",
            "nvidia-cuda-runtime>=13.3.29",
        }

        for extra_name in ("cuda13", "cuda13-nightly"):
            with self.subTest(extra=extra_name):
                self.assertTrue(expected.isdisjoint(set(extras_require[extra_name])))

        for extra_name in ("transformerengine-cuda13", "cuda13-transformerengine"):
            with self.subTest(extra=extra_name):
                self.assertTrue(expected.issubset(set(extras_require[extra_name])))
                self.assertIn("transformer_engine[pytorch]>=2.16.0,<2.17.0", extras_require[extra_name])

    def test_cuda13_vllm_is_limited_to_captioning_extras(self):
        extras_require = load_setup_kwargs()["extras_require"]
        expected = "vllm>=0.20.0,<0.26.0"

        for extra_name in ("cuda", "cuda13", "cuda-nightly", "cuda13-nightly", "rocm", "apple", "cpu"):
            with self.subTest(extra=extra_name):
                dependencies = extras_require[extra_name]
                self.assertNotIn(expected, dependencies)
                self.assertFalse(any(dependency.startswith("vllm") for dependency in dependencies))

        for extra_name in ("captioning-cuda13", "cuda13-captioning"):
            with self.subTest(extra=extra_name):
                dependencies = extras_require[extra_name]
                self.assertIn(expected, dependencies)
                self.assertFalse(any(dependency.startswith("vllm @ ") for dependency in dependencies))
                self.assertNotIn("vllm>=0.19.1,<0.20.0", dependencies)


if __name__ == "__main__":
    unittest.main()

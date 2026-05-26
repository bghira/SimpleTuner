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
    def test_platform_extras_use_latest_torchao_with_torch_211_floor(self):
        extras_require = load_setup_kwargs()["extras_require"]

        for extra_name in ("apple", "cuda", "cuda13", "rocm", "cpu"):
            with self.subTest(extra=extra_name):
                dependencies = extras_require[extra_name]
                self.assertIn("torch>=2.11.0", dependencies)
                self.assertIn("torchvision>=0.26.0", dependencies)
                self.assertIn("torchaudio>=2.11.0", dependencies)
                self.assertIn("torchao>=0.17.0,<0.18.0", dependencies)

    def test_cuda_nightly_extras_use_latest_torchao_range(self):
        extras_require = load_setup_kwargs()["extras_require"]

        for extra_name in ("cuda-nightly", "cuda13-nightly"):
            with self.subTest(extra=extra_name):
                self.assertIn("torchao>=0.17.0,<0.18.0", extras_require[extra_name])


if __name__ == "__main__":
    unittest.main()

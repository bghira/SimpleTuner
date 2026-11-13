import runpy
import unittest
from pathlib import Path

import setuptools
from packaging.requirements import Requirement


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

        for extra_name in ("cuda", "cuda13", "cpu", "apple"):
            with self.subTest(extra=extra_name):
                dependencies = extras_require[extra_name]
                self.assertIn("torch>=2.13.0", dependencies)
                self.assertIn("torchvision>=0.28.0", dependencies)
                self.assertIn("torchaudio>=2.11.0", dependencies)
                self.assertIn("torchao>=0.17.0,<0.18.0", dependencies)

    def test_rocm_extra_selects_rocm_wheels_without_generic_triton(self):
        dependencies = {
            Requirement(value).name: Requirement(value) for value in load_setup_kwargs()["extras_require"]["rocm"]
        }
        expected = {
            "torch": "2.15.0+rocm10.0",
            "torchvision": "0.29.2+rocm10.0",
            "torchaudio": "2.11.1+rocm10.0",
            "torchao": "0.17.0+cpu",
        }
        self.assertEqual(set(dependencies), set(expected))
        for name, version in expected.items():
            with self.subTest(package=name):
                self.assertTrue(dependencies[name].specifier.contains(version))
                self.assertFalse(dependencies[name].specifier.contains(version.split("+")[0]))
                self.assertFalse(dependencies[name].specifier.contains(version.split("+")[0] + "+cu130"))

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

    def test_captioning_aliases_delegate_worker_dependencies_to_captionflow(self):
        extras_require = load_setup_kwargs()["extras_require"]

        for extra_name in ("cuda", "cuda13", "cuda-nightly", "cuda13-nightly", "rocm", "apple", "cpu"):
            with self.subTest(extra=extra_name):
                dependencies = extras_require[extra_name]
                self.assertFalse(any(dependency.startswith("vllm") for dependency in dependencies))

        for extra_name in ("captioning", "captioning-cuda13", "cuda13-captioning"):
            with self.subTest(extra=extra_name):
                dependencies = extras_require[extra_name]
                self.assertEqual(dependencies, extras_require["captioning"])
                self.assertEqual(len(dependencies), 1)
                captionflow = Requirement(dependencies[0])
                self.assertEqual(captionflow.name, "caption-flow")
                self.assertEqual(captionflow.extras, {"vllm"})
                self.assertFalse(captionflow.specifier.contains("0.5.6"))
                self.assertTrue(captionflow.specifier.contains("0.5.7"))


if __name__ == "__main__":
    unittest.main()

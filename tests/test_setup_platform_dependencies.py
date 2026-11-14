import runpy
import sys
import tomllib
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

        for extra_name in ("cpu", "apple"):
            with self.subTest(extra=extra_name):
                dependencies = extras_require[extra_name]
                self.assertIn("torch>=2.15.0", dependencies)
                self.assertIn("torchvision>=0.29.2", dependencies)
                self.assertIn("torchaudio>=2.11.1", dependencies)
                self.assertIn("torchao>=0.18.0,<0.19.0", dependencies)

    def test_cuda_aliases_select_only_cuda132_wheels(self):
        extras = load_setup_kwargs()["extras_require"]
        self.assertEqual(extras["cuda"], extras["cuda13"])
        dependencies = {Requirement(value).name: Requirement(value) for value in extras["cuda"]}
        for name, version in {"torch": "2.15.0", "torchvision": "0.29.2", "torchaudio": "2.11.1"}.items():
            with self.subTest(package=name):
                specifier = dependencies[name].specifier
                self.assertTrue(specifier.contains(version + "+cu132"))
                for suffix in ("", "+cu126", "+cu130", "+cu134", "+cpu", "+rocm10.0"):
                    self.assertFalse(specifier.contains(version + suffix))
        self.assertEqual(dependencies["nvidia-cutlass-dsl"].extras, {"cu13"})
        self.assertNotIn("triton", dependencies)
        self.assertFalse(any(name.startswith(("nvidia-cudnn-", "nvidia-nccl-")) for name in dependencies))

    def test_nightly_aliases_use_published_cuda132_wheel_abis(self):
        extras = load_setup_kwargs()["extras_require"]
        self.assertEqual(extras["cuda-nightly"], extras["cuda13-nightly"])
        dependencies = {Requirement(value).name: Requirement(value) for value in extras["cuda-nightly"]}
        for name in ("torch", "torchvision", "torchaudio"):
            with self.subTest(package=name):
                url = dependencies[name].url
                self.assertIn("/nightly/cu132/", url)
                self.assertIn("%2Bcu132-", url)
                if name == "torch":
                    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
                    self.assertIn(f"-{tag}-{tag}-", url)
                else:
                    self.assertIn("-py3-none-", url)
        self.assertNotIn("triton", dependencies)

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
                self.assertIn("torchao>=0.18.0,<0.19.0", extras_require[extra_name])

    def test_transformerengine_aliases_leave_cuda_runtime_selection_to_torch(self):
        extras = load_setup_kwargs()["extras_require"]
        self.assertEqual(extras["transformerengine"], ["transformer_engine[pytorch]>=2.21.0,<2.22.0"])
        for name in ("transformerengine-cuda13", "cuda13-transformerengine"):
            self.assertEqual(extras[name], extras["transformerengine"] + extras["cuda"])

    def test_python_version_metadata_is_consistent(self):
        project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())["project"]
        self.assertEqual(load_setup_kwargs()["python_requires"], project["requires-python"])

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

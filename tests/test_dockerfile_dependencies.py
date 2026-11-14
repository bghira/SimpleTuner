#!/usr/bin/env python
"""Tests for Docker image dependency wiring."""

import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DockerfileDependenciesTests(unittest.TestCase):
    def test_editable_install_avoids_cuda_extra(self):
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        editable_install = re.search(r"pip install --no-cache-dir -e [\"']\.\[([^\"']+)\][\"']", dockerfile)

        self.assertIsNotNone(editable_install, "Dockerfile should install SimpleTuner with explicit extras")
        extras = {extra.strip() for extra in editable_install.group(1).split(",")}
        self.assertIn("jxl", extras)
        self.assertNotIn("cuda", extras)

    def test_container_cuda_dependencies_are_explicit(self):
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        expected_dependencies = [
            "bitsandbytes>=0.50.2",
            "deepspeed>=0.20.0",
            "torchao>=0.18.0,<0.19.0",
            "nvidia-ml-py>=12.555",
            "lm-eval>=0.4.4",
        ]

        for dependency in expected_dependencies:
            self.assertIn(dependency, dockerfile)

    def test_container_toolkit_and_pytorch_use_cuda132(self):
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("FROM nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04", dockerfile)
        self.assertIn("ENV CUDA_HOME=/usr/local/cuda-13.2", dockerfile)
        self.assertIn("https://download.pytorch.org/whl/cu132", dockerfile)
        for package in ("torch==2.15.0+cu132", "torchvision==0.29.2+cu132", "torchaudio==2.11.1+cu132"):
            self.assertIn(package, dockerfile)

    def test_container_cuda_dependencies_skip_conflicting_nvidia_wheels(self):
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertNotIn("nvidia-cudnn-cu12", dockerfile)
        self.assertNotIn("nvidia-nccl-cu12", dockerfile)


if __name__ == "__main__":
    unittest.main()

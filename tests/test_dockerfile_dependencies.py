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
            "bitsandbytes>=0.45.0",
            "deepspeed>=0.17.2",
            "torchao>=0.17.0,<0.18.0",
            "nvidia-ml-py>=12.555",
            "lm-eval>=0.4.4",
            "ramtorch",
        ]

        for dependency in expected_dependencies:
            self.assertIn(dependency, dockerfile)

    def test_container_cuda_dependencies_skip_conflicting_nvidia_wheels(self):
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertNotIn("nvidia-cudnn-cu12", dockerfile)
        self.assertNotIn("nvidia-nccl-cu12", dockerfile)


if __name__ == "__main__":
    unittest.main()

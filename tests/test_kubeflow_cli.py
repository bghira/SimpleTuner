"""Tests for Kubeflow server command configuration."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from simpletuner.cli import create_parser
from simpletuner.cli.server import _configure_kubeflow_environment


class KubeflowServerCliTestCase(unittest.TestCase):
    """Test explicit Kubeflow server flags and environment mapping."""

    def test_parser_accepts_single_gpu_kubeflow_options(self) -> None:
        """Verify the server command exposes all required integration values."""
        args = create_parser().parse_args(
            [
                "server",
                "--kubeflow",
                "--kubeflow-queue",
                "gpu-training",
                "--kubeflow-worker-image",
                "registry.example.com/simpletuner:4.5.0",
                "--kubeflow-orchestrator-url",
                "http://simpletuner.simpletuner.svc:8001",
            ]
        )

        self.assertTrue(args.kubeflow)
        self.assertEqual(args.kubeflow_namespace, "default")
        self.assertEqual(args.kubeflow_runtime, "simpletuner-worker")
        self.assertEqual(args.kubeflow_poll_interval, 5.0)

    def test_environment_mapping_requires_and_exports_cluster_values(self) -> None:
        """Verify validated CLI values become the server integration contract."""
        args = SimpleNamespace(
            kubeflow=True,
            mode="trainer",
            kubeflow_namespace="simpletuner",
            kubeflow_runtime="simpletuner-worker",
            kubeflow_queue="gpu-training",
            kubeflow_worker_image="registry.example.com/simpletuner:4.5.0",
            kubeflow_orchestrator_url="http://simpletuner.simpletuner.svc:8001",
            kubeflow_poll_interval=3.0,
        )

        with patch.dict(os.environ, {}, clear=True):
            error = _configure_kubeflow_environment(args)

            self.assertIsNone(error)
            self.assertEqual(os.environ["SIMPLETUNER_KUBEFLOW_ENABLED"], "true")
            self.assertEqual(os.environ["SIMPLETUNER_KUBEFLOW_QUEUE"], "gpu-training")
            self.assertEqual(
                os.environ["SIMPLETUNER_KUBEFLOW_WORKER_IMAGE"],
                "registry.example.com/simpletuner:4.5.0",
            )

    def test_callback_only_server_rejects_kubeflow(self) -> None:
        """Verify the scheduler cannot run without trainer APIs."""
        args = SimpleNamespace(kubeflow=True, mode="callback")

        self.assertIn("trainer", _configure_kubeflow_environment(args))

"""Tests for single-GPU Kubeflow TrainJob provisioning."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from simpletuner.simpletuner_sdk.server.services.kubeflow import (
    KubeflowPhase,
    KubeflowSettings,
    KubeflowWorkerProvisioner,
)


class KubeflowSettingsTestCase(unittest.TestCase):
    """Test Kubeflow server configuration validation."""

    def test_enabled_settings_require_runtime_values(self) -> None:
        """Verify enabled integration rejects incomplete configuration."""
        settings = KubeflowSettings(enabled=True)

        with self.assertRaisesRegex(ValueError, "queue_name"):
            settings.validate()


class KubeflowWorkerProvisionerTestCase(unittest.IsolatedAsyncioTestCase):
    """Test TrainJob and Secret lifecycle without a live cluster."""

    def setUp(self) -> None:
        """Create provisioner fixtures."""
        self.settings = KubeflowSettings(
            enabled=True,
            namespace="simpletuner",
            runtime_name="simpletuner-worker",
            queue_name="gpu-training",
            worker_image="registry.example.com/simpletuner:4.5.0",
            orchestrator_url="http://simpletuner.simpletuner.svc:8001",
        )
        self.core_api = MagicMock()
        self.custom_api = MagicMock()
        self.provisioner = KubeflowWorkerProvisioner(
            self.settings,
            core_api=self.core_api,
            custom_objects_api=self.custom_api,
        )

    def test_trainjob_requests_exactly_one_gpu(self) -> None:
        """Verify every generated TrainJob is one node, one process, one GPU."""
        manifest = self.provisioner.build_trainjob_manifest(
            job_id="kjob-123",
            worker_id="worker-123",
            secret_name="simpletuner-worker-kjob-123",
        )

        trainer = manifest["spec"]["trainer"]
        self.assertEqual(manifest["apiVersion"], "trainer.kubeflow.org/v1alpha1")
        self.assertTrue(manifest["spec"]["suspend"])
        self.assertEqual(trainer["numNodes"], 1)
        self.assertEqual(trainer["numProcPerNode"], 1)
        self.assertEqual(trainer["resourcesPerNode"]["requests"], {"nvidia.com/gpu": "1"})
        self.assertEqual(trainer["resourcesPerNode"]["limits"], {"nvidia.com/gpu": "1"})
        self.assertEqual(
            manifest["metadata"]["labels"]["kueue.x-k8s.io/queue-name"],
            "gpu-training",
        )

    def test_worker_token_is_loaded_from_secret(self) -> None:
        """Verify the TrainJob never contains the plaintext worker token."""
        manifest = self.provisioner.build_trainjob_manifest(
            job_id="kjob-123",
            worker_id="worker-123",
            secret_name="simpletuner-worker-kjob-123",
        )

        token_env = next(
            item
            for item in manifest["spec"]["trainer"]["env"]
            if item["name"] == "SIMPLETUNER_WORKER_TOKEN"
        )
        self.assertEqual(
            token_env["valueFrom"]["secretKeyRef"],
            {"name": "simpletuner-worker-kjob-123", "key": "worker-token"},
        )
        self.assertNotIn("test-token", str(manifest))

    def test_worker_uses_standard_simpletuner_environment(self) -> None:
        """Verify Kubernetes does not require a Kubeflow-aware Worker."""
        manifest = self.provisioner.build_trainjob_manifest(
            job_id="kjob-123",
            worker_id="worker-123",
            secret_name="simpletuner-worker-kjob-123",
        )

        trainer = manifest["spec"]["trainer"]
        env_names = {item["name"] for item in trainer["env"]}

        self.assertEqual(trainer["command"], ["simpletuner", "worker"])
        self.assertNotIn("SIMPLETUNER_WORKER_PROVIDER", env_names)
        self.assertNotIn("SIMPLETUNER_WORKER_BOUND_JOB_ID", env_names)

    async def test_create_rolls_back_secret_when_trainjob_creation_fails(self) -> None:
        """Verify a failed TrainJob creation does not leave a token Secret."""
        self.custom_api.create_namespaced_custom_object.side_effect = RuntimeError("create failed")

        with self.assertRaisesRegex(RuntimeError, "create failed"):
            await self.provisioner.create(
                job_id="kjob-123",
                worker_id="worker-123",
                worker_token="test-token",
            )

        self.core_api.create_namespaced_secret.assert_called_once()
        self.core_api.delete_namespaced_secret.assert_called_once_with(
            name="simpletuner-worker-kjob-123",
            namespace="simpletuner",
        )

    async def test_get_phase_maps_complete_condition(self) -> None:
        """Verify completed TrainJobs map to a terminal provisioning phase."""
        self.custom_api.get_namespaced_custom_object.return_value = {
            "status": {
                "conditions": [
                    {"type": "Complete", "status": "True"},
                ]
            }
        }

        phase = await self.provisioner.get_phase("simpletuner-kjob-123")

        self.assertEqual(phase, KubeflowPhase.COMPLETED)


    async def test_get_logs_reads_trainjob_pod(self) -> None:
        """Verify Server-side polling reads the ephemeral Worker Pod log."""
        self.core_api.list_namespaced_pod.return_value = SimpleNamespace(
            items=[SimpleNamespace(metadata=SimpleNamespace(name="worker-pod"))]
        )
        self.core_api.read_namespaced_pod_log.return_value = b"training step 1/1"

        logs = await self.provisioner.get_logs("simpletuner-kjob-123")

        self.core_api.list_namespaced_pod.assert_called_once_with(
            namespace="simpletuner",
            label_selector="jobset.sigs.k8s.io/jobset-name=simpletuner-kjob-123",
        )
        self.core_api.read_namespaced_pod_log.assert_called_once_with(
            name="worker-pod",
            namespace="simpletuner",
            timestamps=True,
        )
        self.assertEqual(logs, "training step 1/1")

    async def test_get_logs_decodes_stringified_bytes(self) -> None:
        """Verify Kubernetes client byte representations become plain text."""
        self.core_api.list_namespaced_pod.return_value = SimpleNamespace(
            items=[SimpleNamespace(metadata=SimpleNamespace(name="worker-pod"))]
        )
        self.core_api.read_namespaced_pod_log.return_value = (
            'b"training step 1/1\\ntraining complete"'
        )

        logs = await self.provisioner.get_logs("simpletuner-kjob-123")

        self.assertEqual(logs, "training step 1/1\ntraining complete")


if __name__ == "__main__":
    unittest.main()

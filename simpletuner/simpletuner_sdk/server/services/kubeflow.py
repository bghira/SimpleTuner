"""Kubernetes resources for one-shot, single-GPU SimpleTuner Workers."""

from __future__ import annotations

import ast
import asyncio
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

KUBEFLOW_PROVIDER = "kubeflow"
LOCAL_UPLOAD_BUCKET = "outputs"
TRAINJOB_GROUP = "trainer.kubeflow.org"
TRAINJOB_VERSION = "v1alpha1"
TRAINJOB_PLURAL = "trainjobs"
GPU_RESOURCE_NAME = "nvidia.com/gpu"
SINGLE_GPU_QUANTITY = "1"
WORKER_TOKEN_KEY = "worker-token"


def _env_flag(value: Optional[str]) -> bool:
    """Parse a conventional boolean environment value.

    Args:
        value: Raw environment value.

    Returns:
        Whether the value enables the feature.
    """
    return bool(value and value.strip().lower() in {"1", "true", "yes", "on"})


@dataclass(frozen=True, slots=True)
class KubeflowSettings:
    """Configuration for namespace-scoped Kubeflow Worker provisioning."""

    enabled: bool = False
    namespace: str = "default"
    runtime_name: str = "simpletuner-worker"
    queue_name: Optional[str] = None
    worker_image: Optional[str] = None
    orchestrator_url: Optional[str] = None
    poll_interval_seconds: float = 5.0

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "KubeflowSettings":
        """Load Kubeflow settings from environment variables.

        Args:
            environ: Environment mapping, defaulting to the process environment.

        Returns:
            Parsed and validated settings.
        """
        values = os.environ if environ is None else environ
        settings = cls(
            enabled=_env_flag(values.get("SIMPLETUNER_KUBEFLOW_ENABLED")),
            namespace=values.get("SIMPLETUNER_KUBEFLOW_NAMESPACE", "default"),
            runtime_name=values.get("SIMPLETUNER_KUBEFLOW_RUNTIME", "simpletuner-worker"),
            queue_name=values.get("SIMPLETUNER_KUBEFLOW_QUEUE"),
            worker_image=values.get("SIMPLETUNER_KUBEFLOW_WORKER_IMAGE"),
            orchestrator_url=values.get("SIMPLETUNER_KUBEFLOW_ORCHESTRATOR_URL"),
            poll_interval_seconds=float(values.get("SIMPLETUNER_KUBEFLOW_POLL_INTERVAL", "5")),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Validate settings required by an enabled integration.

        Raises:
            ValueError: If an enabled integration is incomplete or invalid.
        """
        if not self.enabled:
            return
        required = {
            "queue_name": self.queue_name,
            "worker_image": self.worker_image,
            "orchestrator_url": self.orchestrator_url,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Kubeflow configuration requires: {', '.join(missing)}")
        if self.poll_interval_seconds <= 0:
            raise ValueError("Kubeflow poll_interval_seconds must be greater than zero")


@dataclass(frozen=True, slots=True)
class KubeflowResources:
    """Kubernetes resources allocated for one SimpleTuner job."""

    namespace: str
    trainjob_name: str
    secret_name: str
    trainjob_uid: Optional[str] = None

    def to_dict(self) -> dict[str, Optional[str]]:
        """Serialize resource references into job metadata.

        Returns:
            JSON-serializable resource references.
        """
        return {
            "namespace": self.namespace,
            "trainjob_name": self.trainjob_name,
            "secret_name": self.secret_name,
            "trainjob_uid": self.trainjob_uid,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "KubeflowResources":
        """Restore resource references from job metadata.

        Args:
            values: Kubernetes metadata stored with the job.

        Returns:
            Resource reference object.
        """
        return cls(
            namespace=str(values["namespace"]),
            trainjob_name=str(values["trainjob_name"]),
            secret_name=str(values["secret_name"]),
            trainjob_uid=values.get("trainjob_uid"),
        )


class KubeflowPhase(str, Enum):
    """Infrastructure phase of a provisioned TrainJob."""

    WAITING = "waiting_resource"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    MISSING = "missing"


def _resource_name(prefix: str, identifier: str) -> str:
    """Build a deterministic DNS-compatible Kubernetes resource name.

    Args:
        prefix: Resource-specific prefix.
        identifier: SimpleTuner job identifier.

    Returns:
        Kubernetes resource name no longer than 63 characters.
    """
    normalized = re.sub(r"[^a-z0-9-]+", "-", identifier.lower()).strip("-")
    return f"{prefix}-{normalized}"[:63].rstrip("-")


def _decode_log_payload(payload: Any) -> str:
    """Normalize Kubernetes client log responses into plain UTF-8 text.

    Some Kubernetes client versions expose a response body as a stringified
    bytes literal such as ``b"line\\n"``. This helper handles both raw bytes
    and that representation without changing ordinary string logs.

    Args:
        payload: Raw value returned by ``read_namespaced_pod_log``.

    Returns:
        Decoded log text suitable for API responses and central archival.
    """
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")

    text = str(payload or "")
    if len(text) >= 3 and text.startswith(("b'", 'b"')):
        try:
            decoded = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
        if isinstance(decoded, bytes):
            return decoded.decode("utf-8", errors="replace")
    return text


class KubeflowWorkerProvisioner:
    """Create and observe one-shot SimpleTuner TrainJobs."""

    def __init__(
        self,
        settings: KubeflowSettings,
        *,
        core_api: Any = None,
        custom_objects_api: Any = None,
    ) -> None:
        """Initialize the provisioner.

        Args:
            settings: Validated Kubeflow settings.
            core_api: Optional injected CoreV1Api for tests.
            custom_objects_api: Optional injected CustomObjectsApi for tests.
        """
        settings.validate()
        self.settings = settings
        if core_api is None or custom_objects_api is None:
            core_api, custom_objects_api = self._load_incluster_clients()
        self.core_api = core_api
        self.custom_objects_api = custom_objects_api

    @staticmethod
    def _load_incluster_clients() -> tuple[Any, Any]:
        """Load official Kubernetes clients using the Pod ServiceAccount.

        Returns:
            CoreV1Api and CustomObjectsApi instances.

        Raises:
            RuntimeError: If the optional dependency is missing.
        """
        try:
            from kubernetes import client, config
        except ImportError as exc:
            raise RuntimeError(
                "Kubeflow support requires the optional dependency: pip install 'simpletuner[kubernetes]'"
            ) from exc
        config.load_incluster_config()
        return client.CoreV1Api(), client.CustomObjectsApi()

    def build_trainjob_manifest(
        self,
        *,
        job_id: str,
        worker_id: str,
        secret_name: str,
    ) -> dict[str, Any]:
        """Build a TrainJob fixed to one Pod, process, and GPU.

        Args:
            job_id: SimpleTuner job identifier.
            worker_id: Pre-created Worker identifier.
            secret_name: Secret containing the Worker token.

        Returns:
            Kubeflow Trainer v2 TrainJob manifest.
        """
        labels = {
            "kueue.x-k8s.io/queue-name": str(self.settings.queue_name),
            "simpletuner.ai/job-id": job_id,
            "simpletuner.ai/worker-id": worker_id,
        }
        return {
            "apiVersion": f"{TRAINJOB_GROUP}/{TRAINJOB_VERSION}",
            "kind": "TrainJob",
            "metadata": {
                "name": _resource_name("simpletuner", job_id),
                "namespace": self.settings.namespace,
                "labels": labels,
            },
            "spec": {
                "suspend": True,
                "runtimeRef": {
                    "name": self.settings.runtime_name,
                    "kind": "TrainingRuntime",
                    "apiGroup": TRAINJOB_GROUP,
                },
                "trainer": {
                    "image": self.settings.worker_image,
                    "command": ["simpletuner", "worker"],
                    "numNodes": 1,
                    "numProcPerNode": 1,
                    "resourcesPerNode": {
                        "requests": {GPU_RESOURCE_NAME: SINGLE_GPU_QUANTITY},
                        "limits": {GPU_RESOURCE_NAME: SINGLE_GPU_QUANTITY},
                    },
                    "env": [
                        {
                            "name": "SIMPLETUNER_ORCHESTRATOR_URL",
                            "value": self.settings.orchestrator_url,
                        },
                        {
                            "name": "SIMPLETUNER_WORKER_TOKEN",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": secret_name,
                                    "key": WORKER_TOKEN_KEY,
                                }
                            },
                        },
                        {"name": "SIMPLETUNER_WORKER_NAME", "value": worker_id},
                        {"name": "SIMPLETUNER_WORKER_PERSISTENT", "value": "false"},
                    ],
                },
            },
        }

    def build_secret_manifest(
        self,
        *,
        job_id: str,
        worker_token: str,
    ) -> dict[str, Any]:
        """Build the short-lived Worker authentication Secret.

        Args:
            job_id: SimpleTuner job identifier.
            worker_token: Plaintext Worker startup token.

        Returns:
            Kubernetes Secret manifest.
        """
        return {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": _resource_name("simpletuner-worker", job_id),
                "namespace": self.settings.namespace,
                "labels": {"simpletuner.ai/job-id": job_id},
            },
            "type": "Opaque",
            "stringData": {WORKER_TOKEN_KEY: worker_token},
        }

    async def create(
        self,
        *,
        job_id: str,
        worker_id: str,
        worker_token: str,
    ) -> KubeflowResources:
        """Create the Worker Secret and single-GPU TrainJob.

        Args:
            job_id: SimpleTuner job identifier.
            worker_id: Bound Worker identifier.
            worker_token: Plaintext Worker startup token.

        Returns:
            Created Kubernetes resource references.

        Raises:
            Exception: If either Kubernetes resource cannot be created.
        """
        secret = self.build_secret_manifest(job_id=job_id, worker_token=worker_token)
        secret_name = secret["metadata"]["name"]
        trainjob = self.build_trainjob_manifest(
            job_id=job_id,
            worker_id=worker_id,
            secret_name=secret_name,
        )
        trainjob_name = trainjob["metadata"]["name"]

        await asyncio.to_thread(
            self.core_api.create_namespaced_secret,
            namespace=self.settings.namespace,
            body=secret,
        )
        try:
            created = await asyncio.to_thread(
                self.custom_objects_api.create_namespaced_custom_object,
                group=TRAINJOB_GROUP,
                version=TRAINJOB_VERSION,
                namespace=self.settings.namespace,
                plural=TRAINJOB_PLURAL,
                body=trainjob,
            )
        except Exception:
            await asyncio.to_thread(
                self.core_api.delete_namespaced_secret,
                name=secret_name,
                namespace=self.settings.namespace,
            )
            raise

        return KubeflowResources(
            namespace=self.settings.namespace,
            trainjob_name=trainjob_name,
            secret_name=secret_name,
            trainjob_uid=created.get("metadata", {}).get("uid"),
        )

    async def get_phase(self, trainjob_name: str) -> KubeflowPhase:
        """Read and normalize a TrainJob infrastructure phase.

        Args:
            trainjob_name: Kubernetes TrainJob name.

        Returns:
            Normalized infrastructure phase.
        """
        try:
            trainjob = await asyncio.to_thread(
                self.custom_objects_api.get_namespaced_custom_object,
                group=TRAINJOB_GROUP,
                version=TRAINJOB_VERSION,
                namespace=self.settings.namespace,
                plural=TRAINJOB_PLURAL,
                name=trainjob_name,
            )
        except Exception as exc:
            if getattr(exc, "status", None) == 404:
                return KubeflowPhase.MISSING
            raise

        conditions = trainjob.get("status", {}).get("conditions", [])
        true_conditions = {
            condition.get("type")
            for condition in conditions
            if str(condition.get("status", "")).lower() == "true"
        }
        if "Failed" in true_conditions:
            return KubeflowPhase.FAILED
        if "Complete" in true_conditions or "Succeeded" in true_conditions:
            return KubeflowPhase.COMPLETED
        if "Running" in true_conditions:
            return KubeflowPhase.RUNNING
        if trainjob.get("spec", {}).get("suspend", False):
            return KubeflowPhase.WAITING
        return KubeflowPhase.STARTING

    async def get_logs(self, trainjob_name: str) -> str:
        """Read the current Worker Pod log for a TrainJob.

        Args:
            trainjob_name: Kubernetes TrainJob name.

        Returns:
            Combined Worker log text, or an empty string before Pod creation.
        """
        pod_list = await asyncio.to_thread(
            self.core_api.list_namespaced_pod,
            namespace=self.settings.namespace,
            label_selector=f"jobset.sigs.k8s.io/jobset-name={trainjob_name}",
        )
        pods = list(getattr(pod_list, "items", None) or [])
        if not pods:
            return ""
        pod_name = pods[0].metadata.name
        logs = await asyncio.to_thread(
            self.core_api.read_namespaced_pod_log,
            name=pod_name,
            namespace=self.settings.namespace,
            timestamps=True,
        )
        return _decode_log_payload(logs)

    async def delete(self, resources: KubeflowResources) -> None:
        """Delete the TrainJob and Worker token Secret.

        Args:
            resources: Kubernetes resources owned by the job.
        """
        calls = (
            (
                self.custom_objects_api.delete_namespaced_custom_object,
                {
                    "group": TRAINJOB_GROUP,
                    "version": TRAINJOB_VERSION,
                    "namespace": resources.namespace,
                    "plural": TRAINJOB_PLURAL,
                    "name": resources.trainjob_name,
                },
            ),
            (
                self.core_api.delete_namespaced_secret,
                {
                    "name": resources.secret_name,
                    "namespace": resources.namespace,
                },
            ),
        )
        for operation, kwargs in calls:
            try:
                await asyncio.to_thread(operation, **kwargs)
            except Exception as exc:
                if getattr(exc, "status", None) != 404:
                    raise

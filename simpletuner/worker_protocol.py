"""Payloads and limits shared by workers and the orchestrator."""

import re

WORKER_LOG_TAIL_SIZE = 50_000
WORKER_HEARTBEAT_SECONDS = 30
WORKER_JOB_PROVIDERS = {"worker", "kubeflow"}
WORKER_JOB_ACTIVE_STATUSES = {"starting", "training", "running"}


def bundled_dataloader(metadata: dict) -> list[dict] | None:
    return metadata.get("dataloader", (metadata.get("config") or {}).get("dataloader_config"))


_CREDENTIAL_FIELDS = {
    "access_key",
    "access_key_id",
    "api_key",
    "authorization",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "cookie",
    "credentials",
    "headers",
    "password",
    "secret",
    "token",
    "secret_key",
    "storage_options",
}
_CREDENTIAL_SUFFIXES = ("_password", "_secret", "_token", "_credential", "_credentials", "_api_key")
_REMOTE_URL = re.compile(r"^([a-z][a-z0-9+.-]*://)([^/?#]*)([^?#]*)", re.IGNORECASE)


def redact_dataset_config(value):
    """Copy authored settings with credentials and URL authentication removed."""
    if isinstance(value, dict):
        return {key: redact_dataset_config(item) for key, item in value.items() if not _is_credential_field(key)}
    if isinstance(value, list):
        return [redact_dataset_config(item) for item in value]
    if isinstance(value, str) and (match := _REMOTE_URL.match(value)):
        scheme, host, path = match.groups()
        return scheme + host.rsplit("@", 1)[-1] + path
    return value


def _is_credential_field(key: str) -> bool:
    key = key.lower().replace("-", "_")
    return key in _CREDENTIAL_FIELDS or key.endswith(_CREDENTIAL_SUFFIXES)

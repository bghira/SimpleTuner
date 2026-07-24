"""Credential creation shared by manual and provisioned GPU workers."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from ..models.worker import Worker, WorkerStatus, WorkerType


@dataclass(frozen=True, slots=True)
class WorkerCredentials:
    """A persisted Worker and its one-time plaintext token."""

    worker: Worker
    token: str


def hash_token(token: str) -> str:
    """Hash a worker token using SHA-256.

    Args:
        token: Plaintext Worker token.

    Returns:
        Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_worker_token() -> str:
    """Generate a cryptographically secure Worker token.

    Returns:
        URL-safe plaintext token.
    """
    return secrets.token_urlsafe(32)


async def create_worker_credentials(
    worker_repository,
    *,
    worker_id: str,
    name: str,
    user_id: int,
    worker_type: WorkerType = WorkerType.EPHEMERAL,
    provider: Optional[str] = None,
    current_job_id: Optional[str] = None,
) -> WorkerCredentials:
    """Create a Worker record and return its plaintext startup token.

    Args:
        worker_repository: Repository used to persist the Worker.
        worker_id: Stable Worker identifier.
        name: Human-readable Worker name.
        user_id: Owning SimpleTuner user identifier.
        worker_type: Persistent or ephemeral Worker type.
        provider: Infrastructure provider that owns the Worker.
        current_job_id: Job permanently assigned to this Worker, if any.

    Returns:
        Persisted Worker and its plaintext token.
    """
    token = generate_worker_token()
    worker = Worker(
        worker_id=worker_id,
        name=name,
        worker_type=worker_type,
        status=WorkerStatus.CONNECTING,
        token_hash=hash_token(token),
        user_id=user_id,
        provider=provider,
        current_job_id=current_job_id,
        created_at=datetime.now(timezone.utc),
    )
    await worker_repository.create_worker(worker)
    return WorkerCredentials(worker=worker, token=token)

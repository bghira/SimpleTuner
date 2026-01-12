"""Webhook handling endpoints for cloud training."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from ...services.cloud import CloudJobStatus
from ...services.cloud.cache import TTLCache
from ...services.cloud.secrets import get_secrets_manager
from ._shared import ReplicateWebhookPayload, check_ip_allowlist, emit_cloud_event, get_client_ip, get_job_store

logger = logging.getLogger(__name__)

# Configurable webhook timestamp tolerance (seconds) - prevents replay attacks
WEBHOOK_TIMESTAMP_TOLERANCE = int(os.environ.get("SIMPLETUNER_WEBHOOK_TIMESTAMP_TOLERANCE", "300"))

router = APIRouter()


# Webhook signing secret cache (5 minute TTL - short to pick up secret rotation quickly)
_webhook_secret_cache: TTLCache[str] = TTLCache(default_ttl=300.0)


async def _get_webhook_signing_secret() -> Optional[str]:
    """Fetch and cache the webhook signing secret from Replicate."""
    cached = _webhook_secret_cache.get("signing_secret")
    if cached is not None:
        return cached

    secrets = get_secrets_manager()
    api_token = secrets.get_replicate_token()
    if not api_token:
        return None

    try:
        from ...services.cloud.http_client import get_async_client

        async with get_async_client(timeout=10.0) as client:
            response = await client.get(
                "https://api.replicate.com/v1/webhooks/default/secret",
                headers={"Authorization": f"Bearer {api_token}"},
            )
            if response.status_code == 200:
                data = response.json()
                secret = data.get("key")
                if secret:
                    _webhook_secret_cache.set("signing_secret", secret)
                return secret
            else:
                logger.warning("Failed to fetch webhook signing secret: %s", response.status_code)
                return None
    except Exception as exc:
        logger.warning("Error fetching webhook signing secret: %s", exc)
        return None


def _verify_webhook_signature(
    body: bytes,
    webhook_id: str,
    webhook_timestamp: str,
    webhook_signature: str,
    secret: str,
) -> bool:
    """Verify the webhook signature using HMAC-SHA256."""
    # Check timestamp to prevent replay attacks
    try:
        ts = int(webhook_timestamp)
        now = int(time.time())
        if abs(now - ts) > WEBHOOK_TIMESTAMP_TOLERANCE:
            logger.warning(
                "Webhook timestamp too old or in future: %s (tolerance=%ds)", webhook_timestamp, WEBHOOK_TIMESTAMP_TOLERANCE
            )
            return False
    except (ValueError, TypeError):
        logger.warning("Invalid webhook timestamp: %s", webhook_timestamp)
        return False

    # Construct the signed content: id.timestamp.body
    signed_content = f"{webhook_id}.{webhook_timestamp}.".encode() + body

    # Decode the secret (base64 encoded, may have whsec_ prefix)
    try:
        secret_key = secret
        if secret_key.startswith("whsec_"):
            secret_key = secret_key[6:]
        secret_bytes = base64.b64decode(secret_key)
    except Exception as exc:
        logger.warning("Failed to decode webhook secret: %s", exc)
        return False

    # Compute expected signature
    expected_sig = hmac.new(secret_bytes, signed_content, hashlib.sha256).digest()
    expected_sig_b64 = base64.b64encode(expected_sig).decode()

    # Parse the signature header and compare
    for sig_part in webhook_signature.split():
        if "," in sig_part:
            version, sig = sig_part.split(",", 1)
            if version == "v1":
                if hmac.compare_digest(sig, expected_sig_b64):
                    return True

    logger.warning("Webhook signature verification failed")
    return False


@router.post("/webhook/replicate")
async def replicate_webhook(request: Request) -> Dict[str, Any]:
    """Receive webhook events from Replicate Cog."""
    store = get_job_store()
    provider_config = await store.get_provider_config("replicate")
    client_ip = get_client_ip(request)

    # Note: Rate limiting is handled by RateLimitMiddleware (100 req/min for /api/cloud/webhooks/)

    # IP allowlist check
    allowed_ips = provider_config.get("webhook_allowed_ips", [])
    if allowed_ips:
        if not check_ip_allowlist(client_ip, allowed_ips):
            logger.warning("Webhook from non-allowed IP: %s", client_ip)
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="IP address not in allowlist")

    # Get raw body for signature verification
    body = await request.body()

    # Signature verification
    require_signature = provider_config.get("webhook_require_signature", True)
    webhook_id = request.headers.get("webhook-id")
    webhook_timestamp = request.headers.get("webhook-timestamp")
    webhook_signature = request.headers.get("webhook-signature")

    if webhook_id and webhook_timestamp and webhook_signature:
        signing_secret = await _get_webhook_signing_secret()
        if signing_secret:
            if not _verify_webhook_signature(body, webhook_id, webhook_timestamp, webhook_signature, signing_secret):
                logger.warning("Invalid webhook signature from IP: %s", client_ip)
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook signature")
        elif require_signature:
            logger.warning("Webhook signature required but no signing secret configured")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Webhook signature verification failed: no signing secret configured.",
            )
    elif require_signature:
        logger.warning("Webhook received without signature headers from IP: %s (signature required)", client_ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Webhook signature required but not provided.",
        )

    # Parse and validate payload
    try:
        raw_payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload")

    try:
        # Validate with Pydantic model - allows extra fields
        payload = ReplicateWebhookPayload.model_validate(raw_payload)
    except ValidationError as exc:
        logger.warning("Webhook payload validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid webhook payload: {exc.error_count()} validation errors"
        )

    # Extract fields - prefer explicit fields, fall back to raw payload for additional lookups
    event_type = payload.event_type or raw_payload.get("event_type")
    job_id = payload.job_id or raw_payload.get("job_id")

    if not job_id:
        logger.warning("Webhook received without job_id: %s", raw_payload)
        return {"status": "ignored", "reason": "no job_id"}

    # Map Cog event types to job status updates
    # Keys are external event type strings from Cog webhooks
    event_status_map = {
        "lifecycle.stage": None,
        "training.status": None,
        "training.checkpoint": None,
        "completed": CloudJobStatus.COMPLETED,
        "failed": CloudJobStatus.FAILED,
        "canceled": CloudJobStatus.CANCELLED,  # Cog uses American spelling
        "error": CloudJobStatus.FAILED,
    }

    mapped_status = event_status_map.get(event_type)
    new_status = mapped_status.value if mapped_status else None

    # For lifecycle events, check the status field
    if event_type in ("lifecycle.stage", "training.status"):
        stage_status = payload.status or raw_payload.get("status")
        if stage_status:
            # Use from_external to handle spelling variations
            normalized = CloudJobStatus.from_external(stage_status)
            new_status = normalized.value

    updates: Dict[str, Any] = {}
    if new_status:
        updates["status"] = new_status

    # Extract error message
    error_msg = payload.error or payload.message
    if error_msg and event_type in ("error", "failed"):
        updates["error_message"] = str(error_msg)

    # Extract output_url
    output = payload.output
    if output:
        if isinstance(output, str):
            updates["output_url"] = output
        elif isinstance(output, dict) and output.get("url"):
            updates["output_url"] = output["url"]

    # Update completed_at for terminal states
    if new_status in CloudJobStatus.terminal_values():
        updates["completed_at"] = datetime.now(timezone.utc).isoformat()

    target_job_id = job_id
    if updates:
        success = await store.update_job(job_id, updates)
        if not success:
            matched = await store.find_job_by_external_id(job_id, provider="replicate")
            if matched:
                target_job_id = matched.job_id
                success = await store.update_job(target_job_id, updates)
        if not success:
            logger.debug("Job %s not found in store, webhook ignored", job_id)

    # Emit admin events for terminal states
    if new_status == CloudJobStatus.COMPLETED.value:
        emit_cloud_event(
            "cloud.job.completed", target_job_id, f"Cloud job completed: {target_job_id[:12]}", severity="success"
        )
    elif new_status == CloudJobStatus.FAILED.value:
        error_msg = updates.get("error_message", "Unknown error")
        emit_cloud_event("cloud.job.failed", target_job_id, f"Cloud job failed: {error_msg}", severity="error")

    # Forward to SSE for live updates
    try:
        from ...services.sse_manager import get_sse_manager

        sse_manager = get_sse_manager()
        cloud_event = {
            "type": "cloud_job_update",
            "job_id": target_job_id,
            "event_type": event_type,
            "status": new_status,
            "updates": updates,
            "payload": {k: v for k, v in raw_payload.items() if k not in ("logs",)},
            "external_job_id": job_id,
        }
        asyncio.create_task(sse_manager.broadcast(cloud_event, event_type="cloud_job_update"))
    except Exception as exc:
        logger.debug("Could not broadcast to SSE: %s", exc)

    return {"status": "received", "job_id": job_id, "updates": updates}


@router.post("/webhook/test")
async def test_webhook(request: Request) -> Dict[str, Any]:
    """Test webhook connectivity before submitting expensive jobs.

    Performs a pre-flight check to verify the webhook URL is reachable.
    Uses the Replicate webhook-check cog when available (cheaper than GPU),
    otherwise falls back to a direct HTTP test.

    Parameters:
        webhook_url: The URL to test.
        provider: Cloud provider (default: "replicate").
        timeout: Request timeout in seconds (default: 30).
        method: Force a specific test method:
            - "direct": Test from this server (internal).
            - "replicate_cog": Test from Replicate's cloud (external).
            - omit/null: Auto-select (tries cog first for Replicate).
    """
    from typing import Literal

    from pydantic import BaseModel

    class TestRequest(BaseModel):
        webhook_url: str
        provider: str = "replicate"
        timeout: float = 30.0
        method: Optional[Literal["direct", "replicate_cog"]] = None

    body = await request.json()
    try:
        req = TestRequest.model_validate(body)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {exc}",
        )

    from ...services.cloud.webhook_test_service import get_webhook_test_service

    service = get_webhook_test_service()
    result = await service.test_webhook(
        webhook_url=req.webhook_url,
        provider=req.provider,
        timeout=req.timeout,
        force_method=req.method,
    )

    return {
        "success": result.success,
        "latency_ms": result.latency_ms,
        "status_code": result.status_code,
        "error": result.error,
        "method": result.provider_method,
    }


@router.get("/upload/progress/{upload_id}")
async def upload_progress_sse(upload_id: str):
    """SSE endpoint for monitoring upload progress."""
    store = get_job_store()

    async def event_generator():
        while True:
            progress = store.get_upload_progress(upload_id)
            if progress:
                event = f"data: {json.dumps(progress)}\n\n"
                yield event

                if progress.get("done") or progress.get("error"):
                    store.cleanup_upload_progress(upload_id)
                    break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )

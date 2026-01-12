"""Webhook connectivity testing service.

Provides pre-flight webhook checks to avoid expensive GPU job failures
due to unreachable webhook endpoints.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class WebhookTestResult:
    """Result of a webhook connectivity test."""

    success: bool
    latency_ms: Optional[float] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    provider_method: str = "direct"
    details: Optional[Dict[str, Any]] = None


class WebhookTestService:
    """Tests webhook connectivity before expensive cloud training jobs."""

    def __init__(self, timeout: float = 30.0):
        """Initialize the service.

        Args:
            timeout: Default timeout for webhook tests in seconds.
        """
        self.default_timeout = timeout

    async def test_webhook(
        self,
        webhook_url: str,
        provider: str = "generic",
        timeout: Optional[float] = None,
        force_method: Optional[str] = None,
    ) -> WebhookTestResult:
        """Test webhook connectivity.

        For Replicate: Uses simpletuner/webhook-check cog (cheap CPU) if available.
        For others: Direct HTTP POST test.

        Args:
            webhook_url: The webhook URL to test.
            provider: The cloud provider (affects test method).
            timeout: Request timeout in seconds.
            force_method: Force a specific test method ("direct" or "replicate_cog").
                If None, tries replicate_cog first for Replicate provider.

        Returns:
            WebhookTestResult with connectivity status.
        """
        timeout = timeout or self.default_timeout

        # Force direct method if requested
        if force_method == "direct":
            return await self._test_direct(webhook_url, timeout)

        # Force replicate cog method if requested
        if force_method == "replicate_cog":
            result = await self._test_via_replicate_cog(webhook_url, timeout)
            if result is not None:
                return result
            # Return error if cog method was forced but failed
            return WebhookTestResult(
                success=False,
                error="Replicate cog test unavailable. Ensure 'replicate' package is installed and API key is configured.",
                provider_method="replicate_cog",
            )

        # Default behavior: try cog first for Replicate, fall back to direct
        if provider == "replicate":
            try:
                result = await self._test_via_replicate_cog(webhook_url, timeout)
                if result is not None:
                    return result
            except Exception as exc:
                logger.debug("Replicate cog test failed, falling back to direct: %s", exc)

        # Fall back to direct test
        return await self._test_direct(webhook_url, timeout)

    async def _test_via_replicate_cog(
        self,
        webhook_url: str,
        timeout: float,
    ) -> Optional[WebhookTestResult]:
        """Test webhook using the Replicate webhook-check cog.

        This runs a minimal CPU-only prediction that sends a test POST
        to the webhook URL, simulating how Replicate will call the webhook.

        Args:
            webhook_url: The webhook URL to test.
            timeout: Timeout for the cog run.

        Returns:
            WebhookTestResult or None if cog unavailable.
        """
        try:
            import replicate
        except ImportError:
            logger.debug("replicate package not available, skipping cog test")
            return None

        try:
            # Run the webhook-check cog
            output = await replicate.async_run(
                "simpletuner/webhook-check",
                input={
                    "webhook_url": webhook_url,
                    "timeout_seconds": min(timeout, 120.0),
                },
            )

            # Parse the cog output
            if isinstance(output, dict):
                return WebhookTestResult(
                    success=output.get("success", False),
                    latency_ms=output.get("latency_ms"),
                    status_code=output.get("status_code"),
                    error=output.get("error"),
                    provider_method="replicate_cog",
                    details=output.get("details"),
                )

            logger.warning("Unexpected webhook-check output format: %s", type(output))
            return None

        except Exception as exc:
            logger.debug("Replicate cog webhook test error: %s", exc)
            return None

    async def _test_direct(
        self,
        webhook_url: str,
        timeout: float,
    ) -> WebhookTestResult:
        """Test webhook with direct HTTP POST.

        Sends a test payload directly to the webhook URL to verify
        connectivity without using cloud resources.

        Args:
            webhook_url: The webhook URL to test.
            timeout: Request timeout in seconds.

        Returns:
            WebhookTestResult with test outcome.
        """
        test_payload = {
            "event": "webhook_check",
            "status": "test",
            "message": "SimpleTuner webhook connectivity test",
            "timestamp": time.time(),
        }

        result = WebhookTestResult(
            success=False,
            provider_method="direct",
            details={"webhook_url": webhook_url, "timeout_seconds": timeout},
        )

        start_time = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    webhook_url,
                    json=test_payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-SimpleTuner-Event": "webhook_check",
                    },
                )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            result.latency_ms = round(elapsed_ms, 2)
            result.status_code = response.status_code

            if 200 <= response.status_code < 300:
                result.success = True
            else:
                result.error = f"HTTP {response.status_code}: {response.reason_phrase}"

        except httpx.ConnectError as exc:
            result.error = f"Connection failed: {exc}"
            if result.details:
                result.details["error_type"] = "connect_error"

        except httpx.TimeoutException:
            result.error = f"Request timed out after {timeout}s"
            if result.details:
                result.details["error_type"] = "timeout"

        except httpx.RequestError as exc:
            result.error = f"Request error: {exc}"
            if result.details:
                result.details["error_type"] = "request_error"

        except Exception as exc:
            result.error = f"Unexpected error: {type(exc).__name__}: {exc}"
            if result.details:
                result.details["error_type"] = "unexpected"

        return result


# Singleton instance
_service: Optional[WebhookTestService] = None


def get_webhook_test_service() -> WebhookTestService:
    """Get the singleton webhook test service instance."""
    global _service
    if _service is None:
        _service = WebhookTestService()
    return _service

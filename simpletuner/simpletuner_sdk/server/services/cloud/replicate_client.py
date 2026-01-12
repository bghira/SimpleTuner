"""Replicate Cog client for cloud training."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from .base import CloudJobInfo, CloudJobStatus, CloudTrainerService
from .secrets import get_secrets_manager

logger = logging.getLogger(__name__)

# Replicate API base URL
REPLICATE_API_BASE = "https://api.replicate.com/v1"

# Lazy import for credential resolver
_credential_resolver = None


def _get_credential_resolver():
    """Get the credential resolver (lazy import)."""
    global _credential_resolver
    if _credential_resolver is None:
        from .credential_resolver import get_credential_resolver

        _credential_resolver = get_credential_resolver()
    return _credential_resolver


# Default model for SimpleTuner on Replicate (version fetched dynamically)
DEFAULT_MODEL = "simpletuner/advanced-trainer"

# Default hardware info for cost estimation (fallback values if not configured)
# These are based on Replicate's published pricing as of 2024
DEFAULT_HARDWARE_INFO: Dict[str, Dict[str, Any]] = {
    "gpu-l40s": {"name": "L40S (48GB)", "cost_per_second": 0.000975},
    "gpu-a100-large": {"name": "A100 (80GB)", "cost_per_second": 0.001400},
}

# Cached hardware info (loaded from config or defaults)
_hardware_info_cache: Optional[Dict[str, Dict[str, Any]]] = None
_hardware_info_cache_time: Optional[float] = None
_HARDWARE_CACHE_TTL_SECONDS = 300  # 5 minute TTL
_hardware_cache_lock = asyncio.Lock()


async def get_hardware_info_async(store: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get hardware info, preferring configured values over defaults (async).

    Args:
        store: Optional JobStore instance for loading configured values

    Returns:
        Dict of hardware_id -> hardware info (name, cost_per_second)
    """
    global _hardware_info_cache, _hardware_info_cache_time

    # Check cache with TTL
    import time

    now = time.monotonic()
    if _hardware_info_cache is not None and _hardware_info_cache_time is not None:
        if now - _hardware_info_cache_time < _HARDWARE_CACHE_TTL_SECONDS:
            return _hardware_info_cache

    # Use lock to prevent concurrent cache updates
    async with _hardware_cache_lock:
        # Double-check after acquiring lock
        if _hardware_info_cache is not None and _hardware_info_cache_time is not None:
            if now - _hardware_info_cache_time < _HARDWARE_CACHE_TTL_SECONDS:
                return _hardware_info_cache

        # Try to load from provider config
        if store is not None:
            try:
                config = await store.get_provider_config("replicate")
                custom_hardware = config.get("hardware_info")
                if isinstance(custom_hardware, dict) and custom_hardware:
                    _hardware_info_cache = custom_hardware
                    _hardware_info_cache_time = now
                    logger.debug("Using configured hardware info: %s", list(custom_hardware.keys()))
                    return _hardware_info_cache
            except Exception as exc:
                logger.debug("Could not load hardware config: %s", exc)

        # Fall back to defaults
        _hardware_info_cache = DEFAULT_HARDWARE_INFO.copy()
        _hardware_info_cache_time = now
        return _hardware_info_cache


async def get_default_hardware_cost_per_hour(store: Optional[Any] = None) -> float:
    """
    Get the default hardware cost per hour (L40S).

    Args:
        store: Optional JobStore instance for loading configured values

    Returns:
        Cost per hour in USD
    """
    hardware = await get_hardware_info_async(store)
    l40s = hardware.get("gpu-l40s", DEFAULT_HARDWARE_INFO["gpu-l40s"])
    return l40s.get("cost_per_second", 0.000975) * 3600


def update_hardware_info_cache(hardware_info: Dict[str, Dict[str, Any]]) -> None:
    """
    Update the cached hardware info.

    Args:
        hardware_info: New hardware info dict
    """
    import time

    global _hardware_info_cache, _hardware_info_cache_time
    _hardware_info_cache = hardware_info
    _hardware_info_cache_time = time.monotonic()


def clear_hardware_info_cache() -> None:
    """Clear the hardware info cache to force reload from config."""
    global _hardware_info_cache, _hardware_info_cache_time
    _hardware_info_cache = None
    _hardware_info_cache_time = None


class ReplicateCogClient(CloudTrainerService):
    """Client for Replicate's Cog-based training API."""

    def __init__(self, version_override: Optional[str] = None):
        """
        Initialize the Replicate client.

        Args:
            version_override: Custom Cog deployment version (e.g., "username/model:hash")
        """
        self._version = version_override  # None means fetch latest
        self._model = DEFAULT_MODEL
        self._secrets = get_secrets_manager()
        self._http_client: Optional[httpx.AsyncClient] = None

    @property
    def _token(self) -> Optional[str]:
        """Get the API token from secrets manager (global fallback)."""
        return self._secrets.get_replicate_token()

    async def get_token_for_user(self, user_id: Optional[int] = None) -> Optional[str]:
        """Get the effective API token, checking user credentials first.

        Args:
            user_id: Optional user ID for per-user credential lookup

        Returns:
            API token (user's if available, otherwise global)
        """
        resolver = _get_credential_resolver()
        return await resolver.get_replicate_token(user_id)

    async def get_token_source(self, user_id: Optional[int] = None) -> Optional[str]:
        """Determine where the token would come from.

        Args:
            user_id: Optional user ID

        Returns:
            "user", "global", or None if no token available
        """
        resolver = _get_credential_resolver()
        return await resolver.get_credential_source(
            provider="replicate",
            credential_name="api_token",
            user_id=user_id,
            global_key="REPLICATE_API_TOKEN",
        )

    @property
    def provider_name(self) -> str:
        return "replicate"

    @property
    def supports_cost_tracking(self) -> bool:
        return True

    @property
    def supports_live_logs(self) -> bool:
        return True

    def _get_headers(self, token: Optional[str] = None) -> Dict[str, str]:
        """Get HTTP headers for Replicate API requests.

        Args:
            token: Optional specific token to use. If not provided, uses the global token.

        Returns:
            Headers dict with Authorization
        """
        effective_token = token or self._token
        return {
            "Authorization": f"Bearer {effective_token}",
            "Content-Type": "application/json",
        }

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create an async HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=60.0)
        return self._http_client

    async def get_latest_version(self, model_id: Optional[str] = None) -> str:
        """
        Get the latest version for a model.

        Args:
            model_id: Model ID in format "owner/model". Defaults to DEFAULT_MODEL.

        Returns:
            Full version string "owner/model:hash"

        Raises:
            ValueError: If no versions are available
        """
        versions = await self.list_model_versions(model_id or self._model)
        if not versions:
            raise ValueError(f"No versions available for model {model_id or self._model}")
        # versions are sorted newest first
        return versions[0]["full_version"]

    async def get_effective_version(self) -> str:
        """Get the version to use - either the override or fetch latest."""
        if self._version:
            return self._version
        return await self.get_latest_version()

    async def validate_credentials(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Validate credentials and return user info or error.

        Args:
            user_id: Optional user ID for per-user credential validation
        """
        token = await self.get_token_for_user(user_id)
        if not token:
            return {"error": "REPLICATE_API_TOKEN not set"}

        try:
            client = await self._get_http_client()
            response = await client.get(
                f"{REPLICATE_API_BASE}/account",
                headers=self._get_headers(token),
            )
            response.raise_for_status()
            account = response.json()

            # Include token source in response
            token_source = await self.get_token_source(user_id)

            return {
                "valid": True,
                "username": account.get("username"),
                "name": account.get("name"),
                "github_url": account.get("github_url"),
                "token_source": token_source,
            }
        except httpx.HTTPStatusError as exc:
            logger.warning("Failed to validate Replicate credentials: %s", exc)
            return {"error": f"HTTP {exc.response.status_code}: {exc.response.text}"}
        except Exception as exc:
            logger.warning("Failed to validate Replicate credentials: %s", exc)
            return {"error": str(exc)}

    async def list_jobs(self, limit: int = 50) -> List[CloudJobInfo]:
        """List recent predictions from Replicate."""
        if not self._token:
            return []

        try:
            client = await self._get_http_client()
            response = await client.get(
                f"{REPLICATE_API_BASE}/predictions",
                headers=self._get_headers(),
            )
            response.raise_for_status()
            data = response.json()

            # API returns paginated results with "results" key
            predictions = data.get("results", [])[:limit]

            jobs = []
            for pred in predictions:
                # Include all predictions from this account
                status = self._map_prediction_status(pred.get("status", "unknown"))
                created_at = pred.get("created_at")
                started_at = pred.get("started_at")
                completed_at = pred.get("completed_at")
                model_name = pred.get("model")

                # Calculate cost if metrics available
                cost = None
                metrics = pred.get("metrics") or {}
                if "predict_time" in metrics:
                    predict_time = metrics["predict_time"]
                    # Assume L40S pricing
                    cost = predict_time * DEFAULT_HARDWARE_INFO["gpu-l40s"]["cost_per_second"]

                jobs.append(
                    CloudJobInfo(
                        job_id=pred["id"],
                        provider="replicate",
                        status=status,
                        created_at=self._format_datetime(created_at),
                        started_at=self._format_datetime(started_at),
                        completed_at=self._format_datetime(completed_at),
                        cost_usd=cost,
                        hardware_type="L40S (48GB)",
                        logs_url=pred.get("logs"),
                        output_url=self._get_output_url_from_dict(pred),
                        metadata={"prediction_id": pred["id"], "model": model_name},
                    )
                )

            return jobs
        except Exception as exc:
            logger.warning("Failed to list Replicate jobs: %s", exc)
            return []

    async def run_job(
        self,
        config: Dict[str, Any],
        dataloader: List[Dict[str, Any]],
        data_archive_url: Optional[str] = None,
        webhook_url: Optional[str] = None,
        hf_token: Optional[str] = None,
        hub_model_id: Optional[str] = None,
        user_id: Optional[int] = None,
        lycoris_config: Optional[Dict[str, Any]] = None,
    ) -> CloudJobInfo:
        """Submit a new training job to Replicate.

        Args:
            config: Training configuration dict
            dataloader: Dataloader configuration list
            data_archive_url: Optional URL to data archive
            webhook_url: Optional webhook URL for status updates
            hf_token: Optional HuggingFace token for pushing
            hub_model_id: Optional HuggingFace Hub model ID
            user_id: Optional user ID for per-user API key delegation
            lycoris_config: Optional LyCORIS configuration dict
        """
        # Get effective token for this user
        token = await self.get_token_for_user(user_id)
        if not token:
            raise ValueError("REPLICATE_API_TOKEN not set (neither user nor global)")

        try:
            client = await self._get_http_client()

            # Get effective version (override or fetch latest)
            effective_version = await self.get_effective_version()
            logger.info("Using Replicate model version: %s", effective_version)

            # Prepare input for the Cog predictor
            prediction_input: Dict[str, Any] = {}

            # Serialize config to JSON string
            prediction_input["config_json"] = json.dumps(config)

            # Serialize dataloader config to JSON string
            prediction_input["dataloader_json"] = json.dumps(dataloader)

            # Serialize lycoris config if provided
            if lycoris_config:
                prediction_input["lycoris_json"] = json.dumps(lycoris_config)

            # Add data archive if provided
            if data_archive_url:
                prediction_input["images"] = data_archive_url

            # Add HuggingFace Hub params if provided
            if hf_token:
                prediction_input["hf_token"] = hf_token
            if hub_model_id:
                prediction_input["hub_model_id"] = hub_model_id

            # Add webhook config if provided (passed to Cog, not Replicate's platform webhook)
            if webhook_url:
                webhook_config_entry = {
                    "url": webhook_url,
                    "events": ["all"],
                }
                prediction_input["webhook_config"] = json.dumps([webhook_config_entry])

            # Build prediction create payload
            payload: Dict[str, Any] = {
                "version": effective_version,
                "input": prediction_input,
            }

            # Submit prediction
            response = await client.post(
                f"{REPLICATE_API_BASE}/predictions",
                headers=self._get_headers(token),
                json=payload,
            )
            response.raise_for_status()
            prediction = response.json()

            status = self._map_prediction_status(prediction.get("status", "starting"))

            return CloudJobInfo(
                job_id=prediction["id"],
                provider="replicate",
                status=status,
                created_at=self._format_datetime(prediction.get("created_at")) or datetime.now(timezone.utc).isoformat(),
                hardware_type="L40S (48GB)",
                metadata={
                    "prediction_id": prediction["id"],
                    "webhook_url": webhook_url,
                },
            )
        except httpx.HTTPStatusError as exc:
            logger.error("Failed to submit Replicate job: HTTP %s: %s", exc.response.status_code, exc.response.text)
            raise ValueError(f"Replicate API error: {exc.response.text}")
        except Exception as exc:
            logger.error("Failed to submit Replicate job: %s", exc)
            raise

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running prediction."""
        if not self._token:
            return False

        try:
            client = await self._get_http_client()
            response = await client.post(
                f"{REPLICATE_API_BASE}/predictions/{job_id}/cancel",
                headers=self._get_headers(),
            )
            response.raise_for_status()
            return True
        except httpx.HTTPStatusError as exc:
            # 404 means job doesn't exist on Replicate - treat as success
            if exc.response.status_code == 404:
                logger.info("Replicate job %s not found (already gone or never existed)", job_id)
                return True
            logger.warning("Failed to cancel Replicate job %s: %s", job_id, exc)
            return False
        except Exception as exc:
            logger.warning("Failed to cancel Replicate job %s: %s", job_id, exc)
            return False

    async def get_job_logs(self, job_id: str) -> str:
        """Fetch logs for a prediction."""
        if not self._token:
            return "(No API token configured)"

        try:
            client = await self._get_http_client()
            response = await client.get(
                f"{REPLICATE_API_BASE}/predictions/{job_id}",
                headers=self._get_headers(),
            )
            response.raise_for_status()
            prediction = response.json()

            logs = prediction.get("logs")
            if logs:
                return logs

            # If no logs attribute, try to get from output
            output = prediction.get("output")
            if output and isinstance(output, str):
                return output

            return "(No logs available)"
        except Exception as exc:
            logger.warning("Failed to fetch logs for job %s: %s", job_id, exc)
            return f"(Error fetching logs: {exc})"

    async def get_job_status(self, job_id: str) -> CloudJobInfo:
        """Get current status of a prediction."""
        if not self._token:
            raise ValueError("REPLICATE_API_TOKEN not set")

        try:
            client = await self._get_http_client()
            response = await client.get(
                f"{REPLICATE_API_BASE}/predictions/{job_id}",
                headers=self._get_headers(),
            )
            response.raise_for_status()
            prediction = response.json()

            status = self._map_prediction_status(prediction.get("status", "unknown"))
            created_at = prediction.get("created_at")
            started_at = prediction.get("started_at")
            completed_at = prediction.get("completed_at")

            # Calculate cost
            cost = None
            metrics = prediction.get("metrics") or {}
            if "predict_time" in metrics:
                predict_time = metrics["predict_time"]
                cost = predict_time * DEFAULT_HARDWARE_INFO["gpu-l40s"]["cost_per_second"]

            error_msg = None
            if status == CloudJobStatus.FAILED:
                error_msg = prediction.get("error")

            return CloudJobInfo(
                job_id=prediction["id"],
                provider="replicate",
                status=status,
                created_at=self._format_datetime(created_at),
                started_at=self._format_datetime(started_at),
                completed_at=self._format_datetime(completed_at),
                cost_usd=cost,
                hardware_type="L40S (48GB)",
                logs_url=prediction.get("logs"),
                output_url=self._get_output_url_from_dict(prediction),
                error_message=error_msg,
                metadata={"prediction_id": prediction["id"]},
            )
        except Exception as exc:
            logger.error("Failed to get status for job %s: %s", job_id, exc)
            raise

    async def get_billing_info(self) -> Dict[str, Any]:
        """
        Get billing/hardware information for Replicate.

        Note: Credit balance is not available via Replicate's public API.
        Use the Replicate dashboard to check your balance.
        """
        if not self._token:
            return {
                "balance": None,
                "error": "No API token configured",
            }

        # Return hardware info - balance must be checked on Replicate dashboard
        return {
            "balance": None,
            "balance_note": "Check balance on replicate.com/account",
            "hardware": DEFAULT_HARDWARE_INFO,
            "default_hardware": "gpu-l40s",
        }

    async def list_model_versions(self, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available versions for a model.

        Args:
            model_id: Model ID in format "owner/model". Defaults to the configured model.

        Returns:
            List of version info dicts sorted by created_at descending (newest first).
        """
        if not self._token:
            return []

        # Use provided model_id, or extract from version override, or use default model
        if not model_id:
            if self._version and ":" in self._version:
                model_id = self._version.split(":")[0]
            else:
                model_id = self._model

        try:
            client = await self._get_http_client()

            # Split model_id into owner/name
            if "/" not in model_id:
                logger.warning("Invalid model_id format (expected owner/name): %s", model_id)
                return []

            response = await client.get(
                f"{REPLICATE_API_BASE}/models/{model_id}/versions",
                headers=self._get_headers(),
            )
            response.raise_for_status()
            data = response.json()

            # API returns paginated results with "results" key
            versions = data.get("results", [])

            # Convert to dicts and sort by created_at descending
            result = []
            for v in versions:
                created_at = v.get("created_at")
                version_id = v.get("id", "")
                result.append(
                    {
                        "id": version_id,
                        "created_at": self._format_datetime(created_at),
                        "model_id": model_id,
                        "full_version": f"{model_id}:{version_id}",
                    }
                )

            # Sort by created_at descending (newest first)
            result.sort(key=lambda x: x.get("created_at") or "", reverse=True)
            return result

        except Exception as exc:
            logger.warning("Failed to list model versions for %s: %s", model_id, exc)
            return []

    def _map_prediction_status(self, status: str) -> CloudJobStatus:
        """Map Replicate prediction status to CloudJobStatus.

        Uses CloudJobStatus.from_external() to handle external API variations.
        """
        return CloudJobStatus.from_external(status)

    def _format_datetime(self, dt: Any) -> Optional[str]:
        """Format datetime to ISO string."""
        if dt is None:
            return None
        if isinstance(dt, str):
            return dt
        if hasattr(dt, "isoformat"):
            return dt.isoformat()
        return str(dt)

    def _get_output_url_from_dict(self, prediction: Dict[str, Any]) -> Optional[str]:
        """Extract output URL from prediction dict."""
        output = prediction.get("output")
        if isinstance(output, str) and output.startswith("http"):
            return output
        if isinstance(output, list) and output:
            first = output[0]
            if isinstance(first, str) and first.startswith("http"):
                return first
        return None

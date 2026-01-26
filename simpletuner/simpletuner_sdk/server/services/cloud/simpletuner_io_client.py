"""SimpleTuner.io client for cloud training."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

from .base import CloudJobInfo, CloudJobStatus, CloudTrainerService
from .exceptions import (
    InvalidConfigError,
    InvalidTokenError,
    JobNotFoundError,
    MissingTokenError,
    ProviderAPIError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from .http_client import get_async_client
from .secrets import get_secrets_manager
from .storage.provider_config_store import ProviderConfigStore

logger = logging.getLogger(__name__)

PROVIDER_NAME = "simpletuner_io"
DEFAULT_BASE_URL = "https://simpletuner.io"
BASE_URL_ENV = "SIMPLETUNER_IO_API_BASE"
REFRESH_TOKEN_KEY = "SIMPLETUNER_IO_REFRESH_TOKEN"
ACCESS_TOKEN_SKEW_SECONDS = 60

CONFIG_ORG_ID_KEY = "org_id"
CONFIG_ACCESS_TOKEN_KEY = "access_token"
CONFIG_ACCESS_TOKEN_EXPIRES_AT_KEY = "access_token_expires_at"
CONFIG_ACCESS_TOKEN_ORG_KEY = "access_token_org_id"
CONFIG_API_BASE_URL_KEY = "api_base_url"
CONFIG_MAX_RUNTIME_MINUTES_KEY = "max_runtime_minutes"


class SimpleTunerIOClient(CloudTrainerService):
    """Client for SimpleTuner.io cloud training."""

    def __init__(self):
        self._config_store = ProviderConfigStore()
        self._token_lock = asyncio.Lock()
        self._secrets = get_secrets_manager()

    @property
    def provider_name(self) -> str:
        return PROVIDER_NAME

    @property
    def supports_cost_tracking(self) -> bool:
        return True

    @property
    def supports_live_logs(self) -> bool:
        return True

    def _get_base_url(self, config: Dict[str, Any]) -> str:
        base_url = config.get(CONFIG_API_BASE_URL_KEY) or os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL
        return str(base_url).rstrip("/")

    def _build_url(self, config: Dict[str, Any], path: str) -> str:
        return f"{self._get_base_url(config)}{path}"

    def _parse_datetime(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    def _parse_expires_at(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value)
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                return None
        return None

    def _is_token_valid(self, token: str, expires_at: Optional[datetime], org_id: Optional[str], token_org: Optional[str]) -> bool:
        if not token or not expires_at:
            return False
        if token_org and org_id and token_org != org_id:
            return False
        now = datetime.now(timezone.utc)
        return now + timedelta(seconds=ACCESS_TOKEN_SKEW_SECONDS) < expires_at

    async def _exchange_token(self, refresh_token: str, org_id: str) -> Dict[str, Any]:
        payload = {"refresh_token": refresh_token, "org_id": org_id}
        config = await self._config_store.get(PROVIDER_NAME)
        url = self._build_url(config, "/v1/auth/cli/token")
        try:
            async with get_async_client(timeout=30.0, circuit_breaker_name="simpletuner-io-auth") as client:
                response = await client.post(url, json=payload, headers={"Accept": "application/json"})
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(PROVIDER_NAME, f"Token exchange timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(PROVIDER_NAME, f"Token exchange failed: {exc}") from exc

        if response.status_code == 401:
            raise InvalidTokenError("SimpleTuner.io refresh token is invalid or expired.")
        if response.status_code >= 400:
            message = self._extract_error_message(response)
            raise ProviderAPIError(PROVIDER_NAME, message, status_code=response.status_code)

        data = response.json()
        if not data.get("access_token"):
            raise ProviderAPIError(PROVIDER_NAME, "Token exchange response missing access_token.")
        return data

    def _extract_error_message(self, response: httpx.Response) -> str:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                detail = payload.get("detail") or payload.get("message") or payload.get("error")
                if detail:
                    return str(detail)
        except ValueError:
            pass
        return f"HTTP {response.status_code}: {response.text}"

    async def _invalidate_access_token(self) -> None:
        await self._config_store.update(
            PROVIDER_NAME,
            {
                CONFIG_ACCESS_TOKEN_KEY: None,
                CONFIG_ACCESS_TOKEN_EXPIRES_AT_KEY: None,
                CONFIG_ACCESS_TOKEN_ORG_KEY: None,
            },
        )

    async def _get_access_token(self) -> str:
        config = await self._config_store.get(PROVIDER_NAME)
        org_id = config.get(CONFIG_ORG_ID_KEY)
        if not org_id:
            raise InvalidConfigError("SimpleTuner.io org_id is not configured.")

        token = config.get(CONFIG_ACCESS_TOKEN_KEY)
        expires_at = self._parse_expires_at(config.get(CONFIG_ACCESS_TOKEN_EXPIRES_AT_KEY))
        token_org = config.get(CONFIG_ACCESS_TOKEN_ORG_KEY)
        if token and self._is_token_valid(token, expires_at, org_id, token_org):
            return token

        async with self._token_lock:
            config = await self._config_store.get(PROVIDER_NAME)
            org_id = config.get(CONFIG_ORG_ID_KEY)
            if not org_id:
                raise InvalidConfigError("SimpleTuner.io org_id is not configured.")

            token = config.get(CONFIG_ACCESS_TOKEN_KEY)
            expires_at = self._parse_expires_at(config.get(CONFIG_ACCESS_TOKEN_EXPIRES_AT_KEY))
            token_org = config.get(CONFIG_ACCESS_TOKEN_ORG_KEY)
            if token and self._is_token_valid(token, expires_at, org_id, token_org):
                return token

            refresh_token = self._secrets.get(REFRESH_TOKEN_KEY)
            if not refresh_token:
                raise MissingTokenError("SimpleTuner.io refresh token is not configured.")

            exchange = await self._exchange_token(refresh_token, org_id)
            expires_in = exchange.get("expires_in")
            if not isinstance(expires_in, (int, float)):
                raise ProviderAPIError(PROVIDER_NAME, "Token exchange response missing expires_in.")

            expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
            access_token = exchange["access_token"]
            await self._config_store.update(
                PROVIDER_NAME,
                {
                    CONFIG_ACCESS_TOKEN_KEY: access_token,
                    CONFIG_ACCESS_TOKEN_EXPIRES_AT_KEY: expires_at.isoformat(),
                    CONFIG_ACCESS_TOKEN_ORG_KEY: org_id,
                },
            )
            return access_token

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        requires_auth: bool = True,
        retry_on_unauthorized: bool = True,
    ) -> Dict[str, Any]:
        config = await self._config_store.get(PROVIDER_NAME)
        token = await self._get_access_token() if requires_auth else None
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        url = self._build_url(config, path)

        try:
            async with get_async_client(timeout=30.0, circuit_breaker_name="simpletuner-io-api") as client:
                response = await client.request(method, url, json=json_body, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(PROVIDER_NAME, f"Request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(PROVIDER_NAME, f"Request failed: {exc}") from exc

        if response.status_code == 401 and requires_auth and retry_on_unauthorized:
            await self._invalidate_access_token()
            return await self._request(
                method,
                path,
                json_body=json_body,
                requires_auth=requires_auth,
                retry_on_unauthorized=False,
            )

        if response.status_code == 404:
            if path.startswith("/v1/jobs/"):
                raise JobNotFoundError(path.split("/")[-1])
            raise ProviderAPIError(PROVIDER_NAME, "Resource not found.", status_code=response.status_code)

        if response.status_code >= 400:
            message = self._extract_error_message(response)
            if response.status_code == 401:
                raise InvalidTokenError(message)
            raise ProviderAPIError(PROVIDER_NAME, message, status_code=response.status_code)

        if response.status_code == 204 or not response.content:
            return {}
        return response.json()

    def _map_status(self, status: Optional[str]) -> CloudJobStatus:
        if not status:
            return CloudJobStatus.PENDING
        status_lower = status.lower().strip()
        status_map = {
            "pending": CloudJobStatus.PENDING,
            "created": CloudJobStatus.PENDING,
            "provisioning": CloudJobStatus.PENDING,
            "waiting_for_capacity": CloudJobStatus.QUEUED,
            "queued": CloudJobStatus.QUEUED,
            "running": CloudJobStatus.RUNNING,
            "training": CloudJobStatus.RUNNING,
            "retrying": CloudJobStatus.PENDING,
            "cancelling": CloudJobStatus.RUNNING,
            "canceled": CloudJobStatus.CANCELLED,
            "cancelled": CloudJobStatus.CANCELLED,
            "completed": CloudJobStatus.COMPLETED,
            "failed": CloudJobStatus.FAILED,
            "preempted": CloudJobStatus.FAILED,
        }
        return status_map.get(status_lower, CloudJobStatus.from_external(status_lower))

    async def validate_credentials(self) -> Dict[str, Any]:
        """Validate credentials and return user/org info."""
        try:
            data = await self._request("GET", "/v1/account/me")
            return {
                "valid": True,
                "user_id": data.get("user_id"),
                "org_id": data.get("org_id"),
                "org_role": data.get("org_role"),
                "email": data.get("email"),
            }
        except Exception as exc:
            logger.warning("Failed to validate SimpleTuner.io credentials: %s", exc)
            return {"error": str(exc)}

    async def list_jobs(self, limit: int = 50) -> List[CloudJobInfo]:
        """List recent jobs from SimpleTuner.io."""
        try:
            response = await self._request("GET", f"/v1/jobs?page_size={limit}")
        except (MissingTokenError, InvalidConfigError):
            return []

        items = response.get("items", []) if isinstance(response, dict) else []
        jobs: List[CloudJobInfo] = []
        for item in items[:limit]:
            status = self._map_status(item.get("current_attempt_status") or item.get("status"))
            jobs.append(
                CloudJobInfo(
                    job_id=item.get("id"),
                    provider=PROVIDER_NAME,
                    status=status,
                    created_at=self._parse_datetime(item.get("created_at")) or datetime.now(timezone.utc).isoformat(),
                    completed_at=None,
                    cost_usd=None,
                    hardware_type=None,
                    metadata={"current_attempt_status": item.get("current_attempt_status")},
                )
            )
        return jobs

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
        """Submit a new training job to SimpleTuner.io."""
        provider_config = await self._config_store.get(PROVIDER_NAME)
        max_runtime_minutes = (
            config.get("--max_runtime_minutes")
            or config.get("max_runtime_minutes")
            or provider_config.get(CONFIG_MAX_RUNTIME_MINUTES_KEY)
        )
        if not max_runtime_minutes:
            raise InvalidConfigError("max_runtime_minutes is required for SimpleTuner.io jobs.")

        payload: Dict[str, Any] = {
            "max_runtime_minutes": int(max_runtime_minutes),
            "config": config,
            "dataloader_config": dataloader,
        }
        if lycoris_config:
            payload["lycoris_config"] = lycoris_config
        resume_key = config.get("--resume_from_checkpoint") or config.get("resume_from_checkpoint")
        if resume_key:
            payload["resume_from_checkpoint"] = resume_key

        response = await self._request("POST", "/v1/jobs", json_body=payload)
        job_id = response.get("id")
        if not job_id:
            raise ProviderAPIError(PROVIDER_NAME, "Job submission response missing job id.")
        status = self._map_status(response.get("status"))
        created_at = datetime.now(timezone.utc).isoformat()

        return CloudJobInfo(
            job_id=job_id,
            provider=PROVIDER_NAME,
            status=status,
            created_at=created_at,
            metadata={
                "attempt_id": response.get("attempt_id"),
                "estimated_cost_cents": response.get("estimated_cost_cents"),
                "estimated_total_cents": response.get("estimated_total_cents"),
            },
        )

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        try:
            response = await self._request("POST", f"/v1/jobs/{job_id}/cancel")
        except ProviderAPIError as exc:
            status_code = exc.details.get("provider_status_code")
            if status_code == 409:
                return False
            raise
        status = response.get("status")
        return status in {"cancelling", "canceled", "cancelled"}

    async def get_job_logs(self, job_id: str) -> str:
        """Fetch logs for a job."""
        response = await self._request("GET", f"/v1/jobs/{job_id}/logs")
        logs = response.get("logs")
        return logs or ""

    async def get_job_status(self, job_id: str) -> CloudJobInfo:
        """Get current status of a job."""
        response = await self._request("GET", f"/v1/jobs/{job_id}")
        job = response.get("job", {}) if isinstance(response, dict) else {}
        attempts = response.get("attempts", []) if isinstance(response, dict) else []

        latest_attempt = None
        if attempts:
            latest_attempt = max(attempts, key=lambda item: item.get("attempt_number", 0))

        status_source = None
        if latest_attempt:
            status_source = latest_attempt.get("status")
        if not status_source:
            status_source = job.get("status")

        status = self._map_status(status_source)
        started_at = None
        completed_at = None
        error_message = None
        if latest_attempt:
            started_at = self._parse_datetime(
                latest_attempt.get("training_started_at") or latest_attempt.get("started_at")
            )
            completed_at = self._parse_datetime(latest_attempt.get("ended_at"))
            error_message = latest_attempt.get("failure_reason")

        return CloudJobInfo(
            job_id=job.get("id") or job_id,
            provider=PROVIDER_NAME,
            status=status,
            created_at=self._parse_datetime(job.get("created_at")) or datetime.now(timezone.utc).isoformat(),
            started_at=started_at,
            completed_at=completed_at,
            error_message=error_message,
            metadata={"attempt_id": latest_attempt.get("id") if latest_attempt else None},
        )

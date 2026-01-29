"""Provider configuration endpoints for cloud training."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from ...services.cloud.base import CloudJobStatus
from ...services.cloud.factory import ProviderFactory
from ...services.cloud.secrets import get_secrets_manager
from ._shared import (
    ExternalWebhookTestResponse,
    ModelVersionInfo,
    ModelVersionsResponse,
    ProviderConfigResponse,
    ProviderConfigUpdate,
    ProvidersListResponse,
    PublishingStatusResponse,
    ValidateResponse,
    WebhookTestResponse,
    get_active_config,
    get_job_store,
    get_local_upload_dir,
    validate_webhook_url,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Default webhook-check Cog model
WEBHOOK_CHECK_MODEL = "simpletuner/webhook-check"


@router.get("/providers", response_model=ProvidersListResponse)
async def list_providers() -> ProvidersListResponse:
    """List available cloud providers."""
    from ...services.cloud.provider_registry import get_enriched_providers

    providers = await get_enriched_providers()
    return ProvidersListResponse(providers=providers)


@router.get("/providers/{provider}/config", response_model=ProviderConfigResponse)
async def get_provider_config(provider: str) -> ProviderConfigResponse:
    """Get configuration for a specific provider."""
    store = get_job_store()
    config = await store.get_provider_config(provider)
    return ProviderConfigResponse(provider=provider, config=config)


@router.put("/providers/{provider}/config", response_model=ProviderConfigResponse)
async def update_provider_config(provider: str, update: ProviderConfigUpdate) -> ProviderConfigResponse:
    """Update configuration for a specific provider."""
    import ipaddress

    store = get_job_store()
    config = await store.get_provider_config(provider)

    if update.version_override is not None:
        config["version_override"] = update.version_override or None
    if update.webhook_url is not None:
        config["webhook_url"] = validate_webhook_url(update.webhook_url)
    if update.preferences is not None:
        config["preferences"] = update.preferences

    # Cost limit settings
    if update.cost_limit_enabled is not None:
        config["cost_limit_enabled"] = update.cost_limit_enabled
    if update.cost_limit_amount is not None:
        if update.cost_limit_amount < 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cost limit amount must be non-negative")
        config["cost_limit_amount"] = update.cost_limit_amount
    if update.cost_limit_period is not None:
        if update.cost_limit_period not in {"daily", "weekly", "monthly"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Cost limit period must be 'daily', 'weekly', or 'monthly'"
            )
        config["cost_limit_period"] = update.cost_limit_period
    if update.cost_limit_action is not None:
        if update.cost_limit_action not in {"warn", "block"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Cost limit action must be 'warn' or 'block'"
            )
        config["cost_limit_action"] = update.cost_limit_action

    # Hardware pricing configuration
    if update.hardware_info is not None:
        for hw_id, hw_info in update.hardware_info.items():
            if not isinstance(hw_info, dict):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=f"Hardware info for '{hw_id}' must be a dict"
                )
            if "cost_per_second" not in hw_info:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Hardware info for '{hw_id}' must include 'cost_per_second'",
                )
            if not isinstance(hw_info["cost_per_second"], (int, float)) or hw_info["cost_per_second"] < 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"cost_per_second for '{hw_id}' must be a non-negative number",
                )
        config["hardware_info"] = update.hardware_info

        from ...services.cloud.replicate_client import clear_hardware_info_cache

        clear_hardware_info_cache()

    # Security settings
    if update.webhook_require_signature is not None:
        config["webhook_require_signature"] = update.webhook_require_signature
    if update.webhook_allowed_ips is not None:
        validated_ips = []
        for ip_str in update.webhook_allowed_ips:
            try:
                if "/" in ip_str:
                    ipaddress.ip_network(ip_str, strict=False)
                else:
                    ipaddress.ip_address(ip_str)
                validated_ips.append(ip_str)
            except ValueError:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid IP address or CIDR: {ip_str}")
        config["webhook_allowed_ips"] = validated_ips

    # TLS settings
    if update.ssl_verify is not None:
        config["ssl_verify"] = update.ssl_verify
    if update.ssl_ca_bundle is not None:
        if update.ssl_ca_bundle:
            ca_path = Path(update.ssl_ca_bundle).expanduser()
            if not ca_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=f"CA bundle path does not exist: {update.ssl_ca_bundle}"
                )
            if not ca_path.is_file():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=f"CA bundle path is not a file: {update.ssl_ca_bundle}"
                )
            config["ssl_ca_bundle"] = str(ca_path.resolve())
        else:
            config.pop("ssl_ca_bundle", None)

    # Proxy settings
    if update.proxy_url is not None:
        if update.proxy_url:
            config["proxy_url"] = update.proxy_url
        else:
            config.pop("proxy_url", None)

    # HTTP timeout
    if update.http_timeout is not None:
        if update.http_timeout > 0:
            config["http_timeout"] = update.http_timeout
        else:
            config.pop("http_timeout", None)

    # Rate limiting settings
    if update.webhook_rate_limit_max is not None:
        config["webhook_rate_limit_max"] = update.webhook_rate_limit_max
    if update.webhook_rate_limit_window is not None:
        config["webhook_rate_limit_window"] = update.webhook_rate_limit_window
    if update.s3_rate_limit_max is not None:
        config["s3_rate_limit_max"] = update.s3_rate_limit_max
    if update.s3_rate_limit_window is not None:
        config["s3_rate_limit_window"] = update.s3_rate_limit_window

    # SimpleTuner.io settings
    if update.org_id is not None:
        org_id = update.org_id.strip() if isinstance(update.org_id, str) else update.org_id
        if org_id:
            config["org_id"] = org_id
        else:
            config.pop("org_id", None)
    if update.api_base_url is not None:
        if update.api_base_url:
            from urllib.parse import urlparse

            parsed = urlparse(update.api_base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="api_base_url must include scheme and host (e.g., https://simpletuner.io)",
                )
            config["api_base_url"] = update.api_base_url.rstrip("/")
        else:
            config.pop("api_base_url", None)
    if update.max_runtime_minutes is not None:
        config["max_runtime_minutes"] = update.max_runtime_minutes

    await store.save_provider_config(provider, config)

    # Reload HTTP client factory to pick up new settings
    from ...services.cloud.http_client import HTTPClientFactory

    HTTPClientFactory.reset()

    # Reconfigure rate limiters with new settings
    from ._shared import configure_rate_limits_from_provider

    await configure_rate_limits_from_provider(provider)

    return ProviderConfigResponse(provider=provider, config=config)


class AdvancedConfigResponse(BaseModel):
    """Response for advanced provider configuration."""

    ssl_verify: bool = True
    ssl_ca_bundle: Optional[str] = None
    proxy_url: Optional[str] = None
    http_timeout: int = 30
    webhook_ip_allowlist_enabled: bool = False
    webhook_allowed_ips: List[str] = []


class AdvancedConfigUpdate(BaseModel):
    """Partial update for advanced configuration."""

    ssl_verify: Optional[bool] = None
    ssl_ca_bundle: Optional[str] = None
    proxy_url: Optional[str] = None
    http_timeout: Optional[int] = None
    webhook_ip_allowlist_enabled: Optional[bool] = None
    webhook_allowed_ips: Optional[List[str]] = None


@router.get("/providers/{provider}/advanced", response_model=AdvancedConfigResponse)
async def get_advanced_config(provider: str) -> AdvancedConfigResponse:
    """Get advanced configuration for a provider."""
    store = get_job_store()
    config = await store.get_provider_config(provider)

    return AdvancedConfigResponse(
        ssl_verify=config.get("ssl_verify", True),
        ssl_ca_bundle=config.get("ssl_ca_bundle"),
        proxy_url=config.get("proxy_url"),
        http_timeout=config.get("http_timeout", 30),
        webhook_ip_allowlist_enabled=bool(config.get("webhook_allowed_ips")),
        webhook_allowed_ips=config.get("webhook_allowed_ips", []),
    )


@router.patch("/providers/{provider}/advanced", response_model=AdvancedConfigResponse)
async def update_advanced_config(provider: str, update: AdvancedConfigUpdate) -> AdvancedConfigResponse:
    """Update advanced configuration for a provider."""
    import ipaddress

    store = get_job_store()
    config = await store.get_provider_config(provider)

    if update.ssl_verify is not None:
        config["ssl_verify"] = update.ssl_verify

    if update.ssl_ca_bundle is not None:
        if update.ssl_ca_bundle:
            ca_path = Path(update.ssl_ca_bundle).expanduser()
            if not ca_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=f"CA bundle path does not exist: {update.ssl_ca_bundle}"
                )
            config["ssl_ca_bundle"] = str(ca_path.resolve())
        else:
            config.pop("ssl_ca_bundle", None)

    if update.proxy_url is not None:
        if update.proxy_url:
            config["proxy_url"] = update.proxy_url
        else:
            config.pop("proxy_url", None)

    if update.http_timeout is not None:
        if update.http_timeout > 0:
            config["http_timeout"] = update.http_timeout
        else:
            config.pop("http_timeout", None)

    if update.webhook_ip_allowlist_enabled is not None:
        # Just track the intent; actual IPs control the allowlist
        pass

    if update.webhook_allowed_ips is not None:
        validated_ips = []
        for ip_str in update.webhook_allowed_ips:
            try:
                if "/" in ip_str:
                    ipaddress.ip_network(ip_str, strict=False)
                else:
                    ipaddress.ip_address(ip_str)
                validated_ips.append(ip_str)
            except ValueError:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid IP address or CIDR: {ip_str}")
        config["webhook_allowed_ips"] = validated_ips

    await store.save_provider_config(provider, config)

    # Reload HTTP client factory to pick up new settings
    from ...services.cloud.http_client import HTTPClientFactory

    HTTPClientFactory.reset()

    return AdvancedConfigResponse(
        ssl_verify=config.get("ssl_verify", True),
        ssl_ca_bundle=config.get("ssl_ca_bundle"),
        proxy_url=config.get("proxy_url"),
        http_timeout=config.get("http_timeout", 30),
        webhook_ip_allowlist_enabled=bool(config.get("webhook_allowed_ips")),
        webhook_allowed_ips=config.get("webhook_allowed_ips", []),
    )


@router.post("/providers/{provider}/test-webhook", response_model=WebhookTestResponse)
async def test_webhook_connection(provider: str) -> WebhookTestResponse:
    """Test webhook connection by sending a test event."""
    from ...services.cloud.http_client import get_async_client

    store = get_job_store()
    config = await store.get_provider_config(provider)
    webhook_url = config.get("webhook_url")

    if not webhook_url:
        return WebhookTestResponse(
            success=False, message="No webhook URL configured", error="Configure a webhook URL in provider settings first"
        )

    try:
        test_payload = {
            "event_type": "test.connection",
            "job_id": "test-webhook-connection",
            "message": "SimpleTuner webhook connection test",
            "timestamp": time.time(),
        }

        start_time = time.time()

        async with get_async_client(timeout=10.0) as client:
            response = await client.post(
                webhook_url,
                json=test_payload,
                headers={"Content-Type": "application/json", "X-SimpleTuner-Test": "true"},
            )

        latency_ms = (time.time() - start_time) * 1000

        if response.status_code < 400:
            return WebhookTestResponse(
                success=True, message=f"Webhook reachable (HTTP {response.status_code})", latency_ms=round(latency_ms, 2)
            )
        else:
            return WebhookTestResponse(
                success=False,
                message="Webhook returned error status",
                latency_ms=round(latency_ms, 2),
                error=f"HTTP {response.status_code}: {response.text[:200]}",
            )

    except Exception as exc:
        if "ConnectError" in type(exc).__name__:
            return WebhookTestResponse(
                success=False, message="Connection failed", error=f"Could not connect to {webhook_url}: {exc}"
            )
        elif "TimeoutException" in type(exc).__name__:
            return WebhookTestResponse(
                success=False, message="Connection timeout", error="Request timed out after 10 seconds"
            )
        return WebhookTestResponse(success=False, message="Test failed", error=str(exc))


@router.post("/providers/{provider}/test-webhook-external", response_model=ExternalWebhookTestResponse)
async def test_webhook_external(provider: str) -> ExternalWebhookTestResponse:
    """Test webhook reachability from Replicate's infrastructure."""
    from ...services.cloud.http_client import get_async_client

    if provider != "replicate":
        return ExternalWebhookTestResponse(
            success=False,
            message="External test only supported for Replicate",
            error=f"Provider '{provider}' does not support external webhook testing",
        )

    store = get_job_store()
    config = await store.get_provider_config(provider)
    webhook_url = config.get("webhook_url")

    if not webhook_url:
        return ExternalWebhookTestResponse(
            success=False, message="No webhook URL configured", error="Configure a webhook URL in provider settings first"
        )

    api_token = get_secrets_manager().get_replicate_token()
    if not api_token:
        return ExternalWebhookTestResponse(
            success=False, message="No API token", error="REPLICATE_API_TOKEN environment variable not set"
        )

    try:
        webhook_check_model = config.get("webhook_check_model", WEBHOOK_CHECK_MODEL)

        async with get_async_client(timeout=120.0) as client:
            model_response = await client.get(
                f"https://api.replicate.com/v1/models/{webhook_check_model}",
                headers={"Authorization": f"Bearer {api_token}"},
            )

            if model_response.status_code == 404:
                return ExternalWebhookTestResponse(
                    success=False,
                    message="Webhook check model not found",
                    error=f"Model '{webhook_check_model}' not found. Deploy the webhook-check Cog first.",
                )
            elif model_response.status_code != 200:
                return ExternalWebhookTestResponse(
                    success=False,
                    message="Failed to get model info",
                    error=f"HTTP {model_response.status_code}: {model_response.text[:200]}",
                )

            model_data = model_response.json()
            latest_version = model_data.get("latest_version", {}).get("id")

            if not latest_version:
                return ExternalWebhookTestResponse(
                    success=False,
                    message="No model version available",
                    error=f"Model '{webhook_check_model}' has no versions. Push a version first.",
                )

            prediction_response = await client.post(
                "https://api.replicate.com/v1/predictions",
                headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"},
                json={"version": latest_version, "input": {"webhook_url": webhook_url, "timeout_seconds": 30.0}},
            )

            if prediction_response.status_code != 201:
                return ExternalWebhookTestResponse(
                    success=False,
                    message="Failed to start test",
                    error=f"HTTP {prediction_response.status_code}: {prediction_response.text[:200]}",
                )

            prediction = prediction_response.json()
            prediction_id = prediction.get("id")

            start_time = time.time()
            max_wait = 60.0
            poll_interval = 1.0

            while time.time() - start_time < max_wait:
                status_response = await client.get(
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                    headers={"Authorization": f"Bearer {api_token}"},
                )

                if status_response.status_code != 200:
                    await asyncio.sleep(poll_interval)
                    continue

                status_data = status_response.json()
                pred_status = status_data.get("status")

                # Normalize external status using CloudJobStatus
                normalized_status = CloudJobStatus.from_external(pred_status) if pred_status else None

                if normalized_status == CloudJobStatus.COMPLETED:
                    output = status_data.get("output", {})
                    return ExternalWebhookTestResponse(
                        success=output.get("success", False),
                        message=(
                            "Webhook reachable from Replicate"
                            if output.get("success")
                            else "Webhook not reachable from Replicate"
                        ),
                        latency_ms=output.get("latency_ms"),
                        status_code=output.get("status_code"),
                        error=output.get("error"),
                        prediction_id=prediction_id,
                        details=output.get("details"),
                    )
                elif normalized_status == CloudJobStatus.FAILED:
                    return ExternalWebhookTestResponse(
                        success=False,
                        message="Test prediction failed",
                        error=status_data.get("error", "Unknown error"),
                        prediction_id=prediction_id,
                    )
                elif normalized_status == CloudJobStatus.CANCELLED:
                    return ExternalWebhookTestResponse(
                        success=False, message="Test was cancelled", prediction_id=prediction_id
                    )

                await asyncio.sleep(poll_interval)

            return ExternalWebhookTestResponse(
                success=False,
                message="Test timed out",
                error=f"Prediction did not complete within {max_wait} seconds",
                prediction_id=prediction_id,
            )

    except Exception as exc:
        logger.exception("External webhook test failed")
        return ExternalWebhookTestResponse(success=False, message="Test failed", error=str(exc))


@router.get("/providers/replicate/validate", response_model=ValidateResponse)
async def validate_replicate_key() -> ValidateResponse:
    """Check if REPLICATE_API_TOKEN is set and valid."""
    api_token = get_secrets_manager().get_replicate_token()

    if not api_token:
        return ValidateResponse(valid=False, error="REPLICATE_API_TOKEN environment variable not set")

    try:
        client = ProviderFactory.get_provider("replicate")
        user_info = await client.validate_credentials()

        if user_info.get("error"):
            return ValidateResponse(valid=False, error=user_info.get("error"))

        return ValidateResponse(valid=True, user_info=user_info)
    except Exception as exc:
        return ValidateResponse(valid=False, error=str(exc))


@router.get("/providers/simpletuner_io/validate", response_model=ValidateResponse)
async def validate_simpletuner_io_token() -> ValidateResponse:
    """Check if SimpleTuner.io refresh token is set and valid."""
    refresh_token = get_secrets_manager().get(SIMPLETUNER_IO_REFRESH_TOKEN_KEY)
    if not refresh_token:
        return ValidateResponse(valid=False, error="SIMPLETUNER_IO_REFRESH_TOKEN not set")

    try:
        client = ProviderFactory.get_provider("simpletuner_io")
        user_info = await client.validate_credentials()

        if user_info.get("error"):
            return ValidateResponse(valid=False, error=user_info.get("error"))

        return ValidateResponse(valid=True, user_info=user_info)
    except Exception as exc:
        return ValidateResponse(valid=False, error=str(exc))


class SaveTokenRequest(BaseModel):
    """Request to save an API token."""

    api_token: str


class SaveTokenResponse(BaseModel):
    """Response for saving an API token."""

    success: bool
    file_path: Optional[str] = None
    error: Optional[str] = None


SIMPLETUNER_IO_REFRESH_TOKEN_KEY = "SIMPLETUNER_IO_REFRESH_TOKEN"


@router.put("/providers/replicate/token", response_model=SaveTokenResponse)
async def save_replicate_token(request: SaveTokenRequest) -> SaveTokenResponse:
    """Save the Replicate API token to the secrets file.

    The token is saved to ~/.simpletuner/secrets.json with restrictive permissions.
    Environment variable REPLICATE_API_TOKEN still takes precedence if set.
    """
    token = request.api_token.strip()

    if not token:
        return SaveTokenResponse(success=False, error="Token cannot be empty")

    # Basic validation - Replicate tokens start with r8_
    if not token.startswith("r8_"):
        return SaveTokenResponse(success=False, error="Invalid token format. Replicate tokens start with 'r8_'")

    secrets_mgr = get_secrets_manager()

    # Save the token
    success = secrets_mgr.set_replicate_token(token)

    if not success:
        return SaveTokenResponse(success=False, error="Failed to save token to file")

    file_path = str(secrets_mgr.get_secrets_file_path())

    # Validate the saved token by trying to connect
    try:
        client = ProviderFactory.get_provider("replicate")
        user_info = await client.validate_credentials()

        if user_info.get("error"):
            # Token saved but validation failed - keep it saved but warn user
            return SaveTokenResponse(
                success=True, file_path=file_path, error=f"Token saved but validation failed: {user_info.get('error')}"
            )

        return SaveTokenResponse(success=True, file_path=file_path)
    except Exception as exc:
        # Token saved but validation failed
        return SaveTokenResponse(success=True, file_path=file_path, error=f"Token saved but validation failed: {exc}")


@router.put("/providers/simpletuner_io/token", response_model=SaveTokenResponse)
async def save_simpletuner_io_token(request: SaveTokenRequest) -> SaveTokenResponse:
    """Save the SimpleTuner.io CLI refresh token to the secrets file."""
    token = request.api_token.strip()

    if not token:
        return SaveTokenResponse(success=False, error="Token cannot be empty")

    secrets_mgr = get_secrets_manager()
    success = secrets_mgr.set_secret(SIMPLETUNER_IO_REFRESH_TOKEN_KEY, token)

    if not success:
        return SaveTokenResponse(success=False, error="Failed to save token to file")

    file_path = str(secrets_mgr.get_secrets_file_path())

    # Validate token if org_id is configured
    try:
        store = get_job_store()
        config = await store.get_provider_config("simpletuner_io")
        if config.get("org_id"):
            client = ProviderFactory.get_provider("simpletuner_io")
            user_info = await client.validate_credentials()
            if user_info.get("error"):
                return SaveTokenResponse(
                    success=True,
                    file_path=file_path,
                    error=f"Token saved but validation failed: {user_info.get('error')}",
                )
    except Exception as exc:
        return SaveTokenResponse(success=True, file_path=file_path, error=f"Token saved but validation failed: {exc}")

    return SaveTokenResponse(success=True, file_path=file_path)


@router.delete("/providers/replicate/token", response_model=SaveTokenResponse)
async def delete_replicate_token() -> SaveTokenResponse:
    """Delete the Replicate API token from the secrets file."""
    secrets_mgr = get_secrets_manager()
    success = secrets_mgr.delete_secret(secrets_mgr.REPLICATE_API_TOKEN)

    if success:
        return SaveTokenResponse(success=True, file_path=str(secrets_mgr.get_secrets_file_path()))
    return SaveTokenResponse(success=False, error="Failed to delete token")


@router.delete("/providers/simpletuner_io/token", response_model=SaveTokenResponse)
async def delete_simpletuner_io_token() -> SaveTokenResponse:
    """Delete the SimpleTuner.io CLI refresh token from the secrets file."""
    secrets_mgr = get_secrets_manager()
    success = secrets_mgr.delete_secret(SIMPLETUNER_IO_REFRESH_TOKEN_KEY)

    if success:
        return SaveTokenResponse(success=True, file_path=str(secrets_mgr.get_secrets_file_path()))
    return SaveTokenResponse(success=False, error="Failed to delete token")


@router.get("/providers/replicate/versions", response_model=ModelVersionsResponse)
async def list_replicate_versions(
    model_id: Optional[str] = Query(None, description="Model ID (owner/model). Defaults to configured model.")
) -> ModelVersionsResponse:
    """List available versions for a Replicate model, sorted newest first."""
    api_token = get_secrets_manager().get_replicate_token()
    if not api_token:
        return ModelVersionsResponse(error="REPLICATE_API_TOKEN not set")

    from ...services.cloud.replicate_client import ReplicateCogClient

    try:
        store = get_job_store()
        provider_config = await store.get_provider_config("replicate")
        version_override = provider_config.get("version_override")

        client = ReplicateCogClient()
        versions = await client.list_model_versions(model_id)

        return ModelVersionsResponse(
            versions=[ModelVersionInfo(**v) for v in versions],
            current_version=version_override,
        )
    except ImportError:
        return ModelVersionsResponse(error="Replicate SDK not installed")
    except Exception as exc:
        logger.warning("Failed to list versions: %s", exc)
        return ModelVersionsResponse(error=str(exc))


@router.get("/providers/replicate/billing")
async def get_replicate_billing() -> Dict[str, Any]:
    """Get Replicate billing/credit information."""
    api_token = get_secrets_manager().get_replicate_token()

    if not api_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="REPLICATE_API_TOKEN not set")

    try:
        client = ProviderFactory.get_provider("replicate")
        return await client.get_billing_info()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error fetching billing info: {exc}")


@router.get("/publishing/status", response_model=PublishingStatusResponse)
async def get_publishing_status() -> PublishingStatusResponse:
    """Check if publishing is configured (HF token, hub_model_id, S3, etc)."""
    from ...services.publishing_service import PublishingService

    result = PublishingStatusResponse()

    try:
        service = PublishingService()
        token_status = service.validate_token()
        result.hf_token_valid = token_status.get("valid", False)
        result.hf_username = token_status.get("username")
        if result.hf_token_valid:
            result.hf_configured = True
        else:
            result.message = token_status.get("message")
    except Exception as exc:
        logger.warning("HF token validation failed: %s", exc)
        result.message = str(exc)

    try:
        config_resp = await get_active_config()
        if config_resp:
            result.push_to_hub = config_resp.get("push_to_hub") or config_resp.get("--push_to_hub", False)
            result.hub_model_id = config_resp.get("hub_model_id") or config_resp.get("--hub_model_id")
            publishing_config = config_resp.get("publishing_config") or config_resp.get("--publishing_config")
            if publishing_config:
                result.s3_configured = True
            logger.debug(
                "Publishing status: hf_valid=%s, hub_model_id=%s, push_to_hub=%s",
                result.hf_token_valid,
                result.hub_model_id,
                result.push_to_hub,
            )
        else:
            logger.warning("No active config found for publishing status")
    except Exception as exc:
        logger.warning("Failed to get config for publishing status: %s", exc)

    try:
        store = get_job_store()
        provider_config = await store.get_provider_config("replicate")
        webhook_url = provider_config.get("webhook_url")
        if webhook_url:
            result.local_upload_available = True
            upload_dir = get_local_upload_dir()
            result.local_upload_dir = str(upload_dir / "outputs")
    except Exception as exc:
        logger.debug("Could not check local upload status: %s", exc)

    return result


@router.post("/publishing/enable-recommended")
async def enable_recommended_publishing() -> Dict[str, Any]:
    """Enable recommended publishing settings for cloud training."""
    from ...services.config_store import ConfigStore
    from ...services.webui_state import WebUIStateStore

    try:
        defaults = WebUIStateStore().load_defaults()
        if not defaults.configs_dir or not defaults.active_config:
            return {"success": False, "error": "No active configuration"}

        store = ConfigStore(config_dir=Path(defaults.configs_dir).expanduser(), config_type="model")
        config, metadata = store.load_config(defaults.active_config)

        config["push_to_hub"] = True
        config["push_checkpoints_to_hub"] = True
        config["push_to_hub_background"] = True

        store.save_config(defaults.active_config, config, metadata=metadata, overwrite=True)

        return {
            "success": True,
            "message": "Publishing settings enabled",
            "settings": {"push_to_hub": True, "push_checkpoints_to_hub": True, "push_to_hub_background": True},
        }
    except Exception as exc:
        logger.error("Failed to enable publishing settings: %s", exc)
        return {"success": False, "error": str(exc)}

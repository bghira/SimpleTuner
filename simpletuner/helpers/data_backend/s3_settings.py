"""Resolve WebShart S3 settings from the host without changing its environment."""

import os

from botocore.session import Session

_NATIVE_CREDENTIAL_ENV = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
)
_CREDENTIAL_KEYS = ("aws_access_key_id", "aws_secret_access_key", "aws_session_token")


def _profile_endpoint(profile: dict, services: dict) -> str | None:
    service = services.get(profile.get("services"), {}).get("s3", {})
    return service.get("endpoint_url") or profile.get("endpoint_url")


def resolve_s3_options(options: dict, *, use_profile_credentials: bool = True) -> dict:
    """Apply host S3 settings without serializing credentials into a dataset config."""
    session = Session()
    config = session.full_config
    profiles = config["profiles"]
    services = config.get("services", {})
    profile_name = session.get_config_variable("profile")
    profile = profiles.get(profile_name or "default", {})

    resolved = dict(options)
    endpoint = (
        resolved.get("endpoint_url")
        or os.environ.get("AWS_ENDPOINT_URL_S3")
        or os.environ.get("AWS_ENDPOINT_URL")
        or os.environ.get("AWS_ENDPOINT")
        or _profile_endpoint(profile, services)
    )
    if endpoint:
        resolved["endpoint_url"] = endpoint

    if (
        use_profile_credentials
        and not any(key in resolved for key in _CREDENTIAL_KEYS)
        and not any(os.environ.get(key) for key in _NATIVE_CREDENTIAL_ENV)
    ):
        if endpoint and not profile_name:
            matched = [
                candidate
                for candidate in profiles.values()
                if (_profile_endpoint(candidate, services) or "").rstrip("/") == endpoint.rstrip("/")
                and candidate.get("aws_access_key_id")
                and candidate.get("aws_secret_access_key")
            ]
            if len(matched) > 1:
                raise ValueError("More than one AWS profile matches this S3 endpoint. Set AWS_PROFILE to choose one.")
            if matched:
                profile = matched[0]
        if profile.get("aws_access_key_id") and profile.get("aws_secret_access_key"):
            resolved.update({key: profile[key] for key in _CREDENTIAL_KEYS if profile.get(key)})

    if "region_name" not in resolved:
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or profile.get("region")
        if region:
            resolved["region_name"] = region
    return resolved

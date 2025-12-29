"""Secrets management abstraction with multiple backend support.

Supports:
- Environment variables (default, plaintext)
- File-based secrets (JSON/YAML file)
- AWS Secrets Manager
- HashiCorp Vault
"""

from __future__ import annotations

import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SecretProvider(ABC):
    """Abstract base class for secret providers."""

    @abstractmethod
    def get_secret(self, key: str) -> Optional[str]:
        """Get a secret value by key.

        Args:
            key: The secret key/name to retrieve

        Returns:
            The secret value, or None if not found
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available and configured.

        Returns:
            True if the provider can be used
        """
        pass


class EnvironmentSecretProvider(SecretProvider):
    """Secret provider that reads from environment variables.

    This is the default provider for simple deployments.
    Keys are automatically uppercased and dashes converted to underscores.
    """

    def __init__(self, prefix: str = ""):
        """Initialize the environment secret provider.

        Args:
            prefix: Optional prefix to prepend to all key lookups
        """
        self._prefix = prefix

    def _normalize_key(self, key: str) -> str:
        """Normalize a key for environment variable lookup."""
        normalized = key.upper().replace("-", "_").replace(".", "_")
        if self._prefix:
            return f"{self._prefix}_{normalized}"
        return normalized

    def get_secret(self, key: str) -> Optional[str]:
        """Get a secret from environment variables."""
        env_key = self._normalize_key(key)
        value = os.environ.get(env_key)
        if value:
            logger.debug("Retrieved secret '%s' from environment variable '%s'", key, env_key)
        return value

    def is_available(self) -> bool:
        """Environment provider is always available."""
        return True


class FileSecretProvider(SecretProvider):
    """Secret provider that reads from a JSON or YAML file.

    The file should contain a flat dictionary of key-value pairs.
    Supports both JSON and YAML formats (auto-detected by extension).
    """

    def __init__(self, file_path: Optional[str] = None):
        """Initialize the file secret provider.

        Args:
            file_path: Path to the secrets file. If not provided,
                      looks for ~/.simpletuner/secrets.json or secrets.yaml
        """
        self._file_path = self._resolve_file_path(file_path)
        self._secrets: Optional[Dict[str, str]] = None
        self._loaded = False

    def _resolve_file_path(self, provided_path: Optional[str]) -> Optional[Path]:
        """Resolve the secrets file path."""
        if provided_path:
            path = Path(provided_path).expanduser()
            if path.exists():
                return path
            return None

        # Check default locations
        home_config = Path.home() / ".simpletuner"
        for name in ["secrets.json", "secrets.yaml", "secrets.yml"]:
            path = home_config / name
            if path.exists():
                return path

        return None

    def _load_secrets(self) -> Dict[str, str]:
        """Load secrets from the file."""
        if self._loaded:
            return self._secrets or {}

        self._loaded = True

        if not self._file_path or not self._file_path.exists():
            self._secrets = {}
            return self._secrets

        try:
            content = self._file_path.read_text(encoding="utf-8")

            if self._file_path.suffix in (".yaml", ".yml"):
                try:
                    import yaml

                    self._secrets = yaml.safe_load(content) or {}
                except ImportError:
                    logger.warning("PyYAML not installed, cannot read YAML secrets file")
                    self._secrets = {}
            else:
                self._secrets = json.loads(content)

            logger.info("Loaded secrets from %s", self._file_path)
        except Exception as exc:
            logger.warning("Failed to load secrets from %s: %s", self._file_path, exc)
            self._secrets = {}

        return self._secrets

    def get_secret(self, key: str) -> Optional[str]:
        """Get a secret from the file."""
        secrets = self._load_secrets()
        value = secrets.get(key)
        if value:
            logger.debug("Retrieved secret '%s' from file", key)
        return value

    def is_available(self) -> bool:
        """Check if a secrets file exists."""
        return self._file_path is not None and self._file_path.exists()

    def set_secret(self, key: str, value: str) -> bool:
        """Save a secret to the file.

        Args:
            key: The secret key
            value: The secret value

        Returns:
            True if saved successfully
        """
        # Ensure we have a file path
        if not self._file_path:
            self._file_path = Path.home() / ".simpletuner" / "secrets.json"

        # Create parent directory if needed
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing secrets
        secrets = self._load_secrets()

        # Update the secret
        secrets[key] = value
        self._secrets = secrets

        # Write back to file
        try:
            with open(self._file_path, "w", encoding="utf-8") as f:
                json.dump(secrets, f, indent=2)

            # Set restrictive permissions (owner read/write only)
            self._file_path.chmod(0o600)

            logger.info("Saved secret '%s' to %s", key, self._file_path)
            return True
        except Exception as exc:
            logger.error("Failed to save secret to %s: %s", self._file_path, exc)
            return False

    def delete_secret(self, key: str) -> bool:
        """Delete a secret from the file.

        Args:
            key: The secret key to delete

        Returns:
            True if deleted successfully
        """
        if not self._file_path or not self._file_path.exists():
            return False

        secrets = self._load_secrets()
        if key not in secrets:
            return True  # Already doesn't exist

        del secrets[key]
        self._secrets = secrets

        try:
            with open(self._file_path, "w", encoding="utf-8") as f:
                json.dump(secrets, f, indent=2)
            logger.info("Deleted secret '%s' from %s", key, self._file_path)
            return True
        except Exception as exc:
            logger.error("Failed to delete secret from %s: %s", self._file_path, exc)
            return False

    def get_file_path(self) -> Optional[Path]:
        """Get the secrets file path."""
        return self._file_path


class AWSSecretsManagerProvider(SecretProvider):
    """Secret provider that reads from AWS Secrets Manager.

    Requires boto3 and AWS credentials configured.
    """

    def __init__(
        self,
        secret_name: Optional[str] = None,
        region_name: Optional[str] = None,
    ):
        """Initialize the AWS Secrets Manager provider.

        Args:
            secret_name: The name of the secret in AWS Secrets Manager.
                        If not provided, uses SIMPLETUNER_AWS_SECRET_NAME env var.
            region_name: AWS region. If not provided, uses default region.
        """
        self._secret_name = secret_name or os.environ.get("SIMPLETUNER_AWS_SECRET_NAME")
        self._region_name = region_name or os.environ.get("AWS_DEFAULT_REGION")
        self._client = None
        self._secrets: Optional[Dict[str, str]] = None
        self._loaded = False

    def _get_client(self):
        """Lazily initialize the boto3 client."""
        if self._client is None:
            try:
                import boto3

                self._client = boto3.client(
                    "secretsmanager",
                    region_name=self._region_name,
                )
            except ImportError:
                raise ImportError("boto3 is required for AWS Secrets Manager. " "Install it with: pip install boto3")
        return self._client

    def _load_secrets(self) -> Dict[str, str]:
        """Load secrets from AWS Secrets Manager."""
        if self._loaded:
            return self._secrets or {}

        self._loaded = True

        if not self._secret_name:
            self._secrets = {}
            return self._secrets

        try:
            client = self._get_client()
            response = client.get_secret_value(SecretId=self._secret_name)

            secret_string = response.get("SecretString")
            if secret_string:
                self._secrets = json.loads(secret_string)
                logger.info("Loaded secrets from AWS Secrets Manager: %s", self._secret_name)
            else:
                self._secrets = {}

        except ImportError:
            self._secrets = {}
        except Exception as exc:
            logger.warning("Failed to load secrets from AWS: %s", exc)
            self._secrets = {}

        return self._secrets

    def get_secret(self, key: str) -> Optional[str]:
        """Get a secret from AWS Secrets Manager."""
        secrets = self._load_secrets()
        value = secrets.get(key)
        if value:
            logger.debug("Retrieved secret '%s' from AWS Secrets Manager", key)
        return value

    def is_available(self) -> bool:
        """Check if AWS Secrets Manager is configured."""
        if not self._secret_name:
            return False

        try:
            import boto3  # noqa: F401

            return True
        except ImportError:
            return False


class HashiCorpVaultProvider(SecretProvider):
    """Secret provider that reads from HashiCorp Vault.

    Requires hvac library and Vault configuration.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        path: Optional[str] = None,
        mount_point: str = "secret",
    ):
        """Initialize the HashiCorp Vault provider.

        Args:
            url: Vault server URL. Defaults to VAULT_ADDR env var.
            token: Vault token. Defaults to VAULT_TOKEN env var.
            path: Path to the secret in Vault. Defaults to SIMPLETUNER_VAULT_PATH.
            mount_point: KV mount point. Defaults to "secret".
        """
        self._url = url or os.environ.get("VAULT_ADDR")
        self._token = token or os.environ.get("VAULT_TOKEN")
        self._path = path or os.environ.get("SIMPLETUNER_VAULT_PATH", "simpletuner")
        self._mount_point = mount_point
        self._client = None
        self._secrets: Optional[Dict[str, str]] = None
        self._loaded = False

    def _get_client(self):
        """Lazily initialize the hvac client."""
        if self._client is None:
            try:
                import hvac

                self._client = hvac.Client(url=self._url, token=self._token)
            except ImportError:
                raise ImportError("hvac is required for HashiCorp Vault. " "Install it with: pip install hvac")
        return self._client

    def _load_secrets(self) -> Dict[str, str]:
        """Load secrets from HashiCorp Vault."""
        if self._loaded:
            return self._secrets or {}

        self._loaded = True

        if not self._url or not self._token:
            self._secrets = {}
            return self._secrets

        try:
            client = self._get_client()

            # Try KV v2 first, fall back to v1
            try:
                response = client.secrets.kv.v2.read_secret_version(
                    path=self._path,
                    mount_point=self._mount_point,
                )
                self._secrets = response.get("data", {}).get("data", {})
            except Exception:
                # Try KV v1
                response = client.secrets.kv.v1.read_secret(
                    path=self._path,
                    mount_point=self._mount_point,
                )
                self._secrets = response.get("data", {})

            logger.info("Loaded secrets from Vault: %s/%s", self._mount_point, self._path)

        except ImportError:
            self._secrets = {}
        except Exception as exc:
            logger.warning("Failed to load secrets from Vault: %s", exc)
            self._secrets = {}

        return self._secrets

    def get_secret(self, key: str) -> Optional[str]:
        """Get a secret from HashiCorp Vault."""
        secrets = self._load_secrets()
        value = secrets.get(key)
        if value:
            logger.debug("Retrieved secret '%s' from Vault", key)
        return value

    def is_available(self) -> bool:
        """Check if Vault is configured."""
        if not self._url or not self._token:
            return False

        try:
            import hvac  # noqa: F401

            return True
        except ImportError:
            return False


class SecretsManager:
    """Unified secrets manager that chains multiple providers.

    Providers are checked in order until a value is found.
    Default order: Environment -> File -> AWS -> Vault
    """

    # Well-known secret keys
    REPLICATE_API_TOKEN = "REPLICATE_API_TOKEN"
    HF_TOKEN = "HF_TOKEN"
    WEBHOOK_SECRET = "CLOUD_WEBHOOK_SECRET"

    _instance: Optional["SecretsManager"] = None

    def __new__(cls, *args, **kwargs) -> "SecretsManager":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        providers: Optional[list[SecretProvider]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the secrets manager.

        Args:
            providers: List of providers to use. If not provided,
                      auto-configures based on available providers.
            config: Optional configuration dict with provider settings.
        """
        if getattr(self, "_initialized", False):
            return

        self._config = config or {}
        self._providers = providers or self._auto_configure_providers()
        self._cache: Dict[str, Optional[str]] = {}
        self._initialized = True

        available = [p.__class__.__name__ for p in self._providers if p.is_available()]
        logger.info("SecretsManager initialized with providers: %s", available)

    def _auto_configure_providers(self) -> list[SecretProvider]:
        """Auto-configure providers based on what's available."""
        providers: list[SecretProvider] = []

        # Environment is always first (allows overrides)
        providers.append(EnvironmentSecretProvider())

        # File-based secrets
        file_path = self._config.get("file_path")
        file_provider = FileSecretProvider(file_path)
        if file_provider.is_available():
            providers.append(file_provider)

        # AWS Secrets Manager
        aws_secret = self._config.get("aws_secret_name")
        aws_region = self._config.get("aws_region")
        aws_provider = AWSSecretsManagerProvider(aws_secret, aws_region)
        if aws_provider.is_available():
            providers.append(aws_provider)

        # HashiCorp Vault
        vault_url = self._config.get("vault_url")
        vault_token = self._config.get("vault_token")
        vault_path = self._config.get("vault_path")
        vault_provider = HashiCorpVaultProvider(vault_url, vault_token, vault_path)
        if vault_provider.is_available():
            providers.append(vault_provider)

        return providers

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a secret value.

        Checks providers in order until a value is found.

        Args:
            key: The secret key to retrieve
            default: Default value if not found

        Returns:
            The secret value, or default if not found
        """
        # Check cache first
        if key in self._cache:
            return self._cache[key] if self._cache[key] is not None else default

        # Try each provider in order
        for provider in self._providers:
            if not provider.is_available():
                continue

            try:
                value = provider.get_secret(key)
                if value is not None:
                    self._cache[key] = value
                    return value
            except Exception as exc:
                logger.warning(
                    "Error retrieving secret '%s' from %s: %s",
                    key,
                    provider.__class__.__name__,
                    exc,
                )

        # Cache the miss too
        self._cache[key] = None
        return default

    def get_replicate_token(self) -> Optional[str]:
        """Get the Replicate API token."""
        return self.get(self.REPLICATE_API_TOKEN)

    def get_hf_token(self) -> Optional[str]:
        """Get the HuggingFace token."""
        return self.get(self.HF_TOKEN)

    def get_webhook_secret(self) -> Optional[str]:
        """Get the webhook secret for HMAC validation."""
        return self.get(self.WEBHOOK_SECRET)

    def clear_cache(self) -> None:
        """Clear the secrets cache."""
        self._cache.clear()

    def set_secret(self, key: str, value: str) -> bool:
        """Save a secret to the file-based provider.

        This always saves to ~/.simpletuner/secrets.json.
        The environment variable still takes precedence when reading.

        Args:
            key: The secret key
            value: The secret value

        Returns:
            True if saved successfully
        """
        # Find or create file provider
        file_provider = None
        for provider in self._providers:
            if isinstance(provider, FileSecretProvider):
                file_provider = provider
                break

        if file_provider is None:
            file_provider = FileSecretProvider()
            self._providers.insert(1, file_provider)  # After env provider

        success = file_provider.set_secret(key, value)
        if success:
            # Update cache
            self._cache[key] = value
        return success

    def delete_secret(self, key: str) -> bool:
        """Delete a secret from the file-based provider.

        Args:
            key: The secret key to delete

        Returns:
            True if deleted successfully
        """
        for provider in self._providers:
            if isinstance(provider, FileSecretProvider):
                success = provider.delete_secret(key)
                if success:
                    # Clear from cache
                    self._cache.pop(key, None)
                return success
        return False

    def get_secrets_file_path(self) -> Path:
        """Get the path where secrets are stored."""
        for provider in self._providers:
            if isinstance(provider, FileSecretProvider):
                path = provider.get_file_path()
                if path:
                    return path
        return Path.home() / ".simpletuner" / "secrets.json"

    def set_replicate_token(self, token: str) -> bool:
        """Save the Replicate API token."""
        return self.set_secret(self.REPLICATE_API_TOKEN, token)

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing only).

        Raises:
            RuntimeError: If called outside of a test environment.
        """
        if "pytest" not in sys.modules and "unittest" not in sys.modules:
            raise RuntimeError("SecretsManager.reset() is only allowed in test environments")
        cls._instance = None


def get_secrets_manager() -> SecretsManager:
    """Get the global SecretsManager instance."""
    return SecretsManager()

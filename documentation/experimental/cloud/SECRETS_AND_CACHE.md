# Secrets Providers and Cache System

This document covers SimpleTuner's secrets management system and caching infrastructure for cloud training.

## Overview

SimpleTuner uses a layered secrets management system with multiple provider backends and an in-memory caching layer to optimize performance.

## Secrets Providers

The `SecretsManager` class chains multiple secret providers in a priority order. When retrieving a secret, providers are checked sequentially until a value is found.

### Provider Priority Order

1. **Environment Variables** (highest priority)
2. **File-based Secrets**
3. **AWS Secrets Manager**
4. **HashiCorp Vault** (lowest priority)

This ordering allows environment variables to always override other sources, which is useful for:
- Local development overrides
- Container deployments with injected secrets
- CI/CD pipelines

### Well-Known Secret Keys

<details>
<summary>Secret key constants</summary>

```python
REPLICATE_API_TOKEN  # Replicate API authentication
HF_TOKEN             # HuggingFace Hub token
CLOUD_WEBHOOK_SECRET # Webhook HMAC validation secret
```
</details>

---

## Provider Configuration

### 1. Environment Variables (Default)

The environment provider is always available and takes precedence over all other providers.

**Key normalization rules:**
- Keys are uppercased
- Dashes (`-`) are converted to underscores (`_`)
- Dots (`.`) are converted to underscores (`_`)

**Optional prefix:**

<details>
<summary>Python configuration example</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import EnvironmentSecretProvider

# Without prefix
provider = EnvironmentSecretProvider()
# Key "replicate-api-token" -> REPLICATE_API_TOKEN

# With prefix
provider = EnvironmentSecretProvider(prefix="SIMPLETUNER")
# Key "replicate-api-token" -> SIMPLETUNER_REPLICATE_API_TOKEN
```
</details>

**Usage:**

<details>
<summary>Setting secrets via environment</summary>

```bash
# Set secrets via environment
export REPLICATE_API_TOKEN="r8_your_token_here"
export HF_TOKEN="hf_your_token_here"
export CLOUD_WEBHOOK_SECRET="your_webhook_secret"
```
</details>

### 2. File-based Secrets

Secrets can be stored in a JSON or YAML file with restrictive permissions.

**Default locations (checked in order):**
- `~/.simpletuner/secrets.json`
- `~/.simpletuner/secrets.yaml`
- `~/.simpletuner/secrets.yml`

<details>
<summary>File format examples (JSON/YAML)</summary>

**File format (JSON):**

```json
{
  "REPLICATE_API_TOKEN": "r8_your_token_here",
  "HF_TOKEN": "hf_your_token_here",
  "CLOUD_WEBHOOK_SECRET": "your_webhook_secret"
}
```

**File format (YAML):**

```yaml
REPLICATE_API_TOKEN: r8_your_token_here
HF_TOKEN: hf_your_token_here
CLOUD_WEBHOOK_SECRET: your_webhook_secret
```
</details>

<details>
<summary>Manual configuration</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import FileSecretProvider

# Use a custom file path
provider = FileSecretProvider(file_path="/path/to/secrets.json")
```
</details>

**Security note:** When saving secrets via the API, the file is created with `chmod 0o600` (owner read/write only).

### 3. AWS Secrets Manager

For production deployments, AWS Secrets Manager provides secure, centralized secret storage.

<details>
<summary>Environment variables</summary>

```bash
# Required: Name of the secret in AWS Secrets Manager
export SIMPLETUNER_AWS_SECRET_NAME="simpletuner/production"

# Optional: AWS region (uses default if not set)
export AWS_DEFAULT_REGION="us-west-2"
```
</details>

<details>
<summary>AWS secret format</summary>

The secret in AWS Secrets Manager should be a JSON object:

```json
{
  "REPLICATE_API_TOKEN": "r8_your_token_here",
  "HF_TOKEN": "hf_your_token_here",
  "CLOUD_WEBHOOK_SECRET": "your_webhook_secret"
}
```
</details>

<details>
<summary>Manual configuration</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import AWSSecretsManagerProvider

provider = AWSSecretsManagerProvider(
    secret_name="simpletuner/production",
    region_name="us-west-2"
)
```
</details>

<details>
<summary>AWS IAM permissions required</summary>

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "secretsmanager:GetSecretValue",
      "Resource": "arn:aws:secretsmanager:REGION:ACCOUNT:secret:simpletuner/*"
    }
  ]
}
```
</details>

### 4. HashiCorp Vault

For enterprises using HashiCorp Vault for secrets management.

**Required dependencies:**

```bash
pip install hvac
```

<details>
<summary>Environment variables</summary>

```bash
# Required: Vault server URL
export VAULT_ADDR="https://vault.example.com:8200"

# Required: Vault authentication token
export VAULT_TOKEN="s.your_vault_token"

# Optional: Path to secrets (default: "simpletuner")
export SIMPLETUNER_VAULT_PATH="simpletuner"
```
</details>

<details>
<summary>Manual configuration</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import HashiCorpVaultProvider

provider = HashiCorpVaultProvider(
    url="https://vault.example.com:8200",
    token="s.your_vault_token",
    path="simpletuner",
    mount_point="secret"  # KV mount point (default: "secret")
)
```
</details>

<details>
<summary>Vault KV setup</summary>

```bash
# KV v2 (recommended)
vault kv put secret/simpletuner \
    REPLICATE_API_TOKEN="r8_your_token" \
    HF_TOKEN="hf_your_token"

# KV v1
vault write secret/simpletuner \
    REPLICATE_API_TOKEN="r8_your_token" \
    HF_TOKEN="hf_your_token"
```

The provider automatically tries KV v2 first, falling back to v1 if that fails.
</details>

---

## Using the Secrets Manager

### Basic Usage

<details>
<summary>Getting secrets</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import get_secrets_manager

# Get the singleton instance
secrets = get_secrets_manager()

# Get any secret
api_token = secrets.get("REPLICATE_API_TOKEN")
api_token = secrets.get("REPLICATE_API_TOKEN", default="fallback_value")

# Convenience methods
replicate_token = secrets.get_replicate_token()
hf_token = secrets.get_hf_token()
webhook_secret = secrets.get_webhook_secret()
```
</details>

### Saving Secrets (File Provider Only)

<details>
<summary>Setting and deleting secrets</summary>

```python
secrets = get_secrets_manager()

# Save a secret (writes to ~/.simpletuner/secrets.json)
secrets.set_secret("MY_CUSTOM_KEY", "my_value")

# Convenience method for Replicate token
secrets.set_replicate_token("r8_new_token")

# Delete a secret
secrets.delete_secret("MY_CUSTOM_KEY")
```
</details>

### Custom Provider Configuration

<details>
<summary>Custom configuration example</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import SecretsManager

# Reset singleton for custom configuration
SecretsManager.reset()

# Create with custom config
secrets = SecretsManager(config={
    "file_path": "/custom/path/secrets.json",
    "aws_secret_name": "myapp/secrets",
    "aws_region": "eu-west-1",
    "vault_url": "https://vault.internal:8200",
    "vault_token": "s.token",
    "vault_path": "myapp/simpletuner"
})
```
</details>

---

## Cache System

SimpleTuner uses an in-memory TTL (Time-To-Live) cache to reduce database queries and improve response times for frequently accessed data.

### TTLCache Features

- **Thread-safe operations** using `RLock`
- **Per-key TTL** with configurable defaults
- **Automatic cleanup** of expired entries
- **LRU eviction** when at capacity
- **Prefix-based invalidation** for bulk cache clearing

### Cache Configuration

<details>
<summary>TTLCache initialization</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import TTLCache

cache = TTLCache[str](
    default_ttl=300.0,      # 5 minutes default TTL
    max_size=1000,          # Maximum entries before eviction
    cleanup_interval=60.0   # Cleanup expired entries every 60s
)
```
</details>

### Global Cache Instances

SimpleTuner maintains two global cache instances:

| Cache | Default TTL | Max Size | Purpose |
|-------|-------------|----------|---------|
| Provider Config Cache | 300s (5 min) | 100 | Provider settings, webhook URLs, cost limits |
| User Permission Cache | 60s (1 min) | 500 | User permissions (shorter TTL for security) |

<details>
<summary>Accessing global caches</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import (
    get_provider_config_cache,
    get_user_permission_cache
)

# Access global caches
provider_cache = get_provider_config_cache()
user_cache = get_user_permission_cache()
```
</details>

### Basic Cache Operations

<details>
<summary>Cache operations reference</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import TTLCache

cache = TTLCache[dict](default_ttl=120.0)

# Set a value (uses default TTL)
cache.set("my_key", {"data": "value"})

# Set with custom TTL
cache.set("my_key", {"data": "value"}, ttl=60.0)

# Get a value (returns None if expired or missing)
value = cache.get("my_key")

# Get or compute if missing
value = cache.get_or_set("my_key", lambda: compute_value())

# Async version
value = await cache.get_or_set_async("my_key", async_compute_value)

# Delete a specific key
cache.delete("my_key")

# Clear all entries
count = cache.clear()

# Get cache statistics
stats = cache.stats()
# {"size": 42, "max_size": 1000, "expired_count": 3, "default_ttl": 120.0}
```
</details>

---

## Provider Metadata Caching

Provider configuration (webhook URLs, cost limits, hardware info) is cached to reduce database load.

### Cache Key Format

```
provider:{provider_name}:config
```

Example: `provider:replicate:config`

### Caching Behavior

**On read (`get_provider_config`):**
1. Check cache for `provider:{name}:config`
2. If found and not expired, return cached value
3. If missing/expired, load from database
4. Store result in cache with 5-minute TTL
5. Return the value

**On write (`save_provider_config`):**
1. Write to database
2. Invalidate cache for `provider:{name}:*` (prefix invalidation)

### Cache Invalidation

<details>
<summary>Invalidation examples</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import (
    invalidate_provider_cache,
    invalidate_user_cache
)

# Invalidate all cached config for a provider
invalidate_provider_cache("replicate")

# Invalidate all cached data for a user
invalidate_user_cache(user_id=123)
```
</details>

### Prefix-Based Invalidation

<details>
<summary>Prefix invalidation example</summary>

```python
cache = get_provider_config_cache()

# Invalidate all keys starting with "provider:replicate:"
count = cache.invalidate_prefix("provider:replicate:")
```
</details>

---

## Hardware Info Caching

Hardware pricing information is cached separately with manual invalidation.

### Cache Behavior

<details>
<summary>Hardware info cache operations</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.replicate_client import (
    get_hardware_info,
    get_hardware_info_async,
    clear_hardware_info_cache,
    set_hardware_info
)

# Get hardware info (uses cache or defaults)
info = get_hardware_info()
info = await get_hardware_info_async(store)

# Clear cache to force reload from config
clear_hardware_info_cache()

# Manually set hardware info (updates cache)
set_hardware_info({
    "gpu-l40s": {"name": "L40S (48GB)", "cost_per_second": 0.000975}
})
```
</details>

### Default Hardware Pricing

<details>
<summary>Default hardware info values</summary>

```python
DEFAULT_HARDWARE_INFO = {
    "gpu-l40s": {"name": "L40S (48GB)", "cost_per_second": 0.000975},
    "gpu-a100-large": {"name": "A100 (80GB)", "cost_per_second": 0.001400},
}
```
</details>

---

## Decorator-Based Caching

For caching function results, use the `@cached` decorator:

<details>
<summary>Cached decorator usage</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import cached

@cached(ttl=120, key_prefix="mymodule")
def get_user(user_id: int) -> dict:
    # Expensive database query
    return {"id": user_id, "name": "..."}

@cached(ttl=60)
async def fetch_external_data() -> dict:
    # Async API call
    return await api_client.get("/data")

# Cache key is auto-generated from function name and arguments:
# "mymodule:get_user:123" for get_user(123)

# Access the underlying cache for manual invalidation
get_user._cache.clear()
```
</details>

---

## Credential Encryption

SimpleTuner encrypts sensitive credentials (like API tokens) at rest using Fernet symmetric encryption. This provides an additional layer of security beyond file permissions.

### How It Works

1. **Key Derivation**: A master secret is derived using PBKDF2-HMAC-SHA256 (100,000 iterations)
2. **Encryption**: Credentials are encrypted with Fernet (AES-128-CBC with HMAC)
3. **Storage**: Encrypted values are stored base64-encoded in the database

### Key Management

The encryption key is sourced in priority order:

| Priority | Source | Use Case |
|----------|--------|----------|
| 1 | `SIMPLETUNER_CREDENTIAL_KEY` env var | Production deployments, containers |
| 2 | `~/.simpletuner/credential.key` file | Local development, persistent key |
| 3 | Auto-generated | First-time setup (saved to file) |

<details>
<summary>Setting the key via environment</summary>

```bash
# Generate a secure key
export SIMPLETUNER_CREDENTIAL_KEY=$(openssl rand -base64 32)
```
</details>

**Key file location:**

```
~/.simpletuner/credential.key
```

The key file is created with `chmod 0600` (owner read/write only).

### Usage

Credential encryption is automatic when storing provider API tokens through the UI or API. You don't need to call these functions directly, but they're available:

<details>
<summary>Encryption/decryption functions</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.auth.credential_encryption import (
    encrypt_credential,
    decrypt_credential,
)

# Encrypt a token
encrypted = encrypt_credential("r8_my_api_token")

# Decrypt it back
plaintext = decrypt_credential(encrypted)
```
</details>

### Requirements

Credential encryption requires the `cryptography` package:

```bash
pip install cryptography
```

If not installed, credentials are stored in plaintext (with a warning in logs).

### Key Rotation

To rotate the encryption key:

1. Export existing credentials (they'll be decrypted with the old key)
2. Set the new key in environment or key file
3. Restart the server
4. Re-enter credentials through the UI (they'll be encrypted with the new key)

**Warning:** Changing the key without re-entering credentials will make existing encrypted credentials unreadable.

### Backup Considerations

When backing up SimpleTuner:

- **Include**: `~/.simpletuner/credential.key` (or note your `SIMPLETUNER_CREDENTIAL_KEY`)
- **Include**: The database containing encrypted credentials
- Both are needed to recover encrypted credentials

---

## Best Practices

### Secrets Management

1. **Use environment variables for CI/CD** - They're always available and take precedence
2. **Use file-based secrets for local development** - Easy to manage, persisted across restarts
3. **Use AWS/Vault for production** - Centralized, auditable, rotatable secrets
4. **Never commit secrets to version control** - Add `secrets.json` to `.gitignore`

### Cache Tuning

1. **Provider config (5 min TTL)** - Configuration rarely changes; longer TTL is fine
2. **User permissions (1 min TTL)** - Shorter TTL ensures permission changes propagate quickly
3. **Invalidate on write** - Always invalidate cache when updating the underlying data
4. **Monitor cache stats** - Use `cache.stats()` to check hit rates and expired counts

### Security Considerations

1. **File permissions** - Secrets files are created with `0o600` (owner only)
2. **Memory caching** - Secrets are cached in memory after first retrieval
3. **Clear sensitive caches** - Call `secrets.clear_cache()` if needed for security
4. **Audit logging** - All provider config changes are logged to the audit trail

---

## Troubleshooting

### Secret Not Found

<details>
<summary>Check provider availability</summary>

```python
secrets = get_secrets_manager()

# Check which providers are available
for provider in secrets._providers:
    print(f"{provider.__class__.__name__}: {provider.is_available()}")
```
</details>

### Cache Not Updating

<details>
<summary>Force cache invalidation</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import (
    get_provider_config_cache,
    invalidate_provider_cache
)

# Option 1: Invalidate specific provider
invalidate_provider_cache("replicate")

# Option 2: Clear entire cache
cache = get_provider_config_cache()
cache.clear()
```
</details>

### Vault Connection Issues

<details>
<summary>Verify Vault is accessible</summary>

```bash
# Check Vault status
curl -s $VAULT_ADDR/v1/sys/health

# Verify token is valid
curl -s -H "X-Vault-Token: $VAULT_TOKEN" $VAULT_ADDR/v1/auth/token/lookup-self
```
</details>

### AWS Secrets Manager Issues

<details>
<summary>Verify AWS credentials and permissions</summary>

```bash
# Check current identity
aws sts get-caller-identity

# Test secret access
aws secretsmanager get-secret-value --secret-id $SIMPLETUNER_AWS_SECRET_NAME
```
</details>

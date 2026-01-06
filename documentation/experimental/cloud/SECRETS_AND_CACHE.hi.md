# Secrets Providers और Cache System

यह दस्तावेज़ SimpleTuner के secrets management सिस्टम और cloud training के लिए caching infrastructure को कवर करता है।

## Overview

SimpleTuner layered secrets management सिस्टम इस्तेमाल करता है जिसमें कई provider backends और performance optimize करने के लिए in-memory caching layer शामिल है।

## Secrets Providers

`SecretsManager` क्लास कई secret providers को priority order में chain करती है। Secret प्राप्त करते समय providers को क्रम में जांचा जाता है जब तक value मिल न जाए।

### Provider Priority Order

1. **Environment Variables** (सबसे उच्च priority)
2. **File-based Secrets**
3. **AWS Secrets Manager**
4. **HashiCorp Vault** (सबसे निम्न priority)

यह क्रम environment variables को हमेशा override करने देता है, जो उपयोगी है:
- Local development overrides
- Injected secrets वाले container deployments
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

Environment provider हमेशा उपलब्ध होता है और बाकी सभी providers पर precedence लेता है।

**Key normalization rules:**
- Keys uppercase की जाती हैं
- Dashes (`-`) को underscores (`_`) में बदला जाता है
- Dots (`.`) को underscores (`_`) में बदला जाता है

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

Secrets को JSON या YAML फाइल में restrictive permissions के साथ रखा जा सकता है।

**Default locations (क्रम से चेक होता है):**
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

**Security note:** API के जरिए secrets सेव करने पर फाइल `chmod 0o600` (owner read/write only) के साथ बनाई जाती है।

### 3. AWS Secrets Manager

Production deployments के लिए, AWS Secrets Manager सुरक्षित, centralized secret storage देता है।

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

AWS Secrets Manager में secret JSON object होना चाहिए:

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

HashiCorp Vault उपयोग करने वाले enterprises के लिए।

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

Provider पहले KV v2 आज़माता है, fail होने पर v1 fallback करता है।
</details>

---
## Secrets Manager का उपयोग

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

### Secrets सेव करना (केवल File Provider)

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

SimpleTuner in-memory TTL (Time-To-Live) cache का उपयोग करता है ताकि database queries कम हों और frequently accessed data के लिए response time बेहतर हो।

### TTLCache Features

- `RLock` के साथ **thread-safe operations**
- **Per-key TTL** और configurable defaults
- **Automatic cleanup** of expired entries
- **LRU eviction** जब capacity भर जाए
- bulk cache clearing के लिए **Prefix-based invalidation**

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

SimpleTuner दो global cache instances बनाए रखता है:

| Cache | Default TTL | Max Size | Purpose |
|-------|-------------|----------|---------|
| Provider Config Cache | 300s (5 min) | 100 | Provider settings, webhook URLs, cost limits |
| User Permission Cache | 60s (1 min) | 500 | User permissions (security के लिए छोटा TTL) |

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

Provider configuration (webhook URLs, cost limits, hardware info) को database load कम करने के लिए cache किया जाता है।

### Cache Key Format

```
provider:{provider_name}:config
```

Example: `provider:replicate:config`

### Caching Behavior

**On read (`get_provider_config`):**
1. `provider:{name}:config` के लिए cache चेक करें
2. अगर मिला और expire नहीं हुआ, cached value लौटाएं
3. अगर missing/expired है, database से load करें
4. परिणाम को 5-minute TTL के साथ cache में रखें
5. value लौटाएं

**On write (`save_provider_config`):**
1. database में लिखें
2. `provider:{name}:*` के लिए cache invalidate करें (prefix invalidation)

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

Hardware pricing info अलग से cache किया जाता है और manual invalidation के साथ manage होता है।

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

Function results को cache करने के लिए `@cached` decorator का उपयोग करें:

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

SimpleTuner sensitive credentials (जैसे API tokens) को rest पर Fernet symmetric encryption से encrypt करता है। यह file permissions के अलावा अतिरिक्त सुरक्षा देता है।

### यह कैसे काम करता है

1. **Key Derivation**: master secret PBKDF2-HMAC-SHA256 (100,000 iterations) से derive होता है
2. **Encryption**: Credentials को Fernet (AES-128-CBC with HMAC) से encrypt किया जाता है
3. **Storage**: Encrypted values base64-encoded होकर database में स्टोर होते हैं

### Key Management

Encryption key priority order में लिया जाता है:

| Priority | Source | Use Case |
|----------|--------|----------|
| 1 | `SIMPLETUNER_CREDENTIAL_KEY` env var | Production deployments, containers |
| 2 | `~/.simpletuner/credential.key` file | Local development, persistent key |
| 3 | Auto-generated | First-time setup (file में सेव) |

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

Key file `chmod 0600` (owner read/write only) के साथ बनती है।

### उपयोग

UI या API के जरिए provider API tokens स्टोर करते समय credential encryption अपने आप लागू होता है। आपको इन functions को सीधे कॉल करने की जरूरत नहीं है, लेकिन ये उपलब्ध हैं:

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

Credential encryption के लिए `cryptography` package चाहिए:

```bash
pip install cryptography
```

अगर यह installed नहीं है, तो credentials plaintext में store होते हैं (और logs में warning आती है)।

### Key Rotation

Encryption key rotate करने के लिए:

1. मौजूदा credentials export करें (वे पुराने key से decrypt होंगे)
2. नया key environment या key file में सेट करें
3. server रीस्टार्ट करें
4. UI से credentials दोबारा दर्ज करें (वे नए key से encrypt होंगे)

**Warning:** बिना credentials दोबारा दर्ज किए key बदलने से मौजूदा encrypted credentials unreadable हो जाएंगे।

### Backup Considerations

SimpleTuner का backup लेते समय:

- **Include**: `~/.simpletuner/credential.key` (या अपने `SIMPLETUNER_CREDENTIAL_KEY` का नोट)
- **Include**: encrypted credentials वाले database को
- दोनों encrypted credentials recovery के लिए ज़रूरी हैं

---

## Best Practices

### Secrets Management

1. **CI/CD के लिए environment variables का उपयोग करें** - ये हमेशा उपलब्ध होते हैं और precedence लेते हैं
2. **Local development के लिए file-based secrets** - मैनेज करना आसान, restarts के बाद भी रहते हैं
3. **Production के लिए AWS/Vault** - centralized, auditable, rotatable secrets
4. **Secrets को version control में कभी commit न करें** - `secrets.json` को `.gitignore` में जोड़ें

### Cache Tuning

1. **Provider config (5 min TTL)** - कॉन्फ़िगरेशन शायद ही बदलता है; लंबा TTL ठीक है
2. **User permissions (1 min TTL)** - छोटा TTL permission बदलाव जल्दी propagate करता है
3. **Write पर invalidate करें** - underlying data बदलने पर हमेशा cache invalidate करें
4. **Cache stats मॉनिटर करें** - hit rates और expired counts के लिए `cache.stats()` देखें

### Security Considerations

1. **File permissions** - Secrets files `0o600` के साथ बनती हैं (owner only)
2. **Memory caching** - Secrets पहली retrieval के बाद memory में cache होते हैं
3. **Sensitive caches साफ करें** - जरूरत होने पर `secrets.clear_cache()` कॉल करें
4. **Audit logging** - सभी provider config बदलाव audit trail में log होते हैं

---

## Troubleshooting

### Secret नहीं मिला

<details>
<summary>Check provider availability</summary>

```python
secrets = get_secrets_manager()

# Check which providers are available
for provider in secrets._providers:
    print(f"{provider.__class__.__name__}: {provider.is_available()}")
```
</details>

### Cache अपडेट नहीं हो रहा

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

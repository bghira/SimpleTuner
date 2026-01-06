# Proveedores de secretos y sistema de caché

Este documento cubre el sistema de gestión de secretos de SimpleTuner y la infraestructura de caché para el entrenamiento en la nube.

## Resumen

SimpleTuner usa un sistema de gestión de secretos por capas con múltiples backends de proveedores y una capa de caché en memoria para optimizar el rendimiento.

## Proveedores de secretos

La clase `SecretsManager` encadena múltiples proveedores de secretos en un orden de prioridad. Al recuperar un secreto, los proveedores se revisan secuencialmente hasta encontrar un valor.

### Orden de prioridad de proveedores

1. **Variables de entorno** (prioridad más alta)
2. **Secretos basados en archivos**
3. **AWS Secrets Manager**
4. **HashiCorp Vault** (prioridad más baja)

Este orden permite que las variables de entorno siempre sobrescriban otras fuentes, lo cual es útil para:
- Overrides de desarrollo local
- Despliegues en contenedores con secretos inyectados
- Pipelines de CI/CD

### Claves de secretos conocidas

<details>
<summary>Constantes de claves de secretos</summary>

```python
REPLICATE_API_TOKEN  # Replicate API authentication
HF_TOKEN             # HuggingFace Hub token
CLOUD_WEBHOOK_SECRET # Webhook HMAC validation secret
```
</details>

---

## Configuración de proveedores

### 1. Variables de entorno (predeterminado)

El proveedor de entorno siempre está disponible y tiene prioridad sobre todos los demás proveedores.

**Reglas de normalización de claves:**
- Las claves se convierten a mayúsculas
- Los guiones (`-`) se convierten en guiones bajos (`_`)
- Los puntos (`.`) se convierten en guiones bajos (`_`)

**Prefijo opcional:**

<details>
<summary>Ejemplo de configuración en Python</summary>

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

**Uso:**

<details>
<summary>Configurar secretos vía entorno</summary>

```bash
# Set secrets via environment
export REPLICATE_API_TOKEN="r8_your_token_here"
export HF_TOKEN="hf_your_token_here"
export CLOUD_WEBHOOK_SECRET="your_webhook_secret"
```
</details>

### 2. Secretos basados en archivos

Los secretos pueden almacenarse en un archivo JSON o YAML con permisos restrictivos.

**Ubicaciones predeterminadas (revisadas en orden):**
- `~/.simpletuner/secrets.json`
- `~/.simpletuner/secrets.yaml`
- `~/.simpletuner/secrets.yml`

<details>
<summary>Ejemplos de formato de archivo (JSON/YAML)</summary>

**Formato de archivo (JSON):**

```json
{
  "REPLICATE_API_TOKEN": "r8_your_token_here",
  "HF_TOKEN": "hf_your_token_here",
  "CLOUD_WEBHOOK_SECRET": "your_webhook_secret"
}
```

**Formato de archivo (YAML):**

```yaml
REPLICATE_API_TOKEN: r8_your_token_here
HF_TOKEN: hf_your_token_here
CLOUD_WEBHOOK_SECRET: your_webhook_secret
```
</details>

<details>
<summary>Configuración manual</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import FileSecretProvider

# Use a custom file path
provider = FileSecretProvider(file_path="/path/to/secrets.json")
```
</details>

**Nota de seguridad:** Al guardar secretos vía la API, el archivo se crea con `chmod 0o600` (solo lectura/escritura para el propietario).

### 3. AWS Secrets Manager

Para despliegues en producción, AWS Secrets Manager ofrece almacenamiento seguro y centralizado de secretos.

<details>
<summary>Variables de entorno</summary>

```bash
# Required: Name of the secret in AWS Secrets Manager
export SIMPLETUNER_AWS_SECRET_NAME="simpletuner/production"

# Optional: AWS region (uses default if not set)
export AWS_DEFAULT_REGION="us-west-2"
```
</details>

<details>
<summary>Formato del secreto en AWS</summary>

El secreto en AWS Secrets Manager debe ser un objeto JSON:

```json
{
  "REPLICATE_API_TOKEN": "r8_your_token_here",
  "HF_TOKEN": "hf_your_token_here",
  "CLOUD_WEBHOOK_SECRET": "your_webhook_secret"
}
```
</details>

<details>
<summary>Configuración manual</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import AWSSecretsManagerProvider

provider = AWSSecretsManagerProvider(
    secret_name="simpletuner/production",
    region_name="us-west-2"
)
```
</details>

<details>
<summary>Permisos IAM de AWS requeridos</summary>

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

Para empresas que usan HashiCorp Vault para la gestión de secretos.

**Dependencias requeridas:**

```bash
pip install hvac
```

<details>
<summary>Variables de entorno</summary>

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
<summary>Configuración manual</summary>

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
<summary>Configuración de KV en Vault</summary>

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

El proveedor intenta automáticamente KV v2 primero y vuelve a v1 si falla.
</details>

---

## Uso del Secrets Manager

### Uso básico

<details>
<summary>Obtener secretos</summary>

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

### Guardar secretos (solo proveedor de archivo)

<details>
<summary>Configurar y eliminar secretos</summary>

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

### Configuración de proveedor personalizada

<details>
<summary>Ejemplo de configuración personalizada</summary>

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

## Sistema de caché

SimpleTuner usa un caché en memoria con TTL (Time-To-Live) para reducir consultas a la base de datos y mejorar tiempos de respuesta para datos accedidos con frecuencia.

### Funciones de TTLCache

- **Operaciones thread-safe** usando `RLock`
- **TTL por clave** con predeterminados configurables
- **Limpieza automática** de entradas expiradas
- **Evicción LRU** cuando está a capacidad
- **Invalidación por prefijo** para limpieza masiva de caché

### Configuración de caché

<details>
<summary>Inicialización de TTLCache</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import TTLCache

cache = TTLCache[str](
    default_ttl=300.0,      # 5 minutes default TTL
    max_size=1000,          # Maximum entries before eviction
    cleanup_interval=60.0   # Cleanup expired entries every 60s
)
```
</details>

### Instancias globales de caché

SimpleTuner mantiene dos instancias globales de caché:

| Caché | TTL predeterminado | Tamaño máximo | Propósito |
|-------|---------------------|---------------|-----------|
| Caché de configuración de proveedor | 300s (5 min) | 100 | Ajustes de proveedor, URLs de webhook, límites de costo |
| Caché de permisos de usuario | 60s (1 min) | 500 | Permisos de usuario (TTL más corto por seguridad) |

<details>
<summary>Acceso a cachés globales</summary>

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

### Operaciones básicas de caché

<details>
<summary>Referencia de operaciones de caché</summary>

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

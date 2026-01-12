# Idempotencia de trabajos y reserva de cuota

Este documento describe dos sistemas relacionados que garantizan un envío de trabajos seguro y confiable en el entrenamiento en la nube de SimpleTuner:

1. **Claves de idempotencia** - Evitan envíos de trabajos duplicados
2. **Reservas de cuota** - Consumo atómico de slots para límites de trabajos concurrentes

Ambos sistemas están diseñados para ser resilientes en entornos distribuidos, pipelines CI/CD y escenarios con inestabilidad de red.

## Claves de idempotencia

### Resumen

Las claves de idempotencia permiten a los clientes reintentar envíos de trabajos de forma segura sin crear duplicados. Si una solicitud se interrumpe (timeout de red, caída del proceso, etc.), el cliente puede reenviar con la misma clave y recibir la respuesta del trabajo original.

<details>
<summary>Cómo funciona la idempotencia (diagrama de secuencia)</summary>

```
Client                         Server
  |                              |
  |  POST /jobs/submit           |
  |  idempotency_key: "abc123"   |
  |----------------------------->|
  |                              |  Check idempotency_keys table
  |                              |  Key not found - proceed
  |                              |
  |                              |  Create job, store key->job_id
  |                              |
  |  { job_id: "xyz789" }        |
  |<-----------------------------|
  |                              |
  |  [Connection lost before     |
  |   client receives response]  |
  |                              |
  |  POST /jobs/submit (retry)   |
  |  idempotency_key: "abc123"   |
  |----------------------------->|
  |                              |  Key found - return existing job
  |                              |
  |  { job_id: "xyz789",         |
  |    idempotent_hit: true }    |
  |<-----------------------------|
```

</details>

### Formato de clave y recomendaciones

La clave de idempotencia es una cadena generada por el cliente. Formatos recomendados:

<details>
<summary>Ejemplos de formato de clave</summary>

```bash
# CI/CD builds - use commit SHA or build ID
idempotency_key="ci-build-${GITHUB_SHA}"
idempotency_key="jenkins-${BUILD_NUMBER}"

# Scheduled jobs - include timestamp or run identifier
idempotency_key="nightly-train-$(date +%Y%m%d)"

# User-triggered - combine user ID with action context
idempotency_key="user-42-config-flux-lora-$(date +%s)"

# UUID for guaranteed uniqueness (when duplicates are never desired)
idempotency_key="$(uuidgen)"
```

</details>

**Buenas prácticas:**
- Mantén las claves por debajo de 256 caracteres
- Usa caracteres seguros para URL (alfanuméricos, guiones, guiones bajos)
- Incluye suficiente contexto para identificar la operación lógica
- Para CI/CD, vincula a identificadores de commit/build/deploy

### TTL y expiración

Las claves de idempotencia expiran **a las 24 horas** desde su creación. Esto significa:

- Reintentos dentro de 24 horas devuelven el trabajo original
- Después de 24 horas, la misma clave crea un trabajo nuevo
- El TTL es configurable por clave pero por defecto es 24 horas

<details>
<summary>Ejemplo de configuración de TTL</summary>

```python
# Default: 24-hour TTL
await async_store.store_idempotency_key(
    key="ci-build-abc123",
    job_id="job-xyz789",
    user_id=42,
    ttl_hours=24  # Default
)
```

</details>

### Uso de la API

#### Enviar con clave de idempotencia

```bash
curl -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=replicate' \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name_to_load": "my-training-config",
    "idempotency_key": "ci-build-abc123"
  }'
```

<details>
<summary>Ejemplos de respuesta</summary>

**Respuesta para trabajo nuevo:**

```json
{
  "success": true,
  "job_id": "xyz789abc",
  "status": "uploading",
  "data_uploaded": true,
  "idempotent_hit": false
}
```

**Respuesta para duplicado (clave ya usada):**

```json
{
  "success": true,
  "job_id": "xyz789abc",
  "status": "running",
  "idempotent_hit": true
}
```

El campo `idempotent_hit: true` indica que la respuesta corresponde a un trabajo existente coincidente con la clave de idempotencia.

</details>

<details>
<summary>Esquema de base de datos</summary>

```sql
CREATE TABLE idempotency_keys (
    idempotency_key TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    user_id INTEGER,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE INDEX idx_idempotency_expires ON idempotency_keys(expires_at);
```

Las claves se limpian automáticamente durante las operaciones de búsqueda cuando han expirado.

</details>

## Sistema de reserva de cuota

### El problema: condiciones de carrera

Sin reservas, solicitudes concurrentes pueden saltarse los límites de cuota:

<details>
<summary>Ejemplo de condición de carrera</summary>

```
Request A                    Request B
    |                            |
    |  Check quota: 4/5 used     |
    |  (OK to proceed)           |  Check quota: 4/5 used
    |                            |  (OK to proceed)
    |                            |
    |  Create job (now 5/5)      |  Create job (now 6/5!)
    |                            |
```

Ambas solicitudes pasaron el chequeo de cuota, pero juntas excedieron el límite.

</details>

### La solución: reservas atómicas

El sistema de reservas proporciona semántica atómica de "reservar antes de crear":

<details>
<summary>Flujo de reserva</summary>

```
Request A                    Request B
    |                            |
    |  Reserve slot (4/5 -> 5/5) |
    |  Got reservation R1        |  Reserve slot (5/5)
    |                            |  DENIED - quota exceeded
    |                            |
    |  Create job successfully   |
    |  Consume reservation R1    |
    |                            |
```

</details>

### Cómo funciona

1. **Reserva previa**: Antes de crear un trabajo, solicita una reserva
2. **Chequeo y reclamo atómico**: La reserva solo se concede si la cuota lo permite
3. **Protección por TTL**: Las reservas expiran después de 5 minutos (evita bloqueos huérfanos)
4. **Consumir o liberar**: Después de crear el trabajo, consume la reserva; si falla, libérala

<details>
<summary>Ejemplo de código de reserva</summary>

```python
# The reservation flow
reservation_id = await async_store.reserve_job_slot(
    user_id=42,
    max_concurrent=5,  # User's quota limit
    ttl_seconds=300    # 5-minute expiration
)

if reservation_id is None:
    # Quota exceeded - reject immediately
    return SubmitJobResponse(
        success=False,
        error="Quota exceeded: Maximum 5 concurrent jobs allowed"
    )

try:
    # Proceed with job creation
    result = await submission_service.submit(ctx)

    # Job created successfully - consume the reservation
    await async_store.consume_reservation(reservation_id)

except Exception:
    # Job creation failed - release the reservation
    await async_store.release_reservation(reservation_id)
    raise
```

</details>

### Estados de reserva

| Estado | Descripción | Acción |
|-------|-------------|--------|
| Active | Slot reservado, trabajo aún no creado | Bloquea otras reservas |
| Consumed | El trabajo se creó con éxito | Ya no cuenta contra la cuota |
| Expired | TTL vencido sin consumo | Se ignora automáticamente |
| Released | Liberada explícitamente por fallo | Slot liberado de inmediato |

### TTL y limpieza automática

Las reservas tienen un **TTL de 5 minutos** (300 segundos). Esto cubre:

- **Clientes caídos**: Si un cliente muere en medio del envío, la reserva expira
- **Subidas lentas**: 5 minutos permite subir datasets grandes
- **Problemas de red**: Desconexiones temporales no bloquean slots de forma permanente

<details>
<summary>Consulta de enforcement de TTL</summary>

El TTL se aplica en tiempo de consulta; las reservas expiradas se limpian automáticamente:

```python
# During slot counting, expired reservations are ignored
cursor = await conn.execute("""
    SELECT COUNT(*) FROM job_reservations
    WHERE user_id = ? AND expires_at > ? AND consumed = 0
""", (user_id, now.isoformat()))
```

</details>

<details>
<summary>Esquema de base de datos</summary>

```sql
CREATE TABLE job_reservations (
    reservation_id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    consumed INTEGER DEFAULT 0
);

CREATE INDEX idx_reservations_user ON job_reservations(user_id);
CREATE INDEX idx_reservations_expires ON job_reservations(expires_at);
```

</details>

## Buenas prácticas para clientes de API

### Implementar lógica de reintentos

Al construir clientes que envían trabajos, implementa backoff exponencial con idempotencia:

<details>
<summary>Implementación de reintento en Python</summary>

```python
import time
import uuid
import requests

def submit_job_with_retry(config_name: str, max_retries: int = 3) -> dict:
    """Submit a job with automatic retry and idempotency protection."""

    # Generate idempotency key once, reuse across retries
    idempotency_key = f"client-{uuid.uuid4()}"

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8001/api/cloud/jobs/submit",
                params={"provider": "replicate"},
                json={
                    "config_name_to_load": config_name,
                    "idempotency_key": idempotency_key,
                },
                timeout=120,  # Allow time for data upload
            )

            result = response.json()

            if result.get("success"):
                if result.get("idempotent_hit"):
                    print(f"Retry matched existing job: {result['job_id']}")
                return result

            # Check for quota errors (don't retry these)
            error = result.get("error", "")
            if "Quota exceeded" in error:
                raise QuotaExceededError(error)

            # Other errors might be transient
            raise TransientError(error)

        except requests.exceptions.Timeout:
            # Network timeout - retry with same idempotency key
            pass
        except requests.exceptions.ConnectionError:
            # Connection failed - retry
            pass

        # Exponential backoff: 1s, 2s, 4s, ...
        sleep_time = 2 ** attempt
        print(f"Retry {attempt + 1}/{max_retries} in {sleep_time}s")
        time.sleep(sleep_time)

    raise MaxRetriesExceeded(f"Failed after {max_retries} attempts")
```

</details>

### Patrón de integración CI/CD

<details>
<summary>Ejemplo de GitHub Actions</summary>

```yaml
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - name: Submit Training Job
        id: submit
        run: |
          # Use commit SHA for idempotency - reruns of same commit are safe
          IDEMPOTENCY_KEY="gh-${{ github.repository }}-${{ github.sha }}"

          RESPONSE=$(curl -sf -X POST \
            "${SIMPLETUNER_URL}/api/cloud/jobs/submit?provider=replicate" \
            -H 'Content-Type: application/json' \
            -d "{
              \"config_name_to_load\": \"production\",
              \"idempotency_key\": \"${IDEMPOTENCY_KEY}\",
              \"tracker_run_name\": \"gh-${{ github.run_number }}\"
            }" || echo '{"success":false,"error":"Request failed"}')

          SUCCESS=$(echo "$RESPONSE" | jq -r '.success')
          JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id // empty')
          IDEMPOTENT=$(echo "$RESPONSE" | jq -r '.idempotent_hit // false')

          if [ "$SUCCESS" != "true" ]; then
            echo "::error::Job submission failed: $(echo "$RESPONSE" | jq -r '.error')"
            exit 1
          fi

          if [ "$IDEMPOTENT" = "true" ]; then
            echo "::notice::Matched existing job from previous run"
          fi

          echo "job_id=$JOB_ID" >> $GITHUB_OUTPUT
```

</details>

### Manejo de errores de cuota

Los errores de cuota no deben disparar reintentos: indican un límite que no cambiará:

<details>
<summary>Ejemplo de manejo de errores</summary>

```python
def handle_submission_response(response: dict) -> None:
    """Handle job submission response with appropriate error handling."""

    if not response.get("success"):
        error = response.get("error", "Unknown error")

        # Quota errors - inform user, don't retry
        if "Quota exceeded" in error:
            print(f"Cannot submit: {error}")
            print("Wait for existing jobs to complete or contact administrator")
            return

        # Cost limit errors - similar handling
        if "Cost limit" in error:
            print(f"Spending limit reached: {error}")
            return

        # Other errors might be transient
        raise TransientError(error)

    # Check for warnings even on success
    for warning in response.get("quota_warnings", []):
        print(f"Warning: {warning}")

    if response.get("cost_limit_warning"):
        print(f"Cost warning: {response['cost_limit_warning']}")
```

</details>

## Ejemplos

### Python: envío completo con manejo de errores

<details>
<summary>Ejemplo completo de cliente async</summary>

```python
import asyncio
import aiohttp
import uuid
from dataclasses import dataclass
from typing import Optional


@dataclass
class SubmitResult:
    success: bool
    job_id: Optional[str] = None
    error: Optional[str] = None
    was_duplicate: bool = False


async def submit_cloud_job(
    base_url: str,
    config_name: str,
    idempotency_key: Optional[str] = None,
    tracker_run_name: Optional[str] = None,
) -> SubmitResult:
    """Submit a cloud training job with idempotency protection.

    Args:
        base_url: SimpleTuner server URL (e.g., "http://localhost:8001")
        config_name: Name of the training configuration to use
        idempotency_key: Optional key for deduplication (auto-generated if None)
        tracker_run_name: Optional name for experiment tracking

    Returns:
        SubmitResult with job_id on success, error message on failure
    """
    # Generate idempotency key if not provided
    if idempotency_key is None:
        idempotency_key = f"py-client-{uuid.uuid4()}"

    payload = {
        "config_name_to_load": config_name,
        "idempotency_key": idempotency_key,
    }

    if tracker_run_name:
        payload["tracker_run_name"] = tracker_run_name

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{base_url}/api/cloud/jobs/submit",
                params={"provider": "replicate"},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),  # 5 min for uploads
            ) as response:
                result = await response.json()

                if result.get("success"):
                    return SubmitResult(
                        success=True,
                        job_id=result["job_id"],
                        was_duplicate=result.get("idempotent_hit", False),
                    )
                else:
                    return SubmitResult(
                        success=False,
                        error=result.get("error", "Unknown error"),
                    )

        except asyncio.TimeoutError:
            return SubmitResult(
                success=False,
                error="Request timed out - job may have been created, check with same idempotency key",
            )
        except aiohttp.ClientError as e:
            return SubmitResult(
                success=False,
                error=f"Connection error: {e}",
            )


# Usage
async def main():
    result = await submit_cloud_job(
        base_url="http://localhost:8001",
        config_name="flux-lora-v1",
        idempotency_key="training-batch-2024-01-15",
        tracker_run_name="flux-lora-experiment-42",
    )

    if result.success:
        if result.was_duplicate:
            print(f"Matched existing job: {result.job_id}")
        else:
            print(f"Created new job: {result.job_id}")
    else:
        print(f"Submission failed: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
```

</details>

### Bash: script de CI idempotente

<details>
<summary>Script bash completo</summary>

```bash
#!/bin/bash
set -euo pipefail

# Configuration
SIMPLETUNER_URL="${SIMPLETUNER_URL:-http://localhost:8001}"
CONFIG_NAME="${1:-production-lora}"
MAX_RETRIES=3

# Generate idempotency key from git context if available
if [ -n "${GITHUB_SHA:-}" ]; then
    IDEMPOTENCY_KEY="github-${GITHUB_REPOSITORY}-${GITHUB_SHA}"
elif [ -n "${CI_COMMIT_SHA:-}" ]; then
    IDEMPOTENCY_KEY="gitlab-${CI_PROJECT_PATH}-${CI_COMMIT_SHA}"
else
    IDEMPOTENCY_KEY="manual-$(date +%Y%m%d-%H%M%S)-$$"
fi

echo "Using idempotency key: ${IDEMPOTENCY_KEY}"

# Submit with retry
for i in $(seq 1 $MAX_RETRIES); do
    echo "Attempt $i/$MAX_RETRIES..."

    RESPONSE=$(curl -sf -X POST \
        "${SIMPLETUNER_URL}/api/cloud/jobs/submit?provider=replicate" \
        -H 'Content-Type: application/json' \
        -d "{
            \"config_name_to_load\": \"${CONFIG_NAME}\",
            \"idempotency_key\": \"${IDEMPOTENCY_KEY}\"
        }" 2>&1) || {
        if [ $i -lt $MAX_RETRIES ]; then
            SLEEP_TIME=$((2 ** i))
            echo "Request failed, retrying in ${SLEEP_TIME}s..."
            sleep $SLEEP_TIME
            continue
        else
            echo "Failed after $MAX_RETRIES attempts"
            exit 1
        fi
    }

    SUCCESS=$(echo "$RESPONSE" | jq -r '.success')

    if [ "$SUCCESS" = "true" ]; then
        JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id')
        IDEMPOTENT=$(echo "$RESPONSE" | jq -r '.idempotent_hit')

        if [ "$IDEMPOTENT" = "true" ]; then
            echo "Matched existing job: $JOB_ID"
        else
            echo "Created new job: $JOB_ID"
        fi

        # Output for CI systems
        echo "JOB_ID=$JOB_ID"
        exit 0
    else
        ERROR=$(echo "$RESPONSE" | jq -r '.error')

        # Don't retry quota errors
        if echo "$ERROR" | grep -q "Quota exceeded"; then
            echo "Quota exceeded: $ERROR"
            exit 1
        fi

        if [ $i -lt $MAX_RETRIES ]; then
            echo "Error: $ERROR, retrying..."
            sleep $((2 ** i))
        else
            echo "Failed: $ERROR"
            exit 1
        fi
    fi
done
```

</details>

## Monitoreo y depuración

### Comprobar el estado de la clave de idempotencia

<details>
<summary>Consultas SQL para depuración</summary>

Actualmente, las claves de idempotencia son internas a la base de datos. Para depurar:

```sql
-- Check if a key exists (connect to jobs.db)
SELECT idempotency_key, job_id, created_at, expires_at
FROM idempotency_keys
WHERE idempotency_key = 'your-key-here';

-- List all active keys
SELECT * FROM idempotency_keys
WHERE expires_at > datetime('now')
ORDER BY created_at DESC;
```

</details>

### Comprobar el estado de reservas

<details>
<summary>Consultas SQL para reservas</summary>

```sql
-- Active (unconsumed, unexpired) reservations
SELECT reservation_id, user_id, created_at, expires_at
FROM job_reservations
WHERE consumed = 0 AND expires_at > datetime('now');

-- Count active slots per user
SELECT user_id, COUNT(*) as active_reservations
FROM job_reservations
WHERE consumed = 0 AND expires_at > datetime('now')
GROUP BY user_id;
```

</details>

### Problemas comunes

| Síntoma | Causa | Solución |
|---------|-------|----------|
| Se crean trabajos duplicados | No usar claves de idempotencia | Agrega `idempotency_key` a las solicitudes |
| "Quota exceeded" en el primer trabajo | Reservas huérfanas por solicitudes caídas | Espera 5 minutos a que expire el TTL |
| La clave de idempotencia no coincide | Clave expirada (>24h) | Usa una clave nueva o extiende el TTL |
| Conteos de cuota inesperados | Cuenta reservas + trabajos activos | Es el comportamiento correcto |

## Ver también

- [TUTORIAL.md](TUTORIAL.md) - Recorrido completo de entrenamiento en la nube
- [ENTERPRISE.md](../server/ENTERPRISE.md) - Gestión de cuotas multi‑tenant
- [OPERATIONS_TUTORIAL.md](OPERATIONS_TUTORIAL.md) - Guía de despliegue en producción

# Idempotencia de jobs e reserva de cota

Este documento descreve dois sistemas relacionados que garantem envio seguro e confiavel de jobs no treinamento em nuvem do SimpleTuner:

1. **Chaves de idempotencia** - Evitam envios duplicados de jobs
2. **Reservas de cota** - Consumo atomico de slots para limites de jobs concorrentes

Ambos os sistemas foram projetados para resiliencia em ambientes distribuidos, pipelines CI/CD e cenarios com instabilidade de rede.

## Chaves de idempotencia

### Visao geral

Chaves de idempotencia permitem que clientes repitam envios de jobs com seguranca sem criar duplicatas. Se uma requisicao for interrompida (timeout de rede, crash de processo, etc.), o cliente pode reenviar com a mesma chave e receber a resposta do job original.

<details>
<summary>Como a idempotencia funciona (diagrama de sequencia)</summary>

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

### Formato da chave e recomendacoes

A chave de idempotencia e uma string gerada pelo cliente. Formatos recomendados:

<details>
<summary>Exemplos de formato de chave</summary>

```bash
# Builds de CI/CD - use SHA de commit ou ID de build
idempotency_key="ci-build-${GITHUB_SHA}"
idempotency_key="jenkins-${BUILD_NUMBER}"

# Jobs agendados - inclua timestamp ou identificador de run
idempotency_key="nightly-train-$(date +%Y%m%d)"

# Acionado por usuario - combine ID do usuario com contexto da acao
idempotency_key="user-42-config-flux-lora-$(date +%s)"

# UUID para unicidade garantida (quando duplicatas nunca sao desejadas)
idempotency_key="$(uuidgen)"
```

</details>

**Boas praticas:**
- Mantenha as chaves abaixo de 256 caracteres
- Use caracteres seguros para URL (alfanumericos, hifens, underscores)
- Inclua contexto suficiente para identificar a operacao logica
- Para CI/CD, associe a identificadores de commit/build/deploy

### TTL e expiracao

Chaves de idempotencia expiram apos **24 horas** a partir da criacao. Isso significa:

- Retries dentro de 24 horas retornam o job original
- Depois de 24 horas, a mesma chave cria um novo job
- O TTL e configuravel por chave, mas o padrao e 24 horas

<details>
<summary>Exemplo de configuracao de TTL</summary>

```python
# Default: TTL de 24 horas
await async_store.store_idempotency_key(
    key="ci-build-abc123",
    job_id="job-xyz789",
    user_id=42,
    ttl_hours=24  # Default
)
```

</details>

### Uso da API

#### Envio com chave de idempotencia

```bash
curl -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=replicate' \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name_to_load": "my-training-config",
    "idempotency_key": "ci-build-abc123"
  }'
```

<details>
<summary>Exemplos de resposta</summary>

**Resposta para novo job:**

```json
{
  "success": true,
  "job_id": "xyz789abc",
  "status": "uploading",
  "data_uploaded": true,
  "idempotent_hit": false
}
```

**Resposta para duplicado (chave ja usada):**

```json
{
  "success": true,
  "job_id": "xyz789abc",
  "status": "running",
  "idempotent_hit": true
}
```

O campo `idempotent_hit: true` indica que a resposta corresponde a um job existente associado pela chave de idempotencia.

</details>

<details>
<summary>Schema do banco de dados</summary>

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

Chaves expirada sao limpas automaticamente durante operacoes de lookup.

</details>

## Sistema de reserva de cotas

### O problema: race conditions

Sem reservas, requisicoes concorrentes podem ultrapassar limites de cota:

<details>
<summary>Exemplo de race condition</summary>

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

Ambas as requisicoes passaram na verificacao de cota, mas juntas excederam o limite.

</details>

### A solucao: reservas atomicas

O sistema de reservas fornece semantica atomica de "reservar antes de criar":

<details>
<summary>Fluxo de reserva</summary>

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

### Como funciona

1. **Reserva pre-flight**: Antes de criar um job, solicite uma reserva
2. **Check-and-claim atomico**: A reserva so e bem-sucedida se a cota permitir
3. **Protecao por TTL**: Reservas expiram apos 5 minutos (evita locks orfaos)
4. **Consumir ou liberar**: Apos criar o job, consuma a reserva; em falha, libere-a

<details>
<summary>Exemplo de codigo de reserva</summary>

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

### Estados da reserva

| Estado | Descricao | Acao |
|-------|-------------|--------|
| Active | Slot reservado, job ainda nao criado | Bloqueia outras reservas |
| Consumed | Job criado com sucesso | Nao conta mais contra a cota |
| Expired | TTL expirou sem consumo | Ignorado automaticamente |
| Released | Liberado explicitamente em falha | Slot liberado imediatamente |

### TTL e limpeza automatica

Reservas tem **TTL de 5 minutos** (300 segundos). Isso lida com:

- **Clientes travados**: Se um cliente morrer durante o envio, a reserva expira
- **Uploads lentos**: 5 minutos permitem uploads de datasets grandes
- **Problemas de rede**: Desconexoes temporarias nao bloqueiam slots permanentemente

<details>
<summary>Consulta de enforcement de TTL</summary>

O TTL e imposto no momento da consulta - reservas expiradas sao limpas automaticamente:

```python
# During slot counting, expired reservations are ignored
cursor = await conn.execute("""
    SELECT COUNT(*) FROM job_reservations
    WHERE user_id = ? AND expires_at > ? AND consumed = 0
""", (user_id, now.isoformat()))
```

</details>

<details>
<summary>Schema do banco de dados</summary>

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

## Boas praticas para clientes de API

### Implementando logica de retry

Ao criar clientes que enviam jobs, implemente backoff exponencial com idempotencia:

<details>
<summary>Implementacao de retry em Python</summary>

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

### Padrao de integracao CI/CD

<details>
<summary>Exemplo de GitHub Actions</summary>

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

### Tratando erros de cota

Erros de cota nao devem disparar retries - eles indicam um limite que nao mudara:

<details>
<summary>Exemplo de tratamento de erro</summary>

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

## Exemplos

### Python: envio completo com tratamento de erro

<details>
<summary>Exemplo completo de cliente async</summary>

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

### Bash: Script de CI idempotente

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

## Monitoramento e debugging

### Verificando status da chave de idempotencia

<details>
<summary>Consultas SQL para debugging</summary>

Atualmente, as chaves de idempotencia sao internas ao banco de dados. Para depurar:

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

### Verificando status de reserva

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

### Problemas comuns

| Sintoma | Causa | Solucao |
|---------|-------|----------|
| Jobs duplicados criados | Nao usar chaves de idempotencia | Adicione `idempotency_key` nas requisicoes |
| "Quota exceeded" no primeiro job | Reservas orfas de requisicoes que crasharam | Aguarde 5 minutos pela expiracao do TTL |
| Chave de idempotencia nao corresponde | Chave expirada (>24h) | Use uma chave nova ou estenda o TTL |
| Contagens de cota inesperadas | Contando reservas + jobs ativos | Isso e o comportamento correto |

## Veja tambem

- [TUTORIAL.md](TUTORIAL.md) - Passo a passo completo de treinamento em nuvem
- [ENTERPRISE.md](../server/ENTERPRISE.md) - Gerenciamento de cotas multi-tenant
- [OPERATIONS_TUTORIAL.md](OPERATIONS_TUTORIAL.md) - Orientacoes de deploy em producao

# Infraestrutura de resiliencia

O sistema de treinamento em nuvem do SimpleTuner usa circuit breakers e logica de retry para lidar com falhas de forma graciosa quando servicos externos apresentam problemas.

## Visao geral

Dois padroes principais de resiliencia:

1. **Circuit Breaker** - Evita falhas em cascata ao interromper requisicoes para servicos que falham
2. **Retry com backoff exponencial** - Tenta novamente falhas transitorias com atrasos crescentes

## Padrao Circuit Breaker

Um circuit breaker monitora chamadas para um servico externo. Quando as falhas excedem um limite, o circuito "abre" e bloqueia novas requisicoes por um periodo de cooldown.

### Estados

| Estado | Descricao | Comportamento |
|-------|-------------|----------|
| **CLOSED** | Operacao normal | Requisicoes passam, falhas sao contabilizadas |
| **OPEN** | Servico falhando | Requisicoes sao bloqueadas imediatamente |
| **HALF_OPEN** | Testando recuperacao | Requisicoes limitadas sao permitidas para testar se o servico se recuperou |

<details>
<summary>Diagrama de transicao de estados</summary>

```
                                    Success threshold met
                                   +------------------------+
                                   |                        |
                                   v                        |
+----------+   Failure threshold    +----------+  Timeout    +-------------+
|  CLOSED  | ---------------------->|   OPEN   | ----------->|  HALF_OPEN  |
+----------+                        +----------+             +-------------+
     ^                                   ^                        |
     |                                   |                        |
     |         Success resets            |     Any failure        |
     |          failure count            +------------------------+
     |
     +--------------------------------------------------------------------+
                            Success in CLOSED state
```

</details>

### Configuracao

| Parametro | Padrao | Descricao |
|-----------|---------|-------------|
| `failure_threshold` | 5 | Falhas consecutivas antes do circuito abrir |
| `success_threshold` | 2 | Sucessos em HALF_OPEN para fechar o circuito |
| `timeout_seconds` | 60.0 | Segundos antes de OPEN transicionar para HALF_OPEN |
| `excluded_exceptions` | `()` | Tipos de excecao que nao contam como falhas |

<details>
<summary>Exemplo de configuracao em Python</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
)

config = CircuitBreakerConfig(
    failure_threshold=5,
    success_threshold=2,
    timeout_seconds=60.0,
    excluded_exceptions=(),
)

breaker = CircuitBreaker("replicate-api", config)
```

Para o Replicate, use o breaker pre-configurado:

```python
from simpletuner.simpletuner_sdk.server.services.cloud.resilience import (
    get_replicate_circuit_breaker,
)

breaker = get_replicate_circuit_breaker()
# Uses: failure_threshold=5, success_threshold=2, timeout_seconds=30.0
```

</details>

<details>
<summary>Exemplos de uso</summary>

**Como context manager:**

```python
breaker = CircuitBreaker("replicate-api")

async def submit_job():
    try:
        async with breaker:
            response = await client.post("/api/submit", data=job_data)
            return response.json()
    except CircuitBreakerError as e:
        print(f"Service unavailable. Retry after {e.retry_after:.1f} seconds")
        return None
```

**Como decorator:**

```python
@breaker
async def call_replicate_api():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.replicate.com/v1/predictions")
        return response.json()
```

**Com factory de cliente HTTP:**

```python
async with get_async_client(circuit_breaker_name="replicate-api") as client:
    response = await client.get("https://api.replicate.com/v1/predictions")
```

</details>

<details>
<summary>Como o envio de job usa circuit breakers</summary>

```python
# From job_submission.py (simplified)
async def submit(self, ctx: SubmissionContext) -> SubmissionResult:
    circuit = await get_circuit_breaker(ctx.provider)

    if not await circuit.can_execute():
        return SubmissionResult(
            success=False,
            error=f"Provider '{ctx.provider}' is temporarily unavailable.",
        )

    try:
        cloud_job = await client.run_job(config=config, ...)
        await circuit.record_success()
    except Exception as provider_exc:
        await circuit.record_failure(provider_exc)
        return SubmissionResult(success=False, error=str(provider_exc))
```

Se o circuito estiver aberto (apos 5 falhas consecutivas), o envio de job e bloqueado imediatamente.

</details>

## Padrao de retry

Quando uma requisicao falha com erro transitorio, tente novamente com backoff exponencial:

1. Aguarde um atraso curto
2. Tente novamente a requisicao
3. Se falhar de novo, espere mais
4. Continue com atrasos crescentes ate atingir o maximo de tentativas

### Configuracao

| Parametro | Padrao | Descricao |
|-----------|---------|-------------|
| `max_attempts` | 3 | Maximo de tentativas (incluindo inicial) |
| `base_delay` | 1.0 | Atraso inicial em segundos |
| `max_delay` | 30.0 | Limite maximo de atraso |
| `exponential_base` | 2.0 | Multiplicador por tentativa |
| `jitter` | True | Adiciona jitter aleatorio de 0-25% |
| `retryable_status_codes` | `(429, 500, 502, 503, 504)` | Codigos HTTP para retry |

### Calculo de atraso

```
delay = min(base_delay * (exponential_base ^ attempt), max_delay)
if jitter:
    delay += delay * random(0, 0.25)
```

| Tentativa | Atraso base | Com jitter |
|---------|------------|-------------|
| 1 | 1.0s | 1.0-1.25s |
| 2 | 2.0s | 2.0-2.5s |
| 3 | 4.0s | 4.0-5.0s |
| 4 | 8.0s | 8.0-10.0s |
| 5 | 16.0s | 16.0-20.0s |
| 6+ | 30.0s (cap) | 30.0-37.5s |

<details>
<summary>Exemplos de uso</summary>

**Chamada direta de funcao:**

```python
from simpletuner.simpletuner_sdk.server.services.cloud.resilience import (
    retry_async,
    RetryConfig,
)

async def fetch_predictions():
    async def _call():
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.replicate.com/v1/predictions")
            response.raise_for_status()
            return response.json()

    config = RetryConfig(max_attempts=5, base_delay=2.0)
    return await retry_async(_call, config=config)
```

**Como decorator:**

```python
@retry(config=RetryConfig(max_attempts=5))
async def call_api():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.replicate.com/v1/predictions")
        response.raise_for_status()
        return response.json()
```

**Combinando circuit breaker e retry:**

```python
@retry(config=RetryConfig(max_attempts=3))
@breaker
async def resilient_api_call():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.replicate.com/v1/predictions")
        return response.json()
```

A ordem importa: o retry envolve o circuit breaker, entao falhas se acumulam entre retries.

</details>

## Monitoramento

### Integracao com health check

O endpoint `/api/cloud/health` inclui status do circuit breaker:

```bash
curl http://localhost:8080/api/cloud/health
```

| Estado do circuito | Status de health | Mensagem |
|--------------|---------------|---------|
| `closed` | `healthy` | "Circuit closed - normal operation" |
| `half_open` | `degraded` | "Circuit half-open - testing recovery" |
| `open` | `unhealthy` | "Circuit open - blocking requests" |

<details>
<summary>Exemplo de resposta de health</summary>

```json
{
  "status": "degraded",
  "components": [
    {
      "name": "database",
      "status": "healthy",
      "latency_ms": 1.2
    },
    {
      "name": "circuit_breaker_replicate-api",
      "status": "unhealthy",
      "message": "Circuit open - blocking requests (failures: 5)"
    }
  ]
}
```

</details>

<details>
<summary>Health check programatico</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.resilience import (
    get_all_circuit_breaker_health,
    get_circuit_breaker,
)

# All breakers
health = get_all_circuit_breaker_health()

# Single breaker
breaker = get_circuit_breaker("replicate-api")
health = breaker.get_health()
```

</details>

### Logging

Circuit breakers e logica de retry emitem mensagens de log estruturadas:

```
WARNING - Circuit breaker 'replicate-api' opening after 5 failures: ConnectionError
INFO - Circuit breaker 'replicate-api' transitioning from OPEN to HALF_OPEN
INFO - Circuit breaker 'replicate-api' closing after 2 successful calls

WARNING - Attempt 1/3 failed, retrying in 1.15s: TimeoutError
ERROR - All 3 attempts failed: TimeoutError
```

## Configuracao do operador

### Configuracoes do provedor

```bash
curl -X PUT http://localhost:8080/api/cloud/providers/replicate \
  -H "Content-Type: application/json" \
  -d '{"http_timeout": 60.0}'
```

Timeouts maiores reduzem falsos positivos de requisicoes lentas mas bem-sucedidas.

### Reset manual

<details>
<summary>Resetando circuit breakers</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.resilience import (
    get_circuit_breaker,
    reset_all_circuit_breakers,
)

# Reset a specific breaker
breaker = get_circuit_breaker("replicate-api")
breaker.reset()

# Reset all breakers
reset_all_circuit_breakers()
```

</details>

## Comportamento durante outages do provedor

| Fase | Comportamento |
|-------|----------|
| **Falhas iniciais (1-4)** | Requisicoes tentadas, retry lida com erros transitorios |
| **Circuito abre (5+)** | Todas as requisicoes sao rejeitadas imediatamente com "Provider temporarily unavailable" |
| **Teste de recuperacao** | Apos timeout, requests de teste limitadas sao permitidas |
| **Recuperacao total** | Circuito fecha, operacao normal retorna |

## Solucao de problemas

**Circuit breaker preso em aberto:**
- Verifique se o provedor esta realmente fora do ar
- Verifique se as credenciais de API sao validas
- Verifique conectividade de rede e configuracoes de proxy
- Resete manualmente o breaker se necessario

**Muitos falsos positivos:**
- Aumente `failure_threshold` (ex.: de 5 para 10)
- Aumente `timeout_seconds` para recuperacao mais lenta
- Configure `excluded_exceptions` para ignorar certos tipos de erro

**Nao fazendo retry de erros esperados:**
- Verifique se o tipo de excecao esta em `retryable_exceptions`
- Verifique se o status HTTP esta em `retryable_status_codes`

## Veja tambem

- [Operations Guide](OPERATIONS_TUTORIAL.md) - Deploy e monitoramento em producao
- [Cloud Training Tutorial](TUTORIAL.md) - Guia de inicio
- [Replicate Integration](REPLICATE.md) - Configuracao especifica do provedor

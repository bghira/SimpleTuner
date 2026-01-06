# Infraestructura de resiliencia

El sistema de entrenamiento en la nube de SimpleTuner usa circuit breakers y lógica de reintentos para manejar fallos de forma adecuada cuando los servicios externos tienen problemas.

## Resumen

Dos patrones principales de resiliencia:

1. **Circuit Breaker** - Evita fallos en cascada al detener solicitudes a servicios con fallas
2. **Reintento con backoff exponencial** - Reintenta automáticamente fallos transitorios con demoras crecientes

## Patrón de Circuit Breaker

Un circuit breaker monitorea llamadas a un servicio externo. Cuando las fallas superan un umbral, el circuito "se abre" y bloquea nuevas solicitudes durante un periodo de enfriamiento.

### Estados

| Estado | Descripción | Comportamiento |
|--------|-------------|----------------|
| **CLOSED** | Operación normal | Las solicitudes pasan, se cuentan las fallas |
| **OPEN** | El servicio está fallando | Las solicitudes se bloquean de inmediato |
| **HALF_OPEN** | Probando recuperación | Se permiten solicitudes limitadas para probar si el servicio se recuperó |

<details>
<summary>Diagrama de transición de estados</summary>

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

### Configuración

| Parámetro | Predeterminado | Descripción |
|-----------|----------------|-------------|
| `failure_threshold` | 5 | Fallas consecutivas antes de abrir el circuito |
| `success_threshold` | 2 | Éxitos en HALF_OPEN para cerrar el circuito |
| `timeout_seconds` | 60.0 | Segundos antes de que OPEN pase a HALF_OPEN |
| `excluded_exceptions` | `()` | Tipos de excepción que no cuentan como fallas |

<details>
<summary>Ejemplo de configuración en Python</summary>

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

Para Replicate, usa el breaker preconfigurado:

```python
from simpletuner.simpletuner_sdk.server.services.cloud.resilience import (
    get_replicate_circuit_breaker,
)

breaker = get_replicate_circuit_breaker()
# Uses: failure_threshold=5, success_threshold=2, timeout_seconds=30.0
```

</details>

<details>
<summary>Ejemplos de uso</summary>

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

**Como decorador:**

```python
@breaker
async def call_replicate_api():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.replicate.com/v1/predictions")
        return response.json()
```

**Con factory de clientes HTTP:**

```python
async with get_async_client(circuit_breaker_name="replicate-api") as client:
    response = await client.get("https://api.replicate.com/v1/predictions")
```

</details>

<details>
<summary>Cómo el envío de trabajos usa circuit breakers</summary>

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

Si el circuito está abierto (después de 5 fallas consecutivas), el envío de trabajos se bloquea de inmediato.

</details>

## Patrón de reintentos

Cuando una solicitud falla con un error transitorio, reintenta con backoff exponencial:

1. Espera una demora corta
2. Reintenta la solicitud
3. Si falla nuevamente, espera más tiempo
4. Continúa con demoras crecientes hasta alcanzar los intentos máximos

### Configuración

| Parámetro | Predeterminado | Descripción |
|-----------|----------------|-------------|
| `max_attempts` | 3 | Intentos máximos (incluye el inicial) |
| `base_delay` | 1.0 | Demora inicial en segundos |
| `max_delay` | 30.0 | Tope de demora máxima |
| `exponential_base` | 2.0 | Multiplicador por intento |
| `jitter` | True | Agrega jitter aleatorio de 0-25% |
| `retryable_status_codes` | `(429, 500, 502, 503, 504)` | Códigos HTTP a reintentar |

### Cálculo de demora

```
delay = min(base_delay * (exponential_base ^ attempt), max_delay)
if jitter:
    delay += delay * random(0, 0.25)
```

| Intento | Demora base | Con jitter |
|---------|-------------|------------|
| 1 | 1.0s | 1.0-1.25s |
| 2 | 2.0s | 2.0-2.5s |
| 3 | 4.0s | 4.0-5.0s |
| 4 | 8.0s | 8.0-10.0s |
| 5 | 16.0s | 16.0-20.0s |
| 6+ | 30.0s (capado) | 30.0-37.5s |

<details>
<summary>Ejemplos de uso</summary>

**Llamada de función directa:**

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

**Como decorador:**

```python
@retry(config=RetryConfig(max_attempts=5))
async def call_api():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.replicate.com/v1/predictions")
        response.raise_for_status()
        return response.json()
```

**Combinando circuit breaker y reintento:**

```python
@retry(config=RetryConfig(max_attempts=3))
@breaker
async def resilient_api_call():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.replicate.com/v1/predictions")
        return response.json()
```

El orden importa: el reintento envuelve al circuit breaker, por lo que las fallas se acumulan entre reintentos.

</details>

## Monitoreo

### Integración con chequeos de salud

El endpoint `/api/cloud/health` incluye el estado del circuit breaker:

```bash
curl http://localhost:8080/api/cloud/health
```

| Estado del circuito | Estado de salud | Mensaje |
|---------------------|-----------------|---------|
| `closed` | `healthy` | "Circuit closed - normal operation" |
| `half_open` | `degraded` | "Circuit half-open - testing recovery" |
| `open` | `unhealthy` | "Circuit open - blocking requests" |

<details>
<summary>Respuesta de salud de ejemplo</summary>

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
<summary>Chequeo de salud programático</summary>

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

Los circuit breakers y la lógica de reintento emiten mensajes de log estructurados:

```
WARNING - Circuit breaker 'replicate-api' opening after 5 failures: ConnectionError
INFO - Circuit breaker 'replicate-api' transitioning from OPEN to HALF_OPEN
INFO - Circuit breaker 'replicate-api' closing after 2 successful calls

WARNING - Attempt 1/3 failed, retrying in 1.15s: TimeoutError
ERROR - All 3 attempts failed: TimeoutError
```

## Configuración de operador

### Ajustes del proveedor

```bash
curl -X PUT http://localhost:8080/api/cloud/providers/replicate \
  -H "Content-Type: application/json" \
  -d '{"http_timeout": 60.0}'
```

Tiempos de espera más largos reducen falsos positivos de solicitudes lentas pero exitosas.

### Reinicio manual

<details>
<summary>Reiniciar circuit breakers</summary>

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

## Comportamiento durante caídas del proveedor

| Fase | Comportamiento |
|------|----------------|
| **Fallas iniciales (1-4)** | Se intentan solicitudes; la lógica de reintento maneja errores transitorios |
| **Se abre el circuito (5+)** | Todas las solicitudes se rechazan de inmediato con "Provider temporarily unavailable" |
| **Pruebas de recuperación** | Tras el timeout, se permiten solicitudes de prueba limitadas |
| **Recuperación total** | El circuito se cierra y se reanuda la operación normal |

## Solución de problemas

**Circuit breaker atascado en abierto:**
- Verifica si el proveedor realmente está caído
- Comprueba que las credenciales de API sean válidas
- Revisa la conectividad de red y la configuración de proxy
- Reinicia manualmente el breaker si es necesario

**Demasiados falsos positivos:**
- Incrementa `failure_threshold` (p. ej., de 5 a 10)
- Incrementa `timeout_seconds` para una recuperación más lenta
- Configura `excluded_exceptions` para ignorar ciertos tipos de error

**No reintenta errores esperados:**
- Verifica que el tipo de excepción esté en `retryable_exceptions`
- Revisa si el código HTTP está en `retryable_status_codes`

## Ver también

- [Guía de operaciones](OPERATIONS_TUTORIAL.md) - Despliegue en producción y monitoreo
- [Tutorial de entrenamiento en la nube](TUTORIAL.md) - Guía de inicio
- [Integración con Replicate](REPLICATE.md) - Configuración específica del proveedor

# Resilience Infrastructure

SimpleTuner's cloud training system uses circuit breakers and retry logic to handle failures gracefully when external services experience issues.

## Overview

Two primary resilience patterns:

1. **Circuit Breaker** - Prevents cascading failures by stopping requests to failing services
2. **Retry with Exponential Backoff** - Automatically retries transient failures with increasing delays

## Circuit Breaker Pattern

A circuit breaker monitors calls to an external service. When failures exceed a threshold, the circuit "opens" and blocks further requests for a cooldown period.

### States

| State | Description | Behavior |
|-------|-------------|----------|
| **CLOSED** | Normal operation | Requests flow through, failures are counted |
| **OPEN** | Service is failing | Requests are blocked immediately |
| **HALF_OPEN** | Testing recovery | Limited requests allowed to test if service recovered |

<details>
<summary>State transition diagram</summary>

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

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `failure_threshold` | 5 | Consecutive failures before the circuit opens |
| `success_threshold` | 2 | Successes in HALF_OPEN to close the circuit |
| `timeout_seconds` | 60.0 | Seconds before OPEN transitions to HALF_OPEN |
| `excluded_exceptions` | `()` | Exception types that don't count as failures |

<details>
<summary>Python configuration example</summary>

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

For Replicate, use the pre-configured breaker:

```python
from simpletuner.simpletuner_sdk.server.services.cloud.resilience import (
    get_replicate_circuit_breaker,
)

breaker = get_replicate_circuit_breaker()
# Uses: failure_threshold=5, success_threshold=2, timeout_seconds=30.0
```

</details>

<details>
<summary>Usage examples</summary>

**As a context manager:**

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

**As a decorator:**

```python
@breaker
async def call_replicate_api():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.replicate.com/v1/predictions")
        return response.json()
```

**With HTTP client factory:**

```python
async with get_async_client(circuit_breaker_name="replicate-api") as client:
    response = await client.get("https://api.replicate.com/v1/predictions")
```

</details>

<details>
<summary>How job submission uses circuit breakers</summary>

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

If the circuit is open (after 5 consecutive failures), job submission is blocked immediately.

</details>

## Retry Pattern

When a request fails with a transient error, retry with exponential backoff:

1. Wait a short delay
2. Retry the request
3. If it fails again, wait longer
4. Continue with increasing delays until max attempts reached

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_attempts` | 3 | Maximum attempts (including initial) |
| `base_delay` | 1.0 | Initial delay in seconds |
| `max_delay` | 30.0 | Maximum delay cap |
| `exponential_base` | 2.0 | Multiplier per attempt |
| `jitter` | True | Add 0-25% random jitter |
| `retryable_status_codes` | `(429, 500, 502, 503, 504)` | HTTP codes to retry |

### Delay Calculation

```
delay = min(base_delay * (exponential_base ^ attempt), max_delay)
if jitter:
    delay += delay * random(0, 0.25)
```

| Attempt | Base Delay | With Jitter |
|---------|------------|-------------|
| 1 | 1.0s | 1.0-1.25s |
| 2 | 2.0s | 2.0-2.5s |
| 3 | 4.0s | 4.0-5.0s |
| 4 | 8.0s | 8.0-10.0s |
| 5 | 16.0s | 16.0-20.0s |
| 6+ | 30.0s (capped) | 30.0-37.5s |

<details>
<summary>Usage examples</summary>

**Direct function call:**

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

**As a decorator:**

```python
@retry(config=RetryConfig(max_attempts=5))
async def call_api():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.replicate.com/v1/predictions")
        response.raise_for_status()
        return response.json()
```

**Combining circuit breaker and retry:**

```python
@retry(config=RetryConfig(max_attempts=3))
@breaker
async def resilient_api_call():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.replicate.com/v1/predictions")
        return response.json()
```

The order matters: retry wraps the circuit breaker, so failures accumulate across retries.

</details>

## Monitoring

### Health Check Integration

The `/api/cloud/health` endpoint includes circuit breaker status:

```bash
curl http://localhost:8080/api/cloud/health
```

| Circuit State | Health Status | Message |
|--------------|---------------|---------|
| `closed` | `healthy` | "Circuit closed - normal operation" |
| `half_open` | `degraded` | "Circuit half-open - testing recovery" |
| `open` | `unhealthy` | "Circuit open - blocking requests" |

<details>
<summary>Example health response</summary>

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
<summary>Programmatic health check</summary>

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

Circuit breakers and retry logic emit structured log messages:

```
WARNING - Circuit breaker 'replicate-api' opening after 5 failures: ConnectionError
INFO - Circuit breaker 'replicate-api' transitioning from OPEN to HALF_OPEN
INFO - Circuit breaker 'replicate-api' closing after 2 successful calls

WARNING - Attempt 1/3 failed, retrying in 1.15s: TimeoutError
ERROR - All 3 attempts failed: TimeoutError
```

## Operator Configuration

### Provider Settings

```bash
curl -X PUT http://localhost:8080/api/cloud/providers/replicate \
  -H "Content-Type: application/json" \
  -d '{"http_timeout": 60.0}'
```

Longer timeouts reduce false positives from slow but successful requests.

### Manual Reset

<details>
<summary>Resetting circuit breakers</summary>

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

## Behavior During Provider Outages

| Phase | Behavior |
|-------|----------|
| **Initial failures (1-4)** | Requests attempted, retry logic handles transient errors |
| **Circuit opens (5+)** | All requests immediately rejected with "Provider temporarily unavailable" |
| **Recovery testing** | After timeout, limited test requests allowed |
| **Full recovery** | Circuit closes, normal operation resumes |

## Troubleshooting

**Circuit breaker stuck open:**
- Check if the provider is actually down
- Verify API credentials are valid
- Check network connectivity and proxy settings
- Manually reset the breaker if needed

**Too many false positives:**
- Increase `failure_threshold` (e.g., from 5 to 10)
- Increase `timeout_seconds` for slower recovery
- Configure `excluded_exceptions` to ignore certain error types

**Not retrying expected errors:**
- Verify the exception type is in `retryable_exceptions`
- Check if the HTTP status code is in `retryable_status_codes`

## See Also

- [Operations Guide](OPERATIONS_TUTORIAL.md) - Production deployment and monitoring
- [Cloud Training Tutorial](TUTORIAL.md) - Getting started guide
- [Replicate Integration](REPLICATE.md) - Provider-specific configuration

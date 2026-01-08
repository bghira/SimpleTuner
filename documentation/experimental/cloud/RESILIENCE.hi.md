# Resilience Infrastructure

SimpleTuner का cloud training सिस्टम circuit breakers और retry logic का उपयोग करता है ताकि बाहरी सेवाओं में समस्याओं के समय failures को gracefully हैंडल किया जा सके।

## Overview

दो मुख्य resilience patterns:

1. **Circuit Breaker** - failing services पर requests रोककर cascading failures से बचाता है
2. **Retry with Exponential Backoff** - transient failures को बढ़ते delays के साथ स्वतः retry करता है

## Circuit Breaker Pattern

Circuit breaker बाहरी सेवा के calls को मॉनिटर करता है। जब failures एक threshold पार कर जाते हैं, तो circuit “open” हो जाता है और cooldown अवधि के लिए requests ब्लॉक करता है।

### States

| State | Description | Behavior |
|-------|-------------|----------|
| **CLOSED** | Normal operation | Requests गुजरती हैं, failures गिने जाते हैं |
| **OPEN** | Service failing है | Requests तुरंत ब्लॉक हो जाते हैं |
| **HALF_OPEN** | Recovery टेस्ट | सीमित requests की अनुमति होती है |

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
| `failure_threshold` | 5 | circuit open होने से पहले consecutive failures |
| `success_threshold` | 2 | HALF_OPEN में सफल calls की संख्या ताकि circuit close हो |
| `timeout_seconds` | 60.0 | OPEN से HALF_OPEN में जाने से पहले seconds |
| `excluded_exceptions` | `()` | वे exception types जिन्हें failures में नहीं गिना जाता |

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

Replicate के लिए pre-configured breaker उपयोग करें:

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

**Context manager के रूप में:**

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

**Decorator के रूप में:**

```python
@breaker
async def call_replicate_api():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.replicate.com/v1/predictions")
        return response.json()
```

**HTTP client factory के साथ:**

```python
async with get_async_client(circuit_breaker_name="replicate-api") as client:
    response = await client.get("https://api.replicate.com/v1/predictions")
```

</details>

<details>
<summary>Job submission circuit breakers कैसे उपयोग करता है</summary>

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

यदि circuit open है (5 consecutive failures के बाद), तो job submission तुरंत ब्लॉक हो जाता है।

</details>

## Retry Pattern

जब कोई request transient error के साथ fail होती है, तो exponential backoff के साथ retry करें:

1. छोटी देरी इंतज़ार करें
2. Request retry करें
3. यदि फिर fail हो, अधिक देर इंतज़ार करें
4. Max attempts तक बढ़ती देरी के साथ जारी रखें

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_attempts` | 3 | अधिकतम प्रयास (initial सहित) |
| `base_delay` | 1.0 | प्रारंभिक देरी (seconds) |
| `max_delay` | 30.0 | अधिकतम देरी सीमा |
| `exponential_base` | 2.0 | हर attempt पर multiplier |
| `jitter` | True | 0-25% random jitter जोड़ें |
| `retryable_status_codes` | `(429, 500, 502, 503, 504)` | retry करने योग्य HTTP codes |

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

**Decorator के रूप में:**

```python
@retry(config=RetryConfig(max_attempts=5))
async def call_api():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.replicate.com/v1/predictions")
        response.raise_for_status()
        return response.json()
```

**Circuit breaker और retry संयोजन:**

```python
@retry(config=RetryConfig(max_attempts=3))
@breaker
async def resilient_api_call():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.replicate.com/v1/predictions")
        return response.json()
```

Order मायने रखता है: retry, circuit breaker को wrap करता है, इसलिए failures retries के पार जमा होते हैं।

</details>

## Monitoring

### Health Check Integration

`/api/cloud/health` endpoint में circuit breaker status शामिल होता है:

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

Circuit breakers और retry logic structured log messages emit करते हैं:

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

लंबे timeouts धीमे लेकिन सफल requests से आने वाले false positives कम करते हैं।

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

## Provider Outages के दौरान व्यवहार

| Phase | Behavior |
|-------|----------|
| **Initial failures (1-4)** | Requests की कोशिश होती है, retry logic transient errors संभालता है |
| **Circuit opens (5+)** | सभी requests तुरंत "Provider temporarily unavailable" के साथ reject होती हैं |
| **Recovery testing** | timeout के बाद सीमित test requests allowed |
| **Full recovery** | Circuit close होता है, normal operation resumes |

## Troubleshooting

**Circuit breaker stuck open:**
- provider सच में down है या नहीं जांचें
- API credentials वैध हैं या नहीं देखें
- network connectivity और proxy settings जांचें
- आवश्यकता हो तो manually breaker reset करें

**Too many false positives:**
- `failure_threshold` बढ़ाएँ (जैसे 5 से 10)
- धीमी recovery के लिए `timeout_seconds` बढ़ाएँ
- कुछ error types को ignore करने हेतु `excluded_exceptions` सेट करें

**Expected errors retry नहीं हो रहे:**
- exception type `retryable_exceptions` में है या नहीं देखें
- HTTP status code `retryable_status_codes` में है या नहीं देखें

## See Also

- [Operations Guide](OPERATIONS_TUTORIAL.md) - Production deployment और monitoring
- [Cloud Training Tutorial](TUTORIAL.md) - Getting started गाइड
- [Replicate Integration](REPLICATE.md) - Provider-specific configuration

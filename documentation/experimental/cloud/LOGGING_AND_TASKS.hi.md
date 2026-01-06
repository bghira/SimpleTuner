# Structured Logging और Background Tasks

यह दस्तावेज़ SimpleTuner के cloud training फीचर में structured logging सिस्टम और background task workers को कवर करता है।

## विषय सूची

- [Structured Logging](#structured-logging)
  - [Configuration](#configuration)
  - [JSON Log Format](#json-log-format)
  - [Field Injection के लिए LogContext](#logcontext-for-field-injection)
  - [Correlation IDs](#correlation-ids)
- [Background Tasks](#background-tasks)
  - [Job Status Polling Worker](#job-status-polling-worker)
  - [Queue Processing Worker](#queue-processing-worker)
  - [Approval Expiration Worker](#approval-expiration-worker)
  - [Configuration Options](#configuration-options)
- [Logs से Debugging](#debugging-with-logs)

---

## Structured Logging

SimpleTuner का cloud training structured JSON logging सिस्टम इस्तेमाल करता है जो consistent, parseable log output देता है और distributed tracing के लिए automatic correlation ID tracking जोड़ता है।

### Configuration

Environment variables से logging कॉन्फ़िगर करें:

```bash
# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
export SIMPLETUNER_LOG_LEVEL="INFO"

# Format: "json" (structured) या "text" (traditional)
export SIMPLETUNER_LOG_FORMAT="json"

# Optional: stdout के अलावा file में log करें
export SIMPLETUNER_LOG_FILE="/var/log/simpletuner/cloud.log"
```

<details>
<summary>Programmatic configuration</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.structured_logging import (
    configure_structured_logging,
    init_from_env,
)

# Configure with explicit options
configure_structured_logging(
    level="INFO",
    json_output=True,
    log_file="/var/log/simpletuner/cloud.log",
    include_stack_info=False,  # Include stack traces for errors
)

# Or initialize from environment variables
init_from_env()
```

</details>

### JSON Log Format

जब JSON output enable हो, तो हर log entry में शामिल होता है:

<details>
<summary>Example JSON log entry</summary>

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "INFO",
  "logger": "simpletuner.cloud.jobs",
  "message": "Job submitted successfully",
  "correlation_id": "abc123def456",
  "source": {
    "file": "jobs.py",
    "line": 350,
    "function": "submit_job"
  },
  "extra": {
    "job_id": "xyz789",
    "provider": "replicate",
    "cost_estimate": 2.50
  }
}
```

</details>

| Field | विवरण |
|-------|-------------|
| `timestamp` | UTC में ISO 8601 timestamp |
| `level` | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `logger` | Logger नाम hierarchy |
| `message` | Human-readable log message |
| `correlation_id` | Request tracing ID (auto-generated या propagated) |
| `source` | File, line number, और function नाम |
| `extra` | LogContext से आए अतिरिक्त structured fields |

### Field Injection के लिए LogContext

किसी scope के अंदर सभी logs में structured fields अपने आप जोड़ने के लिए `LogContext` इस्तेमाल करें:

<details>
<summary>LogContext usage example</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.structured_logging import (
    get_logger,
    LogContext,
)

logger = get_logger("simpletuner.cloud.jobs")

async def process_job(job_id: str, provider: str):
    # All logs within this block include job_id and provider
    with LogContext(job_id=job_id, provider=provider):
        logger.info("Starting job processing")

        # Nested context adds more fields
        with LogContext(step="validation"):
            logger.info("Validating configuration")

        with LogContext(step="submission"):
            logger.info("Submitting to provider")

        logger.info("Job processing complete")
```

Output logs में context fields शामिल होंगे:

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "INFO",
  "logger": "simpletuner.cloud.jobs",
  "message": "Starting job processing",
  "correlation_id": "abc123",
  "extra": {
    "job_id": "xyz789",
    "provider": "replicate"
  }
}
```

</details>

आम तौर पर inject किए जाने वाले fields:

| Field | उद्देश्य |
|-------|---------|
| `job_id` | Training job identifier |
| `provider` | Cloud provider (replicate, आदि) |
| `user_id` | Authenticated user |
| `step` | Processing phase (validation, upload, submission) |
| `attempt` | Retry attempt number |

### Correlation IDs

Correlation IDs service boundaries के पार request tracing को सक्षम करते हैं। ये:

1. हर नए request thread के लिए **auto-generate** होते हैं (यदि मौजूद नहीं हैं)
2. `X-Correlation-ID` HTTP header के जरिए **propagate** होते हैं
3. thread-local storage में **stored** रहते हैं ताकि logs में अपने आप inject हों
4. cloud providers को outbound HTTP requests में **include** किए जाते हैं

<details>
<summary>Correlation ID flow diagram</summary>

```
User Request
     |
     v
[X-Correlation-ID: abc123]  <-- Incoming header (or auto-generated)
     |
     v
[Thread-local storage]  <-- set_correlation_id("abc123")
     |
     +---> Log entry: {"correlation_id": "abc123", ...}
     |
     +---> Outbound HTTP: X-Correlation-ID: abc123
           (to Replicate API)
```

</details>

<details>
<summary>Manual correlation ID management</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.http_client import (
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id,
)

# Get current ID (auto-generates if none exists)
current_id = get_correlation_id()

# Set a specific ID (e.g., from incoming request header)
set_correlation_id("request-abc-123")

# Clear when request completes
clear_correlation_id()
```

</details>

<details>
<summary>Correlation ID in HTTP clients</summary>

HTTP client factory outbound requests में correlation ID अपने आप जोड़ता है:

```python
from simpletuner.simpletuner_sdk.server.services.cloud.http_client import (
    get_async_client,
)

# Correlation ID is automatically added to X-Correlation-ID header
async with get_async_client() as client:
    response = await client.get("https://api.replicate.com/v1/predictions")
    # Request includes: X-Correlation-ID: <current-id>
```

</details>

---
## Background Tasks

Cloud training सिस्टम कई background workers चलाता है जो asynchronous operations संभालते हैं।

### Background Task Manager

सभी background tasks `BackgroundTaskManager` singleton द्वारा मैनेज होते हैं:

<details>
<summary>Task manager usage</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.background_tasks import (
    get_task_manager,
    start_background_tasks,
    stop_background_tasks,
)

# Start all configured tasks (typically in app lifespan)
await start_background_tasks()

# Stop gracefully on shutdown
await stop_background_tasks()
```

</details>

### Job Status Polling Worker

Job polling worker cloud providers से job status sync करता है। यह तब उपयोगी है जब webhooks उपलब्ध न हों (जैसे firewall के पीछे)।

**उद्देश्य:**
- active jobs (pending, uploading, queued, running) को cloud providers से poll करना
- local job store को current status से अपडेट करना
- status बदलाव पर SSE events emit करना
- terminal statuses के लिए queue entries अपडेट करना

<details>
<summary>Polling flow diagram</summary>

```
[Every 30 seconds]
     |
     v
List active jobs from local store
     |
     v
Group by provider
     |
     +---> [replicate] --> Get status from API --> Update local job
     |
     v
Emit SSE events for status changes
     |
     v
Update queue on terminal statuses (completed, failed, cancelled)
```

</details>

<details>
<summary>Auto-enable logic</summary>

अगर कोई webhook URL कॉन्फ़िगर नहीं है, तो polling worker अपने आप शुरू हो जाता है:

```python
# In background_tasks.py
async def _should_auto_enable_polling(self) -> bool:
    config = await store.get_config("replicate")
    return not config.get("webhook_url")  # Enable if no webhook
```

</details>

### Queue Processing Worker

Queue priority और concurrency limits के आधार पर job scheduling और dispatch संभालता है।

**उद्देश्य:**
- हर 5 सेकंड में job queue प्रोसेस करना
- priority के अनुसार jobs dispatch करना
- user/organization के लिए concurrency limits का सम्मान करना
- queue entry state transitions संभालना

**Queue Processing Interval:** 5 सेकंड (fixed)

### Approval Expiration Worker

Pending approval requests जो deadline पार कर चुकी हैं उन्हें अपने आप expire और reject करता है।

**उद्देश्य:**
- हर 5 मिनट में expired approval requests जांचना
- expired approvals वाले jobs को auto-reject करना
- queue entries को failed state में अपडेट करना
- expired approvals के लिए SSE notifications emit करना

<details>
<summary>Processing flow diagram</summary>

```
[Every 5 minutes]
     |
     v
List pending approval requests
     |
     v
Filter expired requests (past deadline)
     |
     v
Mark approval requests as expired
     |
     +---> Update queue entries to "failed"
     |
     +---> Update job status to "cancelled"
     |
     +---> Emit SSE "approval_expired" events
```

</details>

### Configuration Options

#### Environment Variable

```bash
# Set custom polling interval (seconds)
export SIMPLETUNER_JOB_POLL_INTERVAL="60"
```

<details>
<summary>Enterprise configuration file</summary>

`simpletuner-enterprise.yaml` बनाएं:

```yaml
background:
  job_status_polling:
    enabled: true
    interval_seconds: 30
  queue_processing:
    enabled: true
    interval_seconds: 5
```

</details>

#### Configuration Properties

| Property | Default | Description |
|----------|---------|-------------|
| `job_polling_enabled` | false (no webhook पर auto) | explicit polling enable करें |
| `job_polling_interval` | 30 seconds | Polling interval |
| Queue processing | Always enabled | Disable नहीं किया जा सकता |
| Approval expiration | Always enabled | हर 5 मिनट में चेक करता है |

<details>
<summary>Accessing configuration programmatically</summary>

```python
from simpletuner.simpletuner_sdk.server.config.enterprise import get_enterprise_config

config = get_enterprise_config()

if config.job_polling_enabled:
    interval = config.job_polling_interval
    print(f"Polling enabled with {interval}s interval")
```

</details>

---

## Logs से Debugging

### संबंधित log entries खोजें

Correlation ID का उपयोग करके request को सभी components में trace करें:

<details>
<summary>Log filtering commands</summary>

```bash
# Find all logs for a specific request
grep '"correlation_id": "abc123"' /var/log/simpletuner/cloud.log

# Or with jq for JSON parsing
cat /var/log/simpletuner/cloud.log | jq 'select(.correlation_id == "abc123")'
```

</details>

<details>
<summary>Filtering by job</summary>

```bash
# Find all logs for a specific job
cat /var/log/simpletuner/cloud.log | \
  jq 'select(.extra.job_id == "xyz789")'
```

</details>

<details>
<summary>Monitoring background tasks</summary>

```bash
# Watch polling activity
cat /var/log/simpletuner/cloud.log | \
  jq 'select(.message | contains("polling")) | {timestamp, message}'

# Monitor approval expirations
cat /var/log/simpletuner/cloud.log | \
  jq 'select(.message | contains("expired"))'
```

</details>

### Log Level की सिफारिशें

| Environment | Level | Reason |
|-------------|-------|-----------|
| Development | DEBUG | Troubleshooting के लिए पूरी visibility |
| Staging | INFO | मुख्य घटनाओं के साथ सामान्य संचालन |
| Production | INFO या WARNING | visibility और volume के बीच संतुलन |

### Common Log Messages

| Message | Level | मतलब |
|---------|-------|---------|
| "Starting job status polling" | INFO | Polling worker शुरू हो गया |
| "Synced N active jobs" | DEBUG | Polling cycle पूरा हुआ |
| "Queue scheduler started" | INFO | Queue processing सक्रिय |
| "Expired N approval requests" | INFO | Approvals auto-reject हुए |
| "Failed to sync job X" | DEBUG | एक job sync फेल (transient) |
| "Error in job polling" | ERROR | Polling loop में error |

### Log Aggregators के साथ इंटीग्रेशन

JSON log format इनसे compatible है:

- **Elasticsearch/Kibana**: JSON logs की direct ingestion
- **Splunk**: field extraction के साथ JSON parsing
- **Datadog**: JSON parsing के साथ log pipeline
- **Loki/Grafana**: `json` parser इस्तेमाल करें

<details>
<summary>Loki/Promtail configuration example</summary>

```yaml
scrape_configs:
  - job_name: simpletuner
    static_configs:
      - targets: [localhost]
        labels:
          job: simpletuner
          __path__: /var/log/simpletuner/cloud.log
    pipeline_stages:
      - json:
          expressions:
            level: level
            correlation_id: correlation_id
            job_id: extra.job_id
      - labels:
          level:
          correlation_id:
```

</details>

### Troubleshooting Checklist

1. **Request trace नहीं हो रही?**
   - चेक करें कि `X-Correlation-ID` header सेट हो रहा है
   - सत्यापित करें कि `CorrelationIDFilter` loggers से जुड़ा है

2. **Context fields दिखाई नहीं दे रहे?**
   - सुनिश्चित करें कि code `LogContext` block के भीतर है
   - JSON output enable है या नहीं, चेक करें

3. **Polling काम नहीं कर रहा?**
   - चेक करें कि webhook URL कॉन्फ़िगर है (auto-polling बंद कर देता है)
   - explicit polling इस्तेमाल कर रहे हों तो enterprise config सत्यापित करें
   - "Starting job status polling" log message देखें

4. **Queue processing नहीं चल रही?**
   - "Queue scheduler started" message देखें
   - "Failed to start queue processing" में errors देखें

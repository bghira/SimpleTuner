# Structured Logging and Background Tasks

This document covers the structured logging system and background task workers in SimpleTuner's cloud training feature.

## Table of Contents

- [Structured Logging](#structured-logging)
  - [Configuration](#configuration)
  - [JSON Log Format](#json-log-format)
  - [LogContext for Field Injection](#logcontext-for-field-injection)
  - [Correlation IDs](#correlation-ids)
- [Background Tasks](#background-tasks)
  - [Job Status Polling Worker](#job-status-polling-worker)
  - [Queue Processing Worker](#queue-processing-worker)
  - [Approval Expiration Worker](#approval-expiration-worker)
  - [Configuration Options](#configuration-options)
- [Debugging with Logs](#debugging-with-logs)

---

## Structured Logging

SimpleTuner's cloud training uses a structured JSON logging system that provides consistent, parseable log output with automatic correlation ID tracking for distributed tracing.

### Configuration

Configure logging via environment variables:

```bash
# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
export SIMPLETUNER_LOG_LEVEL="INFO"

# Format: "json" (structured) or "text" (traditional)
export SIMPLETUNER_LOG_FORMAT="json"

# Optional: Log to file in addition to stdout
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

When JSON output is enabled, each log entry includes:

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

| Field | Description |
|-------|-------------|
| `timestamp` | ISO 8601 timestamp in UTC |
| `level` | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `logger` | Logger name hierarchy |
| `message` | Human-readable log message |
| `correlation_id` | Request tracing ID (auto-generated or propagated) |
| `source` | File, line number, and function name |
| `extra` | Additional structured fields from LogContext |

### LogContext for Field Injection

Use `LogContext` to automatically add structured fields to all logs within a scope:

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

Output logs will include the context fields:

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

Common fields to inject:

| Field | Purpose |
|-------|---------|
| `job_id` | Training job identifier |
| `provider` | Cloud provider (replicate, etc.) |
| `user_id` | Authenticated user |
| `step` | Processing phase (validation, upload, submission) |
| `attempt` | Retry attempt number |

### Correlation IDs

Correlation IDs enable request tracing across service boundaries. They are:

1. **Auto-generated** for each new request thread if not present
2. **Propagated** via the `X-Correlation-ID` HTTP header
3. **Stored** in thread-local storage for automatic log injection
4. **Included** in outbound HTTP requests to cloud providers

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

The HTTP client factory automatically includes the correlation ID in outbound requests:

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

The cloud training system runs several background workers to handle asynchronous operations.

### Background Task Manager

All background tasks are managed by the `BackgroundTaskManager` singleton:

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

The job polling worker synchronizes job statuses from cloud providers. This is useful when webhooks are not available (e.g., behind a firewall).

**Purpose:**
- Poll active jobs (pending, uploading, queued, running) from cloud providers
- Update local job store with current status
- Emit SSE events when status changes
- Update queue entries for terminal statuses

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

The polling worker starts automatically if no webhook URL is configured:

```python
# In background_tasks.py
async def _should_auto_enable_polling(self) -> bool:
    config = await store.get_config("replicate")
    return not config.get("webhook_url")  # Enable if no webhook
```

</details>

### Queue Processing Worker

Handles job scheduling and dispatch based on queue priority and concurrency limits.

**Purpose:**
- Process the job queue every 5 seconds
- Dispatch jobs according to priority
- Respect concurrency limits per user/organization
- Handle queue entry state transitions

**Queue Processing Interval:** 5 seconds (fixed)

### Approval Expiration Worker

Automatically expires and rejects pending approval requests that have passed their deadline.

**Purpose:**
- Check for expired approval requests every 5 minutes
- Auto-reject jobs with expired approvals
- Update queue entries to failed state
- Emit SSE notifications for expired approvals

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

Create `simpletuner-enterprise.yaml`:

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
| `job_polling_enabled` | false (auto if no webhook) | Enable explicit polling |
| `job_polling_interval` | 30 seconds | Polling interval |
| Queue processing | Always enabled | Cannot be disabled |
| Approval expiration | Always enabled | Checks every 5 minutes |

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

## Debugging with Logs

### Finding Related Log Entries

Use the correlation ID to trace a request across all components:

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

### Log Level Recommendations

| Environment | Level | Rationale |
|-------------|-------|-----------|
| Development | DEBUG | Full visibility for troubleshooting |
| Staging | INFO | Normal operation with key events |
| Production | INFO or WARNING | Balance between visibility and volume |

### Common Log Messages

| Message | Level | Meaning |
|---------|-------|---------|
| "Starting job status polling" | INFO | Polling worker started |
| "Synced N active jobs" | DEBUG | Polling cycle completed |
| "Queue scheduler started" | INFO | Queue processing active |
| "Expired N approval requests" | INFO | Approvals auto-rejected |
| "Failed to sync job X" | DEBUG | Single job sync failed (transient) |
| "Error in job polling" | ERROR | Polling loop encountered error |

### Integrating with Log Aggregators

The JSON log format is compatible with:

- **Elasticsearch/Kibana**: Direct ingestion of JSON logs
- **Splunk**: JSON parsing with field extraction
- **Datadog**: Log pipeline with JSON parsing
- **Loki/Grafana**: Use `json` parser

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

1. **Request not being traced?**
   - Check if `X-Correlation-ID` header is being set
   - Verify `CorrelationIDFilter` is attached to loggers

2. **Context fields not appearing?**
   - Ensure code is within `LogContext` block
   - Check that JSON output is enabled

3. **Polling not working?**
   - Check if webhook URL is configured (disables auto-polling)
   - Verify enterprise config if using explicit polling
   - Look for "Starting job status polling" log message

4. **Queue not processing?**
   - Check for "Queue scheduler started" message
   - Look for errors in "Failed to start queue processing"

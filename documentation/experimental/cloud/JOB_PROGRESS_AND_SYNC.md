# Job Progress and Sync API

This document covers SimpleTuner's mechanisms for monitoring cloud training job progress and keeping local job state synchronized with cloud providers.

## Overview

SimpleTuner provides multiple approaches to track job status:

| Method | Use Case | Latency | Resource Usage |
|--------|----------|---------|----------------|
| Inline Progress API | UI polling for running jobs | Low (5s default) | Per-job API calls |
| Job Sync (pull) | Discover jobs from provider | Medium (on-demand) | Batch API call |
| `sync_active` param | Refresh active job statuses | Medium (on-demand) | Per-active-job calls |
| Background Poller | Automatic status updates | Configurable (30s default) | Continuous polling |
| Webhooks | Real-time push notifications | Instant | No polling required |

## Inline Progress API

### Endpoint

```
GET /api/cloud/jobs/{job_id}/inline-progress
```

### Purpose

Returns compact progress information for a single running job, suitable for displaying inline status updates in a job list without fetching full logs.

### Response

```json
{
  "job_id": "abc123",
  "stage": "Training",
  "last_log": "Step 1500/5000 - loss: 0.0234",
  "progress": 30.0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | The job identifier |
| `stage` | string or null | Current training stage: `Preprocessing`, `Warmup`, `Training`, `Validation`, `Saving checkpoint` |
| `last_log` | string or null | Last log line (truncated to 80 chars) |
| `progress` | float or null | Progress percentage (0-100) based on step/epoch parsing |

### Stage Detection

The API parses recent log lines to determine the training stage:

- **Preprocessing**: Detected when logs contain "preprocessing" or "loading"
- **Warmup**: Detected when logs contain "warming up" or "warmup"
- **Training**: Detected when logs contain "step" or "epoch" patterns
- **Validation**: Detected when logs contain "validat"
- **Saving checkpoint**: Detected when logs contain "checkpoint"

### Progress Calculation

Progress percentage is extracted from log patterns like:
- `step 1500/5000` -> 30%
- `epoch 3/10` -> 30%

### When to Use

Use the inline progress API when:
- Displaying compact status in job list cards
- Polling frequently (every 5 seconds) for running jobs only
- You need minimal data transfer per request

<details>
<summary>Client Example (JavaScript)</summary>

```javascript
async function fetchInlineProgress() {
    const runningJobs = jobs.filter(j => j.status === 'running');

    for (const job of runningJobs) {
        try {
            const response = await fetch(
                `/api/cloud/jobs/${job.job_id}/inline-progress`
            );
            if (response.ok) {
                const data = await response.json();
                // Update job card with progress info
                job.inline_stage = data.stage;
                job.inline_log = data.last_log;
                job.inline_progress = data.progress;
            }
        } catch (error) {
            // Silently ignore - job may have completed
        }
    }
}

// Poll every 5 seconds
setInterval(fetchInlineProgress, 5000);
```

</details>

## Job Sync Mechanisms

SimpleTuner provides two sync approaches for keeping local job state current with cloud providers.

### 1. Full Provider Sync

#### Endpoint

```
POST /api/cloud/jobs/sync
```

#### Purpose

Discovers jobs from the cloud provider that may not exist in the local store. This is useful when:
- Jobs were submitted outside SimpleTuner (directly via Replicate API)
- The local job store was reset or corrupted
- You want to import historical jobs

#### Response

```json
{
  "synced": 3,
  "message": "Discovered 3 new jobs from Replicate"
}
```

#### Behavior

1. Fetches up to 100 recent jobs from Replicate
2. For each job:
   - If not in local store: Creates new `UnifiedJob` record
   - If already in store: Updates status, cost, and timestamps
3. Returns count of newly discovered jobs

<details>
<summary>Client Example</summary>

```bash
# Sync jobs from Replicate
curl -X POST http://localhost:8001/api/cloud/jobs/sync

# Response
{"synced": 2, "message": "Discovered 2 new jobs from Replicate"}
```

</details>

#### Web UI Sync Button

The Cloud dashboard includes a sync button for discovering orphaned jobs:

1. Click the **Sync** button in the job list toolbar
2. The button shows a loading spinner during sync
3. On success, a toast notification shows: *"Discovered N jobs from Replicate"*
4. The job list and metrics automatically refresh

**Use Cases:**
- Discovering jobs submitted directly via Replicate API or web console
- Recovering after a database reset
- Importing jobs from a shared team Replicate account

The sync button calls `POST /api/cloud/jobs/sync` internally and then reloads both the job list and dashboard metrics.

### 2. Active Job Status Sync (`sync_active`)

#### Endpoint

```
GET /api/cloud/jobs?sync_active=true
```

#### Purpose

Refreshes the status of all active (non-terminal) cloud jobs before returning the job list. This provides up-to-date status without waiting for background polling.

#### Active States

Jobs in these states are considered "active" and will be synced:
- `pending` - Job submitted but not yet started
- `uploading` - Data upload in progress
- `queued` - Waiting in provider queue
- `running` - Training in progress

#### Behavior

1. Before listing jobs, fetches current status for each active cloud job
2. Updates local store with:
   - Current status
   - `started_at` / `completed_at` timestamps
   - `cost_usd` (accumulated cost)
   - `error_message` (if failed)
3. Returns updated job list

<details>
<summary>Client Example (JavaScript)</summary>

```javascript
// Load jobs with active status sync
async function loadJobs(syncActive = false) {
    const params = new URLSearchParams({
        limit: '50',
        provider: 'replicate',
    });

    if (syncActive) {
        params.set('sync_active', 'true');
    }

    const response = await fetch(`/api/cloud/jobs?${params}`);
    const data = await response.json();
    return data.jobs;
}

// Use sync_active during periodic refresh
setInterval(() => loadJobs(true), 30000);
```

</details>

### Comparison: Sync vs sync_active

| Feature | `POST /jobs/sync` | `GET /jobs?sync_active=true` |
|---------|-------------------|------------------------------|
| Discovers new jobs | Yes | No |
| Updates existing jobs | Yes | Yes (active only) |
| Scope | All provider jobs | Only active local jobs |
| Use case | Initial import, recovery | Regular status refresh |
| Performance | Heavier (batch query) | Lighter (selective) |

## Background Poller Configuration

The background poller automatically syncs active job statuses without client intervention.

### Default Behavior

- **Auto-enabled**: If no webhook URL is configured, polling starts automatically
- **Default interval**: 30 seconds
- **Scope**: All active cloud jobs

<details>
<summary>Enterprise Configuration</summary>

For production deployments, configure polling via `simpletuner-enterprise.yaml`:

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

<details>
<summary>Environment Variables</summary>

```bash
# Set custom polling interval (in seconds)
export SIMPLETUNER_JOB_POLL_INTERVAL=60
```

</details>

### How It Works

1. On server startup, `BackgroundTaskManager` checks:
   - If enterprise config explicitly enables polling, use that interval
   - Otherwise, if no webhook is configured, auto-enable with 30s interval
2. Every interval, the poller:
   - Lists all jobs with active status
   - Groups by provider
   - Fetches current status from each provider
   - Updates local store
   - Emits SSE events for status changes
   - Updates queue entries for terminal states

<details>
<summary>SSE Events</summary>

When the background poller detects status changes, it broadcasts SSE events:

```javascript
// Subscribe to SSE events
const eventSource = new EventSource('/api/events');

eventSource.addEventListener('message', (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'job_status_changed') {
        console.log(`Job ${data.job_id} is now ${data.status}`);
        // Refresh job list
        loadJobs();
    }
});
```

</details>

<details>
<summary>Programmatic Access</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.background_tasks import (
    get_task_manager,
    start_background_tasks,
    stop_background_tasks,
)

# Get the manager
manager = get_task_manager()

# Check if running
if manager._running:
    print("Background tasks are active")

# Manual start/stop (usually handled by app lifespan)
await start_background_tasks()
await stop_background_tasks()
```

</details>

## Best Practices

### 1. Choose the Right Sync Strategy

| Scenario | Recommended Approach |
|----------|---------------------|
| Initial page load | `GET /jobs` without sync (fast) |
| Periodic refresh (30s) | `GET /jobs?sync_active=true` |
| User clicks "Refresh" | `POST /jobs/sync` for discovery |
| Running job details | Inline progress API (5s) |
| Production deployment | Background poller + webhooks |

### 2. Avoid Over-Polling

<details>
<summary>Example</summary>

```javascript
// Good: Poll inline progress only for running jobs
const runningJobs = jobs.filter(j => j.status === 'running');

// Bad: Poll all jobs regardless of status
for (const job of jobs) { /* ... */ }
```

</details>

### 3. Use SSE for Real-Time Updates

<details>
<summary>Example</summary>

Instead of aggressive polling, subscribe to SSE events:

```javascript
// Combine SSE with conservative polling
const eventSource = new EventSource('/api/events');

eventSource.addEventListener('message', (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'job_status_changed') {
        loadJobs();  // Refresh on status change
    }
});

// Fallback: poll every 30 seconds
setInterval(() => loadJobs(true), 30000);
```

</details>

### 4. Handle Terminal States

<details>
<summary>Example</summary>

Stop polling jobs that have reached terminal states:

```javascript
const terminalStates = ['completed', 'failed', 'cancelled'];

function shouldPollJob(job) {
    return !terminalStates.includes(job.status);
}
```

</details>

### 5. Configure Webhooks for Production

<details>
<summary>Example</summary>

Webhooks eliminate the need for polling entirely:

```yaml
# In provider config
webhook_url: "https://your-server.com/api/webhooks/replicate"
```

When webhooks are configured:
- Background polling is disabled (unless explicitly enabled)
- Status updates arrive in real-time via provider callbacks
- Reduced API calls to the provider

</details>

## Troubleshooting

### Jobs Not Updating

<details>
<summary>Debugging steps</summary>

1. Check if background poller is running:
   ```bash
   # Look for log line on startup
   grep "job status polling" server.log
   ```

2. Verify provider connectivity:
   ```bash
   curl http://localhost:8001/api/cloud/replicate/validate
   ```

3. Force a sync:
   ```bash
   curl -X POST http://localhost:8001/api/cloud/jobs/sync
   ```

</details>

### SSE Events Not Received

<details>
<summary>Debugging steps</summary>

1. Check SSE connection limit (5 per IP by default)
2. Verify EventSource is connected:
   ```javascript
   eventSource.addEventListener('open', () => {
       console.log('SSE connected');
   });
   ```

</details>

### High Provider API Usage

<details>
<summary>Solutions</summary>

If you're hitting rate limits:
1. Increase `job_polling_interval` in enterprise config
2. Reduce inline progress polling frequency
3. Configure webhooks to eliminate polling

</details>

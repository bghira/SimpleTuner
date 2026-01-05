# Job Queue

The queue system manages job scheduling, concurrency limits, and GPU allocation for both local and cloud training jobs. It enables features like overnight job scheduling, GPU resource management, and controlled resource usage.

## Overview

When you submit a cloud training job, it's added to the queue and processed based on:

- **Priority** - Higher priority jobs run first
- **Concurrency limits** - Global and per-user limits prevent resource exhaustion
- **FIFO within priority** - Jobs at the same priority level run in submission order

## Queue Status

Access the queue panel by clicking the **queue icon** in the Cloud tab action bar. The panel shows:

| Metric | Description |
|--------|-------------|
| **Queued** | Jobs waiting to run |
| **Running** | Currently executing jobs |
| **Max Concurrent** | Global limit on simultaneous jobs |
| **Avg Wait** | Average time jobs spend in queue |

## Priority Levels

Jobs are assigned priority based on user level:

| User Level | Priority | Value |
|------------|----------|-------|
| Admin | Urgent | 30 |
| Lead | High | 20 |
| Researcher | Normal | 10 |
| Viewer | Low | 0 |

Higher values = higher priority = processed first.

### Priority Override

Leads and admins can override a job's priority for specific situations (e.g., urgent experiments).

## Concurrency Limits

Two limits control how many jobs can run simultaneously:

### Global Limit (`max_concurrent`)

Maximum jobs running across all users. Default: **5 jobs**.

### Per-User Limit (`user_max_concurrent`)

Maximum jobs any single user can run at once. Default: **2 jobs**.

This prevents one user from consuming all available slots.

### Updating Limits

Admins can update limits via the queue panel or API.

<details>
<summary>Example</summary>

```bash
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{"max_concurrent": 10, "user_max_concurrent": 3}'
```

</details>

## Job Lifecycle in Queue

1. **Submitted** - Job created, added to queue with `pending` status
2. **Pending** - Waiting for a slot (concurrency limit)
3. **Running** - Actively training on cloud GPU
4. **Completed/Failed** - Terminal state, removed from active queue

## API Endpoints

### List Queue Entries

```
GET /api/queue
```

Parameters:
- `status` - Filter by status (pending, running, blocked)
- `limit` - Max entries to return (default: 50)
- `include_completed` - Include finished jobs

### Queue Statistics

```
GET /api/queue/stats
```

<details>
<summary>Response example</summary>

```json
{
  "queue_depth": 3,
  "running": 2,
  "max_concurrent": 5,
  "user_max_concurrent": 2,
  "avg_wait_seconds": 45.2,
  "by_status": {"pending": 3, "running": 2},
  "by_user": {"1": 2, "2": 3}
}
```

</details>

### My Queue Status

```
GET /api/queue/me
```

Returns the current user's queue position, pending jobs, and running jobs.

<details>
<summary>Response example</summary>

```json
{
  "pending_count": 2,
  "running_count": 1,
  "blocked_count": 0,
  "best_position": 3,
  "pending_jobs": [...],
  "running_jobs": [...]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `pending_count` | int | Number of jobs waiting in queue |
| `running_count` | int | Number of currently running jobs |
| `blocked_count` | int | Number of jobs awaiting approval |
| `best_position` | int or null | Position of user's highest-priority pending job |
| `pending_jobs` | array | List of pending job details |
| `running_jobs` | array | List of running job details |

The `best_position` field indicates the queue position of the user's best (highest priority or earliest submitted) pending job. This helps users understand when their next job will start. A value of `null` means the user has no pending jobs.

</details>

### Job Position

```
GET /api/queue/position/{job_id}
```

Returns queue position for a specific job.

### Cancel Queued Job

```
POST /api/queue/{job_id}/cancel
```

Cancels a job that hasn't started yet.

### Approve Blocked Job

```
POST /api/queue/{job_id}/approve
```

Admin-only. Approves a job that requires approval (e.g., exceeds cost threshold).

### Reject Blocked Job

```
POST /api/queue/{job_id}/reject?reason=<reason>
```

Admin-only. Rejects a blocked job with a reason.

### Update Concurrency

```
POST /api/queue/concurrency
```

<details>
<summary>Request body</summary>

```json
{
  "max_concurrent": 10,
  "user_max_concurrent": 3
}
```

</details>

### Trigger Processing

```
POST /api/queue/process
```

Admin-only. Manually triggers queue processing (normally automatic).

### Cleanup Old Entries

```
POST /api/queue/cleanup?days=30
```

Admin-only. Removes completed entries older than specified days.

**Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `days` | int | 30 | 1-365 | Retention period in days |

**Behavior:**

Deletes queue entries that:
- Have a terminal status (`completed`, `failed`, or `cancelled`)
- Have a `completed_at` timestamp older than the specified days

Active jobs (pending, running, blocked) are never removed by cleanup.

<details>
<summary>Response and examples</summary>

**Response:**

```json
{
  "success": true,
  "deleted": 42,
  "days": 30
}
```

**Example Usage:**

```bash
# Clean up entries older than 7 days
curl -X POST "http://localhost:8000/api/queue/cleanup?days=7" \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Clean up entries older than 90 days (quarterly cleanup)
curl -X POST "http://localhost:8000/api/queue/cleanup?days=90" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

</details>

## Architecture

<details>
<summary>System diagram</summary>

```
┌─────────────────────────────────────────────────────────────┐
│                     Job Submission                          │
│              (routes/cloud/jobs.py:submit_job)              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  JobSubmissionService                       │
│              Uploads data, submits to provider              │
│                    Enqueues job                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    QueueScheduler                           │
│   ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│   │ Queue Store │  │   Policy     │  │ Background Task │    │
│   │  (SQLite)   │  │  (Priority)  │  │    (5s loop)    │    │
│   └─────────────┘  └──────────────┘  └─────────────────┘    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   QueueDispatcher                           │
│            Updates job status, syncs with provider          │
└─────────────────────────────────────────────────────────────┘
```

</details>

### Components

| Component | Location | Description |
|-----------|----------|-------------|
| `JobRepository` | `storage/job_repository.py` | Unified SQLite persistence for jobs and queue |
| `JobRepoQueueAdapter` | `queue/job_repo_adapter.py` | Adapter for scheduler compatibility |
| `QueueScheduler` | `queue/scheduler.py` | Scheduling logic |
| `SchedulingPolicy` | `queue/scheduler.py` | Priority/fairness algorithm |
| `QueueDispatcher` | `queue/dispatcher.py` | Handles job dispatch |
| `QueueEntry` | `queue/models.py` | Queue entry data model |
| `LocalGPUAllocator` | `services/local_gpu_allocator.py` | GPU allocation for local jobs |

### Database Schema

Queue and job entries are stored in the unified SQLite database (`~/.simpletuner/cloud/jobs.db`).

<details>
<summary>Schema definition</summary>

```sql
CREATE TABLE queue (
    id INTEGER PRIMARY KEY,
    job_id TEXT UNIQUE NOT NULL,
    user_id INTEGER,
    team_id TEXT,
    provider TEXT DEFAULT 'replicate',
    config_name TEXT,
    priority INTEGER DEFAULT 10,
    priority_override INTEGER,
    status TEXT DEFAULT 'pending',
    position INTEGER DEFAULT 0,
    queued_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    estimated_cost REAL DEFAULT 0.0,
    requires_approval INTEGER DEFAULT 0,
    approval_id INTEGER,
    attempt INTEGER DEFAULT 1,
    max_attempts INTEGER DEFAULT 3,
    error_message TEXT,
    metadata TEXT,
    -- GPU allocation (schema v3)
    allocated_gpus TEXT,          -- JSON array of device indices, e.g., "[0,1]"
    job_type TEXT DEFAULT 'cloud', -- "local" or "cloud"
    num_processes INTEGER DEFAULT 1 -- Number of GPUs required
);
```

Migrations run automatically on startup.

</details>

## Local GPU Concurrency

When submitting local training jobs, the queue system tracks GPU allocation to prevent resource conflicts. Jobs are queued if required GPUs are unavailable.

### GPU Allocation Tracking

Each local job specifies:

- **num_processes** - Number of GPUs required (from `--num_processes`)
- **device_ids** - Preferred GPU indices (from `--accelerate_visible_devices`)

The allocator tracks which GPUs are allocated to running jobs and only starts new jobs when resources are available.

### CLI Options

#### Submitting Jobs

<details>
<summary>Examples</summary>

```bash
# Submit a job, queue if GPUs unavailable (default)
simpletuner jobs submit my-config

# Reject immediately if GPUs unavailable
simpletuner jobs submit my-config --no-wait

# Use any available GPUs instead of configured device IDs
simpletuner jobs submit my-config --any-gpu

# Dry-run to check GPU availability
simpletuner jobs submit my-config --dry-run
```

</details>

#### Listing Jobs

<details>
<summary>Examples</summary>

```bash
# List recent jobs
simpletuner jobs list

# List with specific fields
simpletuner jobs list -o job_id,status,config_name

# JSON output with custom fields
simpletuner jobs list --format json -o job_id,status

# Access nested fields using dot notation
simpletuner jobs list --format json -o job_id,metadata.run_name

# Filter by status
simpletuner jobs list --status running
simpletuner jobs list --status queued

# Limit results
simpletuner jobs list -l 10
```

The `-o` (output) option supports dot notation for accessing nested fields in job metadata. For example, `metadata.run_name` extracts the `run_name` field from the job's metadata object.

</details>

### GPU Status API

GPU allocation status is available via the system status endpoint:

```
GET /api/system/status?include_allocation=true
```

<details>
<summary>Response example</summary>

```json
{
  "timestamp": 1704067200.0,
  "load_avg_5min": 2.5,
  "memory_percent": 45.2,
  "gpus": [...],
  "gpu_inventory": {
    "backend": "cuda",
    "count": 4,
    "capabilities": {...}
  },
  "gpu_allocation": {
    "allocated_gpus": [0, 1],
    "available_gpus": [2, 3],
    "running_local_jobs": 1,
    "devices": [
      {"index": 0, "name": "A100", "memory_gb": 40, "allocated": true, "job_id": "abc123"},
      {"index": 1, "name": "A100", "memory_gb": 40, "allocated": true, "job_id": "abc123"},
      {"index": 2, "name": "A100", "memory_gb": 40, "allocated": false, "job_id": null},
      {"index": 3, "name": "A100", "memory_gb": 40, "allocated": false, "job_id": null}
    ]
  }
}
```

</details>

Queue statistics also include local GPU info:

```
GET /api/queue/stats
```

<details>
<summary>Response example</summary>

```json
{
  "queue_depth": 3,
  "running": 2,
  "max_concurrent": 5,
  "local_gpu_max_concurrent": 6,
  "local_job_max_concurrent": 2,
  "local": {
    "running_jobs": 1,
    "pending_jobs": 0,
    "allocated_gpus": [0, 1],
    "available_gpus": [2, 3],
    "total_gpus": 4,
    "max_concurrent_gpus": 6,
    "max_concurrent_jobs": 2
  }
}
```

</details>

### Local Concurrency Limits

Control how many local jobs and GPUs can be used simultaneously via the existing concurrency endpoint:

```
GET /api/queue/concurrency
POST /api/queue/concurrency
```

The concurrency endpoint now accepts local GPU limits alongside cloud limits:

| Field | Type | Description |
|-------|------|-------------|
| `max_concurrent` | int | Maximum cloud jobs running (default: 5) |
| `user_max_concurrent` | int | Maximum cloud jobs per user (default: 2) |
| `local_gpu_max_concurrent` | int or null | Maximum GPUs for local jobs (null = unlimited) |
| `local_job_max_concurrent` | int | Maximum local jobs simultaneously (default: 1) |

<details>
<summary>Example</summary>

```bash
# Allow up to 2 local jobs using up to 6 GPUs total
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{"local_gpu_max_concurrent": 6, "local_job_max_concurrent": 2}'
```

</details>

### Local Job Submission API

```
POST /api/queue/submit
```

<details>
<summary>Request and response</summary>

**Request body:**

```json
{
  "config_name": "my-training-config",
  "no_wait": false,
  "any_gpu": false
}
```

**Response:**

```json
{
  "success": true,
  "job_id": "abc123",
  "status": "running",
  "allocated_gpus": [0, 1],
  "queue_position": null
}
```

</details>

Status values:

| Status | Description |
|--------|-------------|
| `running` | Job started immediately with allocated GPUs |
| `queued` | Job queued, will start when GPUs available |
| `rejected` | GPUs unavailable and `no_wait` was true |

### Automatic Job Processing

When a job completes or fails, its GPUs are released and the queue is processed to start pending jobs. This happens automatically via the process keeper lifecycle hooks.

**Cancellation behavior**: When a job is cancelled, GPUs are released but pending jobs are NOT automatically started. This prevents race conditions during bulk cancellation (`simpletuner jobs cancel --all`) where pending jobs would start before they could be cancelled. Use `POST /api/queue/process` or restart the server to manually trigger queue processing after cancellation.

## Worker Dispatch

Jobs can be dispatched to remote workers instead of running on the orchestrator's local GPUs. See [Worker Orchestration](experimental/server/WORKERS.md) for full worker setup.

### Job Targets

When submitting a job, specify where it should run:

| Target | Behavior |
|--------|----------|
| `auto` (default) | Try remote workers first, fall back to local GPUs |
| `worker` | Dispatch only to remote workers; queue if none available |
| `local` | Run only on orchestrator's local GPUs |

<details>
<summary>Examples</summary>

```bash
# CLI
simpletuner jobs submit my-config --target=worker

# API
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{"config_name": "my-config", "target": "worker"}'
```

</details>

### Worker Selection

Jobs can specify label requirements for worker matching:

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-config",
    "target": "worker",
    "worker_labels": {"gpu_type": "a100*", "location": "us-*"}
  }'
```

Labels support glob patterns. The scheduler matches jobs to workers based on:

1. Label requirements (all must match)
2. GPU count requirements
3. Worker availability (IDLE status)
4. FIFO order within matching workers

### Startup Behavior

On server startup, the queue system automatically processes any pending local jobs. If GPUs are available, queued jobs will start immediately without manual intervention. This ensures jobs submitted before a server restart continue processing once the server is back online.

The startup sequence:
1. Server initializes GPU allocator
2. Pending local jobs are retrieved from the queue
3. For each pending job with available GPUs, the job is started
4. Jobs that cannot start (insufficient GPUs) remain queued

Note: Cloud jobs are handled by the separate cloud queue scheduler, which also resumes on startup.

## Configuration

Queue concurrency limits are configured via the API and persisted in the queue database.

**Via Web UI:** Cloud tab → Queue Panel → Settings

<details>
<summary>API configuration example</summary>

```bash
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{
    "max_concurrent": 5,
    "user_max_concurrent": 2,
    "team_max_concurrent": 10,
    "enable_fair_share": false
  }'
```

</details>

| Setting | Default | Description |
|---------|---------|-------------|
| `max_concurrent` | 5 | Global maximum running jobs |
| `user_max_concurrent` | 2 | Maximum running jobs per user |
| `team_max_concurrent` | 10 | Maximum running jobs per team |
| `enable_fair_share` | false | Enable per-team concurrency limits |

### Fair-Share Scheduling

When `enable_fair_share: true`, the scheduler considers team affiliation to prevent any single team from monopolizing resources.

#### How It Works

Fair-share adds a third layer of concurrency control:

| Layer | Limit | Purpose |
|-------|-------|---------|
| Global | `max_concurrent` | Total jobs across all users/teams |
| Per-User | `user_max_concurrent` | Prevents one user from consuming all slots |
| Per-Team | `team_max_concurrent` | Prevents one team from consuming all slots |

When a job is considered for dispatch:

1. Check global limit → skip if at capacity
2. Check user limit → skip if user at capacity
3. If fair-share enabled AND job has `team_id`:
   - Check team limit → skip if team at capacity

Jobs without a `team_id` are not subject to team limits.

#### Enabling Fair-Share

**Via UI:** Cloud tab → Queue Panel → Toggle "Fair-Share Scheduling"

<details>
<summary>API example</summary>

```bash
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{
    "max_concurrent": 10,
    "user_max_concurrent": 3,
    "team_max_concurrent": 5,
    "enable_fair_share": true
  }'
```

</details>

#### Team Assignment

Teams are assigned to users in the admin panel. When a user submits a job, their team ID is attached to the queue entry. The scheduler tracks running jobs per team and enforces the limit.

<details>
<summary>Example scenario</summary>

With `max_concurrent=6`, `user_max_concurrent=2`, `team_max_concurrent=3`:

| Team | Users | Submitted | Running | Blocked |
|------|-------|-----------|---------|---------|
| Alpha | Alice, Bob | 4 | 3 (at team limit) | 1 |
| Beta | Carol | 3 | 2 | 1 (waiting for global slot) |

- Team Alpha has 3 running (at `team_max_concurrent`)
- Total running is 5 (under `max_concurrent=6`)
- Carol's job is blocked because: 5+1=6, at global limit
- Alice's 4th job is blocked because: team at 3/3

This ensures neither team monopolizes the queue even if they submit many jobs.

</details>

### Starvation Prevention

Jobs waiting longer than `starvation_threshold_minutes` receive a priority boost to prevent indefinite waiting.

## Approval Workflow

Jobs can be marked as requiring approval (e.g., when estimated cost exceeds a threshold):

1. Job submitted with `requires_approval: true`
2. Job enters `blocked` status
3. Admin reviews in queue panel or via API
4. Admin approves or rejects
5. If approved, job moves to `pending` and is scheduled normally

See [Enterprise Guide](experimental/server/ENTERPRISE.md) for approval rule configuration.

## Troubleshooting

### Jobs Stuck in Queue

<details>
<summary>Debugging steps</summary>

Check concurrency limits:
```bash
curl http://localhost:8000/api/queue/stats
```

If `running` equals `max_concurrent`, jobs are waiting for slots.

</details>

### Queue Not Processing

<details>
<summary>Debugging steps</summary>

The background processor runs every 5 seconds. Check server logs for errors:
```
Queue scheduler started with 5s processing interval
```

If not visible, the scheduler may have failed to start.

</details>

### Job Disappeared from Queue

<details>
<summary>Debugging steps</summary>

Check if it was completed or failed:
```bash
curl "http://localhost:8000/api/queue?include_completed=true"
```

</details>

### Local Jobs Show Running But Aren't Training

<details>
<summary>Debugging steps</summary>

If `jobs list` shows local jobs as "running" but no training is happening:

1. Check GPU allocation status:
   ```bash
   simpletuner jobs status --format json
   ```
   Look at the `local.allocated_gpus` field - it should show which GPUs are in use.

2. If allocated GPUs is empty but running count is non-zero, the queue state may be inconsistent. Restart the server to trigger automatic queue reconciliation.

3. Check server logs for GPU allocation errors:
   ```
   Failed to allocate GPUs [0] to job <job_id>
   ```

</details>

### Queue Depth Shows Wrong Count

<details>
<summary>Explanation</summary>

The queue depth and running job counts are calculated separately for local and cloud jobs:

- **Local jobs**: Tracked via `LocalGPUAllocator` based on GPU allocation state
- **Cloud jobs**: Tracked via `QueueScheduler` based on provider status

Use `simpletuner jobs status --format json` to see the breakdown:
- `local.running_jobs` - Running local training jobs
- `local.pending_jobs` - Queued local jobs waiting for GPUs
- `running` - Total running jobs (cloud queue)
- `queue_depth` - Pending cloud jobs

</details>

## See Also

- [Worker Orchestration](experimental/server/WORKERS.md) - Distributed worker registration and job dispatch
- [Cloud Training Tutorial](experimental/cloud/TUTORIAL.md) - Getting started with cloud training
- [Enterprise Guide](experimental/server/ENTERPRISE.md) - Multi-user setup, approvals, governance
- [Operations Guide](experimental/cloud/OPERATIONS_TUTORIAL.md) - Production deployment

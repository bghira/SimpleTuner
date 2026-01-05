# Worker Orchestration

SimpleTuner's worker orchestration allows you to distribute training jobs across multiple GPU machines. Workers register with a central orchestrator, receive job dispatch events in real-time, and report status back.

## Overview

The orchestrator/worker architecture enables:

- **Distributed training** - Run jobs on any machine with GPUs, anywhere
- **Auto-discovery** - Workers self-register with GPU capabilities
- **Real-time dispatch** - Jobs dispatched via SSE (Server-Sent Events)
- **Mixed fleet** - Combine cloud-launched ephemeral workers with persistent on-prem machines
- **Fault tolerance** - Orphaned jobs are automatically requeued

## Worker Types

| Type | Lifecycle | Use Case |
|------|-----------|----------|
| **Ephemeral** | Shuts down after job completion | Cloud spot instances (RunPod, Vast.ai) |
| **Persistent** | Stays online between jobs | On-prem GPUs, reserved instances |

## Quick Start

### 1. Start the Orchestrator

Run the SimpleTuner server on your central machine:

```bash
simpletuner server --host 0.0.0.0 --port 8001
```

For production, enable SSL:

```bash
simpletuner server --host 0.0.0.0 --port 8001 --ssl
```

### 2. Create a Worker Token

**Via Web UI:** Administration → Workers → Create Worker

**Via API:**

```bash
curl -s -X POST http://localhost:8001/api/admin/workers \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gpu-worker-1",
    "worker_type": "persistent",
    "labels": {"location": "datacenter-a", "gpu_type": "a100"}
  }'
```

Response includes the token (shown only once):

```json
{
  "worker_id": "w_abc123",
  "token": "wt_xxxxxxxxxxxx",
  "name": "gpu-worker-1"
}
```

### 3. Start the Worker

On the GPU machine:

```bash
simpletuner worker \
  --orchestrator-url https://orchestrator.example.com:8001 \
  --worker-token wt_xxxxxxxxxxxx \
  --name gpu-worker-1 \
  --persistent
```

Or via environment variables:

```bash
export SIMPLETUNER_ORCHESTRATOR_URL=https://orchestrator.example.com:8001
export SIMPLETUNER_WORKER_TOKEN=wt_xxxxxxxxxxxx
export SIMPLETUNER_WORKER_NAME=gpu-worker-1
export SIMPLETUNER_WORKER_PERSISTENT=true

simpletuner worker
```

The worker will:

1. Connect to the orchestrator
2. Report GPU capabilities (auto-detected)
3. Enter the job dispatch loop
4. Send heartbeats every 30 seconds

### 4. Submit Jobs to Workers

**Via Web UI:** Configure your training, then click **Train in Cloud** → select **Worker** as target.

**Via API:**

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H "Content-Type: application/json" \
  -d '{
    "config_name": "my-training-config",
    "target": "worker"
  }'
```

Target options:

| Target | Behavior |
|--------|----------|
| `worker` | Dispatch only to remote workers |
| `local` | Run on orchestrator's GPUs |
| `auto` | Prefer worker if available, fall back to local |

## CLI Reference

```
simpletuner worker [OPTIONS]

OPTIONS:
  --orchestrator-url URL   Orchestrator panel URL (or SIMPLETUNER_ORCHESTRATOR_URL)
  --worker-token TOKEN     Authentication token (or SIMPLETUNER_WORKER_TOKEN)
  --name NAME              Worker name (default: hostname)
  --persistent             Stay online between jobs (default: ephemeral)
  -v, --verbose            Enable debug logging
```

### Ephemeral vs Persistent Mode

**Ephemeral (default):**
- Worker shuts down after completing one job
- Ideal for cloud spot instances that bill per minute
- Orchestrator cleans up offline ephemeral workers after 1 hour

**Persistent (`--persistent`):**
- Worker stays online waiting for new jobs
- Reconnects automatically if connection drops
- Use for on-prem GPUs or reserved instances

## Worker Lifecycle

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  CONNECTING │ ──▶ │    IDLE     │ ──▶ │    BUSY     │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           │                   │
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  DRAINING   │     │   OFFLINE   │
                    └─────────────┘     └─────────────┘
```

| Status | Description |
|--------|-------------|
| `CONNECTING` | Worker establishing connection |
| `IDLE` | Ready to receive jobs |
| `BUSY` | Currently running a job |
| `DRAINING` | Finishing current job, then shutting down |
| `OFFLINE` | Disconnected (heartbeat timeout) |

## Health Monitoring

The orchestrator monitors worker health:

- **Heartbeat interval:** 30 seconds (worker → orchestrator)
- **Timeout threshold:** 120 seconds without heartbeat → mark offline
- **Health check loop:** Runs every 60 seconds on orchestrator

### Handling Failures

**Worker goes offline during a job:**

1. Job marked as failed after heartbeat timeout
2. If retries remaining (default: 3), job requeued
3. Next available worker picks up the job

**Orchestrator restarts:**

1. Workers automatically reconnect
2. Workers report any in-progress jobs
3. Orchestrator reconciles state and resumes

## GPU Matching

Workers report their GPU capabilities on registration:

```json
{
  "gpu_count": 2,
  "gpu_name": "NVIDIA A100-SXM4-80GB",
  "gpu_vram_gb": 80,
  "accelerator_type": "cuda"
}
```

Jobs can specify GPU requirements:

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H "Content-Type: application/json" \
  -d '{
    "config_name": "my-config",
    "target": "worker",
    "worker_labels": {"gpu_type": "a100*"}
  }'
```

The scheduler matches jobs to workers based on:

1. GPU count requirements
2. Label matching (glob patterns supported)
3. Worker availability (IDLE status)

## Labels

Labels provide flexible worker selection:

**Assign labels on worker creation:**

```bash
curl -s -X POST http://localhost:8001/api/admin/workers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "worker-1",
    "labels": {
      "location": "us-west",
      "gpu_type": "a100",
      "team": "nlp"
    }
  }'
```

**Select workers by label:**

```bash
# Match workers with team=nlp
curl -s -X POST http://localhost:8001/api/queue/submit \
  -d '{"config_name": "my-config", "worker_labels": {"team": "nlp"}}'

# Match workers with gpu_type starting with "a100"
curl -s -X POST http://localhost:8001/api/queue/submit \
  -d '{"config_name": "my-config", "worker_labels": {"gpu_type": "a100*"}}'
```

## Admin Operations

### List Workers

```bash
curl -s http://localhost:8001/api/admin/workers | jq
```

Response:

```json
{
  "workers": [
    {
      "id": "w_abc123",
      "name": "gpu-worker-1",
      "status": "idle",
      "worker_type": "persistent",
      "gpu_count": 2,
      "gpu_name": "A100",
      "labels": {"location": "datacenter-a"},
      "last_heartbeat": "2024-01-15T10:30:00Z"
    }
  ]
}
```

### Drain a Worker

Gracefully finish current job and prevent new dispatch:

```bash
curl -s -X POST http://localhost:8001/api/admin/workers/w_abc123/drain
```

The worker will:

1. Complete any running job
2. Enter DRAINING status
3. Refuse new jobs
4. Disconnect after job completion (ephemeral) or remain in draining state (persistent)

### Rotate Token

Regenerate a worker's authentication token:

```bash
curl -s -X POST http://localhost:8001/api/admin/workers/w_abc123/token
```

The old token is immediately invalidated. Update the worker with the new token.

### Delete a Worker

```bash
curl -s -X DELETE http://localhost:8001/api/admin/workers/w_abc123
```

Only works if the worker is offline.

## Security

### Token Authentication

- Workers authenticate via `X-Worker-Token` header
- Tokens are SHA-256 hashed before storage
- Tokens never leave the orchestrator after creation
- Rotate tokens periodically for security

### Network Security

For production:

1. Use `--ssl` flag or terminate TLS at a reverse proxy
2. Restrict worker registration to trusted networks
3. Use firewall rules to limit access to `/api/workers/*` endpoints

### Audit Logging

All worker actions are logged:

- Registration attempts (success/failure)
- Job dispatch events
- Status transitions
- Token rotations
- Admin operations

See [Audit Guide](AUDIT.md) for log access.

## Troubleshooting

### Worker Can't Connect

**"Connection refused"**
- Verify orchestrator URL and port
- Check firewall rules allow inbound connections
- Ensure orchestrator is running with `--host 0.0.0.0`

**"Invalid token"**
- Token may have been rotated—request a new one
- Check for whitespace in token string

**"SSL certificate verify failed"**
- Use `--ssl-no-verify` for self-signed certs (development only)
- Or add the CA certificate to the system trust store

### Worker Goes Offline Unexpectedly

**Heartbeat timeout (120s)**
- Check network stability between worker and orchestrator
- Look for resource exhaustion (CPU/memory) on worker
- Increase `SIMPLETUNER_HEARTBEAT_TIMEOUT` if on unreliable network

**Process crash**
- Check worker logs for Python exceptions
- Verify GPU drivers are functioning (`nvidia-smi`)
- Ensure sufficient disk space for training

### Jobs Not Dispatching to Workers

**No idle workers**
- Check worker status in admin panel
- Verify workers are connected and IDLE
- Check for label mismatch between job and workers

**GPU requirements not met**
- Job requires more GPUs than any worker has
- Adjust `--num_processes` in training config

## API Reference

### Worker Endpoints (Worker → Orchestrator)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/workers/register` | POST | Register and report capabilities |
| `/api/workers/stream` | GET | SSE stream for job dispatch |
| `/api/workers/heartbeat` | POST | Periodic keepalive |
| `/api/workers/job/{id}/status` | POST | Report job progress |
| `/api/workers/disconnect` | POST | Graceful shutdown notification |

### Admin Endpoints (Requires `admin.workers` permission)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/admin/workers` | GET | List all workers |
| `/api/admin/workers` | POST | Create worker token |
| `/api/admin/workers/{id}` | DELETE | Remove worker |
| `/api/admin/workers/{id}/drain` | POST | Drain worker |
| `/api/admin/workers/{id}/token` | POST | Rotate token |

## See Also

- [Enterprise Guide](ENTERPRISE.md) - SSO, quotas, approval workflows
- [Job Queue](../../JOB_QUEUE.md) - Queue scheduling and priorities
- [Cloud Training](../cloud/README.md) - Cloud provider integration
- [API Tutorial](../../api/TUTORIAL.md) - Local training via REST API

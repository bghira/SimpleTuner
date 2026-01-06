# Queue System

> **Status:** Experimental
> **Available in:** Web UI (Cloud tab)

Queue system cloud training jobs के लिए job scheduling, concurrency limits, और fair-share allocation को प्रबंधित करता है। यह हमेशा active रहता है, single-user mode में भी, जिससे overnight job scheduling और नियंत्रित resource usage जैसे फीचर्स मिलते हैं।

## Overview

जब आप cloud training job submit करते हैं, वह queue में जुड़ता है और निम्न आधारों पर प्रोसेस होता है:

- **Priority** - उच्च priority वाले jobs पहले चलते हैं
- **Concurrency limits** - global और per-user limits resource exhaustion रोकते हैं
- **FIFO within priority** - समान priority वाले jobs submission order में चलते हैं

## Queue Status

Cloud tab action bar में **queue icon** पर क्लिक करके queue panel खोलें। panel यह दिखाता है:

| Metric | Description |
|--------|-------------|
| **Queued** | चलने का इंतज़ार कर रहे jobs |
| **Running** | वर्तमान में चल रहे jobs |
| **Max Concurrent** | simultaneous jobs का global limit |
| **Avg Wait** | jobs का औसत queue समय |

## Priority Levels

Jobs user level के आधार पर priority प्राप्त करते हैं:

| User Level | Priority | Value |
|------------|----------|-------|
| Admin | Urgent | 30 |
| Lead | High | 20 |
| Researcher | Normal | 10 |
| Viewer | Low | 0 |

Higher values = higher priority = पहले प्रोसेस।

### Priority Override

Leads और admins विशेष परिस्थितियों में job priority override कर सकते हैं (जैसे urgent experiments)।

## Concurrency Limits

एक समय में कितने jobs चल सकते हैं, इसे दो limits नियंत्रित करते हैं:

### Global Limit (`max_concurrent`)

सभी users में कुल running jobs की अधिकतम संख्या। डिफ़ॉल्ट: **5 jobs**.

### Per-User Limit (`user_max_concurrent`)

किसी भी एक user के एक साथ चलने वाले jobs की अधिकतम संख्या। डिफ़ॉल्ट: **2 jobs**.

यह एक user को सभी slots consume करने से रोकता है।

### Limits अपडेट करना

Admins queue panel या API के जरिए limits अपडेट कर सकते हैं:

```bash
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{"max_concurrent": 10, "user_max_concurrent": 3}'
```

## Queue में Job Lifecycle

1. **Submitted** - Job बनाया जाता है, `pending` status के साथ queue में जुड़ता है
2. **Pending** - slot के लिए प्रतीक्षा (concurrency limit)
3. **Running** - cloud GPU पर सक्रिय training
4. **Completed/Failed** - terminal state, active queue से हट जाता है

## API Endpoints

### Queue Entries सूचीबद्ध करें

```http
GET /api/queue
```

Parameters:
- `status` - status के आधार पर फ़िल्टर (pending, running, blocked)
- `limit` - लौटाने की अधिकतम entries (डिफ़ॉल्ट: 50)
- `include_completed` - समाप्त jobs शामिल करें

### Queue Statistics

```http
GET /api/queue/stats
```

Returns:
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

### My Queue Status

```http
GET /api/queue/me
```

मौजूदा user के queue position, pending jobs, और running jobs लौटाता है।

**Response:**

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
| `pending_count` | int | Queue में प्रतीक्षा कर रहे jobs की संख्या |
| `running_count` | int | वर्तमान में चल रहे jobs की संख्या |
| `blocked_count` | int | approval की प्रतीक्षा कर रहे jobs की संख्या |
| `best_position` | int या null | user के सबसे उच्च priority pending job की position |
| `pending_jobs` | array | pending job विवरणों की सूची |
| `running_jobs` | array | running job विवरणों की सूची |

`best_position` field user के सबसे अच्छे (उच्च priority या सबसे पहले submit) pending job की queue position बताता है। इससे users को पता चलता है कि उनका अगला job कब शुरू होगा। `null` का मतलब user के पास कोई pending job नहीं है।

### Job Position

```http
GET /api/queue/position/{job_id}
```

किसी specific job के लिए queue position लौटाता है।

### Queued Job Cancel करें

```http
POST /api/queue/{job_id}/cancel
```

ऐसे job को cancel करता है जो अभी शुरू नहीं हुआ।

### Blocked Job Approve करें

```http
POST /api/queue/{job_id}/approve
```

Admin‑only. ऐसे job को approve करता है जिसे approval चाहिए (जैसे cost threshold से अधिक)।

### Blocked Job Reject करें

```http
POST /api/queue/{job_id}/reject?reason=<reason>
```

Admin‑only. कारण के साथ blocked job reject करता है।

### Concurrency Update करें

```http
POST /api/queue/concurrency
```

Body:
```json
{
  "max_concurrent": 10,
  "user_max_concurrent": 3
}
```

### Processing Trigger करें

```http
POST /api/queue/process
```

Admin‑only. मैन्युअली queue processing ट्रिगर करता है (आमतौर पर automatic)।

### पुराने Entries साफ़ करें

```http
POST /api/queue/cleanup?days=30
```

Admin‑only. दिए गए दिनों से पुराने completed entries हटाता है।

**Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `days` | int | 30 | 1-365 | Retention period in days |

**Behavior:**

ऐसी queue entries हटाता है जो:
- terminal status (`completed`, `failed`, या `cancelled`) पर हों
- `completed_at` timestamp दिए गए दिनों से पुराना हो

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
# 7 दिन से पुरानी entries साफ़ करें
curl -X POST "http://localhost:8000/api/queue/cleanup?days=7" \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# 90 दिन से पुरानी entries साफ़ करें (quarterly cleanup)
curl -X POST "http://localhost:8000/api/queue/cleanup?days=90" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

Active jobs (pending, running, blocked) cleanup में कभी नहीं हटते।

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Job Submission                          │
│              (routes/cloud/jobs.py:submit_job)              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  JobSubmissionService                        │
│              Uploads data, submits to provider               │
│                    Enqueues job                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    QueueScheduler                            │
│   ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│   │ Queue Store │  │   Policy     │  │ Background Task │    │
│   │  (SQLite)   │  │  (Priority)  │  │   (5s loop)    │    │
│   └─────────────┘  └──────────────┘  └─────────────────┘    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    QueueDispatcher                           │
│              Updates job status, syncs with provider         │
└─────────────────────────────────────────────────────────────┘
```

### Components

| Component | Location | Description |
|-----------|----------|-------------|
| `JobRepository` | `storage/job_repository.py` | Jobs और queue के लिए unified SQLite persistence |
| `JobRepoQueueAdapter` | `queue/job_repo_adapter.py` | scheduler compatibility के लिए adapter |
| `QueueScheduler` | `queue/scheduler.py` | Scheduling logic |
| `SchedulingPolicy` | `queue/scheduler.py` | Priority/fairness algorithm |
| `QueueDispatcher` | `queue/dispatcher.py` | Job dispatch संभालता है |
| `QueueEntry` | `queue/models.py` | Queue entry data model |
| `LocalGPUAllocator` | `services/local_gpu_allocator.py` | Local jobs के लिए GPU allocation |

### Database Schema

Queue और job entries unified SQLite database (`~/.simpletuner/cloud/jobs.db`) में स्टोर होती हैं:

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

Schema version वर्तमान में **v3** है। Migrations startup पर अपने आप चलती हैं।

## Local GPU Concurrency

Local training jobs सबमिट करते समय, queue system GPU allocation ट्रैक करता है ताकि resource conflicts न हों। यदि आवश्यक GPUs उपलब्ध नहीं हैं, तो jobs queue में चले जाते हैं।

### GPU Allocation Tracking

हर local job में निर्दिष्ट होता है:

- **num_processes** - आवश्यक GPUs की संख्या (`--num_processes` से)
- **device_ids** - पसंदीदा GPU indices (`--accelerate_visible_devices` से)

Allocator यह ट्रैक करता है कि कौन से GPUs running jobs को allocated हैं और नए jobs तभी शुरू करता है जब resources उपलब्ध हों।

### CLI Options

```bash
# Job submit करें, यदि GPUs उपलब्ध नहीं तो queue (default)
simpletuner jobs submit my-config

# GPUs उपलब्ध न हों तो तुरंत reject करें
simpletuner jobs submit my-config --no-wait

# Configured device IDs की जगह कोई भी उपलब्ध GPUs उपयोग करें
simpletuner jobs submit my-config --any-gpu

# GPU availability जांचने के लिए dry-run
simpletuner jobs submit my-config --dry-run
```

### GPU Status API

GPU allocation status system status endpoint के जरिए उपलब्ध है:

```http
GET /api/system/status?include_allocation=true
```

यह GPU allocation सहित system metrics लौटाता है:

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

Queue statistics में local GPU info भी शामिल है:

```http
GET /api/queue/stats
```

Queue stats with local GPU allocation लौटाता है:

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

### Local Concurrency Limits

Local jobs और GPUs की अधिकतम सीमा existing concurrency endpoint के जरिए नियंत्रित करें:

```http
GET /api/queue/concurrency
POST /api/queue/concurrency
```

अब concurrency endpoint cloud limits के साथ local GPU limits भी स्वीकार करता है:

| Field | Type | Description |
|-------|------|-------------|
| `max_concurrent` | int | चल रहे cloud jobs की अधिकतम संख्या (डिफ़ॉल्ट: 5) |
| `user_max_concurrent` | int | प्रति user अधिकतम cloud jobs (डिफ़ॉल्ट: 2) |
| `local_gpu_max_concurrent` | int या null | local jobs के लिए अधिकतम GPUs (null = unlimited) |
| `local_job_max_concurrent` | int | एक साथ चलने वाले local jobs की अधिकतम संख्या (डिफ़ॉल्ट: 1) |

उदाहरण:

```bash
# कुल 6 GPUs तक उपयोग करते हुए 2 local jobs तक अनुमति दें
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{"local_gpu_max_concurrent": 6, "local_job_max_concurrent": 2}'
```

### Local Job Submission API

```http
POST /api/queue/submit
```

Body:

```json
{
  "config_name": "my-training-config",
  "no_wait": false,
  "any_gpu": false
}
```

Response:

```json
{
  "success": true,
  "job_id": "abc123",
  "status": "running",
  "allocated_gpus": [0, 1],
  "queue_position": null
}
```

Status values:

| Status | Description |
|--------|-------------|
| `running` | allocated GPUs के साथ job तुरंत शुरू हुआ |
| `queued` | job queue में है, GPUs उपलब्ध होने पर शुरू होगा |
| `rejected` | GPUs उपलब्ध नहीं और `no_wait` true था |

### Automatic Job Processing

जब job complete, fail, या cancel होता है, उसके GPUs release हो जाते हैं और queue को process किया जाता है ताकि pending jobs शुरू हो सकें। यह process keeper lifecycle hooks के जरिए स्वतः होता है।

## Configuration

Queue concurrency limits API के जरिए कॉन्फ़िगर होते हैं और queue database में persist रहते हैं।

**API के जरिए:**

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

**Web UI के जरिए:** Cloud tab → Queue Panel → Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `max_concurrent` | 5 | Global maximum running jobs |
| `user_max_concurrent` | 2 | प्रति user running jobs की अधिकतम संख्या |
| `team_max_concurrent` | 10 | प्रति team running jobs की अधिकतम संख्या |
| `enable_fair_share` | false | per-team concurrency limits सक्षम करें |

### Fair-Share Scheduling

जब `enable_fair_share: true` हो, scheduler team affiliation को ध्यान में रखता है ताकि कोई टीम संसाधनों पर कब्ज़ा न कर सके।

#### यह कैसे काम करता है

Fair-share एक तीसरा concurrency नियंत्रण जोड़ता है:

| Layer | Limit | Purpose |
|-------|-------|---------|
| Global | `max_concurrent` | सभी users/teams में कुल jobs |
| Per-User | `user_max_concurrent` | किसी एक user को सभी slots लेने से रोकता है |
| Per-Team | `team_max_concurrent` | किसी एक टीम को सभी slots लेने से रोकता है |

जब कोई job dispatch के लिए विचार किया जाता है:

1. Global limit जांचें → capacity हो तो skip
2. User limit जांचें → user capacity हो तो skip
3. यदि fair-share enabled और job के पास `team_id` है:
   - Team limit जांचें → team capacity हो तो skip

जिन jobs में `team_id` नहीं है वे team limits के अधीन नहीं होते।

#### Fair-Share सक्षम करना

**UI के जरिए:** Cloud tab → Queue Panel → "Fair-Share Scheduling" टॉगल करें

**API के जरिए:**

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

#### Team Assignment

Teams admin panel में users को असाइन किए जाते हैं। जब user job submit करता है, उसका team ID queue entry में जुड़ता है। Scheduler हर team के running jobs ट्रैक करता है और limit लागू करता है।

#### उदाहरण Scenario

`max_concurrent=6`, `user_max_concurrent=2`, `team_max_concurrent=3` के साथ:

| Team | Users | Submitted | Running | Blocked |
|------|-------|-----------|---------|---------|
| Alpha | Alice, Bob | 4 | 3 (team limit पर) | 1 |
| Beta | Carol | 3 | 2 | 1 (global slot का इंतज़ार) |

- Team Alpha के पास 3 running ( `team_max_concurrent` पर )
- कुल running 5 है ( `max_concurrent=6` के अंदर )
- Carol का job blocked है क्योंकि: 5+1=6, global limit पर
- Alice का 4th job blocked है क्योंकि: team 3/3 पर

यह सुनिश्चित करता है कि कोई टीम queue पर एकाधिकार न करे, भले ही वे बहुत सारे jobs submit करें।

### Starvation Prevention

`starvation_threshold_minutes` से अधिक इंतज़ार करने वाले jobs को priority boost मिलता है ताकि वे अनिश्चितकाल तक न रुकें।

## Approval Workflow

Jobs को approval की आवश्यकता वाला mark किया जा सकता है (जैसे अनुमानित लागत threshold से ऊपर होने पर):

1. Job `requires_approval: true` के साथ submit होता है
2. Job `blocked` status में जाता है
3. Admin queue panel या API से समीक्षा करता है
4. Admin approve या reject करता है
5. यदि approved, job `pending` में जाता है और सामान्य रूप से schedule होता है

Approval rule कॉन्फ़िगरेशन के लिए [Enterprise Guide](../server/ENTERPRISE.md) देखें।

## Troubleshooting

### Jobs Queue में फँसे हुए हैं

Concurrency limits जांचें:
```bash
curl http://localhost:8000/api/queue/stats
```

यदि `running` == `max_concurrent` है, तो jobs slots का इंतज़ार कर रहे हैं।

### Queue प्रोसेस नहीं हो रही

Background processor हर 5 सेकंड में चलता है। server logs में errors देखें:
```
Queue scheduler started with 5s processing interval
```

यदि यह नहीं दिखता, तो scheduler शायद शुरू नहीं हुआ।

### Job Queue से गायब हो गया

जाँचें कि वह complete या fail हुआ है या नहीं:
```bash
curl "http://localhost:8000/api/queue?include_completed=true"
```

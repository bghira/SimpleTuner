# Job Queue

Queue सिस्टम job scheduling, concurrency limits, और GPU allocation को local और cloud दोनों प्रशिक्षण jobs के लिए मैनेज करता है। यह overnight job scheduling, GPU resource management, और नियंत्रित resource उपयोग जैसे फीचर्स सक्षम करता है।

## अवलोकन

जब आप कोई cloud training job submit करते हैं, यह queue में जोड़ा जाता है और निम्न आधार पर प्रोसेस होता है:

- **Priority** - उच्च priority jobs पहले चलेंगे
- **Concurrency limits** - global और per‑user limits संसाधन exhaustion को रोकते हैं
- **FIFO within priority** - समान priority स्तर पर jobs submission order में चलती हैं

## Queue Status

Cloud टैब action bar में **queue icon** पर क्लिक करके queue panel खोलें। पैनल दिखाता है:

| Metric | Description |
|--------|-------------|
| **Queued** | चलने की प्रतीक्षा में jobs |
| **Running** | वर्तमान में चल रही jobs |
| **Max Concurrent** | एक साथ चलने वाली jobs की global सीमा |
| **Avg Wait** | queue में औसत प्रतीक्षा समय |

## Priority Levels

Jobs को user level के आधार पर priority दी जाती है:

| User Level | Priority | Value |
|------------|----------|-------|
| Admin | Urgent | 30 |
| Lead | High | 20 |
| Researcher | Normal | 10 |
| Viewer | Low | 0 |

ऊँचा value = अधिक priority = पहले प्रोसेस।

### Priority Override

Leads और admins विशेष परिस्थितियों के लिए job की priority override कर सकते हैं (जैसे urgent experiments)।

## Concurrency Limits

दो limits यह नियंत्रित करते हैं कि कितनी jobs एक साथ चल सकती हैं:

### Global Limit (`max_concurrent`)

सभी users में कुल चल रही jobs की अधिकतम संख्या। डिफ़ॉल्ट: **5 jobs**.

### Per‑User Limit (`user_max_concurrent`)

किसी एक user की एक साथ चलने वाली jobs की अधिकतम संख्या। डिफ़ॉल्ट: **2 jobs**.

यह एक user को सभी slots consume करने से रोकता है।

### Limits अपडेट करना

Admins queue panel या API से limits अपडेट कर सकते हैं।

<details>
<summary>Example</summary>

```bash
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{"max_concurrent": 10, "user_max_concurrent": 3}'
```

</details>

## Queue में Job Lifecycle

1. **Submitted** - job बनाई गई, queue में `pending` status के साथ जोड़ी गई
2. **Pending** - slot का इंतज़ार (concurrency limit)
3. **Running** - cloud GPU पर सक्रिय training
4. **Completed/Failed** - terminal स्थिति, active queue से हटाई जाती है

## API Endpoints

### List Queue Entries

```
GET /api/queue
```

Parameters:
- `status` - status के आधार पर फ़िल्टर (pending, running, blocked)
- `limit` - अधिकतम entries (डिफ़ॉल्ट: 50)
- `include_completed` - समाप्त jobs शामिल करें

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

वर्तमान user की queue position, pending jobs, और running jobs लौटाता है।

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
| `pending_count` | int | queue में प्रतीक्षा कर रही jobs की संख्या |
| `running_count` | int | वर्तमान में चल रही jobs की संख्या |
| `blocked_count` | int | approval के लिए रुकी jobs की संख्या |
| `best_position` | int या null | user की सबसे उच्च‑priority pending job की position |
| `pending_jobs` | array | pending job details की सूची |
| `running_jobs` | array | running job details की सूची |

`best_position` फ़ील्ड user की सर्वोत्तम (सबसे उच्च priority या earliest submitted) pending job की queue position दिखाता है। यह users को समझने में मदद करता है कि उनकी अगली job कब शुरू होगी। `null` का अर्थ है कि user की कोई pending job नहीं है।

</details>

### Job Position

```
GET /api/queue/position/{job_id}
```

किसी विशिष्ट job के लिए queue position लौटाता है।

### Cancel Queued Job

```
POST /api/queue/{job_id}/cancel
```

शुरू न हुई job को cancel करता है।

### Approve Blocked Job

```
POST /api/queue/{job_id}/approve
```

Admin‑only. approval आवश्यक job को approve करता है (जैसे cost threshold पार करने पर)।

### Reject Blocked Job

```
POST /api/queue/{job_id}/reject?reason=<reason>
```

Admin‑only. blocked job को reason के साथ reject करता है।

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

Admin‑only. queue processing को manually ट्रिगर करता है (सामान्यतः automatic)।

### Cleanup Old Entries

```
POST /api/queue/cleanup?days=30
```

Admin‑only. निर्दिष्ट दिनों से पुराने completed entries हटाता है।

**Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `days` | int | 30 | 1-365 | दिनों में retention अवधि |

**Behavior:**

ऐसी queue entries हटाता है जो:
- terminal status (`completed`, `failed`, या `cancelled`) में हों
- `completed_at` timestamp निर्दिष्ट दिनों से पुराना हो

Active jobs (pending, running, blocked) कभी नहीं हटाई जातीं।

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
# 7 दिनों से पुराने entries साफ़ करें
curl -X POST "http://localhost:8000/api/queue/cleanup?days=7" \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# 90 दिनों से पुराने entries साफ़ करें (quarterly cleanup)
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
| `JobRepository` | `storage/job_repository.py` | jobs और queue के लिए unified SQLite persistence |
| `JobRepoQueueAdapter` | `queue/job_repo_adapter.py` | scheduler compatibility के लिए adapter |
| `QueueScheduler` | `queue/scheduler.py` | scheduling logic |
| `SchedulingPolicy` | `queue/scheduler.py` | priority/fairness algorithm |
| `QueueDispatcher` | `queue/dispatcher.py` | job dispatch संभालता है |
| `QueueEntry` | `queue/models.py` | queue entry data model |
| `LocalGPUAllocator` | `services/local_gpu_allocator.py` | local jobs के लिए GPU allocation |

### Database Schema

Queue और job entries unified SQLite database (`~/.simpletuner/cloud/jobs.db`) में स्टोर होती हैं।

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

Migrations startup पर स्वचालित रूप से चलते हैं।

</details>

## Local GPU Concurrency

Local training jobs submit करने पर, queue सिस्टम resource conflicts रोकने के लिए GPU allocation को ट्रैक करता है। यदि आवश्यक GPUs उपलब्ध नहीं हैं, तो jobs queue में रखी जाती हैं।

### GPU Allocation Tracking

हर local job यह निर्दिष्ट करती है:

- **num_processes** - आवश्यक GPUs की संख्या (`--num_processes` से)
- **device_ids** - preferred GPU indices (`--accelerate_visible_devices` से)

Allocator चल रही jobs को दिए गए GPUs ट्रैक करता है और नई jobs तभी शुरू करता है जब संसाधन उपलब्ध हों।

### CLI Options

#### Jobs सबमिट करना

<details>
<summary>Examples</summary>

```bash
# Job submit करें, GPUs उपलब्ध न हों तो queue करें (डिफ़ॉल्ट)
simpletuner jobs submit my-config

# GPUs उपलब्ध न हों तो तुरंत reject करें
simpletuner jobs submit my-config --no-wait

# configured device IDs की जगह कोई भी उपलब्ध GPU उपयोग करें
simpletuner jobs submit my-config --any-gpu

# GPU उपलब्धता जाँचने के लिए dry-run
simpletuner jobs submit my-config --dry-run
```

</details>

#### Jobs लिस्ट करना

<details>
<summary>Examples</summary>

```bash
# Recent jobs लिस्ट करें
simpletuner jobs list

# specific fields के साथ लिस्ट करें
simpletuner jobs list -o job_id,status,config_name

# JSON आउटपुट और custom fields
simpletuner jobs list --format json -o job_id,status

# dot notation से nested fields एक्सेस करें
simpletuner jobs list --format json -o job_id,metadata.run_name

# status से फ़िल्टर करें
simpletuner jobs list --status running
simpletuner jobs list --status queued

# परिणाम limit करें
simpletuner jobs list -l 10
```

`-o` (output) विकल्प dot notation सपोर्ट करता है ताकि job metadata के nested fields एक्सेस किए जा सकें। उदाहरण के लिए, `metadata.run_name` job की metadata object से `run_name` निकालता है।

</details>

### GPU Status API

GPU allocation status system status endpoint से उपलब्ध है:

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

Queue statistics में local GPU info भी शामिल होता है:

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

Existing concurrency endpoint के जरिए local jobs और GPUs की एक साथ सीमा नियंत्रित करें:

```
GET /api/queue/concurrency
POST /api/queue/concurrency
```

अब concurrency endpoint cloud limits के साथ local GPU limits भी स्वीकार करता है:

| Field | Type | Description |
|-------|------|-------------|
| `max_concurrent` | int | अधिकतम cloud jobs running (डिफ़ॉल्ट: 5) |
| `user_max_concurrent` | int | प्रति user अधिकतम cloud jobs (डिफ़ॉल्ट: 2) |
| `local_gpu_max_concurrent` | int या null | local jobs के लिए अधिकतम GPUs (null = unlimited) |
| `local_job_max_concurrent` | int | एक साथ स्थानीय jobs की अधिकतम संख्या (डिफ़ॉल्ट: 1) |

<details>
<summary>Example</summary>

```bash
# कुल 6 GPUs तक उपयोग करते हुए 2 local jobs की अनुमति दें
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
| `running` | allocated GPUs के साथ job तुरंत शुरू हुई |
| `queued` | job queue में रखी गई, GPUs उपलब्ध होने पर शुरू होगी |
| `rejected` | GPUs उपलब्ध नहीं थे और `no_wait` true था |

### Automatic Job Processing

जब कोई job पूरी होती है या fail होती है, उसके GPUs release होते हैं और queue process होकर pending jobs शुरू करता है। यह process keeper lifecycle hooks द्वारा स्वतः होता है।

**Cancellation behavior**: जब कोई job cancel होती है, GPUs release होते हैं लेकिन pending jobs स्वतः शुरू नहीं होतीं। इससे bulk cancellation (`simpletuner jobs cancel --all`) के दौरान race conditions से बचाव होता है, जहाँ pending jobs cancel होने से पहले शुरू हो सकती थीं। Cancellation के बाद queue processing manually trigger करने के लिए `POST /api/queue/process` या server restart उपयोग करें।

## Worker Dispatch

Jobs को orchestrator की local GPUs के बजाय remote workers पर dispatch किया जा सकता है। पूरा worker setup देखने के लिए [Worker Orchestration](experimental/server/WORKERS.md) देखें।

### Job Targets

Job submit करते समय target तय करें:

| Target | Behavior |
|--------|----------|
| `auto` (डिफ़ॉल्ट) | पहले remote workers आज़माएँ, फिर local GPUs पर fallback करें |
| `worker` | केवल remote workers पर dispatch करें; उपलब्ध न हों तो queue करें |
| `local` | केवल orchestrator के local GPUs पर चलाएँ |

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

Jobs worker matching के लिए label requirements specify कर सकते हैं:

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-config",
    "target": "worker",
    "worker_labels": {"gpu_type": "a100*", "location": "us-*"}
  }'
```

Labels glob patterns सपोर्ट करते हैं। Scheduler jobs को workers से निम्न आधार पर मैच करता है:

1. Label requirements (सभी match होने चाहिए)
2. GPU count requirements
3. Worker availability (IDLE status)
4. Matching workers में FIFO order

### Startup Behavior

Server startup पर queue system किसी भी pending local jobs को स्वतः process करता है। यदि GPUs उपलब्ध हों, तो queued jobs बिना manual हस्तक्षेप के तुरंत शुरू हो जाएँगी। इससे server restart से पहले submit की गई jobs server के वापस आने पर process होती रहती हैं।

Startup sequence:
1. Server GPU allocator initialize करता है
2. Pending local jobs queue से निकाली जाती हैं
3. हर pending job के लिए, यदि GPUs उपलब्ध हों, job शुरू होती है
4. जिन jobs को शुरू नहीं किया जा सकता (insufficient GPUs), वे queued रहती हैं

Note: Cloud jobs अलग cloud queue scheduler द्वारा संभाली जाती हैं, जो startup पर फिर से resume करता है।

## कॉन्फ़िगरेशन

Queue concurrency limits API के जरिए कॉन्फ़िगर की जाती हैं और queue database में persist होती हैं।

**Web UI से:** Cloud tab → Queue Panel → Settings

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
| `max_concurrent` | 5 | global maximum running jobs |
| `user_max_concurrent` | 2 | प्रति user अधिकतम running jobs |
| `team_max_concurrent` | 10 | प्रति team अधिकतम running jobs |
| `enable_fair_share` | false | per‑team concurrency limits सक्षम करें |

### Fair‑Share Scheduling

`enable_fair_share: true` होने पर scheduler team affiliation का उपयोग करता है ताकि कोई एक टीम resources monopolize न कर सके।

#### यह कैसे काम करता है

Fair‑share concurrency control की तीसरी परत जोड़ता है:

| Layer | Limit | Purpose |
|-------|-------|---------|
| Global | `max_concurrent` | सभी users/teams के बीच कुल jobs |
| Per‑User | `user_max_concurrent` | एक user को सभी slots लेने से रोकता है |
| Per‑Team | `team_max_concurrent` | एक टीम को सभी slots लेने से रोकता है |

जब किसी job को dispatch के लिए विचार किया जाता है:

1. global limit जाँचें → capacity पर हो तो skip करें
2. user limit जाँचें → user capacity पर हो तो skip करें
3. यदि fair‑share सक्षम है और job में `team_id` है:
   - team limit जाँचें → team capacity पर हो तो skip करें

जिन jobs में `team_id` नहीं है, वे team limits के अधीन नहीं होतीं।

#### Fair‑Share सक्षम करना

**UI से:** Cloud tab → Queue Panel → "Fair‑Share Scheduling" toggle करें

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

Teams admin panel में users को assign किए जाते हैं। जब user job submit करता है, उसकी टीम ID queue entry से जुड़ती है। Scheduler running jobs per team ट्रैक करता है और limit लागू करता है।

<details>
<summary>Example scenario</summary>

`max_concurrent=6`, `user_max_concurrent=2`, `team_max_concurrent=3` के साथ:

| Team | Users | Submitted | Running | Blocked |
|------|-------|-----------|---------|---------|
| Alpha | Alice, Bob | 4 | 3 (team limit पर) | 1 |
| Beta | Carol | 3 | 2 | 1 (global slot का इंतज़ार) |

- Team Alpha के 3 running हैं ( `team_max_concurrent` पर)
- कुल running 5 हैं ( `max_concurrent=6` के अंदर)
- Carol की job blocked है क्योंकि: 5+1=6, global limit पर
- Alice की 4th job blocked है क्योंकि: टीम 3/3 पर है

इससे सुनिश्चित होता है कि कोई टीम queue को monopolize न कर सके, भले ही वे कई jobs submit करें।

</details>

### Starvation Prevention

`starvation_threshold_minutes` से अधिक समय तक प्रतीक्षा करने वाली jobs को priority boost मिलता है, ताकि वे अनिश्चितकाल तक न रुकी रहें।

## Approval Workflow

कुछ jobs को approval की आवश्यकता के रूप में चिह्नित किया जा सकता है (जैसे जब अनुमानित लागत threshold पार हो जाए):

1. Job `requires_approval: true` के साथ submit होती है
2. Job `blocked` status में जाती है
3. Admin queue panel या API के जरिए समीक्षा करता है
4. Admin approve या reject करता है
5. Approve होने पर job `pending` में जाती है और सामान्य रूप से schedule होती है

Approval rule कॉन्फ़िगरेशन के लिए [Enterprise Guide](experimental/server/ENTERPRISE.md) देखें।

## Troubleshooting

### Jobs queue में अटकी हैं

<details>
<summary>Debugging steps</summary>

Concurrency limits जाँचें:
```bash
curl http://localhost:8000/api/queue/stats
```

यदि `running` = `max_concurrent` है, jobs slots के लिए प्रतीक्षा कर रही हैं।

</details>

### Queue प्रोसेस नहीं हो रही

<details>
<summary>Debugging steps</summary>

Background processor हर 5 seconds में चलता है। errors के लिए server logs जाँचें:
```
Queue scheduler started with 5s processing interval
```

यदि यह दिख नहीं रहा, scheduler start नहीं हुआ हो सकता है।

</details>

### Job queue से गायब हो गई

<details>
<summary>Debugging steps</summary>

जाँचें कि job complete या fail तो नहीं हुई:
```bash
curl "http://localhost:8000/api/queue?include_completed=true"
```

</details>

### Local jobs running दिख रही हैं लेकिन training नहीं हो रही

<details>
<summary>Debugging steps</summary>

यदि `jobs list` local jobs को "running" दिखाता है लेकिन training नहीं हो रही:

1. GPU allocation status जाँचें:
   ```bash
   simpletuner jobs status --format json
   ```
   `local.allocated_gpus` फ़ील्ड देखें — इसमें उपयोग हो रहे GPUs दिखने चाहिए।

2. यदि allocated GPUs खाली हैं लेकिन running count non‑zero है, तो queue state inconsistent हो सकता है। Server restart करें ताकि automatic queue reconciliation हो सके।

3. GPU allocation errors के लिए server logs देखें:
   ```
   Failed to allocate GPUs [0] to job <job_id>
   ```

</details>

### Queue Depth गलत count दिखा रहा है

<details>
<summary>Explanation</summary>

Queue depth और running job counts local और cloud jobs के लिए अलग‑अलग गणना होते हैं:

- **Local jobs**: `LocalGPUAllocator` के माध्यम से GPU allocation state पर आधारित
- **Cloud jobs**: provider status पर आधारित `QueueScheduler` द्वारा ट्रैक

ब्रेकडाउन देखने के लिए `simpletuner jobs status --format json` उपयोग करें:
- `local.running_jobs` - local training jobs running
- `local.pending_jobs` - GPUs के लिए queued local jobs
- `running` - total running jobs (cloud queue)
- `queue_depth` - pending cloud jobs

</details>

## See Also

- [Worker Orchestration](experimental/server/WORKERS.md) - Distributed worker registration और job dispatch
- [Cloud Training Tutorial](experimental/cloud/TUTORIAL.md) - Cloud training शुरू करने के लिए
- [Enterprise Guide](experimental/server/ENTERPRISE.md) - Multi‑user setup, approvals, governance
- [Operations Guide](experimental/cloud/OPERATIONS_TUTORIAL.md) - Production deployment

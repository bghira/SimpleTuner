# Job Progress और Sync API

यह दस्तावेज़ SimpleTuner में cloud training job progress मॉनिटर करने और local job state को cloud providers के साथ synchronized रखने के तरीकों को कवर करता है।

## Overview

SimpleTuner job status ट्रैक करने के कई तरीके देता है:

| Method | Use Case | Latency | Resource Usage |
|--------|----------|---------|----------------|
| Inline Progress API | running jobs के लिए UI polling | Low (डिफ़ॉल्ट 5s) | प्रति‑job API calls |
| Job Sync (pull) | provider से jobs discover करना | Medium (on-demand) | Batch API call |
| `sync_active` param | active job statuses refresh | Medium (on-demand) | प्रति‑active‑job calls |
| Background Poller | automatic status updates | Configurable (डिफ़ॉल्ट 30s) | Continuous polling |
| Webhooks | real-time push notifications | Instant | polling की जरूरत नहीं |

## Inline Progress API

### Endpoint

```
GET /api/cloud/jobs/{job_id}/inline-progress
```

### उद्देश्य

एक running job के लिए compact progress जानकारी लौटाता है, ताकि job list में inline status update दिखाए जा सकें बिना full logs fetch किए।

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
| `job_id` | string | job identifier |
| `stage` | string या null | वर्तमान training stage: `Preprocessing`, `Warmup`, `Training`, `Validation`, `Saving checkpoint` |
| `last_log` | string या null | अंतिम log line (80 chars तक truncate) |
| `progress` | float या null | step/epoch parsing के आधार पर progress प्रतिशत (0-100) |

### Stage Detection

API recent log lines parse करके training stage निर्धारित करता है:

- **Preprocessing**: logs में "preprocessing" या "loading" मिलने पर
- **Warmup**: logs में "warming up" या "warmup" मिलने पर
- **Training**: logs में "step" या "epoch" patterns मिलने पर
- **Validation**: logs में "validat" मिलने पर
- **Saving checkpoint**: logs में "checkpoint" मिलने पर

### Progress Calculation

Progress percentage ऐसे log patterns से निकाला जाता है:
- `step 1500/5000` -> 30%
- `epoch 3/10` -> 30%

### कब उपयोग करें

Inline progress API तब उपयोग करें जब:
- job list cards में compact status दिखाना हो
- running jobs के लिए frequent polling (हर 5 सेकंड) करना हो
- प्रति‑request न्यूनतम data transfer चाहिए हो

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

SimpleTuner local job state को cloud providers के साथ current रखने के लिए दो sync approaches देता है।

### 1. Full Provider Sync

#### Endpoint

```
POST /api/cloud/jobs/sync
```

#### उद्देश्य

Cloud provider से ऐसे jobs खोजता है जो local store में नहीं हैं। यह उपयोगी है जब:
- jobs SimpleTuner के बाहर (सीधे Replicate API से) submit हुए हों
- local job store reset या corrupted हो गया हो
- आप historical jobs import करना चाहते हों

#### Response

```json
{
  "synced": 3,
  "message": "Discovered 3 new jobs from Replicate"
}
```

#### Behavior

1. Replicate से हाल के 100 jobs तक fetch करता है
2. हर job के लिए:
   - यदि local store में नहीं है: नया `UnifiedJob` record बनाता है
   - यदि store में है: status, cost, और timestamps अपडेट करता है
3. newly discovered jobs की गिनती लौटाता है

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

Cloud dashboard में orphaned jobs खोजने के लिए sync button होता है:

1. job list toolbar में **Sync** बटन क्लिक करें
2. sync के दौरान बटन loading spinner दिखाता है
3. success पर toast दिखता है: *"Discovered N jobs from Replicate"*
4. job list और metrics अपने आप refresh होते हैं

**Use Cases:**
- Replicate API या web console से सीधे submit किए jobs को discover करना
- database reset के बाद recovery
- shared team Replicate account से jobs import करना

Sync बटन अंदर से `POST /api/cloud/jobs/sync` कॉल करता है और फिर job list तथा dashboard metrics reload करता है।

### 2. Active Job Status Sync (`sync_active`)

#### Endpoint

```
GET /api/cloud/jobs?sync_active=true
```

#### उद्देश्य

Job list लौटाने से पहले सभी active (non-terminal) cloud jobs की स्थिति refresh करता है। इससे background polling का इंतज़ार किए बिना up-to-date status मिलता है।

#### Active States

इन states में jobs “active” माने जाते हैं और sync होते हैं:
- `pending` - job submit हुआ लेकिन शुरू नहीं हुआ
- `uploading` - data upload चल रहा है
- `queued` - provider queue में इंतज़ार
- `running` - training चल रही है

#### Behavior

1. jobs list करने से पहले हर active cloud job का current status fetch करता है
2. local store में अपडेट करता है:
   - current status
   - `started_at` / `completed_at` timestamps
   - `cost_usd` (जमा लागत)
   - `error_message` (यदि failed)
3. updated job list लौटाता है

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

### तुलना: Sync बनाम sync_active

| Feature | `POST /jobs/sync` | `GET /jobs?sync_active=true` |
|---------|-------------------|------------------------------|
| New jobs discover करता है | Yes | No |
| Existing jobs अपडेट करता है | Yes | Yes (केवल active) |
| Scope | सभी provider jobs | केवल active local jobs |
| Use case | initial import, recovery | नियमित status refresh |
| Performance | भारी (batch query) | हल्का (selective) |

## बैकग्राउंड पोलर कॉन्फ़िगरेशन

बैकग्राउंड पोलर क्लाइंट इंटरवेंशन के बिना active job statuses को अपने आप sync करता है।

### डिफ़ॉल्ट व्यवहार

- **Auto-enabled**: अगर कोई webhook URL कॉन्फ़िगर नहीं है, तो polling अपने आप शुरू हो जाती है
- **Default interval**: 30 सेकंड
- **Scope**: सभी active cloud jobs

<details>
<summary>Enterprise Configuration</summary>

प्रोडक्शन डिप्लॉयमेंट के लिए `simpletuner-enterprise.yaml` से polling कॉन्फ़िगर करें:

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

### यह कैसे काम करता है

1. सर्वर स्टार्टअप पर `BackgroundTaskManager` जांचता है:
   - अगर enterprise config polling को स्पष्ट रूप से enable करता है, तो वही interval उपयोग होता है
   - अन्यथा, अगर webhook कॉन्फ़िगर नहीं है, तो 30s interval के साथ auto-enable करता है
2. हर interval पर, poller:
   - active status वाले सभी jobs सूचीबद्ध करता है
   - provider के आधार पर group करता है
   - हर provider से current status fetch करता है
   - local store अपडेट करता है
   - status change के लिए SSE events emit करता है
   - terminal states के लिए queue entries अपडेट करता है

<details>
<summary>SSE Events</summary>

जब background poller status changes detect करता है, तो वह SSE events broadcast करता है:

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

## सर्वश्रेष्ठ अभ्यास

### 1. सही Sync रणनीति चुनें

| Scenario | Recommended Approach |
|----------|---------------------|
| Initial page load | sync के बिना `GET /jobs` (फास्ट) |
| Periodic refresh (30s) | `GET /jobs?sync_active=true` |
| User clicks "Refresh" | discovery के लिए `POST /jobs/sync` |
| Running job details | Inline progress API (5s) |
| Production deployment | Background poller + webhooks |

### 2. Over-Polling से बचें

<details>
<summary>Example</summary>

```javascript
// Good: Poll inline progress only for running jobs
const runningJobs = jobs.filter(j => j.status === 'running');

// Bad: Poll all jobs regardless of status
for (const job of jobs) { /* ... */ }
```

</details>

### 3. Real-Time Updates के लिए SSE का उपयोग करें

<details>
<summary>Example</summary>

Aggressive polling की जगह SSE events subscribe करें:

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

### 4. Terminal States हैंडल करें

<details>
<summary>Example</summary>

जो jobs terminal states तक पहुंच चुके हैं, उन्हें poll करना बंद करें:

```javascript
const terminalStates = ['completed', 'failed', 'cancelled'];

function shouldPollJob(job) {
    return !terminalStates.includes(job.status);
}
```

</details>

### 5. प्रोडक्शन के लिए Webhooks कॉन्फ़िगर करें

<details>
<summary>Example</summary>

Webhooks polling की जरूरत पूरी तरह खत्म कर देते हैं:

```yaml
# In provider config
webhook_url: "https://your-server.com/api/webhooks/replicate"
```

Webhooks कॉन्फ़िगर होने पर:
- Background polling बंद हो जाता है (जब तक explicit enable न हो)
- Status updates provider callbacks से real-time में आते हैं
- Provider पर API calls कम हो जाते हैं

</details>

## ट्रबलशूटिंग

### Jobs अपडेट नहीं हो रहीं

<details>
<summary>Debugging steps</summary>

1. जांचें कि background poller चल रहा है या नहीं:
   ```bash
   # Look for log line on startup
   grep "job status polling" server.log
   ```

2. Provider connectivity सत्यापित करें:
   ```bash
   curl http://localhost:8001/api/cloud/providers/replicate/validate
   ```

3. एक sync force करें:
   ```bash
   curl -X POST http://localhost:8001/api/cloud/jobs/sync
   ```

</details>

### SSE Events नहीं मिल रहे

<details>
<summary>Debugging steps</summary>

1. SSE connection limit जांचें (डिफ़ॉल्ट रूप से प्रति IP 5)
2. पुष्टि करें कि EventSource कनेक्ट है:
   ```javascript
   eventSource.addEventListener('open', () => {
       console.log('SSE connected');
   });
   ```

</details>

### Provider API उपयोग बहुत ज्यादा

<details>
<summary>Solutions</summary>

अगर rate limits लग रहे हैं:
1. enterprise config में `job_polling_interval` बढ़ाएं
2. inline progress polling की frequency कम करें
3. polling खत्म करने के लिए webhooks कॉन्फ़िगर करें

</details>

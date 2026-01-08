# Worker Orchestration

SimpleTuner का worker orchestration आपको कई GPU मशीनों पर training jobs वितरित करने देता है। Workers एक central orchestrator के साथ रजिस्टर होते हैं, real-time में job dispatch events प्राप्त करते हैं, और status वापस रिपोर्ट करते हैं।

## ओवरव्यू

Orchestrator/worker आर्किटेक्चर सक्षम करता है:

- **Distributed training** - GPUs वाली किसी भी मशीन पर jobs चलाना, कहीं भी
- **Auto-discovery** - Workers अपनी GPU क्षमताएँ स्वचालित रूप से रजिस्टर करते हैं
- **Real-time dispatch** - SSE (Server-Sent Events) के जरिए jobs dispatch होते हैं
- **Mixed fleet** - cloud-launched ephemeral workers को persistent on-prem मशीनों के साथ मिलाएं
- **Fault tolerance** - orphaned jobs स्वचालित रूप से requeue हो जाते हैं

## Worker प्रकार

| Type | Lifecycle | Use Case |
|------|-----------|----------|
| **Ephemeral** | Job पूरा होने के बाद shutdown | Cloud spot instances (RunPod, Vast.ai) |
| **Persistent** | Jobs के बीच online रहता है | On-prem GPUs, reserved instances |

## Quick Start

### 1. Orchestrator शुरू करें

अपने central machine पर SimpleTuner server चलाएँ:

```bash
simpletuner server --host 0.0.0.0 --port 8001
```

Production के लिए SSL सक्षम करें:

```bash
simpletuner server --host 0.0.0.0 --port 8001 --ssl
```

### 2. Worker Token बनाएँ

**Web UI के जरिए:** Administration → Workers → Create Worker

**API के जरिए:**

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

Response में token मिलता है (केवल एक बार दिखाया जाता है):

```json
{
  "worker_id": "w_abc123",
  "token": "wt_xxxxxxxxxxxx",
  "name": "gpu-worker-1"
}
```

### 3. Worker शुरू करें

GPU मशीन पर:

```bash
simpletuner worker \
  --orchestrator-url https://orchestrator.example.com:8001 \
  --worker-token wt_xxxxxxxxxxxx \
  --name gpu-worker-1 \
  --persistent
```

या environment variables के जरिए:

```bash
export SIMPLETUNER_ORCHESTRATOR_URL=https://orchestrator.example.com:8001
export SIMPLETUNER_WORKER_TOKEN=wt_xxxxxxxxxxxx
export SIMPLETUNER_WORKER_NAME=gpu-worker-1
export SIMPLETUNER_WORKER_PERSISTENT=true

simpletuner worker
```

Worker निम्न करेगा:

1. Orchestrator से कनेक्ट
2. GPU क्षमताएँ रिपोर्ट (auto-detected)
3. Job dispatch loop में प्रवेश
4. हर 30 सेकंड में heartbeats भेजना

### 4. Workers को jobs भेजें

**Web UI के जरिए:** अपनी training कॉन्फ़िगर करें, फिर **Train in Cloud** → target के रूप में **Worker** चुनें।

**API के जरिए:**

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H "Content-Type: application/json" \
  -d '{
    "config_name": "my-training-config",
    "target": "worker"
  }'
```

Target विकल्प:

| Target | Behavior |
|--------|----------|
| `worker` | केवल remote workers पर dispatch |
| `local` | orchestrator के GPUs पर चलाएँ |
| `auto` | यदि worker उपलब्ध हो तो उसे प्राथमिकता दें, अन्यथा local पर fallback |

## CLI Reference

```
simpletuner worker [OPTIONS]

OPTIONS:
  --orchestrator-url URL   Orchestrator panel URL (या SIMPLETUNER_ORCHESTRATOR_URL)
  --worker-token TOKEN     Authentication token (या SIMPLETUNER_WORKER_TOKEN)
  --name NAME              Worker name (default: hostname)
  --persistent             Jobs के बीच online रहें (default: ephemeral)
  -v, --verbose            Debug logging सक्षम करें
```

### Ephemeral बनाम Persistent मोड

**Ephemeral (default):**
- एक job पूरा होने के बाद worker shutdown हो जाता है
- प्रति मिनट बिलिंग वाले cloud spot instances के लिए आदर्श
- Orchestrator offline ephemeral workers को 1 घंटे के बाद साफ़ करता है

**Persistent (`--persistent`):**
- Worker नए jobs का इंतजार करते हुए online रहता है
- कनेक्शन drop होने पर स्वतः reconnect करता है
- On-prem GPUs या reserved instances के लिए उपयोग करें

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
| `CONNECTING` | Worker कनेक्शन स्थापित कर रहा है |
| `IDLE` | Jobs प्राप्त करने के लिए तैयार |
| `BUSY` | वर्तमान में job चला रहा है |
| `DRAINING` | वर्तमान job समाप्त कर रहा है, फिर shutdown करेगा |
| `OFFLINE` | Disconnected (heartbeat timeout) |

## Health Monitoring

Orchestrator worker health मॉनिटर करता है:

- **Heartbeat interval:** 30 seconds (worker → orchestrator)
- **Timeout threshold:** 120 seconds बिना heartbeat → offline मार्क
- **Health check loop:** Orchestrator पर हर 60 सेकंड में चलता है

### Failures को संभालना

**Job के दौरान worker offline हो जाए:**

1. Heartbeat timeout के बाद job failed मार्क
2. यदि retries शेष हैं (डिफ़ॉल्ट: 3), job requeue
3. अगला उपलब्ध worker job उठाता है

**Orchestrator restart:**

1. Workers अपने आप reconnect करते हैं
2. Workers in-progress jobs रिपोर्ट करते हैं
3. Orchestrator state reconcile करता है और resume करता है

## GPU Matching

Workers रजिस्ट्रेशन पर अपनी GPU क्षमताएँ रिपोर्ट करते हैं:

```json
{
  "gpu_count": 2,
  "gpu_name": "NVIDIA A100-SXM4-80GB",
  "gpu_vram_gb": 80,
  "accelerator_type": "cuda"
}
```

Jobs GPU आवश्यकताएँ निर्दिष्ट कर सकते हैं:

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H "Content-Type: application/json" \
  -d '{
    "config_name": "my-config",
    "target": "worker",
    "worker_labels": {"gpu_type": "a100*"}
  }'
```

Scheduler jobs को workers से निम्न आधार पर match करता है:

1. GPU count requirements
2. Label matching (glob patterns समर्थित)
3. Worker availability (IDLE status)

## Labels

Labels flexible worker selection देते हैं:

**Worker creation पर labels असाइन करें:**

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

**Label के आधार पर workers चुनें:**

```bash
# team=nlp वाले workers match करें
curl -s -X POST http://localhost:8001/api/queue/submit \
  -d '{"config_name": "my-config", "worker_labels": {"team": "nlp"}}'

# gpu_type "a100" से शुरू होने वाले workers match करें
curl -s -X POST http://localhost:8001/api/queue/submit \
  -d '{"config_name": "my-config", "worker_labels": {"gpu_type": "a100*"}}'
```

## Admin Operations

### Workers सूचीबद्ध करें

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

### Worker Drain करें

Current job को gracefully पूरा करें और नए dispatch रोकें:

```bash
curl -s -X POST http://localhost:8001/api/admin/workers/w_abc123/drain
```

Worker:

1. कोई भी running job पूरा करता है
2. DRAINING status में जाता है
3. नए jobs को अस्वीकार करता है
4. Job पूरा होने के बाद disconnect (ephemeral) या draining state में रहता है (persistent)

### Token Rotate करें

Worker का authentication token फिर से जनरेट करें:

```bash
curl -s -X POST http://localhost:8001/api/admin/workers/w_abc123/token
```

पुराना token तुरंत invalid हो जाता है। नए token के साथ worker अपडेट करें।

### Worker हटाएँ

```bash
curl -s -X DELETE http://localhost:8001/api/admin/workers/w_abc123
```

यह केवल तब काम करता है जब worker offline हो।

## Security

### Token Authentication

- Workers `X-Worker-Token` header के जरिए authenticate करते हैं
- Tokens स्टोरेज से पहले SHA-256 hashed होते हैं
- Tokens creation के बाद orchestrator से बाहर नहीं जाते
- सुरक्षा के लिए tokens समय-समय पर rotate करें

### Network Security

Production के लिए:

1. `--ssl` flag उपयोग करें या reverse proxy पर TLS terminate करें
2. Worker registration को trusted networks तक सीमित करें
3. `/api/workers/*` endpoints तक पहुँच सीमित करने के लिए firewall rules उपयोग करें

### Audit Logging

सभी worker actions लॉग होते हैं:

- Registration attempts (success/failure)
- Job dispatch events
- Status transitions
- Token rotations
- Admin operations

Log access के लिए [Audit Guide](AUDIT.md) देखें।

## Troubleshooting

### Worker कनेक्ट नहीं हो रहा

**"Connection refused"**
- Orchestrator URL और port सत्यापित करें
- Firewall rules inbound connections की अनुमति दे रहे हैं या नहीं जांचें
- सुनिश्चित करें कि orchestrator `--host 0.0.0.0` के साथ चल रहा है

**"Invalid token"**
- Token rotate हो गया होगा—नया token मांगें
- Token string में whitespace जांचें

**"SSL certificate verify failed"**
- self-signed certs के लिए `--ssl-no-verify` उपयोग करें (केवल development)
- या CA certificate को system trust store में जोड़ें

### Worker अप्रत्याशित रूप से offline हो जाता है

**Heartbeat timeout (120s)**
- worker और orchestrator के बीच network stability जांचें
- worker पर resource exhaustion (CPU/memory) देखें
- unreliable network पर `SIMPLETUNER_HEARTBEAT_TIMEOUT` बढ़ाएँ

**Process crash**
- worker logs में Python exceptions देखें
- GPU drivers काम कर रहे हैं या नहीं जांचें (`nvidia-smi`)
- training के लिए पर्याप्त disk space सुनिश्चित करें

### Jobs workers को dispatch नहीं हो रहे

**No idle workers**
- admin panel में worker status जांचें
- workers connected हैं और IDLE हैं या नहीं देखें
- job और workers के बीच label mismatch जांचें

**GPU requirements पूरी नहीं**
- job को जितने GPUs चाहिए, कोई worker उतने नहीं रखता
- training config में `--num_processes` समायोजित करें

## API Reference

### Worker Endpoints (Worker → Orchestrator)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/workers/register` | POST | Register करें और capabilities रिपोर्ट करें |
| `/api/workers/stream` | GET | Job dispatch के लिए SSE stream |
| `/api/workers/heartbeat` | POST | Periodic keepalive |
| `/api/workers/job/{id}/status` | POST | Job progress रिपोर्ट करें |
| `/api/workers/disconnect` | POST | Graceful shutdown notification |

### Admin Endpoints (`admin.workers` permission आवश्यक)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/admin/workers` | GET | सभी workers सूचीबद्ध करें |
| `/api/admin/workers` | POST | worker token बनाएं |
| `/api/admin/workers/{id}` | DELETE | worker हटाएँ |
| `/api/admin/workers/{id}/drain` | POST | worker drain करें |
| `/api/admin/workers/{id}/token` | POST | token rotate करें |

## See Also

- [Enterprise Guide](ENTERPRISE.md) - SSO, quotas, approval workflows
- [Job Queue](../../JOB_QUEUE.md) - Queue scheduling और priorities
- [Cloud Training](../cloud/README.md) - Cloud provider integration
- [API Tutorial](../../api/TUTORIAL.md) - REST API के जरिए local training

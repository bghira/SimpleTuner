# Cloud Training System

> **Status:** Experimental
>
> **Available in:** Web UI (Cloud tab)

SimpleTuner का cloud training system आपको अपने खुद के infrastructure सेट किए बिना cloud GPU providers पर training jobs चलाने देता है। यह सिस्टम pluggable है, जिससे समय के साथ कई providers जोड़े जा सकते हैं।

## Overview

Cloud training system प्रदान करता है:

- **Unified job tracking** - local और cloud दोनों training jobs को एक जगह ट्रैक करें
- **Automatic data packaging** - local datasets अपने आप पैक और अपलोड होते हैं
- **Results delivery** - प्रशिक्षित मॉडल HuggingFace, S3 या लोकल में डाउनलोड किए जा सकते हैं
- **Cost tracking** - provider‑wise खर्च और job लागत ट्रैक करें, spending limits के साथ
- **Config snapshots** - optional रूप से git के साथ training configs को version-control करें

## Key Concepts

Cloud training उपयोग करने से पहले ये तीन बातें समझें:

### 1. आपके डेटा के साथ क्या होता है

जब आप cloud job सबमिट करते हैं:

1. **Datasets पैक किए जाते हैं** - local datasets (`type: "local"`) zip होते हैं और आपको summary दिखता है
2. **Provider पर अपलोड** - आपकी सहमति के बाद zip सीधे cloud provider पर जाती है
3. **Training चलता है** - cloud GPUs पर आपके samples ट्रेन होने से पहले मॉडल को डाउनलोड करना पड़ सकता है
4. **डेटा हटाया जाता है** - training के बाद अपलोड किया गया डेटा provider servers से हट जाता है और आपका मॉडल deliver किया जाता है

**Upload limit (Replicate):** Packaged archive 100 MiB या उससे कम होना चाहिए। इससे बड़े uploads submission से पहले block हो जाते हैं।

**Security notes:**
- आपका API token कभी आपकी मशीन से बाहर नहीं जाता
- Sensitive files (.env, .git, credentials) स्वतः exclude हो जाती हैं
- हर job से पहले आप uploads की समीक्षा और सहमति देते हैं

### 2. प्रशिक्षित मॉडल कैसे मिलेगा

Training एक मॉडल बनाता है जिसे कहीं deliver करना होता है। इनमें से एक कॉन्फ़िगर करें:

| Destination | Setup | Best For |
|-------------|-------|----------|
| **HuggingFace Hub** | `HF_TOKEN` env var सेट करें, Publishing tab में सक्षम करें | मॉडल शेयरिंग, आसान एक्सेस |
| **Local Download** | webhook URL सेट करें, ngrok के जरिए server expose करें | Privacy, local workflows |
| **S3 Storage** | Publishing tab में endpoint कॉन्फ़िगर करें | टीम एक्सेस, archival |

Step‑by‑step सेटअप के लिए [Receiving Trained Models](TUTORIAL.md#receiving-trained-models) देखें।

### 3. Cost Model

Replicate GPU समय के प्रति सेकंड बिल करता है:

| Hardware | VRAM | Cost | Typical LoRA (2000 steps) |
|----------|------|------|---------------------------|
| L40S | 48GB | ~$3.50/hr | $5-15 |

**Billing तब शुरू होता है** जब training शुरू होती है और **तब रुकता है** जब यह पूरा या fail होता है।

**खुद को सुरक्षित रखें:**
- Cloud settings में spending limit सेट करें
- हर submission से पहले cost estimates दिखते हैं
- चल रहे jobs कभी भी cancel करें (आप उपयोग किए गए समय के लिए भुगतान करेंगे)

Pricing और limits के लिए [Costs](REPLICATE.md#costs) देखें।

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web UI (Cloud Tab)                       │
├─────────────────────────────────────────────────────────────────┤
│  Job List  │  Metrics/Charts  │  Actions/Config  │  Job Details │
└─────────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │   Cloud API Routes  │
                    │   /api/cloud/*      │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
┌────────▼────────┐   ┌────────▼────────┐   ┌────────▼────────┐
│    JobStore     │   │ Upload Service  │   │    Provider     │
│  (Persistence)  │   │ (Data Packaging)│   │   Clients       │
└─────────────────┘   └─────────────────┘   └─────────────────┘
                                                     │
                             ┌───────────────────────┤
                             │                       │
                   ┌─────────▼─────────┐   ┌─────────▼─────────┐
                   │     Replicate     │   │  SimpleTuner.io   │
                   │    Cog Client     │   │   (Coming Soon)   │
                   └───────────────────┘   └───────────────────┘
```

## Supported Providers

| Provider | Status | Features |
|----------|--------|----------|
| [Replicate](REPLICATE.md) | Supported | Cost tracking, live logs, webhooks |
| [Worker Orchestration](../server/WORKERS.md) | Supported | Self-hosted distributed workers, कोई भी GPU |
| SimpleTuner.io | Coming Soon | SimpleTuner टीम द्वारा managed training service |

### Worker Orchestration

Self-hosted distributed training के लिए [Worker Orchestration Guide](../server/WORKERS.md) देखें। Workers यहाँ चल सकते हैं:

- On‑premise GPU servers
- Cloud VMs (any provider)
- Spot instances (RunPod, Vast.ai, Lambda Labs)

Workers आपके SimpleTuner orchestrator के साथ register होते हैं और jobs अपने आप प्राप्त करते हैं।

## Data Flow

### Job सबमिट करना

1. **Config Preparation** - आपकी training config serialize होती है
2. **Data Packaging** - Local datasets (`type: "local"`) zip होते हैं
3. **Upload** - Zip Replicate के file hosting पर अपलोड होती है
4. **Submission** - Job cloud provider को submit होता है
5. **Tracking** - Job status poll होता है और real‑time में अपडेट होता है

### परिणाम प्राप्त करना

परिणाम इस प्रकार deliver किए जा सकते हैं:

1. **HuggingFace Hub** - प्रशिक्षित मॉडल को आपके HuggingFace अकाउंट पर push करें
2. **S3-Compatible Storage** - किसी भी S3 endpoint (AWS, MinIO, आदि) पर अपलोड करें
3. **Local Download** - SimpleTuner में एक built‑in S3‑compatible endpoint है जो uploads को लोकल पर प्राप्त करता है

## Data Privacy & Consent

जब आप cloud job submit करते हैं, SimpleTuner अपलोड कर सकता है:

- **Training datasets** - `type: "local"` वाले datasets की images/files
- **Configuration** - आपके training parameters (learning rate, model settings, आदि)
- **Captions/metadata** - datasets से जुड़े कोई भी text files

डेटा सीधे cloud provider (जैसे Replicate file hosting) पर अपलोड होता है। यह SimpleTuner के servers के जरिए नहीं जाता।

### Consent Settings

Cloud tab में आप data upload व्यवहार कॉन्फ़िगर कर सकते हैं:

| Setting | Behavior |
|---------|----------|
| **Always Ask** | हर upload से पहले datasets सूचीबद्ध करते हुए confirmation dialog दिखाएँ |
| **Always Allow** | trusted workflows के लिए confirmation skip करें |
| **Never Upload** | cloud training बंद करें (केवल local) |

## Local S3 Endpoint

SimpleTuner में trained models प्राप्त करने के लिए built‑in S3‑compatible endpoint शामिल है:

```
PUT /api/cloud/storage/{bucket}/{key}
GET /api/cloud/storage/{bucket}/{key}
GET /api/cloud/storage/{bucket}  (list objects)
GET /api/cloud/storage  (list buckets)
```

Files डिफ़ॉल्ट रूप से `~/.simpletuner/cloud_outputs/` में स्टोर होती हैं।

आप credentials मैन्युअली सेट कर सकते हैं; यदि नहीं करते, तो प्रत्येक training job के लिए ephemeral credentials स्वतः बनते हैं, जो अनुशंसित तरीका है।

यह "download only" मोड को सक्षम करता है जहाँ आप:
1. अपने local SimpleTuner instance की ओर एक webhook URL सेट करते हैं
2. SimpleTuner S3 publishing settings को auto‑configure करता है
3. Trained models वापस आपकी मशीन पर अपलोड हो जाते हैं

**नोट:** cloud provider तक पहुँचने के लिए आपको अपना local SimpleTuner instance ngrok, cloudflared, या समान टूल से expose करना होगा।

## नए Providers जोड़ना

Cloud सिस्टम extensibility के लिए डिज़ाइन किया गया है। नया provider जोड़ने के लिए:

1. `CloudTrainerService` implement करने वाली नई client class बनाएँ:

```python
from .base import CloudTrainerService, CloudJobInfo, CloudJobStatus

class NewProviderClient(CloudTrainerService):
    @property
    def provider_name(self) -> str:
        return "new_provider"

    @property
    def supports_cost_tracking(self) -> bool:
        return True  # or False

    @property
    def supports_live_logs(self) -> bool:
        return True  # or False

    async def validate_credentials(self) -> Dict[str, Any]:
        # Validate API key and return user info
        ...

    async def list_jobs(self, limit: int = 50) -> List[CloudJobInfo]:
        # List recent jobs from the provider
        ...

    async def run_job(self, config, dataloader, ...) -> CloudJobInfo:
        # Submit a new training job
        ...

    async def cancel_job(self, job_id: str) -> bool:
        # Cancel a running job
        ...

    async def get_job_logs(self, job_id: str) -> str:
        # Fetch logs for a job
        ...

    async def get_job_status(self, job_id: str) -> CloudJobInfo:
        # Get current status of a job
        ...
```

2. cloud routes में provider register करें
3. नए provider tab के लिए UI elements जोड़ें

## Files & Locations

| Path | Description |
|------|-------------|
| `~/.simpletuner/cloud/` | Cloud‑related state और job history |
| `~/.simpletuner/cloud/job_history.json` | Unified job tracking database |
| `~/.simpletuner/cloud/provider_configs/` | Per‑provider configuration |
| `~/.simpletuner/cloud_outputs/` | Local S3 endpoint storage |

## Troubleshooting

### "REPLICATE_API_TOKEN not set"

SimpleTuner शुरू करने से पहले environment variable सेट करें:

```bash
export REPLICATE_API_TOKEN="r8_..."
simpletuner --webui
```

### Data upload fails

- अपना internet connection जांचें
- dataset paths मौजूद हैं या नहीं देखें
- browser console में errors देखें
- cloud provider पर पर्याप्त credits और permissions सुनिश्चित करें

### Webhook परिणाम नहीं प्राप्त कर रहा

- सुनिश्चित करें कि आपका local instance publicly accessible है
- webhook URL सही है या नहीं जांचें
- firewall rules incoming connections की अनुमति दें

## Current Limitations

Cloud training सिस्टम **single‑shot training jobs** के लिए डिज़ाइन किया गया है। निम्न फीचर्स फिलहाल समर्थित नहीं हैं:

### Workflow / Pipeline Jobs (DAGs)

SimpleTuner job dependencies या multi‑step workflows सपोर्ट नहीं करता जहाँ एक job का आउटपुट दूसरे में जाता है। हर job independent और self‑contained है।

**यदि आपको workflows चाहिए:**
- external orchestration tools (Airflow, Prefect, Dagster) उपयोग करें
- REST API के जरिए jobs chain करें
- Airflow integration उदाहरण के लिए [ENTERPRISE.md](../server/ENTERPRISE.md#external-orchestration-airflow) देखें

### Resuming Training Runs

interrupt, failure, या early stop के बाद training run resume करने का built‑in सपोर्ट नहीं है। यदि job fail या cancel हो जाए:
- आपको शुरुआत से फिर submit करना होगा
- cloud storage से automatic checkpoint recovery नहीं होती

**Workarounds:**
- intermediate checkpoints सेव करने के लिए frequent HuggingFace Hub pushes (`--push_checkpoints_to_hub`) कॉन्फ़िगर करें
- outputs डाउनलोड करके और नए job के starting point के रूप में re‑upload करके अपना checkpoint management लागू करें
- critical long‑running jobs के लिए छोटे training segments में विभाजित करने पर विचार करें

ये सीमाएँ भविष्य के रिलीज़ में संबोधित हो सकती हैं।

## See Also

### Cloud Training

- [Cloud Training Tutorial](TUTORIAL.md) - शुरू करने के लिए गाइड
- [Replicate Integration](REPLICATE.md) - Replicate provider setup
- [Job Queue](../../JOB_QUEUE.md) - Job scheduling और concurrency
- [Operations Guide](OPERATIONS_TUTORIAL.md) - Production deployment

### Multi-User Features (local और cloud दोनों पर लागू)

- [Enterprise Guide](../server/ENTERPRISE.md) - SSO, approvals, और governance
- [External Authentication](../server/EXTERNAL_AUTH.md) - OIDC और LDAP सेटअप
- [Audit Logging](../server/AUDIT.md) - Security event logging

### General

- [Local API Tutorial](../../api/TUTORIAL.md) - REST API के जरिए local training
- [Datasets Documentation](../../DATALOADER.md) - Dataloader configs समझना

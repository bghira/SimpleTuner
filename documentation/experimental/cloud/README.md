# Cloud Training System

> **Status:** Experimental
>
> **Available in:** Web UI (Cloud tab)

SimpleTuner's cloud training system enables you to run training jobs on cloud GPU providers without setting up your own infrastructure. The system is designed to be pluggable, allowing multiple providers to be added over time.

## Overview

The cloud training system provides:

- **Unified job tracking** - Track both local and cloud training jobs in one place
- **Automatic data packaging** - Local datasets are packaged and uploaded automatically
- **Results delivery** - Trained models can be sent to HuggingFace, S3, or downloaded locally
- **Cost tracking** - Monitor spending and job costs per provider with spending limits
- **Config snapshots** - Optionally version-control your training configs with git

## Key Concepts

Before using cloud training, understand these three things:

### 1. What Happens to Your Data

When you submit a cloud job:

1. **Datasets are packaged** - Local datasets (`type: "local"`) are zipped and you'll see a summary
2. **Uploaded to provider** - The zip goes directly to the cloud provider after consenting
3. **Training runs** - The model may have to download before your samples are trained on cloud GPUs
4. **Data is deleted** - After training, uploaded data is removed from the provider servers and your model is delivered

**Security notes:**
- Your API token never leaves your machine
- Sensitive files (.env, .git, credentials) are automatically excluded
- You review and consent to uploads before each job

### 2. How You Receive Trained Models

Training produces a model that needs somewhere to go. Configure one of:

| Destination | Setup | Best For |
|-------------|-------|----------|
| **HuggingFace Hub** | Set `HF_TOKEN` env var, enable in Publishing tab | Sharing models, easy access |
| **Local Download** | Set webhook URL, expose server via ngrok | Privacy, local workflows |
| **S3 Storage** | Configure endpoint in Publishing tab | Team access, archival |

See [Receiving Trained Models](TUTORIAL.md#receiving-trained-models) for step-by-step setup.

### 3. Cost Model

Replicate bills per second of GPU time:

| Hardware | VRAM | Cost | Typical LoRA (2000 steps) |
|----------|------|------|---------------------------|
| L40S | 48GB | ~$3.50/hr | $5-15 |

**Billing starts** when training begins and **stops** when it completes or fails.

**Protect yourself:**
- Set a spending limit in Cloud settings
- Cost estimates shown before every submission
- Cancel running jobs anytime (you pay for time used)

See [Costs](REPLICATE.md#costs) for pricing and limits.

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
| [Worker Orchestration](../server/WORKERS.md) | Supported | Self-hosted distributed workers, any GPU |
| SimpleTuner.io | Coming Soon | Managed training service by the SimpleTuner team |

### Worker Orchestration

For self-hosted distributed training across multiple machines, see the [Worker Orchestration Guide](../server/WORKERS.md). Workers can run on:

- On-premise GPU servers
- Cloud VMs (any provider)
- Spot instances (RunPod, Vast.ai, Lambda Labs)

Workers register with your SimpleTuner orchestrator and receive jobs automatically.

## Data Flow

### Submitting a Job

1. **Config Preparation** - Your training config is serialized
2. **Data Packaging** - Local datasets (those with `type: "local"`) are zipped
3. **Upload** - The zip is uploaded to Replicate's file hosting
4. **Submission** - Job is submitted to the cloud provider
5. **Tracking** - Job status is polled and updated in real-time

### Receiving Results

Results can be delivered via:

1. **HuggingFace Hub** - Push trained model to your HuggingFace account
2. **S3-Compatible Storage** - Upload to any S3 endpoint (AWS, MinIO, etc.)
3. **Local Download** - SimpleTuner includes a built-in S3-compatible endpoint that receives uploads locally

## Data Privacy & Consent

When you submit a cloud job, SimpleTuner may upload:

- **Training datasets** - Images/files from datasets with `type: "local"`
- **Configuration** - Your training parameters (learning rate, model settings, etc.)
- **Captions/metadata** - Any text files associated with your datasets

Data is uploaded directly to the cloud provider (e.g., Replicate's file hosting). It is not routed through SimpleTuner's servers.

### Consent Settings

In the Cloud tab, you can configure data upload behavior:

| Setting | Behavior |
|---------|----------|
| **Always Ask** | Show a confirmation dialog listing datasets before each upload |
| **Always Allow** | Skip confirmation for trusted workflows |
| **Never Upload** | Disable cloud training (local only) |

## Local S3 Endpoint

SimpleTuner includes a built-in S3-compatible endpoint for receiving trained models:

```
PUT /api/cloud/storage/{bucket}/{key}
GET /api/cloud/storage/{bucket}/{key}
GET /api/cloud/storage/{bucket}  (list objects)
GET /api/cloud/storage  (list buckets)
```

Files are stored to `~/.simpletuner/cloud_outputs/` by default.

You can configure the credentials manually; if you don't, ephemeral credentials are auto-generated for each training job, which is the recommended approach.

This allows "download only" mode where you:
1. Set a webhook URL pointing to your local SimpleTuner instance
2. SimpleTuner auto-configures the S3 publishing settings
3. Trained models are uploaded back to your machine

**Note:** You'll need to expose your local SimpleTuner instance via ngrok, cloudflared, or similar for the cloud provider to reach it.

## Adding New Providers

The cloud system is designed for extensibility. To add a new provider:

1. Create a new client class implementing `CloudTrainerService`:

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

2. Register the provider in the cloud routes
3. Add UI elements for the new provider tab

## Files & Locations

| Path | Description |
|------|-------------|
| `~/.simpletuner/cloud/` | Cloud-related state and job history |
| `~/.simpletuner/cloud/job_history.json` | Unified job tracking database |
| `~/.simpletuner/cloud/provider_configs/` | Per-provider configuration |
| `~/.simpletuner/cloud_outputs/` | Local S3 endpoint storage |

## Troubleshooting

### "REPLICATE_API_TOKEN not set"

Set the environment variable before starting SimpleTuner:

```bash
export REPLICATE_API_TOKEN="r8_..."
simpletuner --webui
```

### Data upload fails

- Check your internet connection
- Verify the dataset paths exist
- Look for errors in the browser console
- Ensure you have adequate credits and permissions on the cloud provider

### Webhook not receiving results

- Ensure your local instance is publicly accessible
- Check that the webhook URL is correct
- Verify firewall rules allow incoming connections

## Current Limitations

The cloud training system is designed for **single-shot training jobs**. The following features are not currently supported:

### Workflow / Pipeline Jobs (DAGs)

SimpleTuner does not support job dependencies or multi-step workflows where one job's output feeds into another. Each job is independent and self-contained.

**If you need workflows:**
- Use external orchestration tools (Airflow, Prefect, Dagster)
- Chain jobs via the REST API from your pipeline
- See [ENTERPRISE.md](../server/ENTERPRISE.md#external-orchestration-airflow) for an Airflow integration example

### Resuming Training Runs

There is no built-in support for resuming a training run that was interrupted, failed, or stopped early. If a job fails or is cancelled:
- You must resubmit from the beginning
- No automatic checkpoint recovery from cloud storage

**Workarounds:**
- Configure frequent HuggingFace Hub pushes (`--push_checkpoints_to_hub`) to save intermediate checkpoints
- Implement your own checkpoint management by downloading outputs and re-uploading as the starting point for a new job
- For critical long-running jobs, consider breaking into smaller training segments

These limitations may be addressed in future releases.

## See Also

### Cloud Training

- [Cloud Training Tutorial](TUTORIAL.md) - Getting started guide
- [Replicate Integration](REPLICATE.md) - Replicate provider setup
- [Job Queue](../../JOB_QUEUE.md) - Job scheduling and concurrency
- [Operations Guide](OPERATIONS_TUTORIAL.md) - Production deployment

### Multi-User Features (applies to local and cloud)

- [Enterprise Guide](../server/ENTERPRISE.md) - SSO, approvals, and governance
- [External Authentication](../server/EXTERNAL_AUTH.md) - OIDC and LDAP setup
- [Audit Logging](../server/AUDIT.md) - Security event logging

### General

- [Local API Tutorial](../../api/TUTORIAL.md) - Local training via REST API
- [Datasets Documentation](../../DATALOADER.md) - Understanding dataloader configs

# RunPod Template: LongCat Video Full Finetune

Pre-configured Docker image for full finetuning the **LongCat Video (13.6B)** model using SimpleTuner, optimized for 8x A100 80GB or 8x H200.

## Docker Image

```
danielxmed/longcat-video-finetune:latest
```

## Features

- **SimpleTuner** pre-installed and configured
- **FSDP2** for distributed training across multiple GPUs
- **AWS S3** support for large video datasets
- Automatic **Parquet** metadata generation for fast startup
- Configuration via **secrets** and environment variables
- Optional **auto-start** training mode
- Integrated TensorBoard monitoring

---

## Quick Start Guide

This guide walks you through the complete setup process step by step.

### Prerequisites

Before you begin, make sure you have:

- [ ] A RunPod account with GPU credits
- [ ] An AWS S3 bucket (or S3-compatible storage like Cloudflare R2)
- [ ] Your video dataset uploaded to S3 with caption files
- [ ] AWS credentials (Access Key ID and Secret Access Key)

---

## Step 1: Prepare Your Dataset

Your videos must be uploaded to an S3 bucket with matching caption files.

### Expected S3 Structure

```
s3://your-bucket-name/
├── video_001.mp4
├── video_001.txt    ← Caption for video_001.mp4
├── video_002.mp4
├── video_002.txt    ← Caption for video_002.mp4
├── subfolder/
│   ├── clip_a.mp4
│   ├── clip_a.txt
│   └── ...
└── ...
```

### Video Requirements

| Requirement | Details |
|-------------|---------|
| **Formats** | MP4, MOV, AVI, WebM |
| **Minimum duration** | ~3.1 seconds (93 frames @ 30fps) |
| **Caption file** | `.txt` file with the same name as the video |
| **Caption content** | Plain text description of the video |

### Example Caption File

For `video_001.mp4`, create `video_001.txt` containing:
```
A golden retriever running through a sunny meadow, slow motion, cinematic lighting
```

---

## Step 2: Create RunPod Secrets

Secrets keep your AWS credentials secure. You only need to set them up once.

### 2.1 Navigate to Secrets

1. Log in to [RunPod Console](https://www.runpod.io/console)
2. Click on **Settings** in the left sidebar
3. Select **Secrets**

### 2.2 Create Required Secrets

Click **Add Secret** for each of the following:

| Secret Name | Value | Example |
|-------------|-------|---------|
| `AWS_BUCKET_NAME` | Your S3 bucket name | `my-video-dataset` |
| `AWS_ACCESS_KEY_ID` | Your AWS access key | `AKIAIOSFODNN7EXAMPLE` |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key | `wJalrXUtnFEMI/K7MDENG/...` |

> **Security Note**: Never share your AWS credentials. RunPod Secrets are encrypted and secure.

---

## Step 3: Create the Template

### 3.1 Navigate to Templates

1. In RunPod Console, click **Templates** in the left sidebar
2. Click **New Template**

### 3.2 Configure Basic Settings

Fill in the following fields:

| Field | Value |
|-------|-------|
| **Template Name** | `LongCat Video Full Finetune` |
| **Container Image** | `danielxmed/longcat-video-finetune:latest` |
| **Container Disk** | `50 GB` (minimum) |
| **Volume Disk** | `500 GB` or more (for VAE cache and checkpoints) |
| **Volume Mount Path** | `/workspace` |

### 3.3 Configure Environment Variables

In the **Environment Variables** section, add:

#### Required Variables (Using Secrets)

```
AWS_BUCKET_NAME={{ RUNPOD_SECRET_AWS_BUCKET_NAME }}
AWS_ACCESS_KEY_ID={{ RUNPOD_SECRET_AWS_ACCESS_KEY_ID }}
AWS_SECRET_ACCESS_KEY={{ RUNPOD_SECRET_AWS_SECRET_ACCESS_KEY }}
```

#### Optional Variables

Add any of these to customize your training:

```
AWS_REGION=us-east-1
AUTO_START_TRAINING=false
USE_PARQUET=false
MAX_TRAIN_STEPS=30000
LEARNING_RATE=1e-5
```

See [Configuration Reference](#configuration-reference) for all options.

### 3.4 Configure Exposed Ports

| Label | Port | Protocol |
|-------|------|----------|
| TensorBoard | 6006 | HTTP |
| Jupyter | 8888 | HTTP |
| SSH | 22 | TCP |

### 3.5 Save Template

Click **Save Template**. You can now deploy pods using this template.

---

## Step 4: Deploy a Pod

### 4.1 Select GPU Configuration

For full finetuning, you need:

| Configuration | VRAM Required | Recommended |
|---------------|---------------|-------------|
| Full Finetune (bf16) | 8x 80GB | 8x A100 80GB or 8x H200 |
| Full Finetune (int8) | 8x 48GB | 8x A40 or 8x L40S |
| LoRA Training | 4x 24GB | 4x RTX 4090 or 4x L4 |

### 4.2 Launch Pod

1. Go to **GPU Cloud** → **Deploy**
2. Select your GPU configuration (e.g., 8x A100 80GB)
3. Choose your template: `LongCat Video Full Finetune`
4. Click **Deploy**

---

## Step 5: Start Training

### 5.1 Connect to Your Pod

Once the pod is running:

1. Click on the pod name in your RunPod dashboard
2. Choose one of:
   - **Web Terminal** - Browser-based terminal
   - **SSH** - For local terminal access
   - **Jupyter** - Opens JupyterLab interface

### 5.2 Verify Configuration

First, check that everything is set up correctly:

```bash
# View the generated training configuration
cat /workspace/config/config.json

# View the data backend configuration
cat /workspace/config/databackend.json

# Test S3 connection (should list your videos)
aws s3 ls s3://$AWS_BUCKET_NAME/ --human-readable | head -20
```

### 5.3 (Optional) Generate Parquet Metadata

For large datasets (500K+ videos), generating Parquet metadata dramatically speeds up startup:

```bash
/workspace/generate_parquet.sh
```

This may take several hours for very large datasets, but only needs to be done once.

### 5.4 Start Training

```bash
/workspace/start_training.sh
```

### 5.5 Monitor Training

In a separate terminal (or tmux session):

```bash
# Start TensorBoard and GPU monitoring
/workspace/monitor.sh
```

Access TensorBoard via the pod's HTTP port 6006.

---

## Step 6: Auto-Start Mode (Optional)

For subsequent training runs, you can enable auto-start:

1. Edit your template's environment variables
2. Change: `AUTO_START_TRAINING=true`
3. If using Parquet: `USE_PARQUET=true`

Now the pod will automatically begin training when it starts.

---

## Configuration Reference

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `AWS_BUCKET_NAME` | S3 bucket containing your videos |
| `AWS_ACCESS_KEY_ID` | AWS access key for S3 |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key for S3 |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | `us-east-1` | AWS region of your bucket |
| `AWS_DATA_PREFIX` | `""` | Subfolder/prefix in the bucket |
| `AWS_ENDPOINT_URL` | `null` | Custom endpoint (for R2, MinIO, etc.) |
| `USE_PARQUET` | `false` | Use pre-generated Parquet metadata |
| `AUTO_START_TRAINING` | `false` | Start training automatically on pod boot |
| `MODEL_TYPE` | `full` | Training type: `full` or `lora` |
| `LORA_RANK` | - | If set, enables LoRA training with this rank |
| `BASE_MODEL_PRECISION` | `bf16` | Model precision: `bf16`, `int8-quanto`, `fp8-torchao` |
| `LEARNING_RATE` | `1e-5` | Training learning rate |
| `MAX_TRAIN_STEPS` | `30000` | Maximum training steps |
| `TRAIN_BATCH_SIZE` | `1` | Batch size per GPU |
| `GRADIENT_ACCUMULATION_STEPS` | `4` | Gradient accumulation steps |
| `CHECKPOINTING_STEPS` | `2500` | Save checkpoint every N steps |
| `VALIDATION_EVERY_N_STEPS` | `2500` | Run validation every N steps |
| `NUM_GPUS` | `8` | Number of GPUs to use |

---

## Configuration Examples

### Example 1: Basic Full Finetune (8x A100 80GB)

```
AWS_BUCKET_NAME={{ RUNPOD_SECRET_AWS_BUCKET_NAME }}
AWS_ACCESS_KEY_ID={{ RUNPOD_SECRET_AWS_ACCESS_KEY_ID }}
AWS_SECRET_ACCESS_KEY={{ RUNPOD_SECRET_AWS_SECRET_ACCESS_KEY }}
AUTO_START_TRAINING=false
```

### Example 2: Full Finetune with Auto-Start

```
AWS_BUCKET_NAME={{ RUNPOD_SECRET_AWS_BUCKET_NAME }}
AWS_ACCESS_KEY_ID={{ RUNPOD_SECRET_AWS_ACCESS_KEY_ID }}
AWS_SECRET_ACCESS_KEY={{ RUNPOD_SECRET_AWS_SECRET_ACCESS_KEY }}
AUTO_START_TRAINING=true
USE_PARQUET=true
MAX_TRAIN_STEPS=30000
```

### Example 3: LoRA Training (Lower VRAM)

For proof-of-concept or limited VRAM:

```
AWS_BUCKET_NAME={{ RUNPOD_SECRET_AWS_BUCKET_NAME }}
AWS_ACCESS_KEY_ID={{ RUNPOD_SECRET_AWS_ACCESS_KEY_ID }}
AWS_SECRET_ACCESS_KEY={{ RUNPOD_SECRET_AWS_SECRET_ACCESS_KEY }}
LORA_RANK=64
BASE_MODEL_PRECISION=int8-quanto
MAX_TRAIN_STEPS=5000
```

### Example 4: Cloudflare R2 (Free Egress)

```
AWS_BUCKET_NAME={{ RUNPOD_SECRET_AWS_BUCKET_NAME }}
AWS_ACCESS_KEY_ID={{ RUNPOD_SECRET_AWS_ACCESS_KEY_ID }}
AWS_SECRET_ACCESS_KEY={{ RUNPOD_SECRET_AWS_SECRET_ACCESS_KEY }}
AWS_ENDPOINT_URL=https://ACCOUNT_ID.r2.cloudflarestorage.com
AWS_REGION=auto
```

---

## Directory Structure

After the pod starts, you'll find:

```
/workspace/
├── SimpleTuner/              # SimpleTuner source code
├── config/                   # Generated configurations
│   ├── config.json           # Training hyperparameters
│   ├── databackend.json      # Dataset configuration
│   └── metadata.parquet      # (if generated)
├── cache/
│   ├── vae/                  # VAE latent cache
│   └── text/                 # Text embedding cache
├── output/                   # Saved checkpoints
├── logs/                     # TensorBoard logs
├── scripts/                  # Utility scripts
├── generate_parquet.sh       # Generate Parquet metadata
├── start_training.sh         # Start training
└── monitor.sh                # Start monitoring tools
```

---

## Time Estimates

### 8x A100 80GB

| Phase | 100K videos | 500K videos |
|-------|-------------|-------------|
| VAE Cache | ~10 hours | ~50-70 hours |
| Training 10K steps | ~40 hours | ~40 hours |
| Training 30K steps | ~120 hours | ~120 hours |

### 8x H200

| Phase | 100K videos | 500K videos |
|-------|-------------|-------------|
| VAE Cache | ~6 hours | ~30-40 hours |
| Training 10K steps | ~25 hours | ~25 hours |
| Training 30K steps | ~75 hours | ~75 hours |

---

## Troubleshooting

### Out of Memory (OOM) Errors

**Solution 1**: Enable quantization
```
BASE_MODEL_PRECISION=int8-quanto
```

**Solution 2**: Reduce batch size
```
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
```

### Slow Startup with Large Datasets

**Solution**: Generate Parquet metadata once, then reuse:
```bash
# Run once
/workspace/generate_parquet.sh

# Then set in your template
USE_PARQUET=true
```

### Check Training Logs

```bash
# View SimpleTuner logs
tail -f /workspace/logs/*/events.out.tfevents.*

# Start TensorBoard manually
tensorboard --logdir /workspace/logs --port 6006 --bind_all
```

### S3 Connection Issues

```bash
# Test S3 connectivity
aws s3 ls s3://$AWS_BUCKET_NAME/

# Check credentials
echo "Bucket: $AWS_BUCKET_NAME"
echo "Region: $AWS_REGION"
echo "Access Key: ${AWS_ACCESS_KEY_ID:0:5}..."
```

---

## Local Development

For building and testing the Docker image locally:

```bash
cd docker/runpod-longcat

# Build the image
docker build -t longcat-video-finetune:latest .

# Test locally
docker run --rm -it \
    -e AWS_BUCKET_NAME=test \
    -e AWS_ACCESS_KEY_ID=test \
    -e AWS_SECRET_ACCESS_KEY=test \
    longcat-video-finetune:latest /bin/bash
```

---

## Links

- [SimpleTuner Documentation](https://github.com/bghira/SimpleTuner)
- [RunPod Documentation](https://docs.runpod.io)
- [LongCat Video Model](https://huggingface.co/LongCat-Video)

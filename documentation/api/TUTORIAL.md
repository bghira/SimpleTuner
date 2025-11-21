# API Training Tutorial

## Introduction

This guide walks through running SimpleTuner training jobs **entirely through the HTTP API** while keeping setup and dataset management on the command line. It mirrors the structure of the other tutorials but skips the WebUI onboarding. You will:

- install and start the unified server
- discover and download the OpenAPI schema
- create and update environments with REST calls
- validate, launch, and monitor training jobs via `/api/training`
- branch into two proven configurations: a PixArt Sigma 900M full fine-tune and a Flux Kontext LyCORIS LoRA run

## Prerequisites

- Python 3.10–3.12, Git, and `pip`
- SimpleTuner installed in a virtual environment (`pip install 'simpletuner[cuda]'` or the variant that matches your platform)
- Access to required Hugging Face repos (`huggingface-cli login` before pulling gated models)
- Datasets staged locally with captions (caption text files for PixArt, paired edit/reference folders for Kontext)
- A shell with `curl` and `jq`

## Start the server

From your SimpleTuner checkout (or the environment where the package is installed):

```bash
simpletuner server --port 8001
```

The API lives at `http://localhost:8001`. Leave the server running while you issue the following commands in another terminal.

## Discover the API

FastAPI serves interactive docs and the OpenAPI schema:

```bash
# FastAPI Swagger UI
python -m webbrowser http://localhost:8001/docs

# ReDoc view
python -m webbrowser http://localhost:8001/redoc

# Download the schema for local inspection
curl -o openapi.json http://localhost:8001/openapi.json
jq '.info' openapi.json
```

Every endpoint used in this tutorial is documented there under the `configurations` and `training` tags.

## Fast path: run without environments

If you prefer to **skip config/environment management entirely**, you can issue a one-off training run by posting the full CLI-style payload straight to the training endpoints:

1. Author or reuse a dataloader JSON that describes your dataset. The trainer only needs the path referenced by `--data_backend_config`.

   ```bash
   cat <<'JSON' > config/multidatabackend-once.json
   [
     {
       "id": "demo-images",
       "type": "local",
       "dataset_type": "image",
       "instance_data_dir": "/data/datasets/demo",
       "caption_strategy": "textfile",
       "resolution": 1024,
       "resolution_type": "pixel_area"
     },
     {
       "id": "demo-text-embeds",
       "type": "local",
       "dataset_type": "text_embeds",
       "default": true,
       "cache_dir": "/data/cache/text/demo"
     }
   ]
   JSON
   ```

2. Validate the inline config. Provide every required CLI argument (`--model_family`, `--model_type`, `--pretrained_model_name_or_path`, `--output_dir`, `--data_backend_config`, and either `--num_train_epochs` or `--max_train_steps`):

   ```bash
   curl -s -X POST http://localhost:8001/api/training/validate \
     -F __active_tab__=model \
     -F --model_family=pixart_sigma \
     -F --model_type=full \
     -F --model_flavour=900M-1024-v0.6 \
     -F --pretrained_model_name_or_path=terminusresearch/pixart-900m-1024-ft-v0.6 \
     -F --output_dir=/workspace/output/inline-demo \
     -F --data_backend_config=config/multidatabackend-once.json \
     -F --train_batch_size=1 \
     -F --learning_rate=0.0001 \
     -F --max_train_steps=200 \
     -F --num_train_epochs=0
   ```

   A green “Configuration Valid” snippet confirms the trainer will accept the payload.

3. Launch training with the **same** form fields (you can add overrides such as `--seed` or `--validation_prompt`):

   ```bash
   curl -s -X POST http://localhost:8001/api/training/start \
     -F __active_tab__=model \
     -F --model_family=pixart_sigma \
     -F --model_type=full \
     -F --model_flavour=900M-1024-v0.6 \
     -F --pretrained_model_name_or_path=terminusresearch/pixart-900m-1024-ft-v0.6 \
     -F --output_dir=/workspace/output/inline-demo \
     -F --data_backend_config=config/multidatabackend-once.json \
     -F --train_batch_size=1 \
     -F --learning_rate=0.0001 \
     -F --max_train_steps=200 \
     -F --num_train_epochs=0 \
     -F --validation_prompt='test shot of <token>'
   ```

The server automatically merges the submitted settings with its defaults, writes the resolved config into the active file, and begins training. You can reuse the same approach for any model family—the remaining sections cover a fuller workflow when you want reusable environments.

### Monitoring ad-hoc runs

You can track progress through the same status endpoints used later in the guide:

- Poll `GET /api/training/status` for high-level state, active job ID, and startup stage info.
- Fetch incremental logs with `GET /api/training/events?since_index=N` or stream them via the WebSocket at `/api/training/events/stream`.

For push-style updates, supply webhook settings alongside your form fields:

```bash
curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=model \
  -F --model_family=pixart_sigma \
  ... \
  -F --webhook_config='[{"webhook_type":"raw","callback_url":"https://example.com/simpletuner","log_level":"info","ssl_no_verify":false}]' \
  -F --webhook_reporting_interval=10
```

The payload must be JSON serialised as a string; the server posts job lifecycle updates to the `callback_url`. See the `--webhook_config` description in `documentation/OPTIONS.md` or the sample `config/webhooks.json` template for supported fields.

### Trigger manual validation

If you want to force an evaluation pass **between** scheduled validation intervals, call the new endpoint:

```bash
curl -s -X POST http://localhost:8001/api/training/validation/run
```

- The server responds with the active `job_id`.
- The trainer queues a validation run that fires immediately after the next gradient synchronization (it does not interrupt the current micro-batch).
- The run reuses your configured validation prompts/settings so the resulting images appear in the usual event/log streams.
- To offload validation to an external executable instead of the built-in pipeline, set `--validation_method=external-script` in your config (or payload) and point `--validation_external_script` at your script. You can pass training context to the script with placeholders: `{local_checkpoint_path}`, `{global_step}`, `{tracker_run_name}`, `{tracker_project_name}`, `{model_family}`, `{huggingface_path}`, `{remote_checkpoint_path}` (empty for validation), plus any `validation_*` config values (e.g., `validation_num_inference_steps`, `validation_guidance`, `validation_noise_scheduler`). Enable `--validation_external_background` if you want the script to fire-and-forget without blocking training.
- Prefer to keep SimpleTuner's built-in uploads and hand the resulting remote URL to your own tool? Configure `--post_upload_script` instead; it fires once per publishing provider/Hugging Face Hub upload with `{remote_checkpoint_path}` (if provided by the backend) and the same context placeholders. SimpleTuner does not ingest results from your script, so log artifacts/metrics to your tracker yourself.
  - Example: `--post_upload_script='/opt/hooks/notify.sh {remote_checkpoint_path} {tracker_project_name} {tracker_run_name}'` where `notify.sh` calls your tracker API.
  - Working samples:
    - `simpletuner/examples/external-validation/replicate_post_upload.py` triggers a Replicate inference using `{remote_checkpoint_path}`, `{model_family}`, `{model_type}`, `{lora_type}`, and `{huggingface_path}`.
    - `simpletuner/examples/external-validation/wavespeed_post_upload.py` triggers a WaveSpeed inference and polls for completion using the same placeholders.

If no job is active the endpoint returns HTTP 400, so check `/api/training/status` first when scripting retries.

### Trigger manual checkpoint

To persist the current model state immediately (without waiting for the next scheduled checkpoint), hit:

```bash
curl -s -X POST http://localhost:8001/api/training/checkpoint/run
```

- The server responds with the active `job_id`.
- The trainer saves a checkpoint after the next gradient synchronization using the same settings as scheduled checkpoints (upload rules, rolling retention, etc.).
- Rolling cleanup and webhook notifications behave exactly like a scheduled checkpoint.

As with validation, the endpoint returns HTTP 400 if no training job is running.

### Stream validation previews

Models that expose Tiny AutoEncoder (or equivalent) hooks can emit **per-step validation previews** while an image/video is still being sampled. Enable the feature by adding the CLI flags to your payload:

```bash
curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=validation \
  -F --validation_preview=true \
  -F --validation_preview_steps=4 \
  -F --validation_num_inference_steps=20 \
  …other fields…
```

- `--validation_preview` (defaults to `false`) unlocks the preview decoder.
- `--validation_preview_steps` determines how often to emit intermediate frames. With the example above, you receive events at steps 1,5,9,13,17,20 (the first step is always emitted, then every 4th step).

Each preview is published as a `validation.image` event (see `simpletuner/helpers/training/validation.py:899-929`). You can consume them via raw webhooks, `GET /api/training/events`, or the SSE stream at `/api/training/events/stream`. A typical payload looks like:

```json
{
  "type": "validation.image",
  "title": "Validation (step 5/20): night bench",
  "body": "night bench shot of <token>",
  "data": {
    "step": 5,
    "timestep": 563.0,
    "resolution": [1024, 1024],
    "validation_type": "intermediary",
    "prompt": "night bench shot of <token>",
    "step_label": "5/20"
  },
  "images": [
    {"src": "data:image/png;base64,...", "mime_type": "image/png"}
  ]
}
```

Video-capable models attach a `videos` array instead (GIF data URIs with `mime_type: image/gif`). Because these events stream in near-real-time, you can surface them directly in dashboards or send them to Slack/Discord via a raw webhook backend.

## Common API workflow

1. **Create an environment** – `POST /api/configs/environments`
2. **Populate the dataloader file** – edit the generated `config/<env>/multidatabackend.json`
3. **Update training hyperparameters** – `PUT /api/configs/<env>`
4. **Activate the environment** – `POST /api/configs/<env>/activate`
5. **Validate training parameters** – `POST /api/training/validate`
6. **Launch training** – `POST /api/training/start`
7. **Monitor or stop the job** – `/api/training/status`, `/api/training/events`, `/api/training/stop`, `/api/training/cancel`

Each example below follows this flow.

## Example: PixArt Sigma 900M full fine-tune

### 1. Create the environment via REST

```bash
curl -s -X POST http://localhost:8001/api/configs/environments \
  -H 'Content-Type: application/json' \
  -d '{
        "name": "pixart-api-demo",
        "model_family": "pixart_sigma",
        "model_flavour": "900M-1024-v0.6",
        "model_type": "full",
        "description": "PixArt 900M API-driven training"
      }'
```

This creates `config/pixart-api-demo/` and a starter `multidatabackend.json`.

### 2. Wire the dataset

Edit the dataloader file (replace paths with your actual dataset/cache locations):

```bash
cat <<'JSON' > config/pixart-api-demo/multidatabackend.json
[
  {
    "id": "pixart-camera",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "/data/datasets/pseudo-camera-10k",
    "caption_strategy": "filename",
    "resolution": 1.0,
    "resolution_type": "area",
    "minimum_image_size": 0.25,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "cache_dir_vae": "/data/cache/vae/pixart/pseudo-camera-10k",
    "crop": true,
    "crop_style": "random",
    "crop_aspect": "square",
    "metadata_backend": "discovery"
  },
  {
    "id": "pixart-text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "/data/cache/text/pixart/pseudo-camera-10k",
    "write_batch_size": 128
  }
]
JSON
```

### 3. Update hyperparameters through the API

Grab the current config, merge overrides, and push the result back:

```bash
curl -s http://localhost:8001/api/configs/pixart-api-demo \
  | jq '.config + {
      "--output_dir": "/workspace/output/pixart900m",
      "--train_batch_size": 2,
      "--gradient_accumulation_steps": 2,
      "--learning_rate": 0.0001,
      "--optimizer": "adamw_bf16",
      "--lr_scheduler": "cosine",
      "--lr_warmup_steps": 500,
      "--max_train_steps": 1800,
      "--num_train_epochs": 0,
      "--validation_prompt": "a studio portrait of <token> wearing a leather jacket",
      "--validation_guidance": 3.8,
      "--validation_resolution": "1024x1024",
      "--validation_num_inference_steps": 28,
      "--cache_dir_vae": "/data/cache/vae/pixart",
      "--seed": 1337,
      "--resume_from_checkpoint": "latest",
      "--base_model_precision": "bf16",
      "--dataloader_prefetch": true,
      "--report_to": "none",
      "--checkpoints_total_limit": 4,
      "--validation_seed": 12345,
      "--data_backend_config": "pixart-api-demo/multidatabackend.json"
    }' > /tmp/pixart-config.json

jq '{
      "name": "pixart-api-demo",
      "description": "PixArt 900M full tune (API)",
      "tags": ["pixart", "api"],
      "config": .
    }' /tmp/pixart-config.json > /tmp/pixart-update.json

curl -s -X PUT http://localhost:8001/api/configs/pixart-api-demo \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/pixart-update.json
```

### 4. Activate the environment

```bash
curl -s -X POST http://localhost:8001/api/configs/pixart-api-demo/activate
```

### 5. Validate before launching

`validate` consumes form-encoded data. At minimum, ensure one of `num_train_epochs` or `max_train_steps` is 0:

```bash
curl -s -X POST http://localhost:8001/api/training/validate \
  -F __active_tab__=model \
  -F --num_train_epochs=0
```

A success block (`Configuration Valid`) means the trainer accepts the merged configuration.

### 6. Start training

```bash
curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=model \
  -F --num_train_epochs=0
```

The response includes the job ID. Training runs with the parameters saved in step 3.

### 7. Monitor and stop

```bash
# Query coarse status
curl -s http://localhost:8001/api/training/status | jq

# Stream incremental log events
curl -s 'http://localhost:8001/api/training/events?since_index=0' | jq

# Cancel or stop
curl -s -X POST http://localhost:8001/api/training/stop
curl -s -X POST http://localhost:8001/api/training/cancel -F job_id=<JOB_ID>
```

PixArt notes:

- Keep the dataset large enough for the chosen `train_batch_size * gradient_accumulation_steps`
- Set `HF_ENDPOINT` if you need a mirror, and authenticate before downloading `terminusresearch/pixart-900m-1024-ft-v0.6`
- Tune `--validation_guidance` between 3.6 and 4.4 depending on your prompts

## Example: Flux Kontext LyCORIS LoRA

Kontext shares most of its pipeline with Flux Dev but needs paired edit/reference images.

### 1. Provision the environment

```bash
curl -s -X POST http://localhost:8001/api/configs/environments \
  -H 'Content-Type: application/json' \
  -d '{
        "name": "kontext-api-demo",
        "model_family": "flux",
        "model_flavour": "kontext",
        "model_type": "lora",
        "lora_type": "lycoris",
        "description": "Flux Kontext LoRA via API"
      }'
```

### 2. Describe the paired dataloader

Kontext needs edit/reference pairs plus a text-embed cache:

```bash
cat <<'JSON' > config/kontext-api-demo/multidatabackend.json
[
  {
    "id": "kontext-edit",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "/data/datasets/kontext/edit",
    "conditioning_data": ["kontext-reference"],
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "caption_strategy": "textfile",
    "minimum_image_size": 768,
    "maximum_image_size": 1536,
    "target_downsample_size": 1024,
    "cache_dir_vae": "/data/cache/vae/kontext/edit",
    "crop": true,
    "crop_style": "random",
    "crop_aspect": "square"
  },
  {
    "id": "kontext-reference",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/data/datasets/kontext/reference",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "/data/cache/vae/kontext/reference"
  },
  {
    "id": "kontext-text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "/data/cache/text/kontext"
  }
]
JSON
```

Ensure filenames match between edit and reference folders; SimpleTuner stitches embeddings based on names.

### 3. Apply Kontext-specific hyperparameters

```bash
curl -s http://localhost:8001/api/configs/kontext-api-demo \
  | jq '.config + {
      "--output_dir": "/workspace/output/kontext",
      "--train_batch_size": 1,
      "--gradient_accumulation_steps": 4,
      "--learning_rate": 0.00001,
      "--optimizer": "optimi-lion",
      "--lr_scheduler": "cosine",
      "--lr_warmup_steps": 200,
      "--max_train_steps": 12000,
      "--num_train_epochs": 0,
      "--validation_prompt": "a cinematic 1024px product photo of <token>",
      "--validation_guidance": 2.5,
      "--validation_resolution": "1024x1024",
      "--validation_num_inference_steps": 30,
      "--cache_dir_vae": "/data/cache/vae/kontext",
      "--seed": 777,
      "--resume_from_checkpoint": "latest",
      "--base_model_precision": "int8-quanto",
      "--dataloader_prefetch": true,
      "--report_to": "wandb",
      "--lora_rank": 16,
      "--lora_dropout": 0.05,
      "--conditioning_multidataset_sampling": "combined",
      "--clip_skip": 2,
      "--data_backend_config": "kontext-api-demo/multidatabackend.json"
    }' > /tmp/kontext-config.json

jq '{
      "name": "kontext-api-demo",
      "description": "Flux Kontext LyCORIS (API)",
      "tags": ["flux", "kontext", "api"],
      "config": .
    }' /tmp/kontext-config.json > /tmp/kontext-update.json

curl -s -X PUT http://localhost:8001/api/configs/kontext-api-demo \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/kontext-update.json
```

### 4. Activate, validate, and launch

```bash
curl -s -X POST http://localhost:8001/api/configs/kontext-api-demo/activate

curl -s -X POST http://localhost:8001/api/training/validate \
  -F __active_tab__=model \
  -F --num_train_epochs=0

curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=model \
  -F --num_train_epochs=0
```

Kontext tips:

- `conditioning_type=reference_strict` keeps crops aligned; switch to `reference_loose` if your datasets differ in aspect ratio
- Quantise to `int8-quanto` to stay within 24 GB VRAM at 1024 px; full precision requires Hopper/Blackwell-class GPUs
- For multi-node runs, set `--accelerate_config` or `CUDA_VISIBLE_DEVICES` before launching the server

## Useful endpoints at a glance

- `GET /api/configs/` – list environments (pass `?config_type=model` for training configs)
- `GET /api/configs/examples` – enumerate shipped templates
- `POST /api/configs/{name}/dataloader` – regenerate a dataloader file if you want defaults
- `GET /api/training/status` – high-level state, active `job_id`, and startup stage info
- `GET /api/training/events?since_index=N` – incremental trainer log stream
- `POST /api/training/checkpoints` – list checkpoints for the active job’s output directory

## Where to go next

- Explore specific option definitions in `documentation/OPTIONS.md`
- Combine these REST calls with `jq`/`yq` or a Python client for automation
- Hook WebSockets at `/api/training/events/stream` for real-time dashboards
- Reuse the exported configs (`GET /api/configs/<env>/export`) to version-control working setups

With these patterns you can fully script SimpleTuner training without touching the WebUI, while still relying on the battle-tested CLI setup process.

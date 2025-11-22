# External validation and post-upload hooks

This directory contains working samples that demonstrate how to run external scripts/functions from SimpleTuner:

- `replicate_post_upload.py` – triggers a Replicate inference after uploads, using placeholders like `{remote_checkpoint_path}`, `{model_family}`, `{model_type}`, `{lora_type}`, and `{huggingface_path}`.
- `wavespeed_post_upload.py` – triggers a WaveSpeed inference after uploads and polls for completion using the same placeholders.
- `fal_post_upload.py` – triggers a fal.ai Flux LoRA inference after uploads (requires `FAL_KEY`; only runs when `model_family` contains `flux`).
- `use_second_gpu.py` – runs a Flux LoRA on a secondary GPU (defaults to `cuda:1`) using local or Hub LoRA weights; can be used even when no uploads occur.

Both scripts are meant to be used with `--post_upload_script`:

```bash
--post_upload_script="simpletuner/examples/external-validation/replicate_post_upload.py \
  --remote {remote_checkpoint_path} \
  --model_family {model_family} \
  --model_type {model_type} \
  --lora_type {lora_type} \
  --hub_model_id {huggingface_path}"
```

Placeholders available to hook scripts (resolved from the trainer config/state):

- `{local_checkpoint_path}` – last local checkpoint directory (if present)
- `{remote_checkpoint_path}` – URI returned by the publishing provider (if provided)
- `{global_step}` – current global step
- `{tracker_run_name}`, `{tracker_project_name}`
- `{model_family}`, `{model_type}`, `{lora_type}`
- `{huggingface_path}` – value of `--hub_model_id`
- Any `validation_*` config value (e.g., `validation_num_inference_steps`, `validation_guidance`, `validation_noise_scheduler`)

Notes:

- Hooks run asynchronously; failures are logged but do not block training.
- Hooks are invoked even when no remote upload occurs so you can use them for local automation (for example, spinning up inference on another GPU).
- SimpleTuner does not consume outputs from these scripts. If you need tracker updates or artifact logging, emit them directly from your hook.

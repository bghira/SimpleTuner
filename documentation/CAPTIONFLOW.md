# CaptionFlow integration

SimpleTuner can use [CaptionFlow](https://github.com/bghira/CaptionFlow) to caption image datasets from the Web UI. CaptionFlow is a scalable vLLM-powered captioning system with an orchestrator, GPU workers, checkpointed storage, and YAML-driven configuration. In SimpleTuner it is exposed as the **Captioning** sub-tab on the Datasets page, so caption jobs can use the same local GPU queue as training and cache jobs.

Use this integration when you want to generate or refresh captions before training without leaving the SimpleTuner workflow.

## Installation

CaptionFlow is optional. Install the captioning target in the same virtual environment used by SimpleTuner:

```bash
pip install "simpletuner[captioning]"
```

For CUDA 13 environments, use the CUDA 13 target shown by the Web UI install prompt. It includes the vLLM wheel expected by that runtime.

## What SimpleTuner manages

When you start a Captioning job, SimpleTuner:

- maps the selected SimpleTuner dataset to a CaptionFlow processor;
- starts a local CaptionFlow orchestrator on `127.0.0.1`;
- starts one or more local GPU workers through the job queue;
- captures orchestrator and worker logs in the CaptionFlow job workspace;
- gracefully checkpoints CaptionFlow storage before export;
- writes `.txt` captions back to the local dataset directory for local datasets;
- writes JSONL exports into the CaptionFlow job workspace for Hugging Face datasets.

CaptionFlow dependencies are not required for the tab to appear. If they are missing, the tab shows the install command instead of the job builder.

## Builder mode

The **Builder** view covers the common single-stage captioning workflow:

- dataset selection from the active dataloader configuration;
- model, prompt, sampling, batch size, chunk size, and GPU memory settings;
- worker count and queue behavior;
- text file export for local datasets.

The default model is `Qwen/Qwen2.5-VL-3B-Instruct`. Local datasets export text files next to the images, using the output field selected in the form. Hugging Face datasets do not write back to the remote dataset; they export JSONL into the CaptionFlow workspace.

## Raw config mode

Use **Raw Config** when you need CaptionFlow features that the builder does not model, such as multi-stage captioning, per-stage models, per-stage sampling, or prompt chains where one stage consumes another stage's output.

Raw config accepts YAML or JSON. You can paste either the full CaptionFlow config with an `orchestrator:` root or just the orchestrator object.

SimpleTuner intentionally overrides these fields at runtime:

- `host`, `port`, and `ssl`;
- `dataset`, based on the selected SimpleTuner dataset;
- `storage.data_dir` and `storage.checkpoint_dir`, under the job workspace;
- `auth.worker_tokens` and `auth.admin_tokens`.

Other orchestrator settings, including `chunk_size`, `chunks_per_request`, `storage.caption_buffer_size`, `vllm.sampling`, `vllm.inference_prompts`, and `vllm.stages`, are preserved unless SimpleTuner needs a default.

## Multi-stage example

This raw config runs a detailed caption stage, then passes `{caption}` into a shortening stage. The selected SimpleTuner dataset, storage paths, ports, and auth tokens are filled in by SimpleTuner at job launch.

```yaml
orchestrator:
  chunk_size: 1000
  chunks_per_request: 1
  chunk_buffer_multiplier: 2
  min_chunk_buffer: 10

  vllm:
    model: "Qwen/Qwen2.5-VL-3B-Instruct"
    tensor_parallel_size: 1
    max_model_len: 16384
    dtype: "float16"
    gpu_memory_utilization: 0.92
    enforce_eager: true
    disable_mm_preprocessor_cache: true
    limit_mm_per_prompt:
      image: 1
    batch_size: 8
    sampling:
      temperature: 0.7
      top_p: 0.95
      max_tokens: 256
      repetition_penalty: 1.05
      skip_special_tokens: true
      stop:
        - "<|end|>"
        - "<|endoftext|>"
        - "<|im_end|>"
    stages:
      - name: "base_caption"
        model: "Qwen/Qwen2.5-VL-3B-Instruct"
        prompts:
          - "describe this image in detail"
        output_field: "caption"

      - name: "caption_shortening"
        model: "Qwen/Qwen2.5-VL-7B-Instruct"
        prompts:
          - "Please condense this elaborate caption to only the important details: {caption}"
        output_field: "captions"
        requires: ["base_caption"]
        gpu_memory_utilization: 0.35
```

## External CaptionFlow docs

For full standalone CaptionFlow usage and additional config examples, see:

- [CaptionFlow repository](https://github.com/bghira/CaptionFlow)
- [CaptionFlow README](https://github.com/bghira/CaptionFlow#readme)
- [CaptionFlow orchestrator examples](https://github.com/bghira/CaptionFlow/tree/main/examples/orchestrator)

Use the upstream examples as the source of truth for advanced CaptionFlow fields. When running through SimpleTuner, remember that SimpleTuner owns dataset routing, local ports, storage workspace paths, and auth tokens.

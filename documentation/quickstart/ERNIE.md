# ERNIE-Image [base / turbo] Quickstart

In this example, we'll be training an ERNIE-Image LoRA. ERNIE-Image is Baidu's single-stream flow-matching transformer family and uses the same Flux2-style VAE class inside diffusers. SimpleTuner supports both `base` and `turbo` flavours.

## Hardware requirements

ERNIE is not a small model. Plan around the same general class of hardware you would reserve for other large single-stream transformers:

- a realistic target is a 24G+ GPU when using int8 quantisation plus bf16 LoRA weights
- 16G can work with aggressive offload, RamTorch, and slow iteration speed
- multi-GPU, FSDP2, and additional CPU/RAM offload all help if you want to avoid a single large GPU

Apple GPUs are not recommended for training.

### Memory offloading (optional)

RamTorch is already a good default for ERNIE because its text encoder is large. If you still need more VRAM headroom, grouped module offloading is also available:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream
```

- Streams are only effective on CUDA.
- Do not combine multiple unrelated CPU offload systems unless you know why.
- Group offload is not compatible with Quanto quantisation.

## Prerequisites

SimpleTuner works well with Python 3.10 through 3.13.

```bash
python --version
```

If needed on Ubuntu:

```bash
apt -y install python3.13 python3.13-venv
```

### Container image dependencies

For CUDA 12.x images:

```bash
apt -y install nvidia-cuda-toolkit-12-8
```

## Installation

Install SimpleTuner via pip:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

For manual installation or development setup, see the [installation documentation](../INSTALL.md).

### AMD ROCm follow-up steps

For AMD MI300X:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

## Setting up the environment

### Web interface method

Run the server:

```bash
simpletuner server
```

Then open http://localhost:8001 and select the ERNIE model family in the training wizard.

### Manual / command-line method

You can start from the included example:

- example config: `simpletuner/examples/ernie.peft-lora/config.json`
- runnable local env: `config/ernie-example/config.json`

If you prefer to wire it manually, copy `config/config.json.example` to `config/config.json` and change the important values below.

#### Configuration file

```bash
cp config/config.json.example config/config.json
```

Recommended settings:

- `model_type`: `lora`
- `model_family`: `ernie`
- `model_flavour`: `base` or `turbo`
- `pretrained_model_name_or_path`:
  - `base`: `baidu/ERNIE-Image`
  - `turbo`: `baidu/ERNIE-Image-Turbo`
- `output_dir`: where checkpoints and validation images should be written
- `train_batch_size`: start at `1`
- `resolution`: start at `512`
- `mixed_precision`: `bf16` on modern hardware, `fp16` otherwise
- `gradient_checkpointing`: `true`
- `ramtorch`: `true`
- `ramtorch_text_encoder`: `true`

The included example uses:

- `max_train_steps: 100`
- `optimizer: optimi-lion`
- `learning_rate: 1e-4`
- `validation_guidance: 4.0`
- `validation_num_inference_steps: 20`

That exact example can be run with:

```bash
simpletuner train --env ernie-example
```

### Assistant LoRA (Turbo)

SimpleTuner exposes assistant-LoRA support for ERNIE Turbo, but there is no default adapter path bundled for it yet.

- supported flavour: `turbo`
- default weight filename: `pytorch_lora_weights.safetensors`
- required user input: `assistant_lora_path`

If you have a custom assistant adapter, set:

```json
{
  "assistant_lora_path": "your-org/your-ernie-turbo-assistant-lora",
  "assistant_lora_weight_name": "pytorch_lora_weights.safetensors"
}
```

If you do not want to use one, disable it explicitly:

```json
{
  "disable_assistant_lora": true
}
```

### Dataset / caption setup

The example env uses a tiny DreamBooth-style Hugging Face dataset:

- `dataset_name`: `RareConcepts/Domokun`
- `caption_strategy`: `instanceprompt`
- `instance_prompt`: `🟫`

That works as a smoke test, but ERNIE responds better to real text than a one-token trigger. For actual training, prefer richer captions or a more descriptive instance prompt such as `a studio photo of <token>`.

### Validation prompts

The same prompt library workflow used by other models works here:

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

Then point to it from `config.json`:

```json
{
  "--user_prompt_library": "config/user_prompt_library.json"
}
```

### Experimental features

ERNIE also supports the same advanced transformer-side features used by other single-stream families in SimpleTuner:

- TREAD
- LayerSync
- REPA / CREPA-style hidden state capture
- assistant LoRA loading for turbo

These features are optional. Get the base training run working first.

## Notes

- ERNIE uses a patched tokenizer/text-encoder loader because the upstream text encoder config needs minor fixes during load.
- The trainer uses the ERNIE timestep convention expected by the upstream model, so keep custom experimentation aligned with the normal flow-matching schedule unless you are intentionally probing edge cases.
- Start with the provided 512px example before scaling up dataset size or resolution.

## Example configurations

These configurations are provided as an easy way to **immediately** run a training session with SimpleTuner across a large number of architectures.

The options are set up so that a 24G card (NVIDIA 4090) can run training out of the box. In order to do this, compromises were made for resolution, training batch size, or LoRA rank.

It's recommended to use these only as a basic starting point.

### Running an example

All examples can be easily run without modifying the configurations.

We'll assume you don't have any python dependencies installed yet, and that an NVIDIA device is in use.

To run `kontext.peft-lora` example:

**Option 1 (Recommended - pip install):**
```bash
pip install 'simpletuner[cuda]'
# or for CUDA 13 / Blackwell users: pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
simpletuner train example=kontext.peft-lora
```

**Option 2 (Git clone method):**
```bash
git clone https://github.com/bghira/simpletuner
cd simpletuner
python3.13 -m venv .venv
. .venv/bin/activate
pip install -e .
simpletuner train env=examples/kontext.peft-lora
```

**Option 3 (Legacy method - still works):**
```bash
git clone https://github.com/bghira/simpletuner
cd simpletuner
python3.13 -m venv .venv
. .venv/bin/activate
pip install -e .
ENV=examples/kontext.peft-lora ./train.sh
```

This will automatically download an example reference dataset, pre-cache embeds, and run 100 steps of training on a standard PEFT LoRA.

ACE-Step examples are split by model generation:

- `ace_step-v1-0.peft-lora` for the original ACE-Step v1 3.5B path
- `ace_step-v1-5.peft-lora` for the forward-compatible ACE-Step v1.5 LoRA path

LTX-2 conditioning examples are split by conditioning style:

- `ltxvideo2-19b-t2v.peft-lora+first-frame-conditioning` shows the shorthand `ltx2_*` probability fields.
- `ltxvideo2-19b-t2v.peft-lora+intrinsic-conditioning` shows the explicit `ltx2_intrinsic_conditioning` object list.
- `ltxvideo2-19b-t2v.peft-lora+reference-conditioning` shows IC-LoRA reference conditioning with coordinate scale overrides.

Z-Image conditioning examples:

- `z-image-turbo.peft-lora+canny-conditioning` auto-generates Canny edge conditioning data and validates with those references through the IC-LoRA conditioning path.

Cosmos3 examples:

- `cosmos3-image.lycoris-lokr` uses `RareConcepts/Domokun`.
- `cosmos3-video.lycoris-lokr` uses `sayakpaul/video-dataset-disney-organized`.
- `cosmos3-video-audio.lycoris-lokr` uses local synchronized drumming files with `audio.auto_split`.
- `cosmos3-super-i2v.lycoris-lokr` uses `nvidia/Cosmos3-Super-Image2Video` with `video.is_i2v`.

Large multi-GPU video examples are split from the standard 24G examples:

- `wan2.1-t2v-14b-480p-8xh100.peft-lora+cp-fa3`
- `wan2.1-i2v-14b-480p-8xh100.peft-lora+cp-fa3`
- `wan2.1-i2v-14b-720p-8xh100.peft-lora+cp-fa3`
- `ltxvideo2-2.3-dev-720p-8xh100.peft-lora+cp-fa3`
- `ltxvideo2-2.3-dev-1080p-8xh100.peft-lora+cp-fa3`

These profiles assume 8xH100-class hardware, BF16 weights, `context_parallel_size=2`, and the Hugging Face FlashAttention 3 varlen backend. On A100-class systems, copy the example and change `attention_mechanism` to `flash-attn-varlen-hub` before benchmarking.

### Modifying and extending an example

You'll want to copy the folder from `simpletuner/examples` to `config` before modifying anything, otherwise your changes will conflict with newer example config updates.

```bash
cp -R simpletuner/examples/kontext.peft-lora config/kontext.peft-lora
```

Inside the file `config/kontext.peft-lora/config.json` you will need to update the locations of `output_dir` and `dataloader_config`

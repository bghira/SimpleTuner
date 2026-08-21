# InfiniteTalk quickstart

InfiniteTalk is an audio-driven image-to-video model built on Wan 2.1 I2V 14B. SimpleTuner loads the Wan base model, then overlays the official InfiniteTalk single-speaker audio projector and the audio-attention residual in every transformer block.

This integration trains the official single-speaker model. Multi-speaker InfiniteTalk requires multiple synchronized audio streams and speaker masks; the current paired-audio dataloader represents one audio stream per video.

## Requirements

- NVIDIA GPU with bf16 support
- 64 GB system RAM for quantized training; 96 GB or more for RamTorch or unquantized startup
- `ffmpeg` for extracting audio from video files
- Videos sampled at 25 fps
- One aligned audio stream per target video

Install SimpleTuner:

```bash
python -m venv .venv
source .venv/bin/activate
pip install 'simpletuner[cuda]'
```

FlashAttention 3 is used by the examples. Select another supported attention backend if it is unavailable on your GPU.
`trust_remote_code: true` authorizes the example to load the pinned `kernels-community/flash-attn3` Hub kernel. Remove it when selecting a locally installed or built-in backend.

## Starter profiles

The profiles keep batch size at 1. Increase gradient accumulation for a larger effective batch.

| VRAM | Frames | Base weights | Weight residency | Example |
| --- | ---: | --- | --- | --- |
| 24 GB | 17 | bf16 | RamTorch, all blocks streamed | `infinitetalk-14b-480p-24gb.peft-lora` |
| 32 GB | 17 | int8 TorchAO | 20 blocks swapped | `infinitetalk-14b-480p-32gb.peft-lora` |
| 48 GB | 33 | bf16 | 24 blocks swapped | `infinitetalk-14b-480p-48gb.peft-lora` |
| 80 GB | 49 | bf16 | resident | `infinitetalk-14b-480p-80gb.peft-lora` |

CPU RAM, attention backend, optimizer, and frame dimensions affect the actual peak. Start with the matching profile and lower `video.num_frames` before lowering spatial resolution.

## Dataset

Use videos containing their aligned speech audio. Put a caption next to each video:

```text
datasets/infinitetalk/clip-001.mp4
datasets/infinitetalk/clip-001.txt
```

The caption should describe the speaker, scene, camera, expression, and motion. Audio is not a substitute for the text description.

The bundled dataloader configurations extract mono 16 kHz audio automatically:

```json
{
  "id": "infinitetalk-videos-33f",
  "type": "local",
  "dataset_type": "video",
  "instance_data_dir": "datasets/infinitetalk",
  "caption_strategy": "textfile",
  "metadata_backend": "discovery",
  "resolution": 480,
  "resolution_type": "pixel_area",
  "video": {
    "num_frames": 33,
    "min_frames": 33,
    "is_i2v": true,
    "bucket_strategy": "resolution_frames"
  },
  "audio": {
    "auto_split": true,
    "sample_rate": 16000,
    "channels": 1
  },
  "cache_dir_vae": "cache/vae/infinitetalk/33f"
}
```

For separate audio files, disable `audio.auto_split`, create an audio backend, and link it with `s2v_datasets`. Video and audio filenames must have the same stem. See [DATALOADER.md](../DATALOADER.md#audio-configuration-for-s2v-training).

### Alignment rules

- Use 25 fps video. InfiniteTalk constructs Wav2Vec features on that timeline.
- Frame counts must be `4k + 1`; the examples use 17, 33, and 49.
- Audio must cover the same interval as the selected video clip.
- Do not pair random temporal crops with full-track audio.
- Clips without audio are rejected. Zero audio is not a valid training sample.

## Training

Run the 80 GB profile:

```bash
simpletuner train \
  --config simpletuner/examples/infinitetalk-14b-480p-80gb.peft-lora/config.json
```

The important model settings are:

```json
{
  "model_family": "infinitetalk",
  "model_flavour": "single-14b-480p",
  "model_type": "lora",
  "pretrained_model_name_or_path": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
  "framerate": 25,
  "mixed_precision": "bf16"
}
```

SimpleTuner downloads these fixed upstream components:

- Wan base: `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`
- InfiniteTalk audio weights: `MeiGen-AI/InfiniteTalk`, file `single/infinitetalk.safetensors`
- Audio encoder: `TencentGameMate/chinese-wav2vec2-base`

The default PEFT targets include Wan self/text attention and InfiniteTalk audio attention. The default LyCORIS targets also include the audio projector.

## Memory changes

Apply changes in this order:

1. Reduce frame count while keeping `4k + 1`.
2. Enable or increase `musubi_blocks_to_swap`.
3. Use `int8-torchao` with `quantize_via: cpu`.
4. Use the 24 GB RamTorch profile when CPU RAM and PCIe bandwidth are available.
5. Enable feed-forward chunking if activations, rather than weights, are the remaining peak.

InfiniteTalk does not support TREAD or context parallelism. Audio attention is frame-local; token routing or sequence sharding would break its frame-to-audio correspondence.

## Validation

Validation samples need both an image and an audio file. SimpleTuner selects aligned video/audio samples from the configured backend. The validation pipeline uses the first video frame as the identity reference and the paired audio as motion conditioning.

Recommended settings:

```json
{
  "validation_resolution": "832x480",
  "validation_num_video_frames": 49,
  "validation_num_inference_steps": 40,
  "validation_guidance": 5.0
}
```

Built-in validation currently applies text CFG while keeping audio conditioned in both branches. It does not reproduce InfiniteTalk's separate text/audio three-pass guidance. Use the official InfiniteTalk inference project for final quality comparisons.

## Supported training features

| Feature | Support |
| --- | --- |
| PEFT LoRA | Yes |
| LyCORIS | Yes |
| Full-rank training | Yes, multi-GPU only in practice |
| bf16 | Yes |
| TorchAO/Quanto quantization | Yes for adapter training |
| Gradient checkpointing | Yes |
| Checkpoint interval/segment stride | Yes |
| Attention activation offload | Yes |
| Musubi block swap | Yes |
| RamTorch | Yes |
| Feed-forward chunking | Yes |
| CREPA/LayerSync | Yes |
| TREAD | No |
| Context parallelism | No |
| Multi-speaker training | No |

## Licenses

InfiniteTalk and its released weights are Apache 2.0. Wan and the audio encoder retain their own upstream licenses. Review all component licenses and the intended use restrictions before publishing a derivative checkpoint.

## Sources

- [InfiniteTalk implementation](https://github.com/MeiGen-AI/InfiniteTalk)
- [InfiniteTalk technical report](https://arxiv.org/abs/2508.14033)
- [InfiniteTalk weights](https://huggingface.co/MeiGen-AI/InfiniteTalk)

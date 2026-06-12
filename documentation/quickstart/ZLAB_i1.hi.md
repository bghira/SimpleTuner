# zlab i1 Quickstart

यह गाइड [zlab-princeton i1](https://huggingface.co/zlab-princeton/i1-3B) के लिए LoRA training बताती है। i1 एक 3B flow-matching transformer है। इसका original training recipe JAX/TPU के लिए है, लेकिन SimpleTuner इसे native PyTorch path से train करता है और [`bghira/zlab-i1-diffusers`](https://huggingface.co/bghira/zlab-i1-diffusers) वाली Diffusers safetensors conversion इस्तेमाल करता है।

i1 Flux clone नहीं है। यह FLUX.2 VAE, T5Gemma text encoder, 32-channel latents, और CFG के लिए learned null caption इस्तेमाल करता है।

## Hardware requirements

1024px LoRA training के लिए:

- छोटी LoRA के लिए int8 quantisation वाली modern 24G GPU
- ज्यादा आराम के लिए 40G+ GPU
- high rank, बड़े dataset, या कम quantisation के लिए multi-GPU

Example configs `int8-quanto`, `bf16`, `gradient_checkpointing=true`, और `train_batch_size=1` इस्तेमाल करते हैं। CUDA recommended path है। Apple GPU पर i1 training recommended नहीं है।

## Included examples

```bash
simpletuner train example=zlab-i1.peft-lora
simpletuner train example=zlab-i1.lycoris-lokr
```

पहले PEFT LoRA चलाएं। LyCORIS LoKr तब इस्तेमाल करें जब आपको standard LoRA के बजाय LoKr factorisation चाहिए।

## Key configuration

```json
{
  "model_type": "lora",
  "model_family": "zlab_i1",
  "model_flavour": "3b",
  "pretrained_vae_model_name_or_path": "black-forest-labs/FLUX.2-dev",
  "validation_resolution": "1024x1024",
  "validation_guidance": 12.0,
  "validation_guidance_rescale": 0.7,
  "validation_num_inference_steps": 250
}
```

`3b` flavour `bghira/zlab-i1-diffusers` पर resolve होता है, जहाँ transformer standard Diffusers `transformer/` subfolder में safetensors के रूप में है। `pretrained_transformer_model_name_or_path` सिर्फ custom conversion test करते समय सेट करें।

## Validation

Validation native i1 pipeline से चलता है। Quick smoke test के लिए:

```bash
simpletuner train example=zlab-i1.peft-lora validation_num_inference_steps=4 num_eval_images=1
```

4 steps सिर्फ pipeline और image saving check करते हैं। Quality judge करने से पहले 250 steps इस्तेमाल करें।

## Advanced features

i1 SimpleTuner के common transformer feature paths में शामिल है:

- TwinFlow native flow-matching mode में काम करता है। i1 का timestep input upstream की तरह ignore होता है, इसलिए TwinFlow नया time embedding नहीं जोड़ता; यह noisy latent trajectory और target construction बदलता है।
- CREPA Self-Flow और LayerSync i1 image-token hidden-state buffer इस्तेमाल करते हैं। CREPA block indices को i1 की 29 transformer layers के हिसाब से सेट करें।
- TREAD सिर्फ image tokens route करता है। Text tokens intact रहते हैं ताकि T5Gemma conditioning mask की semantics न बदलें।
- Validation CFG Zero*, `validation_no_cfg_until_timestep` से CFG step skip, और `validation_guidance_skip_layers` से skip-layer guidance स्वीकार करता है।
- RamTorch, Musubi block swap, और VAE tiling supported हैं। RamTorch और Musubi को mutually exclusive रखें।

## Dataset

i1 FLUX.2 VAE के 32-channel latents expect करता है। SDXL, Flux.1, PixArt या किसी दूसरे model family का VAE cache reuse न करें।

```json
[
  {
    "id": "my-i1-dataset",
    "type": "local",
    "instance_data_dir": "/datasets/my-subject",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/zlab_i1/my-i1-dataset"
  }
]
```

पहले PEFT example को बिना बदलाव चलाएं। Base benchmark, finite loss, validation image, और `pytorch_lora_weights.safetensors` confirm होने के बाद dataset और prompts बदलें।

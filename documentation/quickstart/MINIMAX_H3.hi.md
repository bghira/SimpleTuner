# MiniMax H3 Quickstart

MiniMax H3 एक 33B flow-matching video/audio model है। SimpleTuner `minimaxh3` family में adapter training support करता है, जिसमें FL2VA first/last-frame conditioning और quantized ConvRot flavours शामिल हैं।

## Starting Configs

इन examples से शुरू करें:

- `simpletuner/examples/minimaxh3/convrot-int8/default/fl2va-convrot-int8.peft-lora`
- `simpletuner/examples/minimaxh3/convrot-int8/24g/fl2va-convrot-int8.peft-lora+ramtorch`
- `simpletuner/examples/minimaxh3/convrot-int8/32g/fl2va-convrot-int8.peft-lora`
- `simpletuner/examples/minimaxh3/convrot-int8/48g/fl2va-convrot-int8.peft-lora`
- `simpletuner/examples/minimaxh3/convrot-int8/80g/fl2va-convrot-int8.peft-lora`

अपनी VRAM के सबसे करीब preset लें, फिर smoke test के बाद resolution, frames, attention backend, और checkpointing बदलें।

## Core Settings

```json
{
  "model_family": "minimaxh3",
  "model_flavour": "convrot-int8",
  "model_type": "lora",
  "mixed_precision": "bf16",
  "base_model_precision": "no_change",
  "text_encoder_1_precision": "int8-quanto",
  "flow_schedule_shift": 12.0,
  "audio_flow_schedule_shift": 3.0,
  "validation_disable_unconditional": true,
  "validation_guidance": 1.0,
  "validation_guidance_real": 1.0
}
```

Examples `convrot-int8` use करते हैं। कम precision checkpoint चाहिए तो उसी family में `convrot-int4` इस्तेमाल कर सकते हैं।

## Distillation बनाए रखना

MiniMax H3 CFG-distilled है। Base checkpoint unconditional branch के बिना चलने के लिए बना है, इसलिए examples guidance `1.0` और `validation_disable_unconditional: true` रखते हैं।

Adapter training base distilled behavior से drift कर सकती है। इसलिए examples default रूप से `h3_drift` enable करते हैं:

```json
{
  "distillation_method": "h3_drift",
  "distillation_config": {
    "h3_drift": {
      "loss_weight": 0.5,
      "sft_loss_weight": 1.0,
      "balance": "token",
      "video_weight": 1.0,
      "audio_weight": 1.0
    }
  }
}
```

यह adapter disabled frozen-base reference pass चलाता है और video/audio prediction drift को penalize करता है। Normal H3 LoRA के लिए इसे enabled रखें। Concept नहीं सीख रहा हो तो `loss_weight` घटाएं; validation base behavior खो रही हो तो बढ़ाएं। पूरी explanation: [MiniMax H3 Drift Distillation](../distillation/MINIMAX_H3_DRIFT.hi.md)।

Negative prompting base H3 contract का हिस्सा नहीं है। SimpleTuner de-distilled checkpoints के लिए real CFG और negative prompts support रखता है, लेकिन `h3_drift` original distilled conditional behavior preserve करता है।

## Audio Target Mode

`minimax_h3_target_mode: "auto"` enabled audio data मिलने पर `av`, अन्यथा `video` में resolve होता है। Validation भी इसी detected-data default का उपयोग करता है, इसलिए audio-only और joint audio/video runs अलग validation override के बिना audio बनाते हैं। Audio VAE work से बचने के लिए `video` explicitly set करें:

```json
{
  "minimax_h3_target_mode": "video"
}
```

जब dataset में target audio latents हों और joint audio/video training चाहिए, तब `"av"` explicitly use कर सकते हैं। Per-backend `h3_target_mode` या `minimax_h3_target_mode` भी set कर सकते हैं।

Audio-only training के लिए `dataset_type: "audio"` पर्याप्त है। H3 fake-video support advertise करता है, इसलिए
SimpleTuner normalized backend config में `audio.audio_only: true` record करता है, placeholder video stream बनाता है,
और video loss mask करता है। Explicit `audio_only` setting valid है, लेकिन required नहीं है।

## Context parallelism

H3 context parallelism Ulysses को `context_parallel_strategy: "alltoall"` के साथ use करता है। Packed sequence को CP
degree तक pad किया जा सकता है, इसलिए local attention backend को mask accept करना होगा। `native` supported
है; CP enabled होने पर SimpleTuner दूसरे backend choices को `native` से replace करता है।

लगभग 8k audio tokens पर CP communication के बदले activation memory घटाता है और हल्का checkpointing देता है। CP अपने
आप weights shard नहीं करता, इसलिए लंबी sequence या FSDP combination के बिना इसे DDP के विरुद्ध benchmark करें।

## Experimental Sparse Attention

MiniMax बताता है कि H3 ने final training stage में video tokens के लिए MoBA-style 3D sparse attention इस्तेमाल किया। Initial public release dense attention use करता है, और MiniMax ने exact block shape, retention budget, layer schedule, या production kernel publish नहीं किया है। इसलिए SimpleTuner इस experimental approximation को default में disabled रखता है।

```json
{
  "minimax_h3_sparse_attention": "moba3d",
  "minimax_h3_sparse_block_shape": "1,8,16",
  "minimax_h3_sparse_video_kv_fraction": 0.25,
  "minimax_h3_sparse_share_heads": false,
  "minimax_h3_sparse_start_layer": 0
}
```

Implementation 3D query/key video blocks को mean-pool करके parameter-free top-k routing करती है। Target-video queries text, audio, और reference context पर dense access रखती हैं; non-target queries dense रहती हैं। Block dimensions का product 128 होना चाहिए। Video KV fraction `1.0` FlexAttention के through dense-connectivity numerical control है।

यह mode CUDA require करता है और FlexAttention के आसपास Dynamo graph boundary introduce करता है। Ulysses context parallelism `context_parallel_strategy=alltoall` के साथ supported है; ring context parallelism और TREAD supported नहीं हैं। 480px पर sparse routing FlashAttention से ज्यादा memory use कर सकता है क्योंकि target lattice और packed context को pad/reorder करना पड़ता है। MiniMax reference implementation publish करे तब तक इसे guaranteed speedup नहीं, fine-tuning ablation मानें।

## Memory Knobs

- VRAM tight हो तो 24G RamTorch example इस्तेमाल करें।
- ज्यादा checkpointing से पहले `musubi_blocks_to_swap` test करें।
- video `flow_schedule_shift` को `12.0` और `audio_flow_schedule_shift` को `3.0` रखें। H3 helper inherited global video default `3.0` को ठीक करता है क्योंकि वह MiniMax H3 schedule से match नहीं करता।
- SimpleTuner H3 video VAE के लिए VAE tiling और temporal roll/chunking force करता है। Tiling geometry upstream जैसी है: `256` tile size और `64` overlap। इन options को false करने पर ignore किया जाएगा, क्योंकि untiled decode से severe colour shift और halftone artifacts आ सकते हैं।
- Target GPU पर `attention_mechanism` benchmark करें।
- `torch.compile` बदलने पर smoke test दोबारा करें, क्योंकि compile cache VRAM बदल सकता है।

## Run

```bash
simpletuner train example=minimaxh3/convrot-int8/default/fl2va-convrot-int8.peft-lora
```

Long run से पहले छोटा smoke test करें और देखें कि `h3_drift_loss`, normal loss, और validation samples साथ-साथ sensible move कर रहे हैं।

## H3 Image

`MiniMaxAI/MiniMax-H3-Image` के लिए `model_family: "minimaxh3"` और `model_flavour: "h3_image"` उपयोग करें। शुरुआती उदाहरण `simpletuner/examples/minimaxh3/h3_image/80g/h3_image.peft-lora` है। साझा H3 transformer `transformer_diffusers` लोड करता है; डिफ़ॉल्ट इमेज VAE `vae_diffusers` में है और इसमें 24-लेयर 2D decoder है। स्पष्ट रूप से दिए गए `pretrained_vae_model_name_or_path` या `vae_path` सुरक्षित रहते हैं; स्थानीय component directories और समर्थित single-file VAE भी स्वीकार हैं। ऑडियो VAE लोड नहीं होता।

2048 या 4096 पिक्सेल के प्रशिक्षण कैनवास उपयोग करें और आकार 32 पिक्सेल के गुणज रखें; 4096 मॉडल का पसंदीदा आकार है। 80GB उदाहरण activation checkpoint offload सक्रिय करता है। प्रशिक्षण जारी किए गए छह PDD noise levels और उनके संबंधित frozen output heads का उपयोग करता है; validation में ठीक वही छह-चरण sigma schedule और guidance 1.0 लागू होते हैं। इस flavour में मनमाने चरणों का sampling और H3 वीडियो distillation recipes समर्थित नहीं हैं। ComfyUI adapters लोड और निर्यात करने के लिए `lora_format: "comfyui"` सेट करें; यह `Comfy-Org/MiniMax-H3-Image` के pruned BF16 checkpoint के साथ भी काम करता है।

वैकल्पिक रूप से `minimax_h3_prompt_expander_url` को `MiniMaxAI/MiniMax-H3-Image-PE` होस्ट करने वाले OpenAI-compatible server पर सेट करें। आधिकारिक text-to-image और editing prompts कैशिंग से पहले positive captions का विस्तार करते हैं; editing में reference images भी भेजी जाती हैं। प्रमाणीकरण के लिए environment variable `MINIMAX_H3_PROMPT_EXPANDER_API_KEY` उपयोग करें। मूल captions रखने के लिए URL खाली छोड़ें। प्रशिक्षित assistant को `assistant_lora_path` से चुन सकते हैं; कोई अप्रकाशित assistant डिफ़ॉल्ट रूप से डाउनलोड नहीं होता।

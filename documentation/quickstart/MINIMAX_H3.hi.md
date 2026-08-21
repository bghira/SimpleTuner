# MiniMax H3 Quickstart

MiniMax H3 एक 33B flow-matching video/audio model है। SimpleTuner `minimaxh3` family में adapter training support करता है, जिसमें FL2VA first/last-frame conditioning और quantized ConvRot flavours शामिल हैं।

## Starting Configs

इन examples से शुरू करें:

- `simpletuner/examples/minimaxh3-fl2va-convrot-int8.peft-lora`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-24g.peft-lora+ramtorch`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-32g.peft-lora`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-48g.peft-lora`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-80g.peft-lora`

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
simpletuner train example=minimaxh3-fl2va-convrot-int8.peft-lora
```

Long run से पहले छोटा smoke test करें और देखें कि `h3_drift_loss`, normal loss, और validation samples साथ-साथ sensible move कर रहे हैं।

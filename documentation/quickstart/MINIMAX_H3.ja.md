# MiniMax H3 Quickstart

MiniMax H3 は 33B の flow-matching video/audio モデルです。SimpleTuner は `minimaxh3` family で adapter training をサポートし、FL2VA first/last-frame conditioning と quantized ConvRot flavours を扱えます。

## Starting Configs

次の examples から始めます。

- `simpletuner/examples/minimaxh3-fl2va-convrot-int8.peft-lora`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-24g.peft-lora+ramtorch`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-32g.peft-lora`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-48g.peft-lora`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-80g.peft-lora`

VRAM に近い preset を選び、smoke test 後に resolution、frame count、attention backend、checkpointing を調整します。

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

Examples は `convrot-int8` を使います。より低い precision の checkpoint を使う場合は同じ family で `convrot-int4` を指定できます。

## Distillation を保つ

MiniMax H3 は CFG-distilled です。base checkpoint は unconditional branch なしで動く想定なので、examples は guidance `1.0` と `validation_disable_unconditional: true` を使います。

Adapter training は distilled behavior から drift することがあります。そのため examples は default で `h3_drift` を有効化します。

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

これは adapter を無効化した frozen-base reference pass を実行し、video/audio prediction drift を penalize します。通常の H3 LoRA では有効のままにしてください。concept を学びにくい場合は `loss_weight` を下げ、validation が base behavior を失う場合は上げます。詳細は [MiniMax H3 Drift Distillation](../distillation/MINIMAX_H3_DRIFT.ja.md) を参照してください。

Negative prompting は base H3 contract の一部ではありません。SimpleTuner は de-distilled checkpoint 向けに real CFG と negative prompts を残していますが、`h3_drift` は original distilled conditional behavior を保つためのものです。

## Audio Target Mode

`minimax_h3_target_mode: "auto"` は、有効な audio data が検出された場合は `av`、それ以外は `video` になります。Validation も同じ検出結果を使うため、audio-only および joint audio/video run では別の validation override なしで audio を生成します。Audio VAE work を避けるには `video` を明示します。

```json
{
  "minimax_h3_target_mode": "video"
}
```

dataset に target audio latents があり joint audio/video training したい場合は `"av"` を明示できます。data backend ごとに `h3_target_mode` または `minimax_h3_target_mode` でも設定できます。

Audio-only training では `dataset_type: "audio"` だけで十分です。H3 は fake-video support を提供するため、
SimpleTuner は normalized backend config に `audio.audio_only: true` を記録し、placeholder video stream を作成して
video loss を mask します。明示的な `audio_only` も使用できますが、必須ではありません。

## Context parallelism

H3 context parallelism は Ulysses と `context_parallel_strategy: "alltoall"` を使用します。packed sequence は CP
degree に合わせて padding される場合があるため、local attention backend は mask を受け付ける必要があります。
`native` と `cudnn` をサポートし、CP 有効時のその他の backend は SimpleTuner が `native` に置き換えます。

約 8k audio tokens では、CP は主に communication と引き換えに activation memory と checkpointing を減らします。
CP 単体では weights を shard しないため、より長い sequence または FSDP との併用でなければ DDP と比較してください。

## Experimental Sparse Attention

MiniMax は、H3 の最終 training stage で MoBA-style 3D sparse attention を video tokens に使ったと述べています。初期 public release は dense attention を使っており、MiniMax は正確な block shape、retention budget、layer schedule、production kernel をまだ公開していません。そのため SimpleTuner ではこの experimental approximation をデフォルトで無効にしています。

```json
{
  "minimax_h3_sparse_attention": "moba3d",
  "minimax_h3_sparse_block_shape": "1,8,16",
  "minimax_h3_sparse_video_kv_fraction": 0.25,
  "minimax_h3_sparse_share_heads": false,
  "minimax_h3_sparse_start_layer": 0
}
```

この実装は 3D query/key video blocks を mean-pool し、parameter-free top-k routing を行います。target-video queries は text、audio、reference context への dense access を保持し、non-target queries は dense のままです。block dimensions の積は 128 である必要があります。video KV fraction `1.0` は FlexAttention 経由の dense-connectivity numerical control です。

この mode は CUDA を必要とし、FlexAttention の周囲に Dynamo graph boundary を導入します。Ulysses context parallelism は `context_parallel_strategy=alltoall` で対応します。ring context parallelism と TREAD は非対応です。480px では、target lattice と packed context の padding/reordering が必要なため、sparse routing が FlashAttention より多くの memory を使う場合があります。MiniMax が reference implementation を公開するまでは、speedup 保証ではなく fine-tuning ablation として扱ってください。

## Memory Knobs

- VRAM が厳しい場合は 24G RamTorch example を使います。
- heavy checkpointing の前に `musubi_blocks_to_swap` を試します。
- video `flow_schedule_shift` は `12.0`、`audio_flow_schedule_shift` は `3.0` のままにします。H3 helper は、MiniMax H3 schedule と一致しない継承された global video default `3.0` を修正します。
- SimpleTuner は H3 video VAE の tiling と temporal roll/chunking を強制的に有効化します。tiling geometry は upstream と同じ `256` tile size / `64` overlap です。これらを false にしても無視されます。untiled decode は大きな color shift や halftone artifacts を出すことがあります。
- 実際の GPU で `attention_mechanism` を benchmark します。
- `torch.compile` を変えたら smoke test をやり直します。compile cache が VRAM を変えるためです。

## Run

```bash
simpletuner train example=minimaxh3-fl2va-convrot-int8.peft-lora
```

long run の前に短い smoke test を行い、`h3_drift_loss`、normal loss、validation sample が一貫して動くことを確認します。

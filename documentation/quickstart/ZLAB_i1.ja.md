# zlab i1 クイックスタート

このガイドでは [zlab-princeton i1](https://huggingface.co/zlab-princeton/i1-3B) の LoRA 学習を扱います。i1 は 3B の flow-matching transformer で、公式には JAX/TPU の学習レシピと PyTorch 推論重みが公開されています。SimpleTuner ではネイティブ PyTorch 実装で学習し、[`bghira/zlab-i1-diffusers`](https://huggingface.co/bghira/zlab-i1-diffusers) の Diffusers safetensors 変換を使います。

i1 は Flux の単純な派生ではありません。FLUX.2 VAE、T5Gemma text encoder、32 チャンネル latent、CFG 用の学習済み null caption を使います。

## ハードウェア要件

1024px LoRA 学習の目安:

- 小さな LoRA なら、int8 量子化ありの 24G GPU
- 余裕を持つなら 40G 以上
- 高 rank、大きな dataset、量子化を弱める場合は multi-GPU

サンプル設定は `int8-quanto`、`bf16`、`gradient_checkpointing=true`、`train_batch_size=1` です。CUDA を想定しています。Apple GPU は推奨しません。

## 付属サンプル

```bash
simpletuner train example=zlab-i1.peft-lora
simpletuner train example=zlab-i1.lycoris-lokr
```

最初は PEFT LoRA を使ってください。LyCORIS LoKr は標準 LoRA ではなく LoKr 分解を使いたい場合に向いています。

## 主要設定

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

`3b` flavour は `bghira/zlab-i1-diffusers` に解決されます。transformer は標準の Diffusers `transformer/` サブフォルダに safetensors として保存されています。独自変換を試す場合だけ `pretrained_transformer_model_name_or_path` を指定してください。

## 検証

検証は i1 のネイティブ pipeline で動きます。短い smoke test:

```bash
simpletuner train example=zlab-i1.peft-lora validation_num_inference_steps=4 num_eval_images=1
```

4 steps は生成と保存の確認だけです。品質を見る場合は 250 steps を使ってください。

## 高度な機能

i1 は SimpleTuner の共通 transformer 機能パスに対応しています:

- TwinFlow は native flow-matching モードで動きます。i1 の timestep 入力は upstream と同じく無視されるため、TwinFlow は新しい time embedding ではなく noisy latent の軌跡と target 構築を変更します。
- CREPA Self-Flow と LayerSync は i1 の image-token hidden-state buffer を使います。CREPA の block index は i1 の 29 transformer layers に合わせて指定してください。
- TREAD は image tokens だけを route します。Text tokens はそのままなので、T5Gemma conditioning mask の意味は保たれます。
- Validation は CFG Zero*、`validation_no_cfg_until_timestep` による CFG step skip、`validation_guidance_skip_layers` による skip-layer guidance を受け付けます。
- RamTorch、Musubi block swap、VAE tiling に対応しています。RamTorch と Musubi は同時に使わないでください。

## Dataset

i1 は FLUX.2 VAE の 32 チャンネル latent を必要とします。SDXL、Flux.1、PixArt など別モデルの VAE cache は再利用しないでください。

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

まず PEFT サンプルをそのまま実行し、base benchmark、有限の loss、検証画像、`pytorch_lora_weights.safetensors` を確認してから dataset と prompt を差し替えてください。

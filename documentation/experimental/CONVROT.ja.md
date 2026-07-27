# ConvRot / Hadamard SDNQ

SimpleTuner は、SDNQ の Hadamard 経路で ConvRot 形式の回転量子化を提供します。凍結したベースモデルを int8 で実行し、LoRA や LyCORIS アダプターを bf16 などの混合精度 dtype のまま学習する大型 PEFT ジョブに向いています。

SimpleTuner は任意の ConvRot sidecar buffer を独立した機能として消費するものではありません。通常の path では元のモデル重みを読み込み、モデルロード後に SimpleTuner が SDNQ で学習対象コンポーネントを量子化します。single-file の quantized transformer weight に対応した loader では、互換性のある INT8 ConvRot transformer safetensors を読み込み、SDNQ Hadamard 経由で実行することもできます。

## クイック設定

```json
{
  "base_model_precision": "int8-sdnq",
  "gradient_checkpointing": true,
  "sdnq_use_hadamard": true,
  "sdnq_hadamard_group_size": 256,
  "sdnq_group_size": -1,
  "sdnq_use_quantized_matmul": true,
  "sdnq_compile_mode": "compile"
}
```

大型モデルでは、モデルガイドで別指定がない限り `quantize_via` は `cpu` のままにしてください。CPU 量子化はセットアップ時のアクセラレータメモリのピークを抑えます。

## オプション

- `base_model_precision: int8-sdnq` は、学習対象のベースコンポーネントに SDNQ int8 のロード後量子化を選びます。
- `sdnq_use_hadamard: true` は Hadamard 回転経路を有効にします。
- `sdnq_hadamard_group_size: 256` は SDNQ が使う回転ブロックサイズです。ConvRot には `256` を使います。より小さいブロックは QuaRot 風の path を選択します。
- `sdnq_group_size: -1` は静的な行単位 weight scale を使います。主にフルファインチューニング向けの動的 grouped 経路で、学習中に重みが再量子化されるのを避けます。
- `sdnq_use_quantized_matmul: true` は SDNQ int8 matmul 経路を有効に保ちます。
- `sdnq_compile_mode: compile` は SDNQ が対応する量子化 helper と kernel をコンパイルします。
- `gradient_checkpointing: true` は PEFT ワークロードで SDNQ の低オーバーヘッドな学習経路を使えるようにします。SimpleTuner はこれを `use_grad_ckpt=True` として SDNQ に渡します。gradient checkpointing が有効なときにこの SDNQ フラグを false にすると、checkpointing がすぐ破棄する量子化 backward 入力を保存するだけで遅くなります。

## PEFT での動作

ベース transformer は SDNQ で量子化されます。アダプター重みは学習可能なままで、通常は bf16 のような混合精度 dtype を使います。

一部のモデルは学習前に固定の補助アダプターを読み込みます。たとえば Z-Image Turbo には assistant LoRA があります。SimpleTuner はこの assistant アダプターを SDNQ 量子化後まで遅延させるため、SDNQ は PEFT wrapper の proxy weight ではなく元の transformer module を処理できます。

## 要件と制限

- SimpleTuner は、対応している install target では SDNQ training dependency をインストールして設定します。
- このプリセットは大型モデルの LoRA と LyCORIS 学習向けです。SDNQ Hadamard でのフルファインチューニングは別途検証が必要です。
- 初期 step は遅くなることがあります。SDNQ と Torch がセットアップや初期学習中に kernel をコンパイルするためです。
- 検証と推論は、学習時と同じく量子化済みベースモデルと有効なアダプターを使います。
- ConvRot は量子化による劣化を減らせますが、すべての model で INT8 が BF16 や FP8 と一致する保証ではありません。長い run の前に loss curve と生成 sample の両方を検証してください。
- SDNQ ConvRot の standalone inference は、この training guide の範囲外です。直接 SDNQ inference API を使う場合は、SimpleTuner の training 設定より API 変更が多いため、[SDNQ upstream documentation](https://github.com/Disty0/sdnq) に従ってください。

## 測定結果

これは synthetic な GEMM-only 結果ではなく、モデル別の SimpleTuner 実 trainer 測定です。`Loop s/step` は wrapper の training loop wall time/step です。`Mean step` は最初の 5 warmup step を除外しています。

| Model | GPU | Steps | Weight path | Loop s/step | Mean step | p50 | p95 | Peak allocated VRAM |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Z-Image Turbo LoRA | H100 80GB | 1000 | SDNQ Hadamard post-load quantization | 1.107 | 1.087 | 1.071 | 1.109 | 9.70 GiB |
| Z-Image Turbo LoRA | L40S | 1000 | SDNQ Hadamard post-load quantization | 1.026 | 1.018 | 1.002 | 1.040 | 9.66 GiB |
| Z-Image Turbo LoRA | L40S | 1000 | baseline SDNQ Hadamard path | 1.131 | 1.072 | 1.055 | 1.102 | 9.66 GiB |
| Krea 2 Raw LoRA | H100 80GB | 100 | `lilcheaty/Krea2-INT8-ConvRot` transformer weights、diffusers attention | 0.787 | 0.399 | 0.397 | 0.411 | 32.15 GiB |
| Krea 2 Raw LoRA | L40S | 100 | `lilcheaty/Krea2-INT8-ConvRot` transformer weights、cuDNN attention | 0.945 | 0.794 | 0.793 | 0.799 | 31.89 GiB |
| Mage-Flow LoRA, square crop | H100 80GB | 100 | SDNQ INT8 vanilla post-load quantization | 1.113 | 0.277 | 0.276 | 0.286 | 20.12 GiB |
| Mage-Flow LoRA, square crop | H100 80GB | 100 | SDNQ ConvRot 256 post-load quantization | 0.436 | 0.299 | 0.297 | 0.308 | 20.15 GiB |

warm cache の L40S Z-Image 比較では、現行 path は baseline SDNQ Hadamard path より train-loop wall time で 10.3%、測定 train-step 平均で 5.2% 高速でした。Krea 2 の各行は、Hugging Face の INT8 ConvRot transformer weight path を実際の 100 step training run で検証したものです。Mage-Flow の行は model-specific validation が重要であることを示します。Square crop は shape compile churn の大半を取り除き、ConvRot は vanilla INT8 より総 train-loop time を短くしましたが、warm 後の measured step は vanilla INT8 より少し遅くなりました。

## サンプルモデル

SimpleTuner には Z-Image Turbo、Krea 2、FLUX.2、Cosmos 3、LTXVideo 2.3 向けの SDNQ Hadamard サンプルがあります。これらは PEFT ワークロードにより合うため、動的 grouped 学習のデフォルトではなく `sdnq_group_size: -1` を使います。

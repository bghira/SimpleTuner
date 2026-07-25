# ConvRot / Hadamard SDNQ

SimpleTuner は、SDNQ の Hadamard 経路で ConvRot 形式の回転量子化を提供します。凍結したベースモデルを int8 で実行し、LoRA や LyCORIS アダプターを bf16 などの混合精度 dtype のまま学習する大型 PEFT ジョブに向いています。

これは外部 ConvRot checkpoint の独自 buffer を直接読み込む機能ではありません。元のモデル重みを読み込み、モデルロード後に SimpleTuner が SDNQ で学習対象コンポーネントを量子化します。

## クイック設定

```json
{
  "base_model_precision": "int8-sdnq",
  "gradient_checkpointing": true,
  "sdnq_use_hadamard": true,
  "sdnq_hadamard_group_size": 128,
  "sdnq_group_size": -1,
  "sdnq_use_quantized_matmul": true,
  "sdnq_compile_mode": "compile"
}
```

大型モデルでは、モデルガイドで別指定がない限り `quantize_via` は `cpu` のままにしてください。CPU 量子化はセットアップ時のアクセラレータメモリのピークを抑えます。

## オプション

- `base_model_precision: int8-sdnq` は、学習対象のベースコンポーネントに SDNQ int8 のロード後量子化を選びます。
- `sdnq_use_hadamard: true` は Hadamard 回転経路を有効にします。
- `sdnq_hadamard_group_size: 128` は SDNQ が使う回転ブロックサイズです。
- `sdnq_group_size: -1` は静的な行単位 weight scale を使います。主にフルファインチューニング向けの動的 grouped 経路で、学習中に重みが再量子化されるのを避けます。
- `sdnq_use_quantized_matmul: true` は SDNQ int8 matmul 経路を有効に保ちます。
- `sdnq_compile_mode: compile` は SDNQ が対応する量子化 helper と kernel をコンパイルします。
- `gradient_checkpointing: true` は PEFT ワークロードで SDNQ の低オーバーヘッドな学習経路を使えるようにします。SimpleTuner はこれを `use_grad_ckpt=True` として SDNQ に渡します。gradient checkpointing が有効なときにこの SDNQ フラグを false にすると、checkpointing がすぐ破棄する量子化 backward 入力を保存するだけで遅くなります。

## PEFT での動作

ベース transformer は SDNQ で量子化されます。アダプター重みは学習可能なままで、通常は bf16 のような混合精度 dtype を使います。

一部のモデルは学習前に固定の補助アダプターを読み込みます。たとえば Z-Image Turbo には assistant LoRA があります。SimpleTuner はこの assistant アダプターを SDNQ 量子化後まで遅延させるため、SDNQ は PEFT wrapper の proxy weight ではなく元の transformer module を処理できます。

## 要件と制限

- Hadamard サポートを含む SDNQ build を使ってください。H100 検証では upstream SDNQ `0.2.3` を使いました。PyPI `0.2.2` には同じ bf16 Hadamard 修正が含まれていません。
- このプリセットは大型モデルの LoRA と LyCORIS 学習向けです。SDNQ Hadamard でのフルファインチューニングは別途検証が必要です。
- 初期 step は遅くなることがあります。SDNQ と Torch がセットアップや初期学習中に kernel をコンパイルするためです。
- 検証と推論は、学習時と同じく量子化済みベースモデルと有効なアダプターを使います。

## サンプルモデル

SimpleTuner には Z-Image Turbo、Krea 2、FLUX.2、Cosmos 3、LTXVideo 2.3 向けの SDNQ Hadamard サンプルがあります。これらは PEFT ワークロードにより合うため、動的 grouped 学習のデフォルトではなく `sdnq_group_size: -1` を使います。

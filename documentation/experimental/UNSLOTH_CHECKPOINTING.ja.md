# Unsloth 形式の Checkpointing

短く言うと、ジョブがあと少しで入るときに使います。モデルが対応しているなら、まず FFN-only を試します。

`unsloth` backend は保存される activation tensor を CPU に offload します。`torch` backend はそれを捨てて backward で再計算します。Unsloth は batch、解像度、frames を少し上げるための最後の数 GiB を作れます。無料の高速化ではありません。`torch` で既に入るなら、普通は `torch` のままでよいです。

## Controls

```json
{
  "gradient_checkpointing": true,
  "gradient_checkpointing_backend": "torch-ffn"
}
```

`gradient_checkpointing_backend` には 4 つの実用値があります:

| Value | Scope | Path | 使う場面 |
| --- | --- | --- | --- |
| `torch` | whole block | recompute | CPU offload の前に、内蔵の最大メモリ削減が必要。 |
| `torch-ffn` | feed-forward | recompute | Flash Attention が attention memory を処理した後の安い削減。 |
| `unsloth` | whole block | CPU offload | torch layer checkpointing でもまだ入らない。 |
| `unsloth-ffn` | feed-forward | CPU offload | torch FFN-only があと少しで入り、CPU offload で最後を詰めたい。 |

対応モデルでは checkpoint する block も減らせます:

```json
{
  "gradient_checkpointing": true,
  "gradient_checkpointing_backend": "torch-ffn",
  "gradient_checkpointing_interval": 2
}
```

`gradient_checkpointing_interval: 2` は対応する whole-block path で連続した 2-block chunk を checkpoint します。値を大きくすると再計算は減り、VRAM に残る activation は増えます。

これらの segmented path では、`gradient_checkpointing_segment_stride` も `unsloth` で使えます。速度目的ではなく fit lever として扱ってください。Skip された blocks は GPU に残り、checkpoint された blocks は保存 tensor に CPU offload を使います。Torch-only の概要とモデル別 benchmark は [Segmented Checkpointing](SEGMENTED_CHECKPOINTING.md) を参照してください。

`gradient_checkpointing_offload_attention` は backend とは別の option です。対応する attention/FFN split blocks では、attention 側の保存 activations を offload します。単体でも実行でき、モデルがその backend をサポートする場合は `torch`、`torch-ffn`、`unsloth`、`unsloth-ffn` と組み合わせられます。

`gradient_checkpointing_offload_pin_memory_max_buckets` は offload された保存 tensor の pinned CPU pooling を制御します。デフォルトは `12` 個の distinct tensor buckets です。`0` にすると通常の CPU memory だけを使います。

`torch-ffn` と `unsloth-ffn` は現在 Chroma、Flux、Krea 2、LTXVideo2、MageFlow、Wan、Z-Image に対応しています。他のモデル族は、同じ安全な境界を expose するまで明示的に失敗します。

## 何を交換するか

- `torch`: 中間 activations を捨て、backward で再計算します。
- `unsloth`: 一部 tensor を CPU に保存し、backward で GPU に戻します。
- `*-ffn`: 明確な FFN 境界があるモデルで feed-forward 側だけを checkpoint します。
- Flash Attention は大きな attention matrix を materialize しません。この「無料 checkpointing」は主に attention の話で、transformer block 全体ではありません。
- CPU offload は、activation が大きく、ピークが parameters や optimizer ではないときに効きます。

CUDA と十分な CPU RAM が必要です。PCIe 帯域も効きます。CPU-GPU copy が隠れないと step は遅くなります。

## Sweep 結果

合成 transformer block、bf16、flash SDPA、base weights frozen、batch 1。モデル保証ではなく、tradeoff の形を見るための数字です。

### Packed Image Latents

2x2 packing では `64x64`、`128x128`、`256x256` が `1024`、`4096`、`16384` transformer tokens になります。

| GPU | Tokens | No checkpoint | Torch FFN | Unsloth FFN | Torch layer | Unsloth layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 80GB | 1024 | 0.0166s / 4.43 GiB | 0.0191s / 4.08 GiB | 0.0233s / 4.00 GiB | 0.0231s / 3.64 GiB | 0.0265s / 3.56 GiB |
| H100 80GB | 4096 | 0.0948s / 7.43 GiB | 0.1029s / 6.02 GiB | 0.1157s / 5.67 GiB | 0.1233s / 4.26 GiB | 0.1358s / 3.93 GiB |
| H100 80GB | 16384 | 0.8781s / 19.39 GiB | 0.9117s / 13.77 GiB | 0.9632s / 12.36 GiB | 1.1157s / 6.72 GiB | 1.1662s / 5.41 GiB |
| L40S | 1024 | 0.0500s / 4.39 GiB | 0.0575s / 4.04 GiB | 0.0627s / 3.95 GiB | 0.0666s / 3.60 GiB | 0.0725s / 3.51 GiB |
| L40S | 4096 | 0.2461s / 7.38 GiB | 0.2729s / 5.97 GiB | 0.2933s / 5.62 GiB | 0.3169s / 4.21 GiB | 0.3369s / 3.88 GiB |
| L40S | 16384 | 1.8153s / 19.35 GiB | 1.9639s / 13.72 GiB | 2.0250s / 12.31 GiB | 2.3360s / 6.67 GiB | 2.4218s / 5.36 GiB |

`1024` tokens では、既に VRAM 限界でない限り追加 offload はほぼ不要です。`16384` tokens では、`torch-ffn` が安い一手で、whole-layer checkpointing が大きな fit lever です。`unsloth` は torch layer checkpointing よりさらに約 `1.3 GiB` 節約しました。

### 大きめの Transformer

Frozen `32` layers、width `4096`、`3072` tokens:

| GPU | No checkpoint | Torch FFN | Unsloth FFN | Torch layer | Unsloth layer |
| --- | ---: | ---: | ---: | ---: | ---: |
| H100 80GB | 0.1943s / 14.56 GiB | 0.2138s / 11.65 GiB | 0.2317s / 10.92 GiB | 0.2527s / 8.01 GiB | 0.2722s / 7.30 GiB |
| L40S | 0.5045s / 14.51 GiB | 0.5640s / 11.60 GiB | 0.5932s / 10.88 GiB | 0.6491s / 7.96 GiB | 0.6864s / 7.26 GiB |

Full weights を trainable にすると絵が変わります。gradients と optimizer state がピークを支配した合成 run では、`unsloth` は `torch` 以上には節約できませんでした。PEFT は frozen-weight のケースに近いです。

## 判断ルール

1. Checkpointing なしで入るなら、オフのまま。
2. 入らないなら、まず `gradient_checkpointing_backend: torch-ffn` を試す。
3. まだ厳しいなら `torch` を試す。
4. torch layer checkpointing でも入らないなら、`unsloth-ffn`、次に `unsloth` を試す。
5. モデルが `gradient_checkpointing_interval` をサポートするなら、まず入る状態にしてから `2` 以上で速度を戻す。

欲しい batch、解像度、frames、rank が入るようになるなら価値があります。小さい token 数や、ピークが trainable weights、gradients、optimizer、VAE cache、validation 由来なら価値は薄いです。

## Notes

- FSDP activation checkpointing が有効な場合、SimpleTuner は競合を避けるため model-level gradient checkpointing を無効にします。
- `torch-ffn` と `unsloth-ffn` はモデル側の対応が必要です。SimpleTuner は別 scope を黙って実行せず、明示的に失敗します。
- `gradient_checkpointing_interval: 1` は通常の every-block checkpointing と同じです。
- 一部モデル族は interval checkpointing に未対応です。SimpleTuner は警告して interval を無視します。
- 私たちの sweep では、`torch.compile` は offload path の決定打にはなりませんでした。

# DiffusionBlocks

DiffusionBlocks は、対応する diffusion Transformer を独立に学習できるレイヤー群へ変換します。各グループは 1 つのノイズ範囲を担当し、1 回の forward ではその batch に対応するグループだけを実行します。

これは [DiffusionBlocks](https://arxiv.org/abs/2506.14202) に基づく実験的なアーキテクチャ変換です。通常の layer freeze ではありません。推論でも学習時と同じ routing が必要です。

## 設定

```json
{
  "diffusion_blocks_config": {
    "layers_per_block": 4,
    "overlap": 0.05
  },
  "find_unused_parameters": true
}
```

DDP では `find_unused_parameters` が自動的に有効になります。`false` はエラーです。

| キー | 既定値 | 意味 |
| --- | --- | --- |
| `layers_per_block` | 必須 | 1 ノイズブロックあたりの連続 Transformer layer の最大数。 |
| `overlap` | `0.05` | 隣接する学習ノイズ範囲の拡張率。`0.0` から `0.5`。 |
| `blocks_to_train` | `"all"` | この run が担当する block index。他のグループは adapter 作成後に freeze されます。 |
| `block_paths` | 自動 | 自動検出で不足する場合の `ModuleList` path。 |
| `timestep_boundaries` | 自動 | `0.0` から `1.0` の昇順境界。要素数は `num_blocks + 1`。 |

自動境界は設定済み timestep 分布を等確率で分割します。block `0` は最大ノイズと先頭 layer、最後の block は最小ノイズと末尾 layer を担当します。

## 対応範囲

同種の Transformer block list を持つ diffusion / flow-matching family に対応します。単一 stage、joint/single stream、double/single stream、`blocks`、`layers` を検出します。

UNet、ControlNet、Musubi block swap、TwinFlow、multi-timestep scheduled sampling、固定 layer capture を使う CREPA、LayerSync は起動時に拒否されます。TREAD route は model 全体の layer index を保持し、active group の global range に clip されます。

Routing は denoiser architecture を変更します。初期 loss と出力品質が通常の full-depth run と一致するとは限りません。この option を有効にしても、既存の通常 LoRA が学習済み DiffusionBlocks adapter に変わることはありません。

`block_paths` は各 path が逐次 denoiser stage であることを確認した場合だけ指定してください。text adapter、VAE block、skip 接続を持つ UNet stage は指定しないでください。
i1 の `in_blocks`/`out_blocks` のような skip 依存 encoder-decoder Transformer stack は検出されません。output group は対応する input group の activation なしでは実行できないためです。

## メモリ

Transformer activation を作るのは active group だけです。全 block を 1 run で学習すると optimizer state は最終的に全 trainable group 分確保されます。

独立 block job では job ごとに `blocks_to_train` を指定します。担当外 group は freeze され、optimizer state を持ちません。推論前に parameter ownership に従って checkpoint を統合する必要があります。

Group offload は併用できます。Musubi block swap は併用できません。

## 推論

SimpleTuner validation は controller を自動使用します。通常の Diffusers pipeline は LoRA weight だけから変換を判断できません。

```python
from simpletuner.helpers.training.diffusion_blocks import DiffusionBlocksConfig, DiffusionBlocksController

config = DiffusionBlocksConfig.from_dict({"layers_per_block": 4, "overlap": 0.05})
controller = DiffusionBlocksController(pipe.transformer, config)
```

pipeline の有効期間中 `controller` を保持し、`simpletuner_config.json` の設定をそのまま使ってください。

## Anima 例

`simpletuner/examples/anima.peft-lora+diffusion-blocks/config.json` を参照してください。Anima v1.0 の 28 layer は `layers_per_block=4` で 7 block になります。

```bash
simpletuner train env=examples/anima.peft-lora+diffusion-blocks max_train_steps=10 validation_steps=10
```

resume では block path、layer 数、境界、`blocks_to_train`、topology、world size、batch sampling、timestep 設定を変更しないでください。推論時に全 layer を実行すると別のアーキテクチャになります。

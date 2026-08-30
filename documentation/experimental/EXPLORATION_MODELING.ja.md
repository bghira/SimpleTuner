# Explorative Modeling (XM)

Explorative Modeling、SimpleTuner では XM と呼ぶ機能は、同じ教師 sample に対して複数の hidden choice を試し、target を最もよく説明した候補だけから学習する training-time 手法です。

元の Explorative Modeling は、generative model の scaling axis として exploration を追加する考え方です。data と parameters だけでなく、training compute を使って複数候補を探索します。SimpleTuner では、対応する画像、動画、音声、autoregressive model family 向けの experimental training objective として実装されています。

Inference は変わりません。XM は training batch の作り方、採点、loss reduction だけを変えます。

## ELI5

目標画像を描く練習で、1 回だけ描かせるのではなく 4 回下書きさせ、一番よかった下書きだけを採点するイメージです。

XM の流れは単純です。

1. 同じ sample から複数候補を作る。
2. 全候補を model に通す。
3. 各候補を正解 target と比べる。
4. sample または token block ごとに最良候補を選ぶ。
5. 選ばれた loss だけで backprop する。

Target に複数の正しい説明がある場合、1 本の道だけを強制すると平均的で曖昧な解を学びやすくなります。XM は候補からよい mode を選ばせることで、その曖昧さを減らします。

## 何が変わるか

XM は新しい inference sampler、checkpoint format、teacher model を追加しません。

- 通常 training は 1 候補を sample して学習します。
- XM は `K` 候補を sample し、lowest-loss candidate から学習します。
- `K` を増やすほど探索は増えますが、training compute も増えます。

Diffusion/flow model では、候補は通常 timestep の noised latent を作る noise です。

Autoregressive token model、特に RVQ/audio planner では、候補は learned route embedding です。同じ supervised token sequence に対して複数の内部 route を与えます。

## SimpleTuner での動作

### Diffusion / Flow Models

対応する diffusion または flow matching family では `xm_training_target=noise` を使います。

SimpleTuner は次を行います。

1. 通常の timestep または sigma を sample する。
2. Batch を `xm_candidate_count` 回繰り返す。
3. 候補ごとに異なる noise tensor を作る。
4. 各 noise から noised latents を作る。
5. Expanded candidate batch を model に通す。
6. 各候補の通常 training loss を計算する。
7. 元 sample ごとに lowest-loss candidate を選ぶ。
8. 選ばれた loss で backprop する。

Model は family ごとの通常 prediction type、つまり flow velocity、epsilon、v-prediction、sample prediction を学習します。

### Autoregressive / RVQ Models

対応する autoregressive planner では `xm_training_target=route` を使います。

SimpleTuner は次を行います。

1. 候補数ぶんの小さな learned route embedding table を追加する。
2. Supervised token sequence を route candidates に展開する。
3. Route signal を model input に加える。
4. Route ごとの token loss を計算する。
5. sample 全体または configured token block ごとに最良 route を選ぶ。
6. 選ばれた route loss だけで backprop する。

これは RVQ audio codes などの discrete token stream を予測する global LM style planner に向いています。Inference-time decoding を変えずに、同じ target sequence へ複数の internal explanation を持てます。

## 疑似コード

```text
for each batch:
    candidates = []

    for candidate_id in 1..K:
        candidate_input = make_candidate(batch, candidate_id)
        prediction = model(candidate_input)
        loss = compare(prediction, target)
        candidates.append(loss)

    selected_loss = minimum_loss_per_sample_or_block(candidates)
    train_on(selected_loss)
```

Diffusion の場合:

```text
candidate_input = add_noise(clean_latent, random_noise_candidate, timestep)
loss = diffusion_or_flow_loss(model(candidate_input), training_target)
```

Autoregressive route selection の場合:

```text
candidate_input = add_route_embedding(token_sequence, route_candidate)
loss = token_loss(model(candidate_input), target_tokens)
```

## Quick Setup

### WebUI

1. **Training → Loss functions** を開く。
2. **XM** を有効にする。
3. **XM Candidates** を `2` または `4` にする。
4. **XM Training Target** を選ぶ。
   - Diffusion/flow は `noise`。
   - Autoregressive/RVQ planner は `route`。
5. Model guide が推奨しない限り **XM Selection Scope** は `sample`。
6. Route block selection を使わない限り **XM Block Size** は `0`。

### Config JSON / CLI

```json
{
  "xm_enabled": true,
  "xm_candidate_count": 4,
  "xm_training_target": "noise",
  "xm_selection_scope": "sample",
  "xm_block_size": 0
}
```

AR/RVQ route training:

```json
{
  "xm_enabled": true,
  "xm_candidate_count": 4,
  "xm_training_target": "route",
  "xm_selection_scope": "block",
  "xm_block_size": 16
}
```

## Settings

- `xm_enabled`: XM を有効化。
- `xm_candidate_count`: sample あたりの候補数。有効時は `2` 以上。
- `xm_training_target`: `noise` は diffusion/flow、`route` は autoregressive token planner。
- `xm_selection_scope`: `sample` は sample 全体、`block` は対応 family で token/frame block ごとに winner を選ぶ。
- `xm_block_size`: block-level selection の token/frame span。`0` は full supervised sequence。

## 値の選び方

| 状況 | 推奨スタート |
| --- | --- |
| Image/video diffusion LoRA | `xm_candidate_count=2`, `xm_training_target=noise`, `xm_selection_scope=sample` |
| Ambiguous dataset または大きめの batch | `xm_candidate_count=4` を試す |
| RVQ/audio planner | `xm_training_target=route`, `xm_selection_scope=block`, model guide の block size |
| 新しい family の初回 | block size `0` のまま non-XM baseline と比較 |

候補数を増やすと cost はほぼ線形に増えます。

## Logs

- `xm_loss`: 選択後の loss。
- `xm_candidate_loss_mean`: 選択前の候補平均 loss。
- `xm_candidate_0_wins`, `xm_candidate_1_wins`: 各候補が勝った回数。
- `xm_route_usage`: AR/RVQ route 使用状況。

よい兆候は、複数候補が勝つこと、validation が改善すること、route usage が長時間 collapse しないことです。

注意すべき兆候は、最初から 1 候補だけが勝つ、training loss は下がるが validation が悪化する、VRAM や step time が大きすぎる、です。

## Compatibility

現在の family-level support は [Quick Start](../QUICKSTART.ja.md) の feature table を参照してください。

一般的なルール:

- Diffusion/flow XM は noise candidates と sample-level selection。
- AR/RVQ XM は route candidates と、family により block-level selection。
- Unsupported family は option を silently ignore せず明示的に fail します。

Diffusion noise-candidate XM では、family が明示しない限り TwinFlow、Scheduled Sampling、`input_perturbation`、CREPA self-flow、stochastic segmentation masked loss は非互換です。

## 他機能との関係

- **MixFlow** は flow model の training trajectory を変えます。XM は候補選択を変えます。
- **Diff2Flow** は legacy diffusion model の target を変えます。
- **NextLat** は hidden-state dynamics を regularize します。XM は route/noise candidate を選びます。
- **LayerSync / CREPA** は representation alignment、XM は best candidate selection です。

## Practical Advice

- Baseline と比べるときは validation seed を固定する。
- VRAM が厳しければ batch size を下げる。
- Training loss だけで判断しない。validation と sample diversity を見る。
- AR/RVQ では guide がない限り block size `1` を避ける。
- 最初は短い ablation にする。

## References

- [Explorative Modeling project page](https://explorative-modeling.github.io/)
- [Explorative Modeling paper](https://arxiv.org/abs/2607.27372)

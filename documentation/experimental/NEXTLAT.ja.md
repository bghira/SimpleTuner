# NextLat

NextLat は、transformer の hidden states が次の hidden state を予測できるようにする auxiliary training objective です。

元の Next-Latent Prediction paper は language-style transformer を対象に、standard next-token prediction だけでは history を compact で stable な internal state に圧縮する圧力が弱いと説明しています。NextLat は latent space の self-supervised transition objective、つまり current hidden state から next hidden state を予測する objective を追加します。SimpleTuner では、対応 transformer family 向けの experimental regularizer として使えます。

Inference は変わりません。NextLat は training loss と小さな predictor を追加するだけです。

## ELI5

通常 training は「これまで見たものから次の output を予測しなさい」と言います。

NextLat はさらに「内部メモも、次の内部メモを予測できる形にしなさい」と言います。

画像、動画、音声 model では、この内部メモは transformer 内の hidden tokens です。Hidden-state transition が滑らかになると、tokens、frames、patches、RVQ code positions の間でより一貫した plan を学びやすくなります。

## 何が変わるか

Training 中:

1. SimpleTuner が 1 つの transformer block から hidden states を capture する。
2. Predictor が最後以外の hidden token を受け取る。
3. 次の hidden token を予測する。
4. 実際の次 hidden token は gradient なしの target になる。
5. Auxiliary loss が通常 training loss に追加される。

Base model は通常 objective で学習し続けます。NextLat は internal state に predictive dynamics を持たせる side objective です。

## 疑似コード

```text
for each batch:
    prediction = model(batch)
    main_loss = normal_training_loss(prediction, target)

    hidden = captured_hidden_states
    current = hidden tokens 0..N-2
    next = hidden tokens 1..N-1

    predicted_next = nextlat_predictor(current)
    nextlat_loss = distance(predicted_next, stop_gradient(next))

    total_loss = main_loss + nextlat_weight * nextlat_loss
    train_on(total_loss)
```

Family が compatible logits head を提供する場合は optional KL も使えます。

```text
pred_logits = logits_head(predicted_next)
target_logits = logits_head(stop_gradient(next))
total_loss += nextlat_kl_weight * agreement_loss(pred_logits, target_logits)
```

通常は `nextlat_kl_weight=0` のままにしてください。

## SimpleTuner での動作

- Hidden states を expose する transformer family で動作します。
- `nextlat_block_index` で選んだ 1 block を capture します。
- `-1` は最後の supported block です。
- Image/video/audio/token hidden states を sequence に flatten します。
- Hidden-token order で 1 step ahead を予測します。
- Target hidden state は detached です。
- Training mode が保存に対応している場合、predictor は extra trainable module として保存されます。

Model guide が別の adapter mode を明記しない限り、standard PEFT LoRA または full-model training を使ってください。

## Quick Setup

### WebUI

1. **Training → Loss functions** を開く。
2. **NextLat** を有効にする。
3. 初回は **NextLat Block Index** を `-1` のままにする。
4. **NextLat Weight** に小さい正の値を入れる。
5. **NextLat State Loss** は `smooth_l1`。
6. **NextLat KL Weight** は guide がない限り `0`。

### Config JSON / CLI

```json
{
  "nextlat_enabled": true,
  "nextlat_block_index": -1,
  "nextlat_weight": 0.05,
  "nextlat_state_loss": "smooth_l1",
  "nextlat_kl_weight": 0.0
}
```

## Settings

- `nextlat_enabled`: NextLat を有効化。
- `nextlat_block_index`: zero-based transformer block。`-1` は最後の supported block。
- `nextlat_weight`: auxiliary hidden-state prediction loss の multiplier。有効時は 0 より大きい必要があります。
- `nextlat_state_loss`: `smooth_l1` または `mse`。
- `nextlat_kl_weight`: compatible logits head がある場合の optional KL weight。

## 値の選び方

| 状況 | 推奨スタート |
| --- | --- |
| 初回 transformer LoRA | `nextlat_block_index=-1`, `nextlat_weight=0.02` から `0.05` |
| AR/RVQ planner | late block、`smooth_l1`、小さい weight |
| Video transformer | final block が強すぎる場合は middle-to-late block |
| Auxiliary loss が不安定 | block より先に `nextlat_weight` を下げる |
| KL 推奨あり | model guide の値だけを使う |

## Logs

- `nextlat_loss`: training objective に加算された weighted auxiliary loss。
- `nextlat_state_loss`: raw hidden-state prediction loss。
- `nextlat_kl_loss`: optional KL term。

Raw state loss は trend を見るための値で、main loss と同じ scale である必要はありません。

## Compatibility

現在の support は [Quick Start](../QUICKSTART.ja.md) の feature table を参照してください。

要件:

- Model が transformer hidden states を expose する。
- 選んだ block が存在し capture できる。
- Captured sequence に hidden token が 2 個以上ある。
- Training mode が NextLat predictor を保存できる。

NextLat は LayerSync、Internal Guidance、CREPA など hidden-state capture を使う機能と自然に併用できますが、auxiliary loss まで hidden states を保持するため VRAM は増えます。

## 期待できる場面

NextLat は internal transition の一貫性が重要な場合に向いています。例: RVQ/audio code planner、temporal structure を持つ video transformer、spatial token order が効く image transformer、stable internal plan が必要な multimodal model。

非常に小さい experiment、weight が main loss を支配する設定、有用な hidden states を expose しない family では効果が弱いことがあります。

## Practical Advice

- まず短い ablation run を行う。
- `nextlat_weight` は低く始める。
- 特別な理由がなければ `smooth_l1`。
- まず `-1`、必要なら middle-to-late block を試す。
- KL は model guide がある場合だけ有効化する。
- VRAM が厳しければ batch size を下げるか他の hidden-state regularizer を切る。

## References

- [Next-Latent Prediction paper](https://arxiv.org/abs/2511.05963)
- [NextLat reference code](https://github.com/JaydenTeoh/NextLat)

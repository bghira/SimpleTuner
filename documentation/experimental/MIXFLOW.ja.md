# MixFlow Training

MixFlow は flow-matching モデル向けの post-training 手法です。時刻 $t$ のモデルを、よりノイズの多い ground-truth interpolation で学習し、学習時の正確な interpolation と sampling 時の不完全な latent の差を縮めます。

## 設定

```json
{
  "mixflow_enabled": true,
  "mixflow_gamma": 0.8
}
```

`mixflow_gamma` は slowed interpolation の範囲を制御します。論文の既定値は `0.8` です。`0.0` は標準 interpolation を維持しつつ MixFlow の timestep sampling を使用します。

MixFlow は data-ward の model timestep を $Beta(2,1)$ から sampling します。SimpleTuner の flow sigma は逆向きの noise-ward 表現なので、$sigma = 1 - sqrt(U)$ を sampling してからモデル設定の flow schedule shift を適用します。モデルには元の timestep を渡し、latent input には次を使います。

$$
sigma_{input} = sigma + U' gamma (1 - sigma)
$$

線形 flow path の velocity target と inference は変更しません。

## サポート

prediction type が `flow_matching` のすべての SimpleTuner model family が共通 MixFlow path を使用します。data-ward timestep、非線形 sigma transform、audio/video joint input は各 model wrapper が処理します。

MixFlow は custom/uniform/Beta/fast flow schedule、Self-Flow、TwinFlow、scheduled sampling、distillation と併用できません。schedule shift は利用できます。

既存 flow model の post-training として使用してください。短い通常継続学習と同じ learning rate と optimizer から始め、固定 seed の validation sample を開始 checkpoint と比較します。

## 参照

- [MixFlow 論文](https://arxiv.org/abs/2512.19311)
- [Reference implementation](https://github.com/fudan-generative-vision/MixFlow)

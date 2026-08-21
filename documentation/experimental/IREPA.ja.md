# iREPA

iREPA は alignment path の空間構造を保持して representation alignment を改善します。token ごとの linear projector を spatial convolution に置き換え、各画像の patch 次元で teacher feature を channel ごとに z-score 正規化します。

SimpleTuner は backbone に応じて既存の alignment engine を使います。Transformer image model は REPA/CREPA、Transformer video model は各 frame に iREPA を独立適用した上で CREPA の temporal-neighbour loss、UNet image model は U-REPA の mid-block と manifold loss を使います。矩形 token grid は clean latent shape から復元され、square bucket は不要です。

```json
{
  "irepa_enabled": true,
  "irepa_spatial_norm_alpha": 0.6,
  "irepa_projector_kernel_size": 3,
  "crepa_enabled": true,
  "crepa_block_index": 8,
  "crepa_lambda": 1.0
}
```

Transformer では iREPA と `crepa_enabled`、UNet では iREPA と `urepa_enabled` を有効にします。対応する `crepa_*` / `urepa_*` options が teacher、weight、capture layer、schedule を制御します。`0.6` は latent-diffusion reference recipe、kernel size `3` は paper の構成です。

iREPA には spatial patch token を持つ hidden state と grid 復元用の clean latent が必要です。video convolution は frame 間を混合しません。

Full-model または standard PEFT LoRA training を使用します。LyCORIS は補助 projector を保存できないため未対応です。

参照: [What Matters for Representation Alignment: Global Information or Spatial Structure?](https://arxiv.org/abs/2512.10794)

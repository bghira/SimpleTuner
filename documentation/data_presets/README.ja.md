# データセット設定プリセット

Hugging Face Hub 上の大規模データセット向けに、動作させるための手がかりとなる設定情報をまとめています。

> **ヒント：** 大規模な正則化データセットの場合、`max_num_samples` を使用してデータセットを決定論的なランダムサブセットに制限できます。詳細は [DATALOADER.md](../DATALOADER.md#max_num_samples) を参照してください。

新しいプリセットを追加する場合は、[このテンプレート](../data_presets/preset.md) を使って新しいプルリクエストを提出してください。

- [DALLE-3 1M](../data_presets/preset_dalle3.md)
- [bghira/photo-concept-bucket](../data_presets/preset_pexels.md)
- [Midjourney v6 520k](../data_presets/preset_midjourney.md)
- [Nijijourney v6 520k](../data_presets/preset_nijijourney.md)
- [Subjects200K](../data_presets/preset_subjects200k.md): `datasets` ライブラリの利用例

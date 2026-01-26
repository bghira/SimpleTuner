# クイックスタートガイド

**注意**: より高度な設定については、[チュートリアル](TUTORIAL.md)および[オプションリファレンス](OPTIONS.md)を参照してください。

## 機能互換性

完全かつ最も正確な機能マトリックスについては、[メインREADME](https://github.com/bghira/SimpleTuner#model-architecture-support)を参照してください。

## モデルクイックスタートガイド

| モデル | パラメータ数 | PEFT LoRA | Lycoris | Full-Rank | 量子化 | 混合精度 | Grad Checkpoint | Flow Shift | TwinFlow | LayerSync | ControlNet | Sliders† | ガイド |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| PixArt Sigma | 0.6B–0.9B | ✗ | ✓ | ✓ | int8 オプション | bf16 | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | [SIGMA.md](quickstart/SIGMA.md) |
| NVLabs Sana | 1.6B–4.8B | ✗ | ✓ | ✓ | int8 オプション | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [SANA.md](quickstart/SANA.md) |
| Kwai Kolors | 2.7B | ✓ | ✓ | ✓ | 非推奨 | bf16 | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | [KOLORS.md](quickstart/KOLORS.md) |
| Stable Diffusion 3 | 2B–8B | ✓ | ✓ | ✓ | int8/fp8/nf4 オプション | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✓ | [SD3.md](quickstart/SD3.md) |
| Flux.1 | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 オプション | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ | [FLUX.md](quickstart/FLUX.md) |
| Flux.2 | 32B | ✓ | ✓ | ✓* | int8/fp8/nf4 オプション | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [FLUX2.md](quickstart/FLUX2.md) |
| Flux Kontext | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 オプション | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ | [FLUX_KONTEXT.md](quickstart/FLUX_KONTEXT.md) |
| Z-Image Turbo | 6B | ✓ | ✗ | ✓* | int8 オプション | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [ZIMAGE.md](quickstart/ZIMAGE.md) |
| ACE-Step | 3.5B | ✓ | ✓ | ✓* | int8 オプション | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [ACE_STEP.md](quickstart/ACE_STEP.md) |
| Chroma 1 | 8.9B | ✓ | ✓ | ✓* | int8/fp8/nf4 オプション | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [CHROMA.md](quickstart/CHROMA.md) |
| Auraflow | 6B | ✓ | ✓ | ✓* | int8/fp8/nf4 オプション | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✓ | [AURAFLOW.md](quickstart/AURAFLOW.md) |
| HiDream I1 | 17B (8.5B MoE) | ✓ | ✓ | ✓* | int8/fp8/nf4 オプション | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | [HIDREAM.md](quickstart/HIDREAM.md) |
| OmniGen | 3.8B | ✓ | ✓ | ✓ | int8/fp8 オプション | bf16 | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | [OMNIGEN.md](quickstart/OMNIGEN.md) |
| Stable Diffusion XL | 2.6B | ✓ | ✓ | ✓ | 非推奨 | bf16 | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | [SDXL.md](quickstart/SDXL.md) |
| Lumina2 | 2B | ✓ | ✓ | ✓ | int8 オプション | bf16 | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [LUMINA2.md](quickstart/LUMINA2.md) |
| Cosmos2 | 2B | ✓ | ✓ | ✓ | 非推奨 | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [COSMOS2IMAGE.md](quickstart/COSMOS2IMAGE.md) |
| LTX Video | ~2.5B | ✓ | ✓ | ✓ | int8/fp8 オプション | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [LTXVIDEO.md](quickstart/LTXVIDEO.md) |
| Hunyuan Video 1.5 | 8.3B | ✓ | ✓ | ✓* | int8 オプション | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [HUNYUANVIDEO.md](quickstart/HUNYUANVIDEO.md) |
| Wan 2.x | 1.3B–14B | ✓ | ✓ | ✓* | int8 オプション | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [WAN.md](quickstart/WAN.md) |
| Qwen Image | 20B | ✓ | ✓ | ✓* | **必須** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [QWEN_IMAGE.md](quickstart/QWEN_IMAGE.md) |
| Qwen Image Edit | 20B | ✓ | ✓ | ✓* | **必須** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [QWEN_EDIT.md](quickstart/QWEN_EDIT.md) |
| Stable Cascade (C) | 1B, 3.6B prior | ✓ | ✓ | ✓* | 非対応 | fp32 (必須) | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | [STABLE_CASCADE_C.md](quickstart/STABLE_CASCADE_C.md) |
| Kandinsky 5.0 Image | 6B (lite) | ✓ | ✓ | ✓* | int8 オプション | bf16 | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [KANDINSKY5_IMAGE.md](quickstart/KANDINSKY5_IMAGE.md) |
| Kandinsky 5.0 Video | 2B (lite), 19B (pro) | ✓ | ✓ | ✓* | int8 オプション | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [KANDINSKY5_VIDEO.md](quickstart/KANDINSKY5_VIDEO.md) |
| LongCat-Video | 13.6B | ✓ | ✓ | ✓* | int8/fp8 オプション | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_VIDEO.md](quickstart/LONGCAT_VIDEO.md) |
| LongCat-Video Edit | 13.6B | ✓ | ✓ | ✓* | int8/fp8 オプション | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_VIDEO_EDIT.md](quickstart/LONGCAT_VIDEO_EDIT.md) |
| LongCat-Image | 6B | ✓ | ✓ | ✓* | int8/fp8 オプション | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_IMAGE.md](quickstart/LONGCAT_IMAGE.md) |
| LongCat-Image Edit | 6B | ✓ | ✓ | ✓* | int8/fp8 オプション | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_EDIT.md](quickstart/LONGCAT_EDIT.md) |

*✓ = サポート、✓* = Full-RankにはDeepSpeed/FSDP2が必要、✗ = 非サポート、`✓+`はVRAMプレッシャーによりチェックポイントが推奨されることを示します。TwinFlow ✓は`twinflow_enabled=true`のときのネイティブサポートを意味します（拡散モデルには`diff2flow_enabled+twinflow_allow_diff2flow`が必要）。LayerSync ✓はバックボーンがセルフアライメント用のトランスフォーマー隠れ状態を公開していることを意味し、✗はそのバッファを持たないUNetスタイルのバックボーンを示します。†SlidersはLoRAおよびLyCORIS（Full-Rank LyCORIS "full"を含む）に適用されます。*

> ℹ️ Wanクイックスタートには2.1 + 2.2ステージプリセットと時間埋め込みトグルが含まれます。Flux Kontextは、Flux.1をベースに構築された編集ワークフローをカバーします。

> ⚠️ これらのクイックスタートは生きたドキュメントです。新しいモデルの登場やトレーニングレシピの改善に伴い、時折更新されることがあります。

### 高速パス: Z-Image TurboとFlux Schnell

- **Z-Image Turbo**: TREADを使用した完全サポートのLoRA。量子化なし（int8も動作）でもNVIDIAとmacOSで高速に動作します。多くの場合、ボトルネックはトレーナーのセットアップだけです。
- **Flux Schnell**: クイックスタート設定が高速ノイズスケジュールとアシスタントLoRAスタックを自動的に処理します。Schnell LoRAをトレーニングするための追加フラグは不要です。

### 高度な実験的機能

- **Diff2Flow**: 標準的なepsilon/v-predictionモデル（SD1.5、SDXL、DeepFloydなど）をFlow Matching損失目的関数を使用してトレーニングできます。これにより、古いアーキテクチャと最新のフローベースのトレーニング間のギャップを埋めます。
- **Scheduled Sampling**: トレーニング中にモデル自身に中間ノイズ潜在変数を生成させる（「ロールアウト」）ことで露出バイアスを軽減します。これにより、モデルが自身の生成エラーから回復する方法を学習できます。

## よくある問題

### データセットのサンプル数が予想より少ない

データセットの使用可能なサンプル数が予想より少ない場合、処理中にファイルがフィルタされた可能性があります。一般的な理由：

- **ファイルが小さすぎる**: `minimum_image_size` 未満の画像はフィルタされます
- **アスペクト比が範囲外**: `minimum_aspect_ratio`/`maximum_aspect_ratio` の範囲外の画像は除外されます
- **時間制限**: 時間制限を超えるオーディオ/ビデオファイルはスキップされます

**フィルタリング統計の確認:**
- WebUI でデータセットディレクトリを参照し、選択するとフィルタリング統計が表示されます
- データセット処理中のログで次のような統計を確認: `Sample processing statistics: {'total_processed': 100, 'skipped': {'too_small': 15, ...}}`

詳細なトラブルシューティングについては、データローダードキュメントの[フィルタされたデータセットのトラブルシューティング](DATALOADER.ja.md#フィルタされたデータセットのトラブルシューティング)を参照してください。

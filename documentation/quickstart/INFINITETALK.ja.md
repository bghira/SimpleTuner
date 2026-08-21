# InfiniteTalk クイックスタート

InfiniteTalk は Wan 2.1 I2V 14B を基盤にした音声駆動動画モデルです。SimpleTuner は Wan 基盤を読み込み、公式の音声プロジェクターと 40 ブロックの音声アテンションを重ねます。

この統合は公式の単一話者モデルを学習します。複数話者には複数の同期音声と話者マスクが必要ですが、現在のデータローダーは動画ごとに 1 音声を表現します。

## 要件

- bf16 対応 NVIDIA GPU
- RAM 64 GB。RamTorch または非量子化ロードでは 96 GB 以上
- `ffmpeg`
- 25 fps で音声が同期した動画

```bash
python -m venv .venv
source .venv/bin/activate
pip install 'simpletuner[cuda]'
```

各例は `trust_remote_code: true` で固定済みの `kernels-community/flash-attn3` Hub カーネルを許可します。ローカルまたは組み込みバックエンドを選ぶ場合は削除してください。

## 初期プロファイル

| VRAM | フレーム | 重み | 配置 | 例 |
| --- | ---: | --- | --- | --- |
| 24 GB | 17 | bf16 | 全ブロック RamTorch | `infinitetalk-14b-480p-24gb.peft-lora` |
| 32 GB | 17 | int8 TorchAO | 20 ブロック交換 | `infinitetalk-14b-480p-32gb.peft-lora` |
| 48 GB | 33 | bf16 | 24 ブロック交換 | `infinitetalk-14b-480p-48gb.peft-lora` |
| 80 GB | 49 | bf16 | 常駐 | `infinitetalk-14b-480p-80gb.peft-lora` |

## データ

`clip-001.mp4` と `clip-001.txt` のように動画と説明文を並べます。同梱設定は 16 kHz モノラル音声を自動抽出します。

```json
"audio": {"auto_split": true, "sample_rate": 16000, "channels": 1}
```

- 25 fps を使用します。
- フレーム数は `4k + 1`、例: 17、33、49。
- 音声は動画クリップと同じ区間を覆う必要があります。
- ランダム時間クロップに曲全体の音声を組み合わせないでください。
- 音声なしクリップは拒否されます。

## 学習

```bash
simpletuner train \
  --config simpletuner/examples/infinitetalk-14b-480p-80gb.peft-lora/config.json
```

```json
{
  "model_family": "infinitetalk",
  "model_flavour": "single-14b-480p",
  "pretrained_model_name_or_path": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
  "framerate": 25
}
```

メモリを減らす順序は、フレーム削減、`musubi_blocks_to_swap` 増加、int8 TorchAO、RamTorch です。音声アテンションはフレーム境界に依存するため、TREAD とコンテキスト並列は未対応です。

検証には画像と音声が必要です。内蔵検証はテキスト CFG の両分岐で音声を保持します。テキストと音声を分離した CFG の比較には公式実装を使用してください。

LoRA、LyCORIS、フル学習、アダプター用量子化、チェックポイント、ブロック交換、RamTorch、FFN 分割、CREPA、LayerSync に対応します。複数話者学習は未対応です。

資料: [コード](https://github.com/MeiGen-AI/InfiniteTalk)、[論文](https://arxiv.org/abs/2508.14033)、[重み](https://huggingface.co/MeiGen-AI/InfiniteTalk)。

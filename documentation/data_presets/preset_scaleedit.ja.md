# ScaleEdit-12M：Webshart メタデータを使うペア Parquet 画像

## 詳細

- **ソースデータセット**：[InternVL-U/ScaleEdit-12M](https://huggingface.co/datasets/InternVL-U/ScaleEdit-12M)。
- **メタデータ補助リポジトリ**：[webshart/ScaleEdit-12M-metadata](https://huggingface.co/datasets/webshart/ScaleEdit-12M-metadata)。
- **説明**：251 個の Parquet シャード、23 タスク分類にわたる 11,845,023 行の画像編集データです。補助メタデータは約 423 MB で、画像バイトは上流データセットに残ります。
- **キャプション形式**：`edit_instruction` 文字列。編集指示であり、単独の情景描写ではありません。
- **ライセンス**：ソース由来の CC-BY-NC-SA-4.0。画像の権利や利用制限は上流カードを参照してください。

## 必要な前処理

tar 展開、画像ごとのキャプションファイル作成、メタデータ生成は不要です。[PR #3325](https://github.com/bghira/SimpleTuner/pull/3325) を含む SimpleTuner の [Webshart Parquet バックエンド](../DATALOADER.md#webshart-datasets)を使い、公開済み Hub 連携をインストールします。

```bash
python -m pip install --upgrade "webshart[hub]>=0.8.0"
```

公開 Hub へのアクセスにトークンは不要です。認証する場合は `HF_TOKEN` を設定できます。補助マニフェストは上流を `05d2eea7b80ba72c077d39be43b7aca1a7857e6a` に固定し、メタデータのチェックサムを検証します。

## データセット構造

各ソース Parquet 行には `source_image`、`edited_image`、`edit_instruction`、両画像の寸法と品質スコアが含まれます。`webshart-parquet.json` がソースと `metadata/<task>/<part>.parquet` を対応付けます。メタデータに画像本体は含まれません。`id` はタスク間で重複するため、結合にはソースパスと物理行番号を使います。

## データローダー設定例

この配列を `multidatabackend.json` に記述します。例は 458 ペアの `3.4_count_change` から開始し、ソースは約 54 MB の単一シャードです。この固定版サブセットでは、すべてのソース画像と編集画像が埋め込まれています。[Flux Kontext](../quickstart/FLUX_KONTEXT.md) など `reference_strict` 対応の画像編集モデルを使ってください。モデルの学習設定は別途必要です。

```json
[
  {
    "id": "scaleedit-edits",
    "type": "webshart",
    "source": "InternVL-U/ScaleEdit-12M",
    "metadata": "webshart/ScaleEdit-12M-metadata",
    "subfolder": "3.4_count_change",
    "caption_strategy": "webshart",
    "metadata_backend": "webshart",
    "resolution": 512,
    "resolution_type": "pixel",
    "crop": false,
    "dataset_type": "image",
    "conditioning_data": [
      "scaleedit-sources"
    ],
    "cache_dir_vae": "cache/vae/scaleedit/edited_image",
    "webshart": {
      "format": "parquet",
      "cache_dir": "cache/webshart/scaleedit/edited_image",
      "shard_cache_gb": 0,
      "parallel_downloads": 4,
      "columns": {
        "image": "edited_image",
        "captions": "edit_instruction",
        "width": "edited_image_width",
        "height": "edited_image_height"
      }
    }
  },
  {
    "id": "scaleedit-sources",
    "type": "webshart",
    "source": "InternVL-U/ScaleEdit-12M",
    "metadata": "webshart/ScaleEdit-12M-metadata",
    "subfolder": "3.4_count_change",
    "caption_strategy": "webshart",
    "metadata_backend": "webshart",
    "resolution": 512,
    "resolution_type": "pixel",
    "crop": false,
    "dataset_type": "conditioning",
    "conditioning_type": "reference_strict",
    "source_dataset_id": "scaleedit-edits",
    "cache_dir_vae": "cache/vae/scaleedit/source_image",
    "webshart": {
      "format": "parquet",
      "cache_dir": "cache/webshart/scaleedit/source_image",
      "shard_cache_gb": 0,
      "parallel_downloads": 4,
      "columns": {
        "image": "source_image",
        "captions": "edit_instruction",
        "width": "source_image_width",
        "height": "source_image_height"
      }
    }
  },
  {
    "id": "scaleedit-text",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/scaleedit"
  }
]
```

## 設定の説明

- `scaleedit-edits` は `edited_image` を学習ターゲット、`edit_instruction` をキャプションとして読みます。寸法列は編集後画像に対応します。
- `scaleedit-sources` は `source_image` を条件画像として読みます。`conditioning_data` はターゲットから条件セットへ、`source_dataset_id` は参照からターゲットへ接続します。ソース寸法列も明示します。
- 両エントリで同じソース、補助リポジトリ、`subfolder` を使い、パート/行の識別子を揃えます。VAE と Webshart のキャッシュ先は分けてください。
- `webshart.shard_cache_gb: 0` は範囲読み込みを保ちます。全体の多くのシャードは約 31 GB で、全シャードキャッシュを有効にすると最初の画像取得でシャード全体をダウンロードする場合があります。
- この初期設定では `crop: false` で独立したクロップを避けます。`reference_strict` はターゲットの準備時ジオメトリを参照画像にも使います。編集モデルに合う前処理を維持してください。

## より大きな範囲への拡張

`subfolder` を変更・削除する前に、全体のうち URL のみで提供される 702,078 枚のソース画像を処理してください。補助データは URL、SHA-256、取得日を保持しますが取得は行いません。埋め込み `source_image` がなければ `None` のままです。明示的に取得・検証するか、パスと行番号で揃えたペアのサブセットを作成します。

ターゲットと参照を別々に制限・フィルタリングしないでください。選ばれるサンプルを揃える必要があります。バケットが異なる場合、同じ `max_num_samples` でも同じペアになる保証はありません。

全シャードキャッシュを無効にしても、大きな行グループや辞書の画像読み込みには数百 MiB が必要になる場合があります。メタデータのバケット作成は補助データを使いますが、画像には別のメモリ・通信コストがあります。大規模実行前に [Webshart ローダーベンチマーク](../DATALOADER.md#parquet-image-bytes-on-s3) で限定サンプルを測定できます。

# Hugging Face Datasets 連携

SimpleTuner は Hugging Face Hub からデータセットを直接読み込むことができ、大規模データセットをローカルに全ダウンロードせずに効率的に学習できます。

## 概要

Hugging Face データセットバックエンドでは次が可能です:
- Hugging Face Hub からデータセットを直接読み込み
- メタデータや品質指標に基づくフィルタリング
- データセット列からキャプションを抽出
- 合成/グリッド画像の処理
- 処理済み埋め込みのみをローカルにキャッシュ

**重要**: SimpleTuner はアスペクト比バケット作成とバッチサイズ計算のためにデータセット全体へのアクセスが必要です。Hugging Face はストリーミングをサポートしていますが、SimpleTuner の構成とは互換性がありません。非常に大きいデータセットはフィルタで適切な規模に絞ってください。

## 基本設定

Hugging Face データセットを使用するには、データローダで `"type": "huggingface"` を指定します:

```json
{
  "id": "my-hf-dataset",
  "type": "huggingface",
  "dataset_name": "username/dataset-name",
  "split": "train",
  "caption_strategy": "huggingface",
  "metadata_backend": "huggingface",
  "caption_column": "text",
  "image_column": "image",
  "cache_dir": "cache/my-hf-dataset"
}
```

### 必須項目

- `type`: `"huggingface"` であること
- `dataset_name`: Hugging Face データセット識別子（例: "laion/laion-aesthetic"）
- `caption_strategy`: `"huggingface"` であること
- `metadata_backend`: `"huggingface"` であること

### 任意項目

- `split`: 使用するデータセット分割（既定: "train"）
- `revision`: 特定のデータセットリビジョン
- `image_column`: 画像列（既定: "image"）
- `caption_column`: キャプション列
- `cache_dir`: データセットファイルのローカルキャッシュディレクトリ
- `streaming`: ⚠️ **現在は機能しません** - SimpleTuner はメタデータとエンコーダキャッシュを効率的に構築するためにデータセットをスキャンします
- `num_proc`: フィルタ処理のプロセス数（既定: 16）

## キャプション設定

Hugging Face バックエンドは柔軟なキャプション抽出をサポートします:

### 単一キャプション列
```json
{
  "caption_column": "caption"
}
```

### 複数キャプション列
```json
{
  "caption_column": ["short_caption", "detailed_caption", "tags"]
}
```

### ネストされた列アクセス
```json
{
  "caption_column": "metadata.caption",
  "fallback_caption_column": "basic_caption"
}
```

### 高度なキャプション設定
```json
{
  "huggingface": {
    "caption_column": "caption",
    "fallback_caption_column": "description",
    "description_column": "detailed_description",
    "width_column": "width",
    "height_column": "height"
  }
}
```

## データセットのフィルタリング

高品質サンプルのみを選択するためにフィルタを適用します:

### 品質ベースのフィルタ
```json
{
  "huggingface": {
    "filter_func": {
      "quality_thresholds": {
        "clip_score": 0.3,
        "aesthetic_score": 5.0,
        "resolution": 0.8
      },
      "quality_column": "quality_assessment"
    }
  }
}
```

### コレクション/サブセットのフィルタ
```json
{
  "huggingface": {
    "filter_func": {
      "collection": ["photo", "artwork"],
      "min_width": 512,
      "min_height": 512
    }
  }
}
```

## 合成画像サポート

グリッド内に複数画像を含むデータセットを処理できます:

```json
{
  "huggingface": {
    "composite_image_config": {
      "enabled": true,
      "image_count": 4,
      "select_index": 0
    }
  }
}
```

この設定で行われること:
- 4 画像グリッドを検出
- 最初の画像（index 0）のみ抽出
- それに合わせて寸法を調整

## 完全な設定例

### 基本の写真データセット
```json
{
  "id": "aesthetic-photos",
  "type": "huggingface",
  "dataset_name": "aesthetic-foundation/aesthetic-photos",
  "split": "train",
  "caption_strategy": "huggingface",
  "metadata_backend": "huggingface",
  "caption_column": "caption",
  "image_column": "image",
  "resolution": 1024,
  "resolution_type": "pixel",
  "minimum_image_size": 512,
  "cache_dir": "cache/aesthetic-photos"
}
```

### フィルタ済み高品質データセット
```json
{
  "id": "high-quality-art",
  "type": "huggingface",
  "dataset_name": "example/art-dataset",
  "caption_strategy": "huggingface",
  "metadata_backend": "huggingface",
  "huggingface": {
    "caption_column": ["title", "description", "tags"],
    "fallback_caption_column": "filename",
    "width_column": "original_width",
    "height_column": "original_height",
    "filter_func": {
      "quality_thresholds": {
        "aesthetic_score": 6.0,
        "technical_quality": 0.8
      },
      "min_width": 768,
      "min_height": 768
    }
  },
  "resolution": 1024,
  "resolution_type": "pixel_area",
  "crop": true,
  "crop_aspect": "square"
}
```

### 動画データセット
```json
{
  "id": "video-dataset",
  "type": "huggingface",
  "dataset_type": "video",
  "dataset_name": "example/video-clips",
  "caption_strategy": "huggingface",
  "metadata_backend": "huggingface",
  "huggingface": {
    "caption_column": "description",
    "num_frames_column": "frame_count",
    "fps_column": "fps"
  },
  "video": {
    "num_frames": 125,
    "min_frames": 100
  },
  "resolution": 480,
  "resolution_type": "pixel"
}
```

## 仮想ファイルシステム

Hugging Face バックエンドは仮想ファイルシステムを使用し、画像はデータセットのインデックスで参照されます:
- `0.jpg` → データセットの最初の項目
- `1.jpg` → データセットの 2 番目の項目
- 以降同様

これにより SimpleTuner の標準パイプラインを変更せずに利用できます。

## キャッシュ挙動

- **データセットファイル**: Hugging Face datasets の既定に従ってキャッシュ
- **VAE 埋め込み**: `cache_dir/vae/{backend_id}/` に保存
- **テキスト埋め込み**: 標準のテキスト埋め込みキャッシュ設定を使用
- **メタデータ**: `cache_dir/huggingface_metadata/{backend_id}/` に保存

## パフォーマンス上の考慮事項

1. **初回スキャン**: 初回実行でデータセットメタデータをダウンロードし、アスペクト比バケットを作成
2. **データセットサイズ**: ファイル一覧と長さ計算のために全メタデータを読み込む必要があります
3. **フィルタ**: 初回ロード時に適用され、フィルタ後の項目はダウンロードされません
4. **キャッシュ再利用**: 2 回目以降はキャッシュされたメタデータと埋め込みを再利用

**注記**: Hugging Face datasets はストリーミングをサポートしますが、SimpleTuner はアスペクトバケット作成とバッチサイズ計算のために全データセットアクセスが必要です。非常に大きなデータセットはフィルタで適切な規模に絞ってください。

## 制限事項

- 読み取り専用（ソースデータセットは変更できません）
- 初回アクセスにはインターネット接続が必要
- 一部のデータセット形式は未対応の場合あり
- ストリーミングモードは未対応（SimpleTuner は全データセットアクセスが必要）
- 超大規模データセットはフィルタで適切な規模にする必要があります
- 初回メタデータ読み込みは巨大データセットでメモリ負荷が高い場合があります

## トラブルシューティング

### データセットが見つからない
```
Error: Dataset 'username/dataset' not found
```
- データセットが Hugging Face Hub に存在するか確認
- データセットがプライベートの場合は認証が必要
- データセット名の綴りを確認

### 初回読み込みが遅い
- 大規模データセットはメタデータ読み込みとバケット作成に時間がかかります
- 厳しめのフィルタでデータセットサイズを削減
- サブセットやフィルタ済みデータセットの使用を検討
- キャッシュファイルにより 2 回目以降は高速化

### メモリ問題
- フィルタでデータセットを縮小してから読み込み
- フィルタ処理の `num_proc` を減らす
- 非常に大規模なデータセットは小さなチャンクに分割
- 品質閾値で高品質サンプルに限定

### キャプション抽出の問題
- 列名がデータセットスキーマと一致しているか確認
- ネストされた列構造を確認
- 欠損キャプションに `fallback_caption_column` を使用

## 高度な使い方

### カスタムフィルタ関数

設定は基本的なフィルタに対応していますが、より複雑なフィルタはコードを修正して実装できます。フィルタ関数は各データ項目を受け取り、True/False を返します。

### 複数データセット学習

Hugging Face データセットとローカルデータを組み合わせる例:

```json
[
  {
    "id": "hf-dataset",
    "type": "huggingface",
    "dataset_name": "laion/laion-art",
    "probability": 0.7
  },
  {
    "id": "local-dataset",
    "type": "local",
    "instance_data_dir": "/path/to/local/data",
    "probability": 0.3
  }
]
```

この設定では Hugging Face データセットから 70%、ローカルデータから 30% をサンプリングします。

## 音声データセット

ACE-Step のような音声モデルでは `dataset_type: "audio"` を指定できます。

```json
{
    "id": "audio-dataset",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "my-audio-data",
    "audio_column": "audio",
    "config": {
        "audio_caption_fields": ["tags"],
        "lyrics_column": "lyrics"
    }
}
```

*   **`audio_column`**: 音声データの列（デコード済みまたは bytes）。既定は `"audio"`。
*   **`audio_caption_fields`**: **プロンプト**（テキスト条件）を作るために結合する列名。既定は `["prompt", "tags"]`。
*   **`lyrics_column`**: 歌詞の列。既定は `"lyrics"`。欠損する場合、SimpleTuner は `"norm_lyrics"` をフォールバックとして確認します。

### 想定される列
*   **`audio`**: 音声データ。
*   **`prompt`** / **`tags`**: テキストエンコーダ用の説明タグやプロンプト。
*   **`lyrics`**: 歌詞エンコーダ用の歌詞。

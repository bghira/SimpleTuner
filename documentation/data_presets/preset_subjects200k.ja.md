# Subjects200K

## 詳細

- **Hub リンク**: [Yuanshi/Subjects200K](https://huggingface.co/datasets/Yuanshi/Subjects200K)
- **説明**: 20 万枚以上の高品質な合成画像と、ペアの説明文・品質評価を含むデータセット。各サンプルは同一被写体を異なる文脈で並べた 2 枚の画像を含みます。
- **キャプション形式**: 画像の左右それぞれに別説明を持つ構造化 JSON
- **特長**: 品質評価スコア、コレクションタグ、合成画像

## データセット構造

Subjects200K がユニークな理由:
- 各 `image` フィールドには **左右に並べた 2 枚の画像** が 1 枚の横長画像として格納されます
- 各サンプルには **2 つのキャプション** があり、左画像は `description.description_0`、右画像は `description.description_1` です
- 品質評価メタデータにより、画像品質指標でフィルタできます
- 画像はコレクション単位で整理されています

データ構造例:
```python
{
    'image': PIL.Image,  # Combined image (e.g., 1056x528 for two 528x528 images)
    'collection': 'collection_1',
    'quality_assessment': {
        'compositeStructure': 5,
        'objectConsistency': 5,
        'imageQuality': 5
    },
    'description': {
        'item': 'Eames Lounge Chair',
        'description_0': 'The Eames Lounge Chair is placed in a modern city living room...',
        'description_1': 'Nestled in a cozy nook of a rustic cabin...',
        'category': 'Furniture',
        'description_valid': True
    }
}
```

## 直接利用（前処理不要）

抽出や前処理が必要なデータセットと異なり、Subjects200K は HuggingFace から直接使えます。すでに適切な形式でホストされています。

`datasets` ライブラリが入っていることを確認してください:
```bash
pip install datasets
```

## データローダ設定

各サンプルが 2 枚の画像を含むため、**2 つのデータセットエントリ** を用意して左右それぞれを扱います:

```json
[
    {
        "id": "subjects200k-left",
        "type": "huggingface",
        "dataset_name": "Yuanshi/Subjects200K",
        "caption_strategy": "huggingface",
        "metadata_backend": "huggingface",
        "resolution": 512,
        "resolution_type": "pixel_area",
        "cache_dir_vae": "cache/vae/subjects-left",
        "huggingface": {
            "caption_column": "description.description_0",
            "image_column": "image",
            "composite_image_config": {
                "enabled": true,
                "image_count": 2,
                "select_index": 0
            },
            "filter_func": {
                "collection": "collection_1",
                "quality_thresholds": {
                    "compositeStructure": 4.5,
                    "objectConsistency": 4.5,
                    "imageQuality": 4.5
                }
            }
        }
    },
    {
        "id": "subjects200k-right",
        "type": "huggingface",
        "dataset_name": "Yuanshi/Subjects200K",
        "caption_strategy": "huggingface",
        "metadata_backend": "huggingface",
        "resolution": 512,
        "resolution_type": "pixel_area",
        "cache_dir_vae": "cache/vae/subjects-right",
        "huggingface": {
            "caption_column": "description.description_1",
            "image_column": "image",
            "composite_image_config": {
                "enabled": true,
                "image_count": 2,
                "select_index": 1
            },
            "filter_func": {
                "collection": "collection_1",
                "quality_thresholds": {
                    "compositeStructure": 4.5,
                    "objectConsistency": 4.5,
                    "imageQuality": 4.5
                }
            }
        }
    },
    {
        "id": "text-embed-cache",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "cache/text/flux"
    }
]
```

### 設定の説明

#### 合成画像設定
- `composite_image_config.enabled`: 合成画像の取り扱いを有効化
- `composite_image_config.image_count`: 合成画像内の枚数（左右 2 枚なら 2）
- `composite_image_config.select_index`: 抽出する画像（0 = 左、1 = 右）

#### 品質フィルタ
`filter_func` で以下の条件を使ってサンプルを絞り込めます:
- `collection`: 特定のコレクションのみを使用
- `quality_thresholds`: 品質指標の最小スコア:
  - `compositeStructure`: 2 枚の画像の整合性
  - `objectConsistency`: 両画像での被写体の一貫性
  - `imageQuality`: 画像全体の品質

#### キャプション選択
- 左画像: `"caption_column": "description.description_0"`
- 右画像: `"caption_column": "description.description_1"`

### カスタマイズの選択肢

1. **品質閾値を調整**: 値を下げる（例: 4.0）と対象が増え、上げる（例: 4.8）とより厳選されます

2. **別のコレクションを使う**: `"collection": "collection_1"` をデータセット内の他のコレクションに変更します

3. **解像度を変更**: 学習に合わせて `resolution` を調整します

4. **フィルタを無効化**: `filter_func` セクションを削除して全画像を使用します

5. **アイテム名をキャプションにする**: キャプション列を `"description.item"` にして被写体名のみを使います

### ヒント

- 初回利用時にデータセットは自動的にダウンロードされ、キャッシュされます
- 各「半分」は独立したデータセットとして扱われるため、学習サンプルが実質 2 倍になります
- バリエーションが欲しい場合は、左右で異なる品質閾値を使うのも有効です
- VAE キャッシュディレクトリは左右で分けて衝突を避けてください

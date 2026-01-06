# Midjourney v6 520k

## 詳細

- **Hub リンク**: [terminusresearch/midjourney-v6-520k-raw](https://huggingface.co/datasets/terminusresearch/midjourney-v6-520k-raw)
- **説明**: 約 520,000 枚の高品質出力。日本語プロンプトは GPT-3.5-Turbo によって再キャプションされています。
- **キャプション形式**: Parquet

## 必要なストレージ

このデータセットにはすべての画像データが含まれるため、十分なディスク容量がないと展開が困難です。**少なくとも 1.5TB の空き容量を確保してください。**

このモデルの T5-XXL テキスト埋め込みは、`--compress_disk_cache` を有効にしても約 520GB を消費します。
VAE 埋め込みは、学習するモデルと埋め込み解像度により約 80〜100GB を消費します。


## ダウンロード

```bash
huggingface-cli download --repo-type=dataset terminusresearch/midjourney-v6-520k-raw --local-dir=midjourney-v6-520k-raw
```

これにより、Hugging Face Hub から分割された tar セグメントを同時にダウンロードします。

## 展開

```bash
cd midjourney-v6-520k-raw
cat *.tar | tar x
```

これにより、現在のディレクトリ内にすべてのサンプルを含むフォルダが作成されます。

## データローダ設定例

```json
{
    "id": "midjourney-v6-520k-raw",
    "type": "local",
    "cache_dir_vae": "cache/vae-mj-520k/",
    "crop": true,
    "crop_aspect": "square",
    "resolution": 1.0,
    "maximum_image_size": 1.0,
    "minimum_image_size": 0.75,
    "target_downsample_size": 1.00,
    "resolution_type": "area",
    "caption_strategy": "parquet",
    "metadata_backend": "parquet",
    "parquet": {
        "path": "/path/to/midjourney-v6-520k-raw/train.parquet",
        "caption_column": "gpt_caption",
        "filename_column": "id",
        "width_column": "width",
        "height_column": "height",
        "identifier_includes_extension": false
    }
}
```

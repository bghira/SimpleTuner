# Photo concept bucket

## 詳細

- **Hub リンク**: [bghira/photo-concept-bucket](https://huggingface.co/datasets/bghira/photo-concept-bucket)
- **説明**: 密度の高い概念分布を持つ約 567,000 枚の高品質写真。CogVLM によりキャプション付けされています。
- **キャプション形式**: Parquet

## 必要な前処理

photo-concept-bucket リポジトリには画像データが含まれていないため、Pexels サーバーから直接取得する必要があります。

ダウンロード用のサンプルスクリプトは用意されていますが、利用時点での Pexels サービスの利用規約に従っていることを必ず確認してください。

キャプションと URL リストのダウンロード:

```bash
huggingface-cli download --repo-type=dataset bghira/photo-concept-bucket --local-dir=/home/user/training/photo-concept-bucket
```

このファイルを `/home/user/training/photo-concept-bucket` に配置します:

`download.py`
```py
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
import os
import requests
from PIL import Image
from io import BytesIO

# Load the Parquet file
parquet_file = 'photo-concept-bucket.parquet'
df = pq.read_table(parquet_file).to_pandas()

# Define the output directory
output_dir = 'train'
os.makedirs(output_dir, exist_ok=True)

def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img

def download_and_save(row):
    img_url = row['url']
    caption = row['cogvlm_caption']
    img_id = row['id']

    try:
        # Download the image
        img_response = requests.get(img_url)
        if img_response.status_code == 200:
            img = Image.open(BytesIO(img_response.content))
            img_path = os.path.join(output_dir, f"{img_id}.png")
            img.save(img_path)

        # Write the caption to a text file
        caption_path = os.path.join(output_dir, f"{img_id}.txt")
        with open(caption_path, 'w') as caption_file:
            caption_file.write(caption)
    except Exception as e:
        print(f"Failed to download or save data for id {img_id}: {e}")

# Run the download in parallel
with ThreadPoolExecutor() as executor:
    executor.map(download_and_save, [row for _, row in df.iterrows()])
```

このスクリプトは Pexels から画像を並列にダウンロードし、キャプションを `train/` ディレクトリの txt ファイルとして保存します。

> ⚠️ このデータセットは非常に大きく、そのまま取得すると 7TB 以上のローカルディスクを消費します。20 メガピクセルの全量を保存したくない場合は、取得時にリサイズ処理を追加することを推奨します。

## データローダ設定例

```json
{
    "id": "photo-concept-bucket",
    "type": "local",
    "instance_data_dir": "/home/user/training/photo-concept-bucket/train",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1.0,
    "minimum_image_size": 1.0,
    "maximum_image_size": 1.5,
    "target_downsample_size": 1.25,
    "resolution_type": "area",
    "cache_dir_vae": "/home/user/training/photo-concept-bucket/cache/vae",
    "caption_strategy": "parquet",
    "metadata_backend": "parquet",
    "parquet": {
        "path": "/home/user/training/photo-concept-bucket/photo-concept-bucket.parquet",
        "caption_column": "cogvlm_caption",
        "fallback_caption_column": "tags",
        "filename_column": "id",
        "width_column": "width",
        "height_column": "height"
    }
}
```

# Photo concept bucket

## 详情

- **Hub 链接**: [bghira/photo-concept-bucket](https://huggingface.co/datasets/bghira/photo-concept-bucket)
- **描述**: 约 567,000 张高质量照片，涵盖密集的概念分布，由 CogVLM 生成字幕。
- **字幕格式**: Parquet

## 必需的预处理步骤

photo-concept-bucket 仓库不包含图像数据，因此需要你直接从 Pexels 服务器获取。

这里提供了示例下载脚本，但请务必确保在使用时遵循 Pexels 服务条款与条件。

下载字幕与 URL 列表：

```bash
huggingface-cli download --repo-type=dataset bghira/photo-concept-bucket --local-dir=/home/user/training/photo-concept-bucket
```

将以下文件放到 `/home/user/training/photo-concept-bucket`：

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

该脚本会从 Pexels 并行下载图像，并将字幕写入 `train/` 目录中的 txt 文件。

> ⚠️ 该数据集体量极大，按原样下载将消耗超过 7TB 的本地磁盘空间。如果不想保存全部 2000 万像素图像，建议在获取时添加缩放步骤。

## 数据加载器配置示例

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

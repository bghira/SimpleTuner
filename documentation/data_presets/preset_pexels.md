# Photo concept bucket

## Details

- **Hub link**: [ptx0/photo-concept-bucket](https://huggingface.co/datasets/ptx0/photo-concept-bucket)
- **Description**: ~567,000 high quality photographs across dense concept distribution, captioned with CogVLM.
- **Caption format(s)**: Parquet

## Required preprocessing steps

As the photo-concept-bucket repository does not include image data, this must be retrieved by you directly from the Pexels server.

An example script for downloading the dataset is provided, but you must ensure you are following the terms and conditions of the Pexels service, at the time of consumption.

To download the captions and URL list:

```bash
huggingface-cli download --repo-type=dataset ptx0/photo-concept-bucket --local-dir=/home/user/training/photo-concept-bucket
```

Place this file into `/home/user/training/photo-concept-bucket`:

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

This script will simultaneously download the images from Pexels and write their captions into the `train/` directory as a txt file.

> ⚠️ This dataset is extremely large, and will consume more than 7TB of local disk space to retrieve as-is. It's recommended that you add a resize step to this retrieval, if you don't wish to store the whole 20 megapixel dataset.

## Dataloader configuration example

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

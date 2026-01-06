# Photo concept bucket

## विवरण

- **Hub लिंक**: [bghira/photo-concept-bucket](https://huggingface.co/datasets/bghira/photo-concept-bucket)
- **विवरण**: ~567,000 उच्च गुणवत्ता वाली फ़ोटोग्राफ़्स, dense concept वितरण में, CogVLM से कैप्शन की गई।
- **कैप्शन फ़ॉर्मेट**: Parquet

## आवश्यक प्री-प्रोसेसिंग चरण

क्योंकि photo-concept-bucket रिपॉज़िटरी में इमेज डेटा शामिल नहीं है, इसे आपको सीधे Pexels सर्वर से प्राप्त करना होगा।

डेटासेट डाउनलोड करने के लिए एक उदाहरण स्क्रिप्ट दी गई है, लेकिन आपको उपयोग के समय Pexels सेवा के नियम और शर्तों का पालन करना होगा।

कैप्शन और URL सूची डाउनलोड करने के लिए:

```bash
huggingface-cli download --repo-type=dataset bghira/photo-concept-bucket --local-dir=/home/user/training/photo-concept-bucket
```

इस फ़ाइल को `/home/user/training/photo-concept-bucket` में रखें:

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

यह स्क्रिप्ट Pexels से इमेजेस एक साथ डाउनलोड करेगी और उनके कैप्शन `train/` डायरेक्टरी में txt फ़ाइल के रूप में लिखेगी।

> ⚠️ यह डेटासेट बहुत बड़ा है, और इसे जस का तस डाउनलोड करने पर 7TB से अधिक लोकल डिस्क स्पेस लगेगा। यदि आप पूरा 20 मेगापिक्सेल डेटासेट स्टोर नहीं करना चाहते, तो डाउनलोड में resize स्टेप जोड़ने की सलाह है।

## डाटालोडर कॉन्फ़िगरेशन उदाहरण

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

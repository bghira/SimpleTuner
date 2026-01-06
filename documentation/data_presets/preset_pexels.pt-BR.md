# Photo concept bucket

## Detalhes

- **Link do Hub**: [bghira/photo-concept-bucket](https://huggingface.co/datasets/bghira/photo-concept-bucket)
- **Descricao**: ~567.000 fotografias de alta qualidade com distribuicao densa de conceitos, legendadas com CogVLM.
- **Formato(s) de caption**: Parquet

## Etapas obrigatorias de pre-processamento

Como o repositorio photo-concept-bucket nao inclui dados de imagem, isso deve ser recuperado por voce diretamente do servidor do Pexels.

Um script de exemplo para baixar o dataset e fornecido, mas voce deve garantir que esta seguindo os termos e condicoes do servico do Pexels no momento do uso.

Para baixar as captions e a lista de URLs:

```bash
huggingface-cli download --repo-type=dataset bghira/photo-concept-bucket --local-dir=/home/user/training/photo-concept-bucket
```

Coloque este arquivo em `/home/user/training/photo-concept-bucket`:

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

Este script vai baixar as imagens do Pexels e escrever suas captions no diretorio `train/` como um arquivo txt.

> AVISO: Este dataset e extremamente grande e consumira mais de 7TB de espaco em disco local se baixado como esta. Recomenda-se adicionar uma etapa de redimensionamento nessa obtencao, se voce nao quiser armazenar todo o dataset de 20 megapixels.

## Exemplo de configuracao do dataloader

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

# DALLE-3

## Detalles

- **Enlace en Hub**: [ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions](https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions)
- **Descripción**: Más de 1 millón de imágenes DALLE-3 combinadas con una pequeña cantidad de imágenes de Midjourney y otras fuentes de IA.
- **Formato(s) de caption**: Archivos JSON que contienen múltiples estilos de caption.

## Pasos de preprocesamiento requeridos

El dataset DALLE-3 contiene archivos con la siguiente estructura:
```
|- data/
|-| data/file.png
|-| data/file.json
```

Usaremos dos scripts para preparar:

- Una tabla parquet que contiene metadatos de imagen, p. ej., ancho, alto y nombre de archivo
- Un archivo .txt por muestra con su caption, en caso de que cargar captions desde parquet sea demasiado lento

### Extraer el dataset

1. Recupera el dataset desde el hub vía el método de descarga que elijas, o:

```bash
huggingface-cli login
huggingface-cli download --repo-type=dataset ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions --local-dir=/home/user/training/data/dalle3
```

2. Entra al directorio de datos de DALLE-3 y extrae todas las entradas:

```bash
cd /home/user/training/data/dalle3
mkdir data
for tar_file_path in *.tar; do
    tar xf "$tar_file_path" -C data/
done
```
**En este punto, tendrás todos los archivos `.png` y `.json` en el subdirectorio `data/`**.

### Compilar una tabla parquet

En el directorio `dalle3`, crea un archivo llamado `compile.py` con el siguiente contenido:

```py
import glob
import json, os
import pandas as pd
from tqdm import tqdm

# Glob for all JSON files in the folder
json_files = glob.glob('data/*.json')

data = []

# Process each JSON file
for file_path in tqdm(json_files):
    with open(file_path, 'r') as file:
        content = json.load(file)
        #print(f"Content: {content}")
        if "width" not in content:
                continue
        # Extract the necessary information
        text_path = os.path.splitext(content['image_name'])[0] + ".txt"
        width = int(content['width'])
        height = int(content['height'])
        caption = content['short_caption']
        filename = content['image_name']

        # Append to data list
        data.append({'width': width, 'height': height, 'caption': caption, 'filename': filename})

# Create a DataFrame
df = pd.DataFrame(data, columns=['width', 'height', 'caption', 'filename'])

# Save the DataFrame to a Parquet file
df.to_parquet('dalle3.parquet', index=False)

print("Data has been successfully compiled and saved as 'dalle3.parquet'")
```

Ejecuta el script de compilación, asegurándote de tener tu venv activo:

```bash
. .venv/bin/activate
python compile.py
```

Verifica que el archivo parquet contiene las filas resultantes. Puede que necesites ejecutar `pip install parquet-tools` primero:

```bash
parquet-tools csv dalle3-parquet | head -n10
```

Esto imprimirá las primeras diez filas del dataset DALLE3. No te preocupes si tarda un poco: estamos procesando más de 1 millón de filas en formato columnar.

### Extraer captions de imagen a archivos de texto

En el directorio `dalle3`, crea un nuevo script llamado `extract-captions.py` con lo siguiente:

```py
import glob
import json, os
import pandas as pd
from tqdm import tqdm

# Glob for all JSON files in the folder
json_files = glob.glob('data/*.json')

data = []
caption_field = 'short_caption'

# Process each JSON file
for file_path in tqdm(json_files, desc="Extracting text captions from JSON"):
    with open(file_path, 'r') as file:
        content = json.load(file)
        if "width" not in content:
                continue
        text_path = "data/" + os.path.splitext(content['image_name'])[0] + ".txt"
        # write content to text path
        with open(text_path, 'w') as text_file:
            text_file.write(content[caption_field])
```

Este script tomará `caption_field` de cada archivo JSON en el subdirectorio `data/` y escribirá ese valor en un archivo `.txt` con el mismo nombre que la imagen.

Si deseas usar un campo de caption distinto del conjunto DALLE-3, actualiza el valor de `caption_field` antes de ejecutarlo.

Ahora ejecuta el script, asegurándote de tener la venv activa:

```bash
. .venv/bin/activate
python extract-captions.py
```

Puedes verificar que haya el número correcto de archivos JSON en el directorio. Ten en cuenta que puede tardar:

```bash
$ find data/ -name \*.json | wc -l
1161957
```

Deberías ver el valor correcto de 1,161,957.


## Entrada de dataloader:

La siguiente entrada de configuración ubicará correctamente los nombres de archivo y captions de tu nuevo dataset DALLE-3:

```json
    {
        "id": "dalle3",
        "type": "local",
        "instance_data_dir": "/home/user/training/data/dalle3/data",
        "resolution": 1.0,
        "maximum_image_size": 2.0,
        "minimum_image_size": 0.75,
        "target_downsample_size": 1.75,
        "resolution_type": "area",
        "cache_dir_vae": "/path/to/cache/vae/",
        "caption_strategy": "textfile",
        "metadata_backend": "parquet",
        "parquet": {
            "path": "/home/user/training/data/dalle3/dalle3.parquet",
            "caption_column": "caption",
            "filename_column": "filename",
            "width_column": "width",
            "height_column": "height",
            "identifier_includes_extension": true
        }
    },
```

**Nota**: Puedes omitir el script `extract-captions.py` y usar `caption_strategy=parquet` si quieres ahorrar inodos en disco.

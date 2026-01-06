# DALLE-3

## Detalhes

- **Link do Hub**: [ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions](https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions)
- **Descricao**: 1 milhao+ de imagens DALLE-3 combinadas com um pequeno numero de imagens Midjourney e outras imagens geradas por IA.
- **Formato(s) de caption**: Arquivos JSON contendo multiplos estilos de caption.

## Etapas obrigatorias de pre-processamento

O dataset DALLE-3 contem arquivos na seguinte estrutura:
```
|- data/
|-| data/file.png
|-| data/file.json
```

Usaremos dois scripts para preparar:

- Uma tabela parquet contendo os metadados das imagens, por exemplo, largura, altura e nome do arquivo
- Um arquivo .txt para cada amostra contendo a caption, caso carregar captions do parquet seja muito lento

### Extraindo o dataset

1. Baixe o dataset do hub pelo metodo de sua escolha, ou:

```bash
huggingface-cli login
huggingface-cli download --repo-type=dataset ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions --local-dir=/home/user/training/data/dalle3
```

2. Entre no diretorio de dados do DALLE-3 e extraia todas as entradas:

```bash
cd /home/user/training/data/dalle3
mkdir data
for tar_file_path in *.tar; do
    tar xf "$tar_file_path" -C data/
done
```
**Neste ponto, voce tera todos os arquivos `.png` e `.json` no subdiretorio `data/`**.

### Compilando uma tabela parquet

No diretorio `dalle3`, crie um arquivo chamado `compile.py` com o seguinte conteudo:

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

Execute o script de compilacao, garantindo que seu venv de Python esteja ativo:

```bash
. .venv/bin/activate
python compile.py
```

Verifique se o arquivo parquet contem as linhas resultantes. Talvez voce precise rodar `pip install parquet-tools` primeiro:

```bash
parquet-tools csv dalle3-parquet | head -n10
```

Isso imprimira as primeiras dez linhas do dataset DALLE3. Nao se preocupe se demorar, estamos processando mais de 1 milhao de linhas em formato colunar.

### Extraindo captions para arquivos de texto

No diretorio `dalle3`, crie um novo script chamado `extract-captions.py` contendo o seguinte:

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

Este script vai ler `caption_field` de cada arquivo JSON no subdiretorio `data/` e escrever esse valor em um arquivo `.txt` com o mesmo nome do arquivo da imagem.

Se voce quiser usar um campo de caption diferente do conjunto DALLE-3, atualize o valor de `caption_field` antes de executa-lo.

Agora, execute o script, garantindo que o venv esteja ativo:

```bash
. .venv/bin/activate
python extract-captions.py
```

Voce pode verificar se ha o numero correto de arquivos JSON no diretorio agora. Atenção: isso pode demorar:

```bash
$ find data/ -name \*.json | wc -l
1161957
```

Voce deve ver o valor correto de 1.161.957.


## Entrada do dataloader:

A entrada de configuracao a seguir vai localizar corretamente os nomes de arquivo e captions do seu dataset DALLE-3 recem-criado:

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

**Nota**: Voce pode pular o script `extract-captions.py` e usar apenas `caption_strategy=parquet` se quiser economizar inodes de disco.

# DALLE-3

## 詳細

- **Hub リンク**: [ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions](https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions)
- **説明**: 100 万枚以上の DALLE-3 画像に、少量の Midjourney など他の AI 由来画像を組み合わせたデータセットです。
- **キャプション形式**: 複数スタイルのキャプションを含む JSON ファイル。

## 必要な前処理

DALLE-3 データセットのファイル構成は以下のとおりです:
```
|- data/
|-| data/file.png
|-| data/file.json
```

以下の 2 つのスクリプトで準備します:

- 画像メタデータ（幅、高さ、ファイル名など）を含む parquet テーブル
- parquet からキャプションを読むのが遅い場合に備えた、各サンプルの `.txt` キャプションファイル

### データセットの展開

1. Hub から任意の方法で取得するか、以下を実行します:

```bash
huggingface-cli login
huggingface-cli download --repo-type=dataset ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions --local-dir=/home/user/training/data/dalle3
```

2. DALLE-3 データディレクトリへ移動し、すべてのエントリを展開します:

```bash
cd /home/user/training/data/dalle3
mkdir data
for tar_file_path in *.tar; do
    tar xf "$tar_file_path" -C data/
done
```
**この時点で、`data/` サブディレクトリ内に `.png` と `.json` がすべて揃います。**

### parquet テーブルの作成

`dalle3` ディレクトリ内に `compile.py` を作成し、以下の内容を記述します:

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

Python venv を有効化した上で、コンパイルスクリプトを実行します:

```bash
. .venv/bin/activate
python compile.py
```

parquet ファイルに行が入っていることを確認してください。必要に応じて `pip install parquet-tools` を先に実行します:

```bash
parquet-tools csv dalle3-parquet | head -n10
```

これは DALLE3 データセットの先頭 10 行を出力します。列指向フォーマットから 100 万行超を処理しているため、時間がかかっても問題ありません。

### 画像キャプションをテキストファイルに抽出

`dalle3` ディレクトリ内に `extract-captions.py` を作成し、以下を記述します:

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

このスクリプトは `data/` サブディレクトリ内の各 JSON から `caption_field` を取得し、画像と同じファイル名の `.txt` に書き込みます。

DALLE-3 の別のキャプションフィールドを使いたい場合は、実行前に `caption_field` を更新してください。

では、venv を有効化して実行します:

```bash
. .venv/bin/activate
python extract-captions.py
```

現在のディレクトリ内の JSON ファイル数が正しいことを確認できます。実行には時間がかかる場合があります:

```bash
$ find data/ -name \*.json | wc -l
1161957
```

正しい値 1,161,957 が表示されるはずです。


## データローダのエントリ:

次の設定エントリで、新しく作成した DALLE-3 データセットのファイル名とキャプションを正しく参照できます:

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

**注記**: ディスク inode を節約したい場合は `extract-captions.py` を省略し、`caption_strategy=parquet` を使うこともできます。

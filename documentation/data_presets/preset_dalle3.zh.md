# DALLE-3

## 详情

- **Hub 链接**: [ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions](https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions)
- **描述**: 超过 100 万张 DALLE-3 图像，混合少量 Midjourney 和其他 AI 来源图像。
- **字幕格式**: 包含多种风格字幕的 JSON 文件。

## 必需的预处理步骤

DALLE-3 数据集文件结构如下：
```
|- data/
|-| data/file.png
|-| data/file.json
```

我们将使用两个脚本来准备：

- 包含图像元数据（例如宽度、高度、文件名）的 parquet 表
- 每个样本对应一个 `.txt` 文件存放字幕，以防从 parquet 读取字幕太慢

### 解压数据集

1. 使用你偏好的方式从 Hub 获取数据集，或执行：

```bash
huggingface-cli login
huggingface-cli download --repo-type=dataset ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions --local-dir=/home/user/training/data/dalle3
```

2. 进入 DALLE-3 数据目录并解压所有条目：

```bash
cd /home/user/training/data/dalle3
mkdir data
for tar_file_path in *.tar; do
    tar xf "$tar_file_path" -C data/
done
```
**此时，`data/` 子目录中将包含所有 `.png` 与 `.json` 文件。**

### 生成 parquet 表

在 `dalle3` 目录中创建 `compile.py`，内容如下：

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

执行编译脚本，确保 Python venv 已激活：

```bash
. .venv/bin/activate
python compile.py
```

确认 parquet 文件中包含结果行。你可能需要先运行 `pip install parquet-tools`：

```bash
parquet-tools csv dalle3-parquet | head -n10
```

这将输出 DALLE3 数据集的前十行。不用担心耗时，我们正在从列式格式中处理超过 100 万行。

### 将图像字幕提取为文本文件

在 `dalle3` 目录中新建 `extract-captions.py`，内容如下：

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

该脚本会从 `data/` 子目录中的每个 JSON 读取 `caption_field`，并写入与图像同名的 `.txt` 文件。

如果想使用 DALLE-3 的其他字幕字段，请在执行前修改 `caption_field` 的值。

现在激活 venv 并执行脚本：

```bash
. .venv/bin/activate
python extract-captions.py
```

可以验证目录中 JSON 文件数量是否正确。注意这一步可能耗时：

```bash
$ find data/ -name \*.json | wc -l
1161957
```

你应该看到正确的值 1,161,957。


## 数据加载器条目：

下面的配置条目可以正确定位你新创建的 DALLE-3 数据集文件名与字幕：

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

**注记**：若希望节省磁盘 inode，可以跳过 `extract-captions.py`，直接使用 `caption_strategy=parquet`。

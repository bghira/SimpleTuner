# DALLE-3

## Details

- **Hub link**: [ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions](https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions)
- **Description**: 1 million+ DALLE-3 images combined with a small number of Midjourney and other AI-sourced images.
- **Caption format(s)**: JSON files containing multiple styles of caption.

## Required preprocessing steps

The DALLE-3 dataset contains files in the following structure:
```
|- data/
|-| data/file.png
|-| data/file.json
```

We will use two scripts to prepare:

- A parquet table containing the image metadata, eg. width, height, and filename
- A .txt file for each sample containing its caption, in case loading captions from parquet is too slow

### Extracting the dataset

1. Retrieve the dataset from the hub via your chosen retrieval method, or:

```bash
huggingface-cli login
huggingface-cli download --repo-type=dataset ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions --local-dir=/home/user/training/data/dalle3
```

2. Enter the DALLE-3 data directory and extract all entries:

```bash
cd /home/user/training/data/dalle3
mkdir data
for tar_file_path in *.tar; do
    tar xf "$tar_file_path" -C data/
done
```
**At this point, you will have all of the `.png` and `.json` files in the `data/` subdirectory**.

### Compiling a parquet table

In the `dalle3` directory, create a file named `compile.py` with the following contents:

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

Execute the compilation script, being sure to have your python venv active:

```bash
. .venv/bin/activate
python compile.py
```

Check that the parquet file contains the resulting rows. You may need to run `pip install parquet-tools` first:

```bash
parquet-tools csv dalle3-parquet | head -n10
```

This will print the first ten rows of the DALLE3 dataset. Don't worry if it takes a while, we're crunching more than 1 million rows from a columnar format.

### Extracting image captions into textfiles

In the `dalle3` directory, create a new script called `extract-captions.py` containing the following:

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

This script will retrieve `caption_field` from each JSON file in the `data/` subdirectory, and write that value to a `.txt` file with the same filename as the image.

If you wish to use a different caption field from the DALLE-3 set, update the value for `caption_field` before executing it.

Now, execute the script, being sure to have the venv active:

```bash
. .venv/bin/activate
python extract-captions.py
```

You can verify that there are the correct number of JSON files in the directory now. Beware that this may take a while to run:

```bash
$ find data/ -name \*.json | wc -l
1161957
```

You should see the correct value of 1,161,957.


## Dataloader entry:

The following configuration entry will correctly locate the filenames and captions for your newly-created DALLE-3 dataset:

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

**Note**: You can skip the `extract-captions.py` script and simply use `caption_strategy=parquet` if you wish to save on disk inodes.
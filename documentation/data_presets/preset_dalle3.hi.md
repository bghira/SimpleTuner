# DALLE-3

## विवरण

- **Hub लिंक**: [ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions](https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions)
- **विवरण**: 1 मिलियन+ DALLE-3 इमेजेस, साथ में थोड़ी संख्या में Midjourney और अन्य AI-सोर्स्ड इमेजेस।
- **कैप्शन फ़ॉर्मेट**: JSON फ़ाइलें जिनमें कई स्टाइल के कैप्शन होते हैं।

## आवश्यक प्री-प्रोसेसिंग चरण

DALLE-3 डेटासेट में फ़ाइलें इस संरचना में हैं:
```
|- data/
|-| data/file.png
|-| data/file.json
```

हम तैयारी के लिए दो स्क्रिप्ट्स का उपयोग करेंगे:

- एक parquet टेबल जिसमें इमेज मेटाडेटा होगा, जैसे चौड़ाई, ऊँचाई और फ़ाइलनाम
- प्रत्येक सैंपल के लिए एक .txt फ़ाइल जिसमें उसका कैप्शन होगा, ताकि parquet से कैप्शन लोड करना बहुत धीमा न हो

### डेटासेट निकालना

1. अपनी पसंदीदा विधि से hub से डेटासेट प्राप्त करें, या:

```bash
huggingface-cli login
huggingface-cli download --repo-type=dataset ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions --local-dir=/home/user/training/data/dalle3
```

2. DALLE-3 डेटा डायरेक्टरी में जाएँ और सभी एंट्रीज़ निकालें:

```bash
cd /home/user/training/data/dalle3
mkdir data
for tar_file_path in *.tar; do
    tar xf "$tar_file_path" -C data/
done
```
**इस बिंदु पर, आपके पास `data/` सबडायरेक्टरी में सभी `.png` और `.json` फ़ाइलें होंगी**।

### Parquet टेबल बनाना

`dalle3` डायरेक्टरी में `compile.py` नाम की फ़ाइल बनाकर यह सामग्री डालें:

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

कम्पाइल स्क्रिप्ट चलाएँ, और सुनिश्चित करें कि आपका python venv सक्रिय है:

```bash
. .venv/bin/activate
python compile.py
```

जाँचें कि parquet फ़ाइल में सही रोज़ हैं। शायद पहले `pip install parquet-tools` चलाना पड़े:

```bash
parquet-tools csv dalle3-parquet | head -n10
```

यह DALLE3 डेटासेट की पहली दस रोज़ प्रिंट करेगा। अगर इसमें समय लगे तो चिंता न करें, हम 1 मिलियन से अधिक रोज़ का columnar डेटा प्रोसेस कर रहे हैं।

### इमेज कैप्शन को टेक्स्ट फ़ाइलों में निकालना

`dalle3` डायरेक्टरी में `extract-captions.py` नाम की नई स्क्रिप्ट बनाएँ और यह सामग्री डालें:

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

यह स्क्रिप्ट `data/` सबडायरेक्टरी की प्रत्येक JSON फ़ाइल से `caption_field` लेगी और उसी नाम की `.txt` फ़ाइल में लिख देगी।

यदि आप DALLE-3 सेट से कोई अलग caption field उपयोग करना चाहते हैं, तो इसे चलाने से पहले `caption_field` का मान बदल दें।

अब स्क्रिप्ट चलाएँ, और सुनिश्चित करें कि venv सक्रिय है:

```bash
. .venv/bin/activate
python extract-captions.py
```

अब जाँचें कि डायरेक्टरी में JSON फ़ाइलों की संख्या सही है। ध्यान दें कि इसमें समय लग सकता है:

```bash
$ find data/ -name \*.json | wc -l
1161957
```

आपको 1,161,957 का सही मान दिखना चाहिए।


## डाटालोडर एंट्री:

नीचे दिया कॉन्फ़िगरेशन एंट्री आपके नए बनाए DALLE-3 डेटासेट के लिए फ़ाइलनाम और कैप्शन सही ढंग से लोकेट करेगा:

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

**नोट**: यदि आप डिस्क inodes बचाना चाहते हैं, तो `extract-captions.py` स्क्रिप्ट को छोड़कर सीधे `caption_strategy=parquet` उपयोग कर सकते हैं।

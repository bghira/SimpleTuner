# Niji v6 520k

## विवरण

- **Hub लिंक**: [terminusresearch/nijijourney-v6-520k-raw](https://huggingface.co/datasets/terminusresearch/nijijourney-v6-520k-raw)
- **विवरण**: ~520,000 उच्च गुणवत्ता वाले आउटपुट, जिनमें कोई भी जापानी user prompts GPT-3.5-Turbo से पुनः कैप्शन किए गए हैं।
- **कैप्शन फ़ॉर्मेट**: Parquet

## आवश्यक स्टोरेज

इस डेटासेट में सभी इमेज डेटा शामिल है, इसलिए पर्याप्त डिस्क स्पेस के बिना इसे निकालना कठिन होगा। **निकालने के लिए कम से कम 1.5TB डिस्क स्पेस उपलब्ध रखें।**

इस मॉडल के लिए T5-XXL text embeds, `--compress_disk_cache` सक्षम होने पर भी ~520GB उपयोग करेंगे।
VAE embeds लगभग 80 से 100GB स्पेस लेंगे, यह इस बात पर निर्भर करेगा कि कौन सा मॉडल ट्रेन हो रहा है और embeds का रेजोल्यूशन क्या है।

## डाउनलोड

```bash
huggingface-cli download --repo-type=dataset terminusresearch/nijijourney-v6-520k-raw --local-dir=nijijourney-v6-520k-raw
```

यह Hugging Face Hub से chunked tar segments एक साथ डाउनलोड करेगा।

## Extract

```bash
cd nijijourney-v6-520k-raw
cat *.tar | tar x
```

इससे मौजूदा डायरेक्टरी के अंदर सभी सैंपल्स वाला फ़ोल्डर बनेगा।

## डाटालोडर कॉन्फ़िगरेशन उदाहरण

```json
{
    "id": "nijijourney-v6-520k-raw",
    "type": "local",
    "cache_dir_vae": "cache/vae-nj-520k/",
    "crop": true,
    "crop_aspect": "square",
    "resolution": 1.0,
    "maximum_image_size": 1.0,
    "minimum_image_size": 0.75,
    "target_downsample_size": 1.00,
    "resolution_type": "area",
    "caption_strategy": "parquet",
    "metadata_backend": "parquet",
    "parquet": {
        "path": "/path/to/nijijourney-v6-520k-raw/train.parquet",
        "caption_column": "gpt_caption",
        "filename_column": "id",
        "width_column": "width",
        "height_column": "height",
        "identifier_includes_extension": false
    }
}
```

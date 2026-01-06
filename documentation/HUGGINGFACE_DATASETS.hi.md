# Hugging Face Datasets एकीकरण

SimpleTuner Hugging Face Hub से सीधे datasets लोड करने का समर्थन करता है, जिससे बिना सभी इमेजेज़ लोकल डाउनलोड किए बड़े datasets पर कुशल प्रशिक्षण संभव होता है।

## अवलोकन

Hugging Face datasets backend आपको यह करने देता है:
- Hugging Face Hub से सीधे datasets लोड करना
- metadata या quality metrics के आधार पर फ़िल्टर लागू करना
- dataset columns से captions निकालना
- composite/grid इमेजेस संभालना
- केवल प्रोसेस्ड embeddings को लोकल cache करना

**महत्वपूर्ण**: SimpleTuner को aspect ratio buckets बनाने और batch sizes गणना के लिए पूरे dataset तक पहुँच चाहिए। Hugging Face streaming datasets सपोर्ट करता है, लेकिन यह फीचर SimpleTuner की architecture के साथ संगत नहीं है। बहुत बड़े datasets को manageable आकार में लाने के लिए filtering का उपयोग करें।

## बेसिक कॉन्फ़िगरेशन

Hugging Face dataset उपयोग करने के लिए, अपने dataloader में `"type": "huggingface"` सेट करें:

```json
{
  "id": "my-hf-dataset",
  "type": "huggingface",
  "dataset_name": "username/dataset-name",
  "split": "train",
  "caption_strategy": "huggingface",
  "metadata_backend": "huggingface",
  "caption_column": "text",
  "image_column": "image",
  "cache_dir": "cache/my-hf-dataset"
}
```

### आवश्यक फ़ील्ड्स

- `type`: `"huggingface"` होना चाहिए
- `dataset_name`: Hugging Face dataset identifier (उदा. "laion/laion-aesthetic")
- `caption_strategy`: `"huggingface"` होना चाहिए
- `metadata_backend`: `"huggingface"` होना चाहिए

### वैकल्पिक फ़ील्ड्स

- `split`: उपयोग करने वाला dataset split (डिफ़ॉल्ट: "train")
- `revision`: विशिष्ट dataset revision
- `image_column`: इमेजेस वाला column (डिफ़ॉल्ट: "image")
- `caption_column`: captions वाले column(s)
- `cache_dir`: dataset फ़ाइलों के लिए लोकल cache डायरेक्टरी
- `streaming`: ⚠️ **वर्तमान में कार्यात्मक नहीं** - SimpleTuner metadata और encoder caches बनाने के लिए dataset स्कैन करने की कोशिश करता है।
- `num_proc`: फ़िल्टरिंग के लिए processes की संख्या (डिफ़ॉल्ट: 16)

## Caption कॉन्फ़िगरेशन

Hugging Face backend लचीला caption extraction सपोर्ट करता है:

### Single Caption Column
```json
{
  "caption_column": "caption"
}
```

### Multiple Caption Columns
```json
{
  "caption_column": ["short_caption", "detailed_caption", "tags"]
}
```

### Nested Column Access
```json
{
  "caption_column": "metadata.caption",
  "fallback_caption_column": "basic_caption"
}
```

### Advanced Caption Configuration
```json
{
  "huggingface": {
    "caption_column": "caption",
    "fallback_caption_column": "description",
    "description_column": "detailed_description",
    "width_column": "width",
    "height_column": "height"
  }
}
```

## Dataset फ़िल्टरिंग

केवल उच्च‑गुणवत्ता वाले samples चुनने के लिए filters लागू करें:

### गुणवत्ता‑आधारित फ़िल्टरिंग
```json
{
  "huggingface": {
    "filter_func": {
      "quality_thresholds": {
        "clip_score": 0.3,
        "aesthetic_score": 5.0,
        "resolution": 0.8
      },
      "quality_column": "quality_assessment"
    }
  }
}
```

### Collection/Subset फ़िल्टरिंग
```json
{
  "huggingface": {
    "filter_func": {
      "collection": ["photo", "artwork"],
      "min_width": 512,
      "min_height": 512
    }
  }
}
```

## Composite Image समर्थन

grid में कई इमेजेस वाले datasets संभालें:

```json
{
  "huggingface": {
    "composite_image_config": {
      "enabled": true,
      "image_count": 4,
      "select_index": 0
    }
  }
}
```

यह कॉन्फ़िगरेशन:
- 4‑image grids का पता लगाएगा
- केवल पहली इमेज (index 0) निकालेगा
- dimensions accordingly एडजस्ट करेगा

## पूर्ण उदाहरण कॉन्फ़िगरेशन्स

### बेसिक Photo Dataset
```json
{
  "id": "aesthetic-photos",
  "type": "huggingface",
  "dataset_name": "aesthetic-foundation/aesthetic-photos",
  "split": "train",
  "caption_strategy": "huggingface",
  "metadata_backend": "huggingface",
  "caption_column": "caption",
  "image_column": "image",
  "resolution": 1024,
  "resolution_type": "pixel",
  "minimum_image_size": 512,
  "cache_dir": "cache/aesthetic-photos"
}
```

### फ़िल्टर्ड उच्च‑गुणवत्ता Dataset
```json
{
  "id": "high-quality-art",
  "type": "huggingface",
  "dataset_name": "example/art-dataset",
  "caption_strategy": "huggingface",
  "metadata_backend": "huggingface",
  "huggingface": {
    "caption_column": ["title", "description", "tags"],
    "fallback_caption_column": "filename",
    "width_column": "original_width",
    "height_column": "original_height",
    "filter_func": {
      "quality_thresholds": {
        "aesthetic_score": 6.0,
        "technical_quality": 0.8
      },
      "min_width": 768,
      "min_height": 768
    }
  },
  "resolution": 1024,
  "resolution_type": "pixel_area",
  "crop": true,
  "crop_aspect": "square"
}
```

### वीडियो Dataset
```json
{
  "id": "video-dataset",
  "type": "huggingface",
  "dataset_type": "video",
  "dataset_name": "example/video-clips",
  "caption_strategy": "huggingface",
  "metadata_backend": "huggingface",
  "huggingface": {
    "caption_column": "description",
    "num_frames_column": "frame_count",
    "fps_column": "fps"
  },
  "video": {
    "num_frames": 125,
    "min_frames": 100
  },
  "resolution": 480,
  "resolution_type": "pixel"
}
```

## Virtual File System

Hugging Face backend एक virtual file system उपयोग करता है जहाँ इमेजेस dataset index द्वारा रेफ़रेंस होती हैं:
- `0.jpg` → dataset में पहला item
- `1.jpg` → dataset में दूसरा item
- आदि

यह standard SimpleTuner pipeline को बिना बदलाव के काम करने देता है।

## Caching व्यवहार

- **Dataset फ़ाइलें**: Hugging Face datasets लाइब्रेरी defaults के अनुसार cache होती हैं
- **VAE embeddings**: `cache_dir/vae/{backend_id}/` में स्टोर होते हैं
- **Text embeddings**: standard text embed cache कॉन्फ़िगरेशन उपयोग करती हैं
- **Metadata**: `cache_dir/huggingface_metadata/{backend_id}/` में स्टोर होती है

## Performance विचार

1. **पहला scan**: पहली रन पर dataset metadata डाउनलोड होगा और aspect ratio buckets बनेंगे
2. **Dataset आकार**: पूरी metadata लोड करनी पड़ती है ताकि file lists और lengths निकाले जा सकें
3. **Filtering**: initial load के दौरान लागू होती है — filtered items डाउनलोड नहीं होंगे
4. **Cache reuse**: बाद के runs cached metadata और embeddings reuse करते हैं

**Note**: Hugging Face datasets streaming सपोर्ट करता है, लेकिन SimpleTuner को aspect buckets और batch sizes के लिए पूर्ण dataset access चाहिए। बहुत बड़े datasets को manageable आकार में फ़िल्टर करें।

## सीमाएँ

- Read‑only access (source dataset को modify नहीं कर सकते)
- initial dataset access के लिए internet connection आवश्यक
- कुछ dataset formats समर्थित नहीं हो सकते
- streaming mode समर्थित नहीं — SimpleTuner को पूरा dataset access चाहिए
- बहुत बड़े datasets को manageable आकार में फ़िल्टर करना होगा
- विशाल datasets के लिए initial metadata loading memory‑intensive हो सकता है

## Troubleshooting

### Dataset नहीं मिला
```
Error: Dataset 'username/dataset' not found
```
- सुनिश्चित करें कि dataset Hugging Face Hub पर मौजूद है
- जाँचें कि dataset private तो नहीं (authentication आवश्यक)
- dataset नाम की spelling सही हो

### Slow initial loading
- बड़े datasets को metadata और buckets बनाने में समय लगता है
- dataset size कम करने के लिए aggressive filtering उपयोग करें
- subset या filtered संस्करण उपयोग करने पर विचार करें
- cache files बाद के runs तेज़ कर देंगी

### Memory समस्याएँ
- loading से पहले filters उपयोग कर dataset size कम करें
- फ़िल्टरिंग ऑपरेशन्स के लिए `num_proc` घटाएँ
- बहुत बड़े datasets को छोटे हिस्सों में बाँटें
- quality thresholds का उपयोग करके केवल high‑quality samples रखें

### Caption extraction समस्याएँ
- column नाम dataset schema से मेल खाते हों
- nested column संरचनाओं की जाँच करें
- missing captions के लिए `fallback_caption_column` उपयोग करें

## Advanced उपयोग

### Custom filter functions

कॉन्फ़िगरेशन basic filtering सपोर्ट करता है, लेकिन आप कोड बदलकर अधिक जटिल filters लागू कर सकते हैं। filter function हर dataset item को लेकर True/False लौटाता है।

### Multi‑Dataset प्रशिक्षण

Hugging Face datasets को local data के साथ मिलाएँ:

```json
[
  {
    "id": "hf-dataset",
    "type": "huggingface",
    "dataset_name": "laion/laion-art",
    "probability": 0.7
  },
  {
    "id": "local-dataset",
    "type": "local",
    "instance_data_dir": "/path/to/local/data",
    "probability": 0.3
  }
]
```

यह कॉन्फ़िगरेशन Hugging Face dataset से 70% और local data से 30% sample करेगा।

## ऑडियो Datasets

Audio मॉडल्स (जैसे ACE-Step) के लिए, आप `dataset_type: "audio"` निर्दिष्ट कर सकते हैं।

```json
{
    "id": "audio-dataset",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "my-audio-data",
    "audio_column": "audio",
    "config": {
        "audio_caption_fields": ["tags"],
        "lyrics_column": "lyrics"
    }
}
```

*   **`audio_column`**: audio data वाला column (decoded या bytes)। डिफ़ॉल्ट `"audio"`.
*   **`audio_caption_fields`**: **prompt** (text conditioning) बनाने के लिए combine किए जाने वाले column नामों की सूची। डिफ़ॉल्ट `"prompt", "tags"`.
*   **`lyrics_column`**: गीत के lyrics वाला column। डिफ़ॉल्ट `"lyrics"`. यदि यह column न मिले, तो SimpleTuner fallback के रूप में `"norm_lyrics"` देखेगा।

### अपेक्षित Columns
*   **`audio`**: audio data.
*   **`prompt`** / **`tags`**: text encoder के लिए descriptive tags या prompts.
*   **`lyrics`**: lyric encoder के लिए गीत के बोल.

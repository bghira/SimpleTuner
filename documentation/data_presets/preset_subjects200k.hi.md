# Subjects200K

## विवरण

- **Hub लिंक**: [Yuanshi/Subjects200K](https://huggingface.co/datasets/Yuanshi/Subjects200K)
- **विवरण**: 200K+ उच्च गुणवत्ता वाली composite इमेजेस जिनमें paired descriptions और quality assessments शामिल हैं। हर सैंपल में एक ही subject की अलग-अलग संदर्भों में दो side-by-side इमेजेस होती हैं।
- **कैप्शन फ़ॉर्मेट**: Structured JSON जिसमें हर इमेज हाफ के लिए अलग-अलग विवरण होते हैं
- **विशेष फीचर्स**: Quality assessment स्कोर, collection tags, composite images

## डेटासेट संरचना

Subjects200K डेटासेट अनोखा है क्योंकि:
- हर `image` फ़ील्ड में **दो इमेजेस side-by-side** मिलाकर एक चौड़ी इमेज होती है
- हर सैंपल में **दो अलग कैप्शन** होते हैं - बाएँ इमेज के लिए (`description.description_0`) और दाएँ के लिए (`description.description_1`)
- Quality assessment मेटाडेटा इमेज गुणवत्ता मीट्रिक्स के आधार पर फ़िल्टरिंग की अनुमति देता है
- इमेजेस पहले से collections में व्यवस्थित हैं

डेटा संरचना का उदाहरण:
```python
{
    'image': PIL.Image,  # Combined image (e.g., 1056x528 for two 528x528 images)
    'collection': 'collection_1',
    'quality_assessment': {
        'compositeStructure': 5,
        'objectConsistency': 5,
        'imageQuality': 5
    },
    'description': {
        'item': 'Eames Lounge Chair',
        'description_0': 'The Eames Lounge Chair is placed in a modern city living room...',
        'description_1': 'Nestled in a cozy nook of a rustic cabin...',
        'category': 'Furniture',
        'description_valid': True
    }
}
```

## Direct उपयोग (प्री-प्रोसेसिंग की आवश्यकता नहीं)

वे डेटासेट जिनमें extraction और preprocessing की आवश्यकता होती है, उनसे अलग, Subjects200K को HuggingFace से सीधे उपयोग किया जा सकता है! यह डेटासेट पहले से सही फ़ॉर्मेट में होस्ट किया गया है।

बस सुनिश्चित करें कि आपके पास `datasets` लाइब्रेरी इंस्टॉल है:
```bash
pip install datasets
```

## डाटालोडर कॉन्फ़िगरेशन

क्योंकि हर सैंपल में दो इमेजेस हैं, हमें **दो अलग dataset entries** कॉन्फ़िगर करनी होंगी - composite इमेज के हर हाफ के लिए एक:

```json
[
    {
        "id": "subjects200k-left",
        "type": "huggingface",
        "dataset_name": "Yuanshi/Subjects200K",
        "caption_strategy": "huggingface",
        "metadata_backend": "huggingface",
        "resolution": 512,
        "resolution_type": "pixel_area",
        "cache_dir_vae": "cache/vae/subjects-left",
        "huggingface": {
            "caption_column": "description.description_0",
            "image_column": "image",
            "composite_image_config": {
                "enabled": true,
                "image_count": 2,
                "select_index": 0
            },
            "filter_func": {
                "collection": "collection_1",
                "quality_thresholds": {
                    "compositeStructure": 4.5,
                    "objectConsistency": 4.5,
                    "imageQuality": 4.5
                }
            }
        }
    },
    {
        "id": "subjects200k-right",
        "type": "huggingface",
        "dataset_name": "Yuanshi/Subjects200K",
        "caption_strategy": "huggingface",
        "metadata_backend": "huggingface",
        "resolution": 512,
        "resolution_type": "pixel_area",
        "cache_dir_vae": "cache/vae/subjects-right",
        "huggingface": {
            "caption_column": "description.description_1",
            "image_column": "image",
            "composite_image_config": {
                "enabled": true,
                "image_count": 2,
                "select_index": 1
            },
            "filter_func": {
                "collection": "collection_1",
                "quality_thresholds": {
                    "compositeStructure": 4.5,
                    "objectConsistency": 4.5,
                    "imageQuality": 4.5
                }
            }
        }
    },
    {
        "id": "text-embed-cache",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "cache/text/flux"
    }
]
```

### कॉन्फ़िगरेशन की व्याख्या

#### Composite इमेज कॉन्फ़िगरेशन
- `composite_image_config.enabled`: composite इमेज हैंडलिंग सक्रिय करता है
- `composite_image_config.image_count`: composite में इमेज की संख्या (side-by-side के लिए 2)
- `composite_image_config.select_index`: कौन सा इमेज निकालना है (0 = बायाँ, 1 = दायाँ)

#### Quality फ़िल्टरिंग
`filter_func` आपको सैंपल्स को निम्न आधारों पर फ़िल्टर करने देता है:
- `collection`: केवल निर्दिष्ट collections से इमेजेस उपयोग करें
- `quality_thresholds`: क्वालिटी मीट्रिक्स के लिए न्यूनतम स्कोर:
  - `compositeStructure`: दोनों इमेजेस कितनी अच्छी तरह साथ काम करती हैं
  - `objectConsistency`: दोनों इमेजेस में subject की संगति
  - `imageQuality`: कुल इमेज गुणवत्ता

#### कैप्शन चयन
- बायाँ इमेज उपयोग करता है: `"caption_column": "description.description_0"`
- दायाँ इमेज उपयोग करता है: `"caption_column": "description.description_1"`

### कस्टमाइज़ेशन विकल्प

1. **Quality thresholds समायोजित करें**: कम मान (जैसे 4.0) अधिक इमेजेस शामिल करते हैं, अधिक मान (जैसे 4.8) अधिक चयनात्मक होते हैं

2. **अलग collections का उपयोग**: डेटासेट में उपलब्ध अन्य collections के लिए `"collection": "collection_1"` बदलें

3. **रेज़ोल्यूशन बदलें**: अपनी ट्रेनिंग आवश्यकताओं के आधार पर `resolution` मान बदलें

4. **फ़िल्टरिंग बंद करें**: सभी इमेजेस उपयोग करने के लिए `filter_func` सेक्शन हटाएँ

5. **item नामों को कैप्शन के रूप में उपयोग करें**: सिर्फ subject नाम उपयोग करने के लिए caption column को `"description.item"` पर बदलें

### टिप्स

- पहली बार उपयोग पर डेटासेट अपने आप डाउनलोड और cache हो जाएगा
- हर "हाफ" को स्वतंत्र डेटासेट माना जाता है, जिससे आपके ट्रेनिंग सैंपल्स लगभग दोगुने हो जाते हैं
- यदि आप विविधता चाहते हैं, तो प्रत्येक हाफ के लिए अलग quality thresholds विचार करें
- VAE cache डायरेक्टरीज़ हर हाफ के लिए अलग होनी चाहिए ताकि conflicts न हों

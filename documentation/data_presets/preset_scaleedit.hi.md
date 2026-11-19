# ScaleEdit-12M: Webshart मेटाडेटा के साथ युग्मित Parquet इमेज

## विवरण

- **मूल डेटासेट**: [InternVL-U/ScaleEdit-12M](https://huggingface.co/datasets/InternVL-U/ScaleEdit-12M)।
- **मेटाडेटा सहायक रिपॉज़िटरी**: [webshart/ScaleEdit-12M-metadata](https://huggingface.co/datasets/webshart/ScaleEdit-12M-metadata)।
- **विवरण**: 251 Parquet shards और 23 कार्य श्रेणियों में 11,845,023 image-editing पंक्तियाँ। सहायक मेटाडेटा लगभग 423 MB है; इमेज बाइट्स मूल डेटासेट में रहती हैं।
- **कैप्शन प्रारूप**: `edit_instruction` स्ट्रिंग। ये संपादन निर्देश हैं, स्वतंत्र दृश्य-विवरण नहीं।
- **लाइसेंस**: मूल स्रोत से प्राप्त CC-BY-NC-SA-4.0। इमेज अधिकार और उपयोग सीमाओं के लिए मूल कार्ड देखें।

## आवश्यक पूर्व-प्रसंस्करण

tar निकालने, हर इमेज का caption बनाने या मेटाडेटा संकलित करने की ज़रूरत नहीं है। [PR #3325](https://github.com/bghira/SimpleTuner/pull/3325) सहित SimpleTuner का [Webshart Parquet backend](../DATALOADER.md#webshart-datasets) उपयोग करें और प्रकाशित Hub एकीकरण इंस्टॉल करें:

```bash
python -m pip install --upgrade "webshart[hub]>=0.8.0"
```

सार्वजनिक Hub पहुँच के लिए token आवश्यक नहीं है। प्रमाणित पहुँच हेतु `HF_TOKEN` सेट करें। सहायक manifest मूल revision को `05d2eea7b80ba72c077d39be43b7aca1a7857e6a` पर स्थिर करता है और metadata checksums सत्यापित करता है।

## डेटासेट की संरचना

हर मूल Parquet पंक्ति में `source_image`, `edited_image`, `edit_instruction`, दोनों इमेज के आयाम और quality scores हैं। `webshart-parquet.json` मूल shards को `metadata/<task>/<part>.parquet` से जोड़ता है। metadata फ़ाइलों में इमेज बाइट्स नहीं हैं। कार्य श्रेणियों में `id` दोहराता है, इसलिए जोड़ने के लिए shard path और वास्तविक row position उपयोग करें।

## Dataloader कॉन्फ़िगरेशन उदाहरण

यह array `multidatabackend.json` में रखें। यह `3.4_count_change` के 458 जोड़ों से शुरू होता है; उसका एक स्रोत shard लगभग 54 MB है। इस स्थिर subset में सभी मूल और संपादित इमेज बाइट्स मौजूद हैं। [Flux Kontext](../quickstart/FLUX_KONTEXT.md) जैसे `reference_strict` समर्थित editing model का उपयोग करें; model training configuration अलग है।

```json
[
  {
    "id": "scaleedit-edits",
    "type": "webshart",
    "source": "InternVL-U/ScaleEdit-12M",
    "metadata": "webshart/ScaleEdit-12M-metadata",
    "subfolder": "3.4_count_change",
    "caption_strategy": "webshart",
    "metadata_backend": "webshart",
    "resolution": 512,
    "resolution_type": "pixel",
    "crop": false,
    "dataset_type": "image",
    "conditioning_data": [
      "scaleedit-sources"
    ],
    "cache_dir_vae": "cache/vae/scaleedit/edited_image",
    "webshart": {
      "format": "parquet",
      "cache_dir": "cache/webshart/scaleedit/edited_image",
      "shard_cache_gb": 0,
      "parallel_downloads": 4,
      "columns": {
        "image": "edited_image",
        "captions": "edit_instruction",
        "width": "edited_image_width",
        "height": "edited_image_height"
      }
    }
  },
  {
    "id": "scaleedit-sources",
    "type": "webshart",
    "source": "InternVL-U/ScaleEdit-12M",
    "metadata": "webshart/ScaleEdit-12M-metadata",
    "subfolder": "3.4_count_change",
    "caption_strategy": "webshart",
    "metadata_backend": "webshart",
    "resolution": 512,
    "resolution_type": "pixel",
    "crop": false,
    "dataset_type": "conditioning",
    "conditioning_type": "reference_strict",
    "source_dataset_id": "scaleedit-edits",
    "cache_dir_vae": "cache/vae/scaleedit/source_image",
    "webshart": {
      "format": "parquet",
      "cache_dir": "cache/webshart/scaleedit/source_image",
      "shard_cache_gb": 0,
      "parallel_downloads": 4,
      "columns": {
        "image": "source_image",
        "captions": "edit_instruction",
        "width": "source_image_width",
        "height": "source_image_height"
      }
    }
  },
  {
    "id": "scaleedit-text",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/scaleedit"
  }
]
```

## कॉन्फ़िगरेशन का विवरण

- `scaleedit-edits` में `edited_image` प्रशिक्षण लक्ष्य और `edit_instruction` कैप्शन है। width/height कॉलम संपादित इमेज के हैं।
- `scaleedit-sources` में `source_image` conditioning इमेज है। `conditioning_data` लक्ष्य को references से और `source_dataset_id` references को वापस लक्ष्य से जोड़ता है। मूल इमेज के आयाम स्पष्ट चुने गए हैं।
- दोनों प्रविष्टियों में source, metadata और `subfolder` समान रखें, जिससे shard/row पहचान मेल खाए। VAE और Webshart cache directories अलग रखें।
- `webshart.shard_cache_gb: 0` range reads बनाए रखता है। पूरे dataset के अधिकतर shards लगभग 31 GB हैं; whole-shard cache पहली इमेज पढ़ते समय पूरा shard डाउनलोड कर सकता है।
- इस शुरुआती recipe में `crop: false` स्वतंत्र crops रोकता है। `reference_strict` reference पर लक्ष्य की तैयारी वाली geometry लगाता है। चुने गए editing model के अनुसार preprocessing रखें।

## उदाहरण से आगे बढ़ना

`subfolder` बदलने या हटाने से पहले पूरे dataset की 702,078 URL-only मूल इमेज संभालें। सहायक metadata URL, SHA-256 और fetch date रखता है, पर डाउनलोड नहीं करता। अनुपस्थित embedded `source_image` का परिणाम `None` रहता है। इन स्रोतों को स्पष्ट रूप से डाउनलोड और सत्यापित करें, या path तथा row position से मेल खाते source/target subsets बनाएँ।

लक्ष्य और reference को अलग-अलग सीमित या filter न करें; चुनी गई samples समान होनी चाहिए। Bucketing अलग होने पर समान `max_num_samples` भी समान जोड़ों की गारंटी नहीं देता।

Whole-shard cache बंद होने पर भी बड़े row groups और dictionaries इमेज पढ़ते समय सैकड़ों MiB ले सकते हैं। Metadata bucketing सहायक रिपॉज़िटरी से होता है, लेकिन इमेज का memory और network खर्च अलग है। बड़ा run शुरू करने से पहले [Webshart loader benchmark](../DATALOADER.md#parquet-image-bytes-on-s3) से सीमित sample माप सकते हैं।

# ACE-Step Songs डेमो डेटासेट

## विवरण

- **Hub लिंक**: [Yi3852/ACEStep-Songs](https://huggingface.co/datasets/Yi3852/ACEStep-Songs)
- **विवरण**: ACE-Step द्वारा स्वयं सिंथेसाइज़ किए गए ~21k गानों का डेटासेट, जिसमें LLMs द्वारा जनरेट किए गए lyrics और GPT-4o द्वारा स्कोरिंग शामिल है। यह ट्रेनिंग पाइपलाइन को बूटस्ट्रैप या टेस्ट करने के लिए उपयोगी है।
  - `score_lyrics`: lyrics गुणवत्ता के लिए GPT-4o स्कोर (1-10); -1 का मतलब instrumental है।
- **कैप्शन फ़ॉर्मेट**: HuggingFace dataset features: `prompt`, `lyrics`। (इसके अलावा `audio_duration` और `score_lyrics` भी हैं जिन्हें उपयोग नहीं किया जाता)।

## डाटालोडर कॉन्फ़िगरेशन उदाहरण

Hugging Face से डेटासेट सीधे लोड करने के लिए अपने `multidatabackend.json` में यह कॉन्फ़िगरेशन उपयोग करें।

```json
[
  {
    "id": "acestep-demo-data",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "Yi3852/ACEStep-Songs",
    "metadata_backend": "huggingface",
    "caption_strategy": "huggingface",
    "cache_dir_vae": "cache/vae/{model_family}/acestep-demo-data"
  },
  {
    "id": "alt-embed-cache",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/{model_family}"
  }
]
```

## Citation

```bibtex
@misc{jiang2025advancingfoundationmodelmusic,
      title={Advancing the Foundation Model for Music Understanding},
      author={Yi Jiang and Wei Wang and Xianwen Guo and Huiyun Liu and Hanrui Wang and Youri Xu and Haoqi Gu and Zhongqian Xie and Chuanjiang Luo},
      year={2025},
      eprint={2508.01178},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2508.01178},
}
```

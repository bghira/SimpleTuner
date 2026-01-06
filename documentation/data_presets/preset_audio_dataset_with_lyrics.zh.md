# ACE-Step Songs 演示数据集

## 详情

- **Hub 链接**: [Yi3852/ACEStep-Songs](https://huggingface.co/datasets/Yi3852/ACEStep-Songs)
- **描述**: 由 ACE-Step 自行合成的约 2.1 万首歌曲数据集，歌词由 LLM 生成并由 GPT-4o 打分。适合用于启动或测试训练流水线。
  - `score_lyrics`: GPT-4o 歌词质量评分（1-10）；-1 表示纯音乐。
- **字幕格式**: HuggingFace 数据集特征为 `prompt`, `lyrics`。（另含 `audio_duration` 与 `score_lyrics`，但未使用）

## 数据加载器配置示例

在 `multidatabackend.json` 中使用以下配置，可直接从 Hugging Face 加载该数据集。

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

## 引用

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

# ACE-Step Songs デモデータセット

## 詳細

- **Hub リンク**: [Yi3852/ACEStep-Songs](https://huggingface.co/datasets/Yi3852/ACEStep-Songs)
- **説明**: ACE-Step 自身が合成した約 2.1 万曲のデータセット。歌詞は LLM が生成し、GPT-4o がスコアリングしています。ブートストラップや学習パイプラインのテストに便利です。
  - `score_lyrics`: 歌詞品質の GPT-4o スコア（1〜10）。-1 はインストゥルメンタルを示します。
- **キャプション形式**: HuggingFace データセットの特徴量は `prompt`, `lyrics`。（`audio_duration` と `score_lyrics` も含まれますが未使用）

## データローダ設定例

`multidatabackend.json` で以下の設定を使うと、Hugging Face から直接読み込めます。

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

# Dataset de demonstracao ACE-Step Songs

## Detalhes

- **Link do Hub**: [Yi3852/ACEStep-Songs](https://huggingface.co/datasets/Yi3852/ACEStep-Songs)
- **Descricao**: Um dataset de ~21k musicas sintetizadas pelo proprio ACE-Step, com letras geradas por LLMs e pontuadas pelo GPT-4o. Isso e util para bootstrap ou para testar o pipeline de treinamento.
  - `score_lyrics`: score do GPT-4o (1-10) para qualidade das letras; -1 indica instrumental.
- **Formato(s) de caption**: Recursos do dataset HuggingFace: `prompt`, `lyrics`. (Tambem contem `audio_duration` e `score_lyrics`, que nao sao usados).

## Exemplo de configuracao do dataloader

Use esta configuracao no seu `multidatabackend.json` para carregar o dataset diretamente do Hugging Face.

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

## Citacao

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

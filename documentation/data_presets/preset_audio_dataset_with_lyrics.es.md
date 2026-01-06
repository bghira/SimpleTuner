# Dataset de demostración de canciones ACE-Step

## Detalles

- **Enlace en Hub**: [Yi3852/ACEStep-Songs](https://huggingface.co/datasets/Yi3852/ACEStep-Songs)
- **Descripción**: Un dataset de ~21k canciones sintetizadas por el propio ACE-Step, con letras generadas por LLMs y puntuadas por GPT-4o. Esto es útil para iniciar o probar el pipeline de entrenamiento.
  - `score_lyrics`: puntuación de GPT-4o (1-10) para la calidad de las letras; -1 indica instrumental.
- **Formato(s) de caption**: Características del dataset de Hugging Face: `prompt`, `lyrics`. (También contiene `audio_duration` y `score_lyrics`, que no se usan).

## Ejemplo de configuración del dataloader

Usa esta configuración en tu `multidatabackend.json` para cargar el dataset directamente desde Hugging Face.

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

## Cita

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

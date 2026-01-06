Um recurso experimental no SimpleTuner implementa as ideias por tras de ["Demystifying SD fine-tuning"](https://github.com/spacepxl/demystifying-sd-finetuning) para fornecer um valor de loss estavel para avaliacao.

Devido a natureza experimental, pode causar problemas ou carecer de funcionalidades / integracao que um recurso totalmente finalizado teria.

E aceitavel usar este recurso em producao, mas esteja atento a possiveis bugs ou mudancas em versoes futuras.

Exemplo de dataloader:

```json
[
    {
        "id": "something-special-to-remember-by",
        "crop": false,
        "type": "local",
        "instance_data_dir": "/datasets/pseudo-camera-10k/train",
        "minimum_image_size": 512,
        "maximum_image_size": 1536,
        "target_downsample_size": 512,
        "resolution": 512,
        "resolution_type": "pixel_area",
        "caption_strategy": "filename",
        "cache_dir_vae": "cache/vae/sana",
        "vae_cache_clear_each_epoch": false,
        "skip_file_discovery": ""
    },
    {
        "id": "sana-eval",
        "type": "local",
        "dataset_type": "eval",
        "instance_data_dir": "/datasets/test_datasets/squares",
        "resolution": 1024,
        "minimum_image_size": 1024,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "resolution_type": "pixel_area",
        "cache_dir_vae": "cache/vae/sana-eval",
        "caption_strategy": "filename"
    },
    {
        "id": "text-embed-cache",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "cache/text/sana"
    }
]
```

- Datasets de imagens de avaliacao podem ser configurados exatamente como um dataset de imagem normal.
- O dataset de avaliacao **nao** e usado para treinamento.
- E recomendado usar imagens que representem conceitos fora do seu conjunto de treino.

Para configurar e habilitar calculos de loss de avaliacao:

```json
{
    "--eval_steps_interval": 10,
    "--eval_epoch_interval": 0.5,
    "--num_eval_images": 1,
    "--report_to": "wandb"
}
```

As avaliacoes agora podem ser agendadas por step ou por epoca. `--eval_epoch_interval` aceita valores decimais, entao `0.5`
rodara avaliacao duas vezes por epoca. Se voce definir `--eval_steps_interval` e `--eval_epoch_interval`, o
trainer registrara um aviso e rodara avaliacoes em ambos os cronogramas.

Para desabilitar calculos de loss de avaliacao enquanto mantem datasets de avaliacao configurados (ex.: apenas para CLIP):

```json
{
    "--eval_loss_disable": true
}
```

Isso e util quando voce quer usar datasets de avaliacao para metricas de CLIP score (`--evaluation_type clip`) sem o overhead de computar loss de validacao a cada timestep.

> Nota: Weights & Biases (wandb) e atualmente necessario para a funcionalidade completa de graficos de avaliacao. Outros trackers recebem apenas o valor medio.

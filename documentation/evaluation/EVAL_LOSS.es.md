Una función experimental en SimpleTuner implementa las ideas de ["Demystifying SD fine-tuning"](https://github.com/spacepxl/demystifying-sd-finetuning) para proporcionar un valor de pérdida estable para evaluación.

Debido a su naturaleza experimental, puede causar problemas o carecer de funcionalidad / integración que tendría una función completamente finalizada.

Está bien usar esta función en producción, pero ten en cuenta el potencial de errores o cambios en versiones futuras.

Ejemplo de dataloader:

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

- Los datasets de imágenes de evaluación se pueden configurar igual que un dataset normal de imágenes.
- El dataset de evaluación **no** se usa para entrenamiento.
- Se recomienda usar imágenes que representen conceptos fuera de tu conjunto de entrenamiento.

Para configurar y habilitar los cálculos de pérdida de evaluación:

```json
{
    "--eval_steps_interval": 10,
    "--eval_epoch_interval": 0.5,
    "--num_eval_images": 1,
    "--report_to": "wandb"
}
```

Las evaluaciones ahora se pueden programar por paso o por época. `--eval_epoch_interval` acepta valores decimales, por lo que `0.5`
ejecutará la evaluación dos veces por época. Si configuras tanto `--eval_steps_interval` como `--eval_epoch_interval`, el
trainer registrará una advertencia y ejecutará evaluaciones en ambos horarios.

Para desactivar el cálculo de pérdida de evaluación mientras mantienes los datasets de evaluación configurados (p. ej., solo para puntuaciones CLIP):

```json
{
    "--eval_loss_disable": true
}
```

Esto es útil cuando quieres usar datasets de evaluación para métricas de puntuación CLIP (`--evaluation_type clip`) sin el overhead de calcular la pérdida de validación en cada timestep.

> **Nota**: Weights & Biases (wandb) es actualmente requerido para la funcionalidad completa de gráficos de evaluación. Otros trackers solo reciben el valor medio único.

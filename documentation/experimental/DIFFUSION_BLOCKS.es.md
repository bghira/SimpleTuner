# DiffusionBlocks

DiffusionBlocks convierte un diffusion Transformer compatible en grupos de capas entrenables de forma independiente. Cada grupo cubre un rango de ruido; cada forward ejecuta solo el grupo asignado al batch.

Es una conversion experimental basada en [DiffusionBlocks](https://arxiv.org/abs/2506.14202), no un congelado normal de capas. La inferencia debe usar el mismo routing que el entrenamiento.

## Configuracion

```json
{
  "diffusion_blocks_config": {
    "layers_per_block": 4,
    "overlap": 0.05
  },
  "find_unused_parameters": true
}
```

DDP activa `find_unused_parameters` automaticamente. Configurarlo como `false` produce un error.

| Clave | Valor predeterminado | Significado |
| --- | --- | --- |
| `layers_per_block` | obligatorio | Maximo de capas Transformer consecutivas por bloque de ruido. |
| `overlap` | `0.05` | Expansion fraccional de rangos vecinos, entre `0.0` y `0.5`. |
| `blocks_to_train` | `"all"` | Indices propiedad de este job. Los demas grupos se congelan despues de crear el adapter. |
| `block_paths` | automatico | Rutas `ModuleList` explicitas cuando el descubrimiento automatico no basta. |
| `timestep_boundaries` | automatico | Limites ascendentes de `0.0` a `1.0`, con `num_blocks + 1` valores. |

Los limites automaticos dividen la distribucion de timestep en rangos de igual probabilidad. El bloque `0` recibe el ruido mas alto y las primeras capas.

## Compatibilidad

La implementacion compartida admite familias diffusion y flow-matching con listas homogeneas de bloques Transformer: una etapa, joint/single stream, double/single stream, `blocks` y `layers`.

UNet, ControlNet, Musubi block swap, TwinFlow, scheduled sampling con varios timesteps, CREPA con captura de capa fija y LayerSync se rechazan al iniciar. Las rutas TREAD conservan los indices globales del modelo y se recortan al rango global del grupo activo.

El routing cambia la arquitectura del denoiser. La perdida inicial y la calidad no tienen por que coincidir con un run normal de profundidad completa. Activar esta opcion no convierte un LoRA normal existente en un adapter DiffusionBlocks entrenado.

Usa `block_paths` solo para etapas secuenciales del denoiser. No selecciones adapters de texto, bloques VAE ni etapas UNet con skip connections.
Los stacks Transformer encoder-decoder con dependencias skip, como `in_blocks`/`out_blocks` de i1, no se detectan porque un grupo de salida no puede ejecutarse sin las activations de su grupo de entrada asociado.

## Memoria

Solo el grupo activo crea activations del Transformer. Un run con todos los bloques termina asignando optimizer state para todos los grupos entrenables.

Para jobs independientes, configura `blocks_to_train` por job. Los grupos no asignados se congelan y no reciben optimizer state. Combina los checkpoints por propiedad de parametros antes de inferencia.

Group offload es compatible. Musubi block swap no lo es.

## Inferencia

La validacion de SimpleTuner usa el controller automaticamente. Un pipeline Diffusers normal no deduce esta conversion desde los pesos LoRA.

```python
from simpletuner.helpers.training.diffusion_blocks import DiffusionBlocksConfig, DiffusionBlocksController

config = DiffusionBlocksConfig.from_dict({"layers_per_block": 4, "overlap": 0.05})
controller = DiffusionBlocksController(pipe.transformer, config)
```

Conserva `controller` durante la vida del pipeline y usa la configuracion exacta de `simpletuner_config.json`.

## Ejemplo Anima

Consulta `simpletuner/examples/anima.peft-lora+diffusion-blocks/config.json`. Las 28 capas de Anima v1.0 forman 7 bloques con `layers_per_block=4`.

```bash
simpletuner train env=examples/anima.peft-lora+diffusion-blocks max_train_steps=10 validation_steps=10
```

Al reanudar no cambies paths, numero de capas, limites, `blocks_to_train`, topologia, world size, batch sampling ni timesteps. Ejecutar todas las capas en inferencia invalida el objetivo entrenado.

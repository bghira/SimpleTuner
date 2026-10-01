# Guía rápida de Qwen Image 2.1

[Qwen Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) usa un transformer de imagen de 7B y 32 bloques, un encoder de texto Qwen3-VL y un VAE de 64 canales con compresión espacial de 16×. SimpleTuner lo selecciona por defecto con `model_family: "qwen_image"` y `model_flavour: "v2.1"`. Esta guía cubre entrenamiento LoRA de texto a imagen.

Para los modelos de 20B `v1.0` / `v2.0`, consulta la [guía anterior de Qwen Image](QWEN_IMAGE.es.md). El entrenamiento con referencias pareadas de las variantes antiguas `edit-*` está en [Qwen Edit](QWEN_EDIT.es.md). Sus adaptadores, embeddings de texto y cachés de latentes no son intercambiables con 2.1.

[Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)

## Instalación

Usa Python 3.12–3.14 y sigue la [guía de instalación](../INSTALL.es.md) para tu plataforma. Los presets siguientes se probaron en GPU NVIDIA. `webshart` ya está incluido en las dependencias; los datos de regularización requieren acceso a la red y espacio de caché local.

```bash
pip install 'simpletuner[cuda]'
```

<a id="vram-presets"></a>

## Elige un preset de VRAM

Los ejemplos combinan [asistente v3](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v3), regularización sintética, REPA y desplazamiento automático del flujo. El ejemplo predeterminado `qwen_image.peft-lora` usa la misma receta 512px + 1024px del preset de 48 GB. Los de 24/32 GB solo usan 512px, incluida la validación.

| VRAM | Ejemplo | Resoluciones base | Actualizaciones | Intervalo de checkpointing de gradientes |
| --- | --- | --- | --- | --- |
| 24 GB | `qwen_image-2.1-24g.peft-lora` | 512px | 2000 | 1 |
| 32 GB | `qwen_image-2.1-32g.peft-lora` | 512px | 2000 | 2 |
| 48 GB | `qwen_image-2.1-48g.peft-lora` | 512px + 1024px | 4000 | 2 |
| 80 GB | `qwen_image-2.1-80g.peft-lora` | 512px + 1024px | 4000 | 2 |
| 144 GB | `qwen_image-2.1-144g.peft-lora` | 512px + 1024px | 4000 | 2 |

```bash
simpletuner train example=qwen_image-2.1-48g.peft-lora
```

Todos usan BF16, rango/alfa 32, lote 1, AdamW BF16 a `1e-4`, 25 pasos de calentamiento y recorte de norma a 1.0. REPA usa `dinov2_vitg14`, bloque 8, peso 0.5, tamaño 518, alineación espacial y distancia temporal 0. El desplazamiento automático está activado y el estático vale 0. El asistente permanece congelado y se desactiva en validación. La regularización toma como objetivo la predicción del modelo base con ambos adaptadores desactivados.

El muestreador probabilístico habitual asigna la mitad del peso a `RareConcepts/Domokun` con el activador `🟫` y la otra mitad a `webshart/qwen-image-2.1-generated-images` con `is_regularisation_data: true`. En multiescala, cada mitad se divide por igual entre 512px y 1024px. Los buckets por área conservan las proporciones: 0.262144 y 1.048576 megapíxeles. Los subconjuntos sintéticos cuadrados, verticales y horizontales comparten por igual el peso de su resolución. No hay alternancia estricta. Se valida y guarda cada 250 pasos.

Los dataloaders limitan cada subconjunto sintético de proporción a 1.024 imágenes y separan las cachés de latentes por resolución y fuente. VAE tiling está desactivado. Estos presupuestos son puntos de partida: revisa las imágenes antes de ampliar el entrenamiento.

<details markdown="1">
<summary>Mediciones de memoria</summary>

Las pruebas de memoria en L40S usaron 16 pasos, cuatro imágenes por backend, validación y guardado. La receta 512px con intervalo 1 pasó con un límite de 24 GiB: pico de 20.06 GiB asignados / 21.10 GiB reservados por PyTorch. La receta multiescala con intervalo 2 pasó en L40S con 32.49 / 41.21 GiB, pero agotó la memoria bajo un límite de 32 GiB. Son pruebas de memoria con subconjuntos pequeños, no de rendimiento ni convergencia; los límites menores se simularon en L40S. El preset final de 32 GB, con 512px e intervalo 2, también pasó: 22.12 GiB asignados / 23.68 GiB reservados.

</details>

## Usa tus propios datos

Copia `config.json` y `dataloader.json` del ejemplo elegido al entorno de entrenamiento y proporciona tu biblioteca de prompts. Ajusta `data_backend_config`, `user_prompt_library` y `output_dir` a esos archivos. La organización se explica en el [tutorial](../TUTORIAL.es.md) y la [referencia del dataloader](../DATALOADER.es.md).

Sustituye los backends Domokun por tu dataset de concepto o fotos. Para multiescala, define backends distintos de 512px y 1024px, con IDs y rutas de caché VAE separados. `resolution_type: "area"` usa megapíxeles: **0.262144** y **1.048576**. `crop: false` conserva las proporciones. Mantén `aspect_bucket_alignment: 32` en la configuración.

En la receta combinada, conserva los backends sintéticos con `is_regularisation_data: true` y marca los de entrenamiento como false. Asigna la mitad del peso al entrenamiento y la otra mitad a regularización; divide cada mitad entre resoluciones. El sampler es probabilístico, sin alternancia estricta. Repeats controla la disponibilidad del dataset, no su probabilidad. Inicia una ejecución nueva al cambiar datos, resolución, lotes o topología distribuida.

## Elige el asistente de entrenamiento

Main y los ejemplos usan actualmente **asistente v3**. Permanece congelado con intensidad 1 durante el entrenamiento y desactivado en validación. El objetivo padre de los lotes de regularización es la **predicción de la base sin adaptadores**, con LoRA entrenable y asistente desactivados.

El [asistente v3](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v3) es el adaptador rank-64 predeterminado, entrenado durante 30k actualizaciones a 512/1024/1536/2048. Las mediciones de memoria de los presets usaron v2; comprueba la VRAM disponible al usar v3. Para reproducir ejecuciones anteriores, selecciona explícitamente el [asistente v2](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v2) con esta configuración:

```json
{
  "assistant_lora_path": "SimpleTuner/Qwen-Image-2.1-training-assistant-v2",
  "assistant_lora_strength": 1.0,
  "assistant_lora_inference_strength": 0.0
}
```

Configura `disable_assistant_lora: true` para entrenar sin asistente. Usa también esa opción al crear un asistente nuevo. En inferencia downstream carga solo la LoRA entrenada; el asistente y el proyector REPA son componentes de entrenamiento.

## Valida el aprendizaje y la calidad

Los ejemplos validan y guardan cada 250 actualizaciones, con 40 pasos de inferencia, true CFG 1 y semilla 42. Validan a 512px en 24/32 GB y a 512px y 1024px en los mayores. Mantén estos ajustes en una configuración propia:

```json
{
  "validation_guidance": 1.0,
  "validation_guidance_real": 1.0,
  "validation_num_inference_steps": 40,
  "validation_seed": 42,
  "validation_step_interval": 250,
  "checkpoint_step_interval": 250
}
```

Incluye el activador en los prompts del concepto y sujetos ajenos para comprobar coherencia y calidad. Compara checkpoints con el mismo prompt, semilla, decoder, resolución y guidance. Resolución de entrenamiento y de salida son distintas: prueba también a 1024px un adaptador entrenado a 512px. Estas recetas no requieren datos del concepto a 2048px.

El [libro de experimentos](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-LoRA-experiments) compara asistentes, regularización, REPA, auto shift y entrenamiento prolongado. Algunas ejecuciones empeoran y después se recuperan; un checkpoint débil no define la trayectoria. La [comparación photo-aesthetics v3](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-LoRA-photo-aesthetics-v3) muestra más detalle de escena tras 50k actualizaciones a 512px que tras 10k a 1024px, incluso generando a 1MP; los presupuestos de actualizaciones difieren.

## VAE e inferencia

Qwen Image 2.1 usa el [VAE con corrección de textura de Ollin](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1) de forma predeterminada para la validación y otras operaciones de decodificación VAE. Solo se ajustó el decodificador; el codificador no cambió, por lo que los latentes de entrenamiento existentes de la versión 2.1 siguen siendo compatibles. La revisión del VAE se fija independientemente del modelo base. Para reproducir resultados con el VAE original, configura `pretrained_vae_model_name_or_path: "Qwen/Qwen-Image-2.1"`. Las sustituciones explícitas del VAE y las variantes anteriores mantienen su comportamiento.

Para adaptadores PEFT estándar, el pipeline incluido en SimpleTuner evita instalar Diffusers desde Git. Este ejemplo lee la LoRA exportada y excluye los tensores del proyector REPA usados solo durante el entrenamiento. Decodifica el fotograma completo y no carga ningún asistente:

El offload de modelos a CPU evita que el encoder de texto, el transformer y el VAE ocupen la GPU a la vez. Con suficiente VRAM, sustituye `pipe.enable_model_cpu_offload()` por `pipe.to("cuda")` para acelerar inferencias repetidas.

```python
import torch
from safetensors.torch import load_file
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from simpletuner.helpers.models.qwen_image.autoencoder_21 import AutoencoderKLQwenImage21
from simpletuner.helpers.models.qwen_image.pipeline_21 import QwenImage21Pipeline
from simpletuner.helpers.models.qwen_image.transformer_21 import QwenImage21Transformer2DModel

vae = AutoencoderKLQwenImage21.from_pretrained(
    "madebyollin/texture-fix-vae-for-qwen-image-2.1",
    revision="e9f84623d22c47f8bc9fb799bc54201fa53cf80b",
    torch_dtype=torch.bfloat16,
)
model_id = "Qwen/Qwen-Image-2.1"
pipe = QwenImage21Pipeline(
    transformer=QwenImage21Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
    ),
    vae=vae,
    text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, subfolder="text_encoder", dtype=torch.bfloat16,
    ),
    processor=Qwen3VLProcessor.from_pretrained(model_id, subfolder="processor"),
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler"),
)
pipe.enable_model_cpu_offload()
pipe.vae.disable_tiling()
pipe.vae.enable_slicing()
weights = load_file("output/my-qwen21/pytorch_lora_weights.safetensors")
weights = {
    key: value for key, value in weights.items()
    if ".lora_A." in key or ".lora_B." in key
}
pipe.load_lora_weights(weights, adapter_name="trained")
pipe.set_adapters("trained", adapter_weights=1.0)
image = pipe(
    prompt="🟫 wearing a red scarf, standing in a snowy forest",
    width=1024, height=1024, num_inference_steps=40, true_cfg_scale=1.0,
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]
image.save("output.png")
```

## Memoria y artefactos de imagen

Empieza con BF16 (`base_model_precision: "no_change"`). Reduce el lote o elige un preset menor antes de cuantizar; la cuantización reduce memoria, pero no garantiza mayor velocidad. Los ejemplos activan compilación regional. Captions largos, más resoluciones, validación y REPA cambian el pico de VRAM. Mide el margen antes de aumentar el lote.

Mantén `vae_enable_tiling: false` cuando haya memoria. El decoder corregido trata la textura de lienzo; las uniones de color del tiling son un problema distinto de contexto espacial. El recorte de norma 1.0 usa `grad_clip_method: "norm"` y `max_grad_norm: 1.0`, distinto del recorte por elemento 0.01 del antiguo piloto AnyFlow.

<details markdown="1">
<summary>Mediciones de memoria: VAE</summary>

Qwen Image 2.1 decodifica imágenes individuales sin conservar cachés temporales de características que no se utilizan. Al decodificar una imagen de 2048×2048 de forma aislada en H200 con BF16, esto redujo el pico de memoria asignada de 26.87 a 15.28 GiB con una salida idéntica. La decodificación por bloques ahorra más memoria, pero puede introducir líneas de color al limitar el contexto espacial; eliminar los cachés sin uso no corrige esas líneas.

</details>

## Recetas históricas y experimentales

Las recetas siguientes crean asistentes o prueban destilación. Son independientes de la receta downstream recomendada arriba; los ajustes y resultados antiguos se conservan como referencia.

<details markdown="1">
<summary>Piloto AnyFlow e historial del optimizador</summary>

<a id="experimental-anyflow-pilot"></a>

### Piloto experimental de AnyFlow

Los tres ejemplos `qwen_image-2.1-anyflow-stage*.peft-lora` prueban si la destilación de intervalos puede conservar el comportamiento de la base al introducir `🟫`. Son experimentales; la ficha de Qwen no confirma que la destilación de guidance causara el deterioro anterior.

Todas las etapas usan 1024px, BF16, AdamW, rank 32 y batch 4. La etapa 1 desactiva el checkpointing de gradientes en H200; las etapas 2 y 3 usan grupos contiguos de dos bloques. Es un entrenamiento de destilación distinto de los presets con REPA anteriores.

Estos ejemplos de AnyFlow usan explícitamente `grad_clip_method: "value"` con `max_grad_norm: 0.01`: cada elemento del gradiente se limita a ±0.01. No es un límite de la norma global. Para probar el recorte por norma a 1.0, configura tanto `grad_clip_method: "norm"` como `max_grad_norm: 1.0`.

1. **Etapa 1:** 10.000 actualizaciones forward de AnyFlow a `1e-5`. CC12M usa `webshart/cc12m-structured-captions` con `caption_key: "long_caption"`; e621 usa `webshart/e621-2024-webp-4Mpixel-webshart-indices`. Cada conjunto de conocimientos previos se limita a 4.096 imágenes aceptadas, con peso 0.49 cada uno. `RareConcepts/Domokun` usa el activador `🟫`, peso 0.02 y `repeats: 0`.
2. **Etapa 2:** 2.000 actualizaciones DMD on-policy a `2e-6`, coentrenando el objetivo forward con la misma mezcla. Es DMD de AnyFlow, no DPO con pares de preferencias. Incluir el personaje en ambas etapas aporta ejemplos reales; el profesor congelado por sí solo no puede enseñarlo.
3. **Etapa 3:** 100 actualizaciones a `5e-7`, con peso 0.5 para Domokun y 0.25 para cada conjunto de regularización, limitado a 64 imágenes cada uno. Todos los intervalos usan `r=t` y el objetivo flow bruto, conservando el embedder de intervalos durante un ajuste supervisado breve. Los batches de regularización usan la predicción de la base con el adapter desactivado.

Revisa los checkpoints de 2.000, 5.000 y 10.000 actualizaciones antes de ampliar la etapa 1 hasta 20.000. La etapa 2 tiene un presupuesto de 2.000 actualizaciones; detenla antes si las imágenes de validación comparables empeoran. El número de pasos por sí solo no determina un modelo final.

La compilación dinámica está habilitada para captions de longitud variable. `schedule_shift: 2.000802574061872` corresponde al scheduler publicado a 1024px (4.096 tokens latentes); vuelve a calcularlo al cambiar la resolución.

```bash
simpletuner train example=qwen_image-2.1-anyflow-stage1.peft-lora
simpletuner train example=qwen_image-2.1-anyflow-stage2.peft-lora
simpletuner train example=qwen_image-2.1-anyflow-stage3.peft-lora
```

Las etapas 2 y 3 cargan el adapter final anterior mediante `init_lora` e inician estados nuevos del optimizador y dataloader. Los pesos controlan la selección entre conjuntos todavía disponibles, sin garantizar porcentajes finales. El piloto limitado no cubre los corpus completos. El campo textual `long_caption` funciona con el selector Webshart existente y no requiere captions como objetos JSON nativos.

Las etapas 1 y 2 usan `diffusion_target: "base_prediction"` y `fuse_guidance_scale: 1.0`: la rama diffusion conserva el campo condicional congelado mientras las otras ramas aprenden de los datos. Es una hipótesis de conservación, no una garantía de aprender el personaje. Compara los prompts de personaje y de conocimientos previos a 4, 16 y 40 pasos de inferencia con una referencia de la base a 40 pasos; la validación automática usa 4. Consulta [AnyFlow](../experimental/ANYFLOW.es.md) para los objetivos y requisitos de checkpoints.

Piloto completado: la etapa 1 llegó a 10.000 actualizaciones y la etapa 2 a 2.000. Tras recargar los adaptadores, las imágenes de 40 pasos del zorro y de los prompts del personaje conservaron coherencia, pero estos últimos produjeron personas en vez de Domokun. Las imágenes de cuatro pasos siguieron borrosas o ruidosas. Otra comparación en L40S probó ambos checkpoints a 1024px, semilla 42 y CFG 1/2/4/6, con prompt negativo vacío. Las aserciones de llamadas verificaron dos pasadas por paso para CFG mayor que 1. Las muestras revisadas del zorro y la playa no mejoraron: aumentar CFG añadió saturación y artefactos. Estos resultados no identifican la causa; la etapa 3 sigue sin validarse.

Dos continuaciones de la etapa 1 desde el mismo checkpoint añadieron 1.000 actualizaciones cada una, comparando umbrales de clipping por valor de 0.01 y 1.0. Tras recargar, las imágenes del zorro a cuatro pasos siguieron ruidosas en ambas; las imágenes de playa a 40 pasos siguieron mostrando personas. El clipping se activó en 65/1.000 actualizaciones con 0.01 y en 0/1.000 con 1.0. Ambas ramas usaron el optimizador sin corregir, y sus gradientes iniciales diferían antes de activarse el clipping; las diferencias pequeñas no pueden atribuirse únicamente al umbral.

<a id="qwen21-optimizer-correction"></a>

Los pilotos de la etapa 1 y del asistente descritos aquí se ejecutaron antes de corregir el helper de suma estocástica de AdamW BF16: calculaba `other + alpha * input` en lugar de `input + alpha * other`. Con β₁ = 0,9, el primer momento seguía `m = 0.09 * m + g` en vez de `m = 0.9 * m + 0.1 * g`. Las pruebas de regresión con aritmética exacta cubren CPU, MPS y CUDA. Las observaciones de las imágenes siguen siendo válidas para esos checkpoints, pero no constituyen una prueba limpia de la arquitectura ni del objetivo de destilación; hay que repetir la comprobación con el optimizador corregido.

Repetir la continuación de 1.000 actualizaciones con el optimizador corregido, el mismo checkpoint inicial y clipping por valor de 1.0 no recuperó las imágenes revisadas del zorro ni de la playa a cuatro pasos. A 40 pasos el zorro conservó coherencia, mientras que el prompt de playa siguió produciendo una persona. Esto evalúa la recuperación del checkpoint existente, no el entrenamiento desde cero con el optimizador corregido; la causa del fallo general sigue sin resolverse.

</details>

<details markdown="1">
<summary>Crear un asistente: recetas online, multirresolución y offline</summary>

<a id="assistant-lora"></a>

### Entrenar un LoRA auxiliar a partir de descripciones

`qwen_image-2.1-assistant-lora.peft-lora` es un punto de partida experimental para L40S: BF16, lote 1, checkpointing con intervalo 2, rango 32 y AdamW BF16 a `1e-4`. Presupuesta 1.000 actualizaciones y valida/guarda cada 50. Sustituye las doce descripciones de prueba por textos diversos antes de entrenar en serio; la convergencia no está validada.

`grad_clip_method: "norm"`, `max_grad_norm: 1.0`.

Selecciona `distillation_method: assistant_lora`. El backend precalcula los embeddings de texto. Cada lote genera latentes nuevos del modelo base con el adaptador desactivado, 40 pasos nativos y CFG 1. La canalización privada reutiliza el transformer sin cargar VAE, procesador ni codificador de texto. Después restaura el adaptador y ejecuta el entrenamiento de eliminación de ruido habitual. No almacena latentes finales. Actualmente solo admite texto a imagen con Qwen Image 2.1.

`distillation_config.assistant_lora` acepta `num_inference_steps` (40 por defecto), `resolutions` (lista no vacía de `[ancho, alto]`, por defecto `[[1024, 1024]]`) y `seed` (42). Las dimensiones deben ser múltiplos de 32. Las resoluciones rotan por lote y las semillas avanzan por muestra, incluidos los lotes incompletos. Los checkpoints guardan estos contadores. Reanuda sin cambiar datos, lote, acumulación ni topología distribuida. La caché de texto bajo demanda no está admitida.

Para comprobar el funcionamiento, usa 8 actualizaciones y 2 pasos del profesor; vuelve a 40 antes de evaluar imágenes. Revisa a las 100, 250, 500 y 1.000 actualizaciones con los mismos prompts y semillas del modelo base. La utilidad del auxiliar requiere otra prueba de entrenamiento de conceptos.

El entrenamiento LoRA de Qwen Image 2.1 carga por defecto el [asistente v3](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v3), congelado durante el entrenamiento y desactivado durante la validación. Usa `disable_assistant_lora: true` para desactivarlo, o `assistant_lora_path` para elegir otro adaptador. Las versiones anteriores conservan su comportamiento. Al entrenar un nuevo asistente, mantén `disable_assistant_lora: true`, como en los ejemplos correspondientes.

Usa este como único método de destilación; no se admite combinarlo con otros destiladores.

Los datasets de descripciones requieren `dataloader_prefetch: false` para que el cursor del checkpoint corresponda a las descripciones consumidas. La reanudación rechaza cambios en identidades/textos, lote, repeticiones, mezcla, semilla, acumulación o distribución. Los checkpoints del auxiliar también rechazan cambios en la semilla de generación, la lista de resoluciones o los pasos de inferencia del profesor.

<a id="assistant-lora-multires"></a>

#### Experimento de asistente con cuatro resoluciones base y buckets de aspecto

`qwen_image-2.1-assistant-lora-multires.peft-lora` recorre 12 buckets de relación de aspecto repartidos entre las resoluciones base 512, 1024, 1536 y 2048. Cada base incluye un bucket cuadrado, uno vertical 4:7 y otro horizontal 7:4; cada lote usa un bucket, con la misma exposición por base y aspecto durante un ciclo completo. Conserva los 40 pasos del profesor, BF16 con lote 1, checkpointing cada 2 bloques, rango 32, AdamW BF16 y 1.000 actualizaciones del ejemplo base; comparte su backend de captions y sus prompts de validación. Sustituye los captions de prueba por el mismo conjunto diverso usado en la referencia. Empieza con un adaptador y un optimizador nuevos en el nuevo directorio de salida; `resume_from_checkpoint: ""` desactiva la reanudación. Conserva los pesos de la primera ejecución para compararlos.

Esto prueba si exponer al asistente a varios tamaños mejora el resultado; no demuestra un requisito de resolución nativa ni una mejora de calidad. La ejecución de 1.000 actualizaciones en L40S completó los 12 buckets, con validación a 1024×1024 y 2048×2048. El zorro y el retrato finales conservaron coherencia, con las conocidas líneas de color del VAE con tiling. El profesor genera objetivos latentes sin decodificar con el VAE. La mejora en el entrenamiento posterior de conceptos sigue sin verificarse.

Assistant LoRA usa el mismo mínimo de 32 entradas de caché Dynamo, limitado a la ejecución, que AnyFlow para las variantes del profesor, alumno y validación. Respeta límites mayores definidos por el usuario y restaura el original al salir. Vigila las recompilaciones al añadir resoluciones o captions más largos.

Una prueba posterior de Domokun entrenó dos adaptadores nuevos durante 250 actualizaciones cada uno a 2048px, batch 1, LR `1e-5` y clipping por valor de 1.0. La intensidad del asistente durante el entrenamiento fue 0 para el control y 1 para la otra ejecución; ambas lo desactivaron durante la inferencia. Los pesos iniciales y las imágenes de validación iniciales coincidieron exactamente. A 40 pasos de inferencia y 1024px, ambos resultados finales siguieron generando personas para los prompts del personaje y conservaron zorros/retratos coherentes; no se demostró una ventaja del asistente. Ambas ejecuciones y la preparación del asistente usaron el optimizador sin corregir descrito [arriba](#qwen21-optimizer-correction).

<a id="assistant-lora-offline"></a>

#### LoRA auxiliar con imágenes generadas reutilizables

`qwen_image-2.1-assistant-lora-offline.peft-lora` entrena con [10.000 imágenes generadas](https://huggingface.co/datasets/webshart/qwen-image-2.1-generated-images) mediante Webshart. El dataset usa prompts `long_caption` de CC12M, 40 pasos nativos del profesor, CFG 1 y decodificación VAE de imagen completa. Doce backends cubren buckets cuadrados, verticales y horizontales con resoluciones base de 512, 1024, 1536 y 2048, pesos de muestreo iguales y sin repeticiones.

La receta inicia un entrenamiento nuevo con `adamw_bf16` corregido, LR `1e-4`, `grad_clip_method: "norm"`, `max_grad_norm: 1.0`, BF16 con lote 1, rango 32 y checkpointing cada 2 bloques. Presupuesta 1.000 actualizaciones, guarda cada 50 y valida cada 100 a 1024. Genera vistas previas adicionales a 2048px antes de publicar. Desactiva el tiling del VAE y codifica con lote 1. `vae_cache_ondemand: true` codifica y almacena cada imagen al muestrearla, evitando codificar las 10.000 imágenes antes de 1.000 actualizaciones. El entrenamiento habitual con imágenes sustituye la generación del profesor en línea; omite `distillation_method` y conserva `disable_assistant_lora: true` al crear el auxiliar.

Puedes reutilizar el dataset en otros experimentos. Los PNG requieren recodificación VAE y no equivalen exactamente a los latentes finales del profesor. Revisa las imágenes de validación antes de publicar y evalúa la utilidad del auxiliar en otro entrenamiento de conceptos. Al cambiar desde la receta de captions, empieza de cero sin reanudar el estado del optimizador ni del dataset.

El [asistente v1 de reemplazo](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v1) completó esta receta de 1.000 actualizaciones en L40S, con revisión final de imágenes a 1024 y 2048. Una comparación equivalente de Domokun durante 1.000 actualizaciones a 2048, LR `1e-4` y clipping de norma 1.0 mantuvo el asistente congelado durante el entrenamiento y desactivado en validación. Tanto el control como la ejecución asistida siguieron generando personas para los dos prompts del personaje. El control introdujo rasgos marcados de Domokun en un prompt de zorro no relacionado; la ejecución asistida conservó un zorro reconocible. Ambas conservaron un retrato coherente. Es evidencia limitada de menor contaminación entre conceptos, no de aprendizaje exitoso del personaje ni de una mejora general de calidad; la evaluación usa una semilla de entrenamiento y cuatro prompts.

```bash
simpletuner train example=qwen_image-2.1-assistant-lora-offline.peft-lora
```

</details>

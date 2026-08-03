# Guía de inicio rápido

**Nota**: Para configuraciones más avanzadas, consulta el [tutorial](TUTORIAL.md) y la [referencia de opciones](OPTIONS.md).

## Compatibilidad de funciones

Para la matriz de funciones completa y más precisa, consulta el [README principal](https://github.com/bghira/SimpleTuner#model-architecture-support).

## Guías de inicio rápido por modelo

| Modelo | Parámetros | PEFT LoRA | Lycoris | Rango completo | Cuantización | Precisión mixta | Checkpointing de gradiente | Flow Shift | TwinFlow | Self-Flow | LayerSync | Ref Inputs | ControlNet | Sliders† | Licencia | Permite uso comercial | Guía |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- | :---: | --- |
| PixArt Sigma | 0.6B–0.9B | ✗ | ✓ | ✓ | int8 opcional | bf16 | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | [OpenRAIL++](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | Aplican condiciones<sup>1</sup> | [SIGMA.md](quickstart/SIGMA.md) |
| NVLabs Sana | 1.6B–4.8B | ✗ | ✓ | ✓ | int8 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sí | [SANA.md](quickstart/SANA.md) |
| Kwai Kolors | 2.7B | ✓ | ✓ | ✓ | no recomendado | bf16 | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [Kwai Kolors License](https://huggingface.co/terminusresearch/kwai-kolors-1.0/blob/main/MODEL_LICENSE) | Aplican condiciones<sup>7</sup> | [KOLORS.md](quickstart/KOLORS.md) |
| Stable Diffusion 3 | 2B–8B | ✓ | ✓ | ✓ | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [Stability AI Community](https://stability.ai/license) | Aplican condiciones<sup>2</sup> | [SD3.md](quickstart/SD3.md) |
| Flux.1 | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) / [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Aplican condiciones<sup>3</sup> | [FLUX.md](quickstart/FLUX.md) |
| Flux.2 | 32B | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ opt | ✗ | ✓ | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) / [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Aplican condiciones<sup>4</sup> | [FLUX2.md](quickstart/FLUX2.md) |
| Flux Kontext | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✓ | ✓ | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) | No<sup>5</sup> | [FLUX_KONTEXT.md](quickstart/FLUX_KONTEXT.md) |
| Z-Image Turbo | 6B | ✓ | ✗ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sí | [ZIMAGE.md](quickstart/ZIMAGE.md) |
| Krea2 | - | ✓ | ✗ | ✓* | int8 opcional | bf16 | ✓+ | ✓ | ✗ | ✗ | ✗ | ✓ opt | ✗ | ✓ | [Krea 2 Community](https://www.krea.ai/krea-2-licensing) | Aplican condiciones<sup>6</sup> | [KREA2.md](quickstart/KREA2.es.md) |
| Mage-Flow | 4B | ✓ | ✓ | ✓* | int8/fp8 opcional | bf16 | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ edit | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | Sí | [MAGEFLOW.md](quickstart/MAGEFLOW.es.md) |
| Boogu-Image 0.1 | - | ✓ | ✓ | ✓* | fp8 opcional | bf16 | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ edit | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sí | [BOOGU_IMAGE.md](quickstart/BOOGU_IMAGE.es.md) |
| zlab i1 | 3B | ✓ | ✓ | ✓ | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Unspecified](https://huggingface.co/bghira/zlab-i1-diffusers) | Aplican condiciones<sup>12</sup> | [ZLAB_i1.md](quickstart/ZLAB_i1.es.md) |
| Ideogram 4 | 9B | ✓ | ✓ | ✓* | fp8 predeterminado, nf4 opcional | bf16 | ✓+ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [Ideogram 4 Non-Commercial](https://huggingface.co/ideogram-ai/ideogram-4-nf4/blob/main/LICENSE.md) | No<sup>5</sup> | [IDEOGRAM4.md](quickstart/IDEOGRAM4.es.md) |
| ERNIE-Image | - | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sí | [ERNIE.md](quickstart/ERNIE.es.md) |
| ACE-Step | 3.5B | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B) / [MIT](https://huggingface.co/ACE-Step/Ace-Step1.5) | Sí | [ACE_STEP.md](quickstart/ACE_STEP.md) |
| Chroma 1 | 8.9B | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sí | [CHROMA.md](quickstart/CHROMA.md) |
| Auraflow | 6B | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) / [Pony License](https://huggingface.co/purplesmartai/pony-v7-base/blob/main/LICENSE) | Aplican condiciones<sup>8</sup> | [AURAFLOW.md](quickstart/AURAFLOW.md) |
| HiDream I1 | 17B (8.5B MoE) | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [MIT](https://opensource.org/license/mit) | Sí | [HIDREAM.md](quickstart/HIDREAM.md) |
| OmniGen | 3.8B | ✓ | ✓ | ✓ | int8/fp8 opcional | bf16 | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | Sí | [OMNIGEN.md](quickstart/OMNIGEN.md) |
| Stable Diffusion XL | 2.6B | ✓ | ✓ | ✓ | no recomendado | bf16 | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ | [OpenRAIL++](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | Aplican condiciones<sup>1</sup> | [SDXL.md](quickstart/SDXL.md) |
| Lumina2 | 2B | ✓ | ✓ | ✓ | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sí | [LUMINA2.md](quickstart/LUMINA2.md) |
| Cosmos2 | 2B | ✓ | ✓ | ✓ | no recomendado | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | Sí<sup>9</sup> | [COSMOS2IMAGE.md](quickstart/COSMOS2IMAGE.md) |
| Cosmos3 | 16B-65B | ✓ | ✓ | ✓* | no_change primero | bf16 | ✓ | ✓ | ✗ | ✗ | ✗ | audio opt | ✗ | ✓ | [OpenMDW 1.1](https://github.com/OpenMDW/openmdw/blob/main/1.1/LICENSE.OpenMDW-1.1) | Sí | [COSMOS3.md](quickstart/COSMOS3.es.md) |
| LTX Video | ~2.5B | ✓ | ✓ | ✓ | int8/fp8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ I2V | ✗ | ✓ | [LTX Video OpenRAIL-M](https://huggingface.co/Lightricks/LTX-Video-0.9.5/blob/main/ltx-video-2b-v0.9.5.license.txt) | Aplican condiciones<sup>10</sup> | [LTXVIDEO.md](quickstart/LTXVIDEO.md) |
| LTX Video 2 | 19B | ✓ | ✓ | ✓* | int8/fp8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ opt | ✗ | ✓ | [LTX-2 Community](https://ltx.io/model/license) | Aplican condiciones<sup>10</sup> | [LTXVIDEO2.md](quickstart/LTXVIDEO2.md) |
| Hunyuan Video 1.5 | 8.3B | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ I2V | ✗ | ✓ | [Tencent Hunyuan Community](https://huggingface.co/tencent/HunyuanVideo-1.5/blob/main/LICENSE) | Aplican condiciones<sup>11</sup> | [HUNYUANVIDEO.md](quickstart/HUNYUANVIDEO.md) |
| SanaVideo | 2B | ✓ | ✓ | ✓* | int8/fp8 opcional | bf16 | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sí | [SANAVIDEO.md](quickstart/SANAVIDEO.es.md) |
| Wan 2.x | 1.3B–14B | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sí | [WAN.md](quickstart/WAN.md) |
| Wan 2.2 S2V | 14B | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sí | [WAN_S2V.md](quickstart/WAN_S2V.md) |
| Qwen Image | 20B | ✓ | ✓ | ✓* | **requerido** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sí | [QWEN_IMAGE.md](quickstart/QWEN_IMAGE.md) |
| Qwen Image Edit | 20B | ✓ | ✓ | ✓* | **requerido** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sí | [QWEN_EDIT.md](quickstart/QWEN_EDIT.md) |
| Stable Cascade (C) | 1B, prior 3.6B | ✓ | ✓ | ✓* | no soportado | fp32 (requerido) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [Stable Cascade NC Community](https://huggingface.co/stabilityai/stable-cascade/blob/main/LICENSE) | No<sup>5</sup> | [STABLE_CASCADE_C.md](quickstart/STABLE_CASCADE_C.md) |
| Kandinsky 5.0 Image | 6B (lite) | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ I2I | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | Sí | [KANDINSKY5_IMAGE.md](quickstart/KANDINSKY5_IMAGE.md) |
| Kandinsky 5.0 Video | 2B (lite), 19B (pro) | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ I2V | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | Sí | [KANDINSKY5_VIDEO.md](quickstart/KANDINSKY5_VIDEO.md) |
| LongCat-Video | 13.6B | ✓ | ✓ | ✓* | int8/fp8 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ opt | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | Sí | [LONGCAT_VIDEO.md](quickstart/LONGCAT_VIDEO.md) |
| LongCat-Video Edit | 13.6B | ✓ | ✓ | ✓* | int8/fp8 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | Sí | [LONGCAT_VIDEO_EDIT.md](quickstart/LONGCAT_VIDEO_EDIT.md) |
| LongCat-Image | 6B | ✓ | ✓ | ✓* | int8/fp8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sí | [LONGCAT_IMAGE.md](quickstart/LONGCAT_IMAGE.md) |
| LongCat-Image Edit | 6B | ✓ | ✓ | ✓* | int8/fp8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sí | [LONGCAT_EDIT.md](quickstart/LONGCAT_EDIT.md) |

*✓ = soportado, ✓* = requiere DeepSpeed/FSDP2 para rango completo, ✗ = no soportado, `✓+` indica que se recomienda checkpointing por presión de VRAM. Ref Inputs marca rutas existentes de condicionamiento por referencia/edición/I2V; `opt` significa opcional y `req` significa requerido por el flavour de edición/I2V. TwinFlow ✓ significa soporte nativo cuando `twinflow_enabled=true` (los modelos de difusión necesitan `diff2flow_enabled+twinflow_allow_diff2flow`). Self-Flow ✓ significa soporte nativo para `crepa_enabled=true` con `crepa_feature_source=self_flow`, `use_ema=true` y `crepa_teacher_block_index` configurado. LayerSync ✓ significa que el backbone expone estados ocultos del transformer para autoalineación; ✗ marca backbones tipo UNet sin ese buffer. †Sliders aplican a LoRA y LyCORIS (incluido LyCORIS de rango completo “full”).*

**Notas de licencia:** El estado de uso comercial cubre pesos del modelo, checkpoints derivados, fine-tunes y uso del modelo alojado. Los derechos sobre salidas generadas pueden diferir; lee el texto de licencia enlazado antes de un despliegue comercial.

<sup>1</sup> Las licencias estilo OpenRAIL suelen permitir uso comercial con restricciones de uso que siguen aplicando al modelo y sus derivados.

<sup>2</sup> La Stability AI Community License está disponible para usuarios que califican por debajo del umbral de ingresos; el uso comercial mayor requiere términos empresariales de Stability.

<sup>3</sup> Flux.1 varía por flavour: Schnell y LibreFlux son Apache-2.0, mientras que Dev, Krea y Kontext usan términos no comerciales de BFL; revisa los metadatos upstream de FluxBooru antes de uso comercial.

<sup>4</sup> Flux.2 varía por flavour: Klein 4B es Apache-2.0, mientras que Dev y Klein 9B usan términos no comerciales de BFL.

<sup>5</sup> Los términos públicos no comerciales no permiten uso comercial de pesos, checkpoints derivados o servicios alojados del modelo sin una licencia separada.

<sup>6</sup> La Krea 2 Community License permite uso comercial solo bajo sus requisitos de ingresos y seguridad/filtrado; de lo contrario se requiere una licencia empresarial.

<sup>7</sup> El uso comercial del modelo Kolors o sus derivados requiere solicitar y recibir permiso explícito del licenciante.

<sup>8</sup> AuraFlow admite flavours upstream Apache-2.0 y un flavour Pony con una licencia personalizada separada; revisa el flavour seleccionado.

<sup>9</sup> La NVIDIA Open Model License permite uso comercial, pero incluye términos de acuerdo, uso aceptable y control de exportación.

<sup>10</sup> LTX Video 0.9.5 usa OpenRAIL-M; LTX Video 2 usa términos comunitarios de LTX con un umbral de ingresos para uso comercial.

<sup>11</sup> La Tencent Hunyuan Community License incluye exclusiones territoriales y un umbral comercial para servicios muy grandes.

<sup>12</sup> Este mirror publica `license: other` sin un texto de licencia estándar; revisa los términos upstream antes de uso comercial.

> ℹ️ El inicio rápido de Wan incluye presets de las etapas 2.1 y 2.2 y el toggle de time-embedding. Flux Kontext cubre flujos de edición construidos sobre Flux.1.

> ⚠️ Estos quickstarts son documentos vivos. Espera actualizaciones ocasionales a medida que llegan nuevos modelos o se mejoran recetas de entrenamiento.

### Rutas rápidas: Z-Image Turbo y Flux Schnell

- **Z-Image Turbo**: LoRA totalmente soportado con TREAD; funciona rápido en NVIDIA y macOS incluso sin quant (int8 también sirve). A menudo el cuello de botella es solo la configuración del trainer.
- **Flux Schnell**: La configuración de quickstart maneja automáticamente el fast noise schedule y la pila de assistant LoRA; no se requieren flags extra para entrenar LoRAs Schnell.

### Funciones experimentales avanzadas

- **Diff2Flow**: Permite entrenar modelos estándar epsilon/v-prediction (SD1.5, SDXL, DeepFloyd, etc.) usando un objetivo de pérdida de Flow Matching. Esto reduce la brecha entre arquitecturas antiguas y el entrenamiento moderno basado en flujo.
- **Scheduled Sampling**: Reduce el sesgo de exposición permitiendo que el modelo genere sus propios latentes ruidosos intermedios durante el entrenamiento ("rollout"). Esto ayuda a que el modelo aprenda a recuperarse de sus propios errores de generación.

## Problemas Comunes

### El dataset tiene menos muestras de lo esperado

Si tu dataset termina con menos muestras utilizables de lo esperado, los archivos pueden haber sido filtrados durante el procesamiento. Razones comunes incluyen:

- **Archivos demasiado pequeños**: Las imágenes por debajo de `minimum_image_size` son filtradas
- **Relación de aspecto fuera de rango**: Las imágenes fuera de los límites de `minimum_aspect_ratio`/`maximum_aspect_ratio` son excluidas
- **Límites de duración**: Los archivos de audio/video que exceden los límites de duración son omitidos

**Ver estadísticas de filtrado:**
- En la WebUI, navega al directorio de tu dataset y selecciónalo para ver estadísticas de filtrado
- Revisa los logs durante el procesamiento del dataset para estadísticas como: `Sample processing statistics: {'total_processed': 100, 'skipped': {'too_small': 15, ...}}`

Para solución de problemas detallada, consulta [Solución de problemas de datasets filtrados](DATALOADER.es.md) en la documentación del dataloader.

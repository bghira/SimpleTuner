# T-LoRA (LoRA dependiente del paso temporal)

## Contexto

El ajuste fino estándar con LoRA aplica una adaptación de bajo rango fija de manera uniforme en todos los pasos temporales de difusión. Cuando los datos de entrenamiento son limitados (especialmente en la personalización con una sola imagen), esto conduce al sobreajuste: el modelo memoriza patrones de ruido en los pasos temporales de alto ruido donde existe poca información semántica.

**T-LoRA** ([Soboleva et al., 2025](https://arxiv.org/abs/2507.05964)) resuelve esto ajustando dinámicamente el número de rangos activos de LoRA en función del paso temporal de difusión actual:

- **Ruido alto** (inicio del proceso de eliminación de ruido, $t \to T$): menos rangos están activos, evitando que el modelo memorice patrones de ruido no informativos.
- **Ruido bajo** (final del proceso de eliminación de ruido, $t \to 0$): más rangos están activos, permitiendo al modelo capturar detalles finos del concepto.

El soporte de T-LoRA en SimpleTuner está construido sobre la biblioteca [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) y requiere una versión de LyCORIS que incluya el módulo `lycoris.modules.tlora`.

> **Experimental:** T-LoRA con modelos de video puede producir resultados deficientes porque la compresión temporal mezcla fotogramas a través de los límites de los pasos temporales.

## Configuración rápida

### 1. Establece tu configuración de entrenamiento

En tu `config.json`, usa LyCORIS con un archivo de configuración T-LoRA separado:

```json
{
    "model_type": "lora",
    "lora_type": "lycoris",
    "lycoris_config": "config/lycoris_tlora.json",
    "validation_lycoris_strength": 1.0
}
```

### 2. Crea la configuración de LyCORIS T-LoRA

Crea `config/lycoris_tlora.json`:

```json
{
    "algo": "tlora",
    "multiplier": 1.0,
    "linear_dim": 64,
    "linear_alpha": 32,
    "apply_preset": {
        "target_module": ["Attention", "FeedForward"]
    }
}
```

Eso es todo lo que necesitas para comenzar a entrenar. Las secciones siguientes cubren el ajuste opcional y la inferencia.

## Referencia de configuración

### Campos obligatorios

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `algo` | string | Debe ser `"tlora"` |
| `multiplier` | float | Multiplicador de intensidad de LoRA. Mantener en `1.0` a menos que sepas lo que estás haciendo |
| `linear_dim` | int | Rango de LoRA. Se convierte en `max_rank` en el programa de enmascaramiento |
| `linear_alpha` | int | Factor de escala de LoRA (separado de `tlora_alpha`) |

### Campos opcionales

| Campo | Tipo | Valor predeterminado | Descripción |
|-------|------|----------------------|-------------|
| `tlora_min_rank` | int | `1` | Rangos activos mínimos en el nivel de ruido más alto |
| `tlora_alpha` | float | `1.0` | Exponente del programa de enmascaramiento. `1.0` es lineal; valores por encima de `1.0` desplazan más capacidad hacia los pasos de detalle fino |
| `apply_preset` | object | — | Selección de módulos objetivo mediante `target_module` y `module_algo_map` |

### Módulos objetivo específicos por modelo

Para la mayoría de los modelos, los objetivos genéricos `["Attention", "FeedForward"]` funcionan. Para Flux 2 (Klein), usa los nombres de clase personalizados:

```json
{
    "algo": "tlora",
    "multiplier": 1.0,
    "linear_dim": 64,
    "linear_alpha": 32,
    "apply_preset": {
        "target_module": [
            "Flux2Attention", "Flux2FeedForward", "Flux2ParallelSelfAttention"
        ]
    }
}
```

Consulta la [documentación de LyCORIS](../LYCORIS.md) para la lista completa de módulos objetivo por modelo.

## Parámetros de ajuste

### `linear_dim` (rango)

Un rango más alto = más parámetros y expresividad, pero más propenso al sobreajuste con datos limitados. El artículo original de T-LoRA usa rango 64 para la personalización de SDXL con una sola imagen.

### `tlora_min_rank`

Controla el mínimo de activación de rangos en el paso temporal más ruidoso. Aumentar esto permite al modelo aprender estructuras más gruesas pero reduce el beneficio contra el sobreajuste. Comienza con el valor predeterminado de `1` y auméntalo solo si la convergencia es demasiado lenta.

### `tlora_alpha` (exponente del programa)

Controla la forma de la curva del programa de enmascaramiento:

- `1.0` — interpolación lineal entre `min_rank` y `max_rank`
- `> 1.0` — enmascaramiento más agresivo con ruido alto; la mayoría de los rangos solo se activan cerca del final del proceso de eliminación de ruido
- `< 1.0` — enmascaramiento más suave; los rangos se activan antes

<details>
<summary>Visualización del programa (rango vs. paso temporal)</summary>

Con `linear_dim=64`, `tlora_min_rank=1`, para un planificador de 1000 pasos:

```
alpha=1.0 (lineal):
  t=0   (limpio) → 64 rangos activos
  t=250 (25%)    → 48 rangos activos
  t=500 (50%)    → 32 rangos activos
  t=750 (75%)    → 16 rangos activos
  t=999 (ruido)  →  1 rango activo

alpha=2.0 (cuadrático — sesgado hacia detalle):
  t=0   (limpio) → 64 rangos activos
  t=250 (25%)    → 60 rangos activos
  t=500 (50%)    → 48 rangos activos
  t=750 (75%)    → 20 rangos activos
  t=999 (ruido)  →  1 rango activo

alpha=0.5 (raíz cuadrada — sesgado hacia estructura):
  t=0   (limpio) → 64 rangos activos
  t=250 (25%)    → 55 rangos activos
  t=500 (50%)    → 46 rangos activos
  t=750 (75%)    → 33 rangos activos
  t=999 (ruido)  →  1 rango activo
```

</details>

## Inferencia con los pipelines de SimpleTuner

Los pipelines incluidos en SimpleTuner tienen soporte integrado para T-LoRA. Durante la validación, los parámetros de enmascaramiento del entrenamiento se reutilizan automáticamente en cada paso de eliminación de ruido; no se necesita configuración adicional.

Para inferencia independiente fuera del entrenamiento, puedes importar directamente el pipeline de SimpleTuner y establecer el atributo `_tlora_config`. Esto asegura que el enmascaramiento por paso coincida con el que se usó durante el entrenamiento del modelo.

### Ejemplo con SDXL

```py
import torch
from lycoris import create_lycoris_from_weights

# Use SimpleTuner's vendored SDXL pipeline (has T-LoRA support built in)
from simpletuner.helpers.models.sdxl.pipeline import StableDiffusionXLPipeline
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
dtype = torch.bfloat16
device = "cuda"

# Load pipeline components
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)

# Load and apply LyCORIS T-LoRA weights
lora_path = "path/to/pytorch_lora_weights.safetensors"
wrapper, _ = create_lycoris_from_weights(1.0, lora_path, unet)
wrapper.merge_to()

unet.to(device)

pipe = StableDiffusionXLPipeline(
    scheduler=scheduler,
    vae=vae,
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    unet=unet,
)

# Enable T-LoRA inference masking — must match training config
pipe._tlora_config = {
    "max_rank": 64,      # linear_dim from your lycoris config
    "min_rank": 1,       # tlora_min_rank (default 1)
    "alpha": 1.0,        # tlora_alpha (default 1.0)
}

with torch.inference_mode():
    image = pipe(
        prompt="a sks dog riding a surfboard",
        width=1024,
        height=1024,
        num_inference_steps=25,
        guidance_scale=5.0,
    ).images[0]

image.save("tlora_output.png")
```

### Ejemplo con Flux

```py
import torch
from lycoris import create_lycoris_from_weights

# Use SimpleTuner's vendored Flux pipeline (has T-LoRA support built in)
from simpletuner.helpers.models.flux.pipeline import FluxPipeline
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

bfl_repo = "black-forest-labs/FLUX.1-dev"
dtype = torch.bfloat16
device = "cuda"

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2")
vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype)
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype)

# Load and apply LyCORIS T-LoRA weights
lora_path = "path/to/pytorch_lora_weights.safetensors"
wrapper, _ = create_lycoris_from_weights(1.0, lora_path, transformer)
wrapper.merge_to()

transformer.to(device)

pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=transformer,
)

# Enable T-LoRA inference masking
pipe._tlora_config = {
    "max_rank": 64,
    "min_rank": 1,
    "alpha": 1.0,
}

with torch.inference_mode():
    image = pipe(
        prompt="a sks dog riding a surfboard",
        width=1024,
        height=1024,
        num_inference_steps=25,
        guidance_scale=3.5,
    ).images[0]

image.save("tlora_flux_output.png")
```

> **Nota:** Debes usar el pipeline incluido en SimpleTuner (por ejemplo, `simpletuner.helpers.models.flux.pipeline.FluxPipeline`), no el pipeline estándar de Diffusers. Solo los pipelines incluidos contienen la lógica de enmascaramiento T-LoRA por paso.

### Por que no simplemente usar `merge_to()` y omitir el enmascaramiento?

`merge_to()` incorpora permanentemente los pesos de LoRA en el modelo base; esto es necesario para que los parámetros de LoRA estén activos durante el pase hacia adelante. Sin embargo, T-LoRA fue **entrenado** con enmascaramiento de rango dependiente del paso temporal: ciertos rangos se pusieron a cero dependiendo del nivel de ruido. Sin reaplicar ese mismo enmascaramiento durante la inferencia, todos los rangos se activan en cada paso temporal, produciendo imágenes sobresaturadas o con apariencia quemada.

Establecer `_tlora_config` en el pipeline le indica al bucle de eliminación de ruido que aplique la máscara correcta antes de cada pase hacia adelante del modelo y la elimine después.

<details>
<summary>Cómo funciona internamente el enmascaramiento</summary>

En cada paso de eliminación de ruido, el pipeline ejecuta:

```python
from simpletuner.helpers.training.lycoris import apply_tlora_inference_mask, clear_tlora_mask

_tlora_cfg = getattr(self, "_tlora_config", None)
if _tlora_cfg:
    apply_tlora_inference_mask(
        timestep=int(t),
        max_timestep=self.scheduler.config.num_train_timesteps,
        max_rank=_tlora_cfg["max_rank"],
        min_rank=_tlora_cfg["min_rank"],
        alpha=_tlora_cfg["alpha"],
    )
try:
    noise_pred = self.unet(...)  # or self.transformer(...)
finally:
    if _tlora_cfg:
        clear_tlora_mask()
```

`apply_tlora_inference_mask` calcula una máscara binaria de forma `(1, max_rank)` usando la fórmula:

$$r = \left\lfloor\left(\frac{T - t}{T}\right)^\alpha \cdot (R_{\max} - R_{\min})\right\rfloor + R_{\min}$$

donde $T$ es el paso temporal máximo del planificador, $R_{\max}$ es `linear_dim`, y $R_{\min}$ es `tlora_min_rank`. Los primeros $r$ elementos de la máscara se establecen en `1.0` y el resto en `0.0`. Esta máscara se establece globalmente en todos los módulos T-LoRA mediante `set_timestep_mask()` de LyCORIS.

Después de que el pase hacia adelante se completa, `clear_tlora_mask()` elimina el estado de la máscara para que no se filtre a operaciones posteriores.

</details>

<details>
<summary>Cómo SimpleTuner pasa la configuración durante la validación</summary>

Durante el entrenamiento, el diccionario de configuración de T-LoRA (`max_rank`, `min_rank`, `alpha`) se almacena en el objeto Accelerator. Cuando se ejecuta la validación, `validation.py` copia esta configuración al pipeline:

```python
# setup_pipeline()
if getattr(self.accelerator, "_tlora_active", False):
    self.model.pipeline._tlora_config = self.accelerator._tlora_config

# clean_pipeline()
if hasattr(self.model.pipeline, "_tlora_config"):
    del self.model.pipeline._tlora_config
```

Esto es completamente automático: no se requiere configuración por parte del usuario para que las imágenes de validación usen el enmascaramiento correcto.

</details>

## Origen: el artículo de T-LoRA

<details>
<summary>Detalles del artículo y algoritmo</summary>

**T-LoRA: Single Image Diffusion Model Customization Without Overfitting**
Vera Soboleva, Aibek Alanov, Andrey Kuznetsov, Konstantin Sobolev
[arXiv:2507.05964](https://arxiv.org/abs/2507.05964) — Aceptado en AAAI 2026

El artículo introduce dos innovaciones complementarias:

### 1. Enmascaramiento de rango dependiente del paso temporal

La idea central es que los pasos temporales de difusión más altos (entradas más ruidosas) son más propensos al sobreajuste que los pasos temporales más bajos. Con ruido alto, el latente contiene principalmente ruido aleatorio con poca señal semántica; entrenar un adaptador de rango completo en esto enseña al modelo a memorizar patrones de ruido en lugar de aprender el concepto objetivo.

T-LoRA aborda esto con un programa de enmascaramiento dinámico que restringe el rango activo de LoRA en función del paso temporal actual.

### 2. Parametrización de pesos ortogonales (opcional)

El artículo también propone inicializar los pesos de LoRA mediante descomposición SVD de los pesos originales del modelo, imponiendo ortogonalidad a través de una pérdida de regularización. Esto asegura la independencia entre los componentes del adaptador.

La integración de LyCORIS en SimpleTuner se centra en el componente de enmascaramiento por paso temporal, que es el principal impulsor de la reducción del sobreajuste. La inicialización ortogonal es parte de la implementación independiente de T-LoRA pero actualmente no es utilizada por el algoritmo `tlora` de LyCORIS.

### Cita

```bibtex
@misc{soboleva2025tlorasingleimagediffusion,
      title={T-LoRA: Single Image Diffusion Model Customization Without Overfitting},
      author={Vera Soboleva and Aibek Alanov and Andrey Kuznetsov and Konstantin Sobolev},
      year={2025},
      eprint={2507.05964},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.05964},
}
```

</details>

## Errores comunes

- **Olvidar `_tlora_config` durante la inferencia:** Las imágenes se ven sobresaturadas o quemadas. Todos los rangos se activan en cada paso temporal en lugar de seguir el programa de enmascaramiento entrenado.
- **Usar el pipeline estándar de Diffusers:** Los pipelines estándar no contienen la lógica de enmascaramiento T-LoRA. Debes usar los pipelines incluidos en SimpleTuner.
- **Discrepancia en `linear_dim`:** El `max_rank` en `_tlora_config` debe coincidir con el `linear_dim` usado durante el entrenamiento, o las dimensiones de la máscara serán incorrectas.
- **Modelos de video:** La compresión temporal mezcla fotogramas a través de los límites de los pasos temporales, lo que puede debilitar la señal de enmascaramiento dependiente del paso temporal. Los resultados pueden ser deficientes.
- **SDXL + módulos FeedForward:** Entrenar módulos FeedForward con LyCORIS en SDXL puede causar pérdida NaN; este es un problema general de LyCORIS, no específico de T-LoRA. Consulta la [documentación de LyCORIS](../LYCORIS.md#potential-problems) para más detalles.

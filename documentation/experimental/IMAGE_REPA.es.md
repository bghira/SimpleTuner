# REPA y U-REPA (regularización de imágenes)

El Alineamiento de Representación (REPA) es una técnica de regularización que alinea los estados ocultos del modelo de difusión con características de un codificador de visión congelado (típicamente DINOv2). Esto mejora la calidad de generación y la eficiencia del entrenamiento aprovechando representaciones visuales preentrenadas.

SimpleTuner soporta dos variantes:

- **REPA** para modelos de imagen basados en DiT (Flux, SD3, Chroma, Sana, PixArt, etc.) - PR #2562
- **U-REPA** para modelos de imagen basados en UNet (SDXL, SD1.5, Kolors) - PR #2563

> **¿Buscas modelos de video?** Consulta [VIDEO_CREPA.es.md](VIDEO_CREPA.es.md) para soporte CREPA en modelos de video con alineamiento temporal.

## Cuándo usarlo

### REPA (modelos DiT)
- Estás entrenando modelos de imagen basados en DiT y quieres convergencia más rápida
- Notas problemas de calidad o quieres un fundamento semántico más fuerte
- Familias de modelos soportadas: `flux`, `flux2`, `sd3`, `chroma`, `sana`, `pixart`, `hidream`, `auraflow`, `lumina2` y otros

### U-REPA (modelos UNet)
- Estás entrenando modelos de imagen basados en UNet (SDXL, SD1.5, Kolors)
- Quieres aprovechar el alineamiento de representación optimizado para arquitecturas UNet
- U-REPA usa alineamiento de **bloque medio** (no capas tempranas) y añade **pérdida de variedad** para mejor estructura de similitud relativa

## Configuración rápida (WebUI)

### Para modelos DiT (REPA)

1. Abre **Entrenamiento → Funciones de pérdida**.
2. Habilita **CREPA** (la misma opción habilita REPA para modelos de imagen).
3. Establece **CREPA Block Index** a una capa temprana del codificador:
   - Flux / Flux2: `8`
   - SD3: `8`
   - Chroma: `8`
   - Sana / PixArt: `10`
4. Establece **Peso** en `0.5` para empezar.
5. Mantén los valores por defecto para el codificador de visión (`dinov2_vitg14`, resolución `518`).

### Para modelos UNet (U-REPA)

1. Abre **Entrenamiento → Funciones de pérdida**.
2. Habilita **U-REPA**.
3. Establece **U-REPA Weight** en `0.5` (valor por defecto del artículo).
4. Establece **U-REPA Manifold Weight** en `3.0` (valor por defecto del artículo).
5. Mantén los valores por defecto para el codificador de visión.

## Configuración rápida (config JSON / CLI)

### Para modelos DiT (REPA)

```json
{
  "crepa_enabled": true,
  "crepa_block_index": 8,
  "crepa_lambda": 0.5,
  "crepa_encoder": "dinov2_vitg14",
  "crepa_encoder_image_size": 518
}
```

### Para modelos UNet (U-REPA)

```json
{
  "urepa_enabled": true,
  "urepa_lambda": 0.5,
  "urepa_manifold_weight": 3.0,
  "urepa_model": "dinov2_vitg14",
  "urepa_encoder_image_size": 518
}
```

## Diferencias clave: REPA vs U-REPA

| Aspecto | REPA (DiT) | U-REPA (UNet) |
|---------|-----------|---------------|
| Arquitectura | Bloques Transformer | UNet con bloque medio |
| Punto de alineamiento | Capas transformer tempranas | Bloque medio (cuello de botella) |
| Forma del estado oculto | `(B, S, D)` secuencia | `(B, C, H, W)` convolucional |
| Componentes de pérdida | Alineamiento coseno | Coseno + Pérdida de variedad |
| Peso por defecto | 0.5 | 0.5 |
| Prefijo de config | `crepa_*` | `urepa_*` |

## Especificidades de U-REPA

U-REPA adapta REPA para arquitecturas UNet con dos innovaciones clave:

### Alineamiento del bloque medio
A diferencia del REPA basado en DiT que usa capas transformer tempranas, U-REPA extrae características del **bloque medio** (cuello de botella) del UNet. Este es el lugar donde el UNet tiene más información semántica comprimida.

- **SDXL/Kolors**: El bloque medio produce `(B, 1280, 16, 16)` para imágenes de 1024x1024
- **SD1.5**: El bloque medio produce `(B, 1280, 8, 8)` para imágenes de 512x512

### Pérdida de variedad
Además del alineamiento coseno, U-REPA añade una **pérdida de variedad** que alinea la estructura de similitud relativa:

```
L_manifold = ||sim(y[i],y[j]) - sim(h[i],h[j])||^2_F
```

Esto asegura que si dos parches del codificador son similares, los parches proyectados correspondientes también deben ser similares. El parámetro `urepa_manifold_weight` (por defecto 3.0) controla el equilibrio entre alineamiento directo y alineamiento de variedad.

## Parámetros de ajuste

### REPA (modelos DiT)
- `crepa_lambda`: Peso de la pérdida de alineamiento (por defecto 0.5)
- `crepa_block_index`: Qué bloque transformer usar (indexado desde 0)
- `crepa_spatial_align`: Interpolar tokens para que coincidan (por defecto true)
- `crepa_encoder`: Modelo del codificador de visión (por defecto `dinov2_vitg14`)
- `crepa_encoder_image_size`: Resolución de entrada (por defecto 518)

### U-REPA (modelos UNet)
- `urepa_lambda`: Peso de la pérdida de alineamiento (por defecto 0.5)
- `urepa_manifold_weight`: Peso de la pérdida de variedad (por defecto 3.0)
- `urepa_model`: Modelo del codificador de visión (por defecto `dinov2_vitg14`)
- `urepa_encoder_image_size`: Resolución de entrada (por defecto 518)
- `urepa_use_tae`: Usar Tiny AutoEncoder para decodificación más rápida

## Programación de coeficientes

Tanto REPA como U-REPA soportan programación para decaer la regularización durante el entrenamiento:

```json
{
  "crepa_scheduler": "cosine",
  "crepa_warmup_steps": 100,
  "crepa_decay_steps": 5000,
  "crepa_lambda_end": 0.0
}
```

Para U-REPA, usa el prefijo `urepa_`:

```json
{
  "urepa_scheduler": "cosine",
  "urepa_warmup_steps": 100,
  "urepa_cutoff_step": 5000
}
```

<details>
<summary>Cómo funciona (practicante)</summary>

### REPA (DiT)
- Captura estados ocultos de un bloque transformer elegido
- Proyecta a través de LayerNorm + Linear a la dimensión del codificador
- Calcula similitud coseno con características DINOv2 congeladas
- Interpola tokens espaciales para que coincidan si los conteos difieren

### U-REPA (UNet)
- Registra un hook forward en el mid_block del UNet
- Captura características convolucionales `(B, C, H, W)`
- Reformatea a secuencia `(B, H*W, C)` y proyecta a dimensión del codificador
- Calcula tanto alineamiento coseno como pérdida de variedad
- La pérdida de variedad alinea la estructura de similitud por pares

</details>

<details>
<summary>Técnico (internos de SimpleTuner)</summary>

### REPA
- Implementación: `simpletuner/helpers/training/crepa.py` (clase `CrepaRegularizer`)
- Detección de modo: `CrepaMode.IMAGE` para modelos de imagen, establecido automáticamente vía propiedad `crepa_mode`
- Estados ocultos almacenados en la clave `crepa_hidden_states` de la salida del modelo

### U-REPA
- Implementación: `simpletuner/helpers/training/crepa.py` (clase `UrepaRegularizer`)
- Captura del bloque medio: `simpletuner/helpers/utils/hidden_state_buffer.py` (`UNetMidBlockCapture`)
- Tamaño oculto inferido de `block_out_channels[-1]` (1280 para SDXL/SD1.5/Kolors)
- Solo habilitado para `MODEL_TYPE == ModelTypes.UNET`
- Estados ocultos almacenados en la clave `urepa_hidden_states` de la salida del modelo

</details>

## Problemas comunes

- **Tipo de modelo incorrecto**: REPA (`crepa_*`) es para modelos DiT; U-REPA (`urepa_*`) es para modelos UNet. Usar el incorrecto no tendrá efecto.
- **Índice de bloque muy alto** (REPA): Baja el índice si recibes errores "hidden states not returned".
- **Picos de VRAM**: Prueba un codificador más pequeño (`dinov2_vits14` + tamaño de imagen `224`) o habilita `use_tae` para decodificación.
- **Peso de variedad muy alto** (U-REPA): Si el entrenamiento se vuelve inestable, reduce `urepa_manifold_weight` de 3.0 a 1.0.

## Referencias

- [Artículo REPA](https://arxiv.org/abs/2402.17750) - Alineamiento de Representación para Generación
- [Artículo U-REPA](https://arxiv.org/abs/2410.xxxxx) - REPA Universal para arquitecturas UNet (NeurIPS 2025)
- [DINOv2](https://github.com/facebookresearch/dinov2) - Codificador de visión auto-supervisado

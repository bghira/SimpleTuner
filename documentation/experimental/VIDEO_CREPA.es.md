# CREPA (regularización de video)

Cross-frame Representation Alignment (CREPA) es un regularizador ligero para modelos de video. Empuja los estados ocultos de cada frame hacia las características de un encoder de visión congelado del frame actual **y de sus vecinos**, mejorando la consistencia temporal sin cambiar la pérdida principal.

> **¿Buscas modelos de imagen?** Consulta [IMAGE_REPA.es.md](IMAGE_REPA.es.md) para soporte REPA en modelos DiT de imagen (Flux, SD3, etc.) y U-REPA para modelos UNet (SDXL, SD1.5, Kolors).

## Cuándo usarlo

- Estás entrenando con videos con movimiento complejo, cambios de escena u oclusiones.
- Estás afinando un DiT de video (LoRA o completo) y ves parpadeo o deriva de identidad entre frames.
- Familias de modelos compatibles: `kandinsky5_video`, `ltxvideo`, `sanavideo` y `wan` (otras familias no exponen los hooks de CREPA).
- Tienes VRAM extra (CREPA añade ~1–2 GB según la configuración) para el encoder DINO y el VAE, que deben permanecer en memoria durante el entrenamiento para decodificar latentes a píxeles.

## Configuración rápida (WebUI)

1. Abre **Training → Loss functions**.
2. Habilita **CREPA**.
3. Configura **CREPA Block Index** en una capa del lado del encoder. Empieza con:
   - Kandinsky5 Video: `8`
   - LTXVideo / Wan: `8`
   - SanaVideo: `10`
4. Deja **Weight** en `0.5` para empezar.
5. Mantén **Adjacent Distance** en `1` y **Temporal Decay** en `1.0` para una configuración que coincide de cerca con el paper original de CREPA.
6. Usa los valores por defecto del encoder de visión (`dinov2_vitg14`, resolución `518`). Cambia solo si sabes que necesitas un encoder más pequeño (p. ej., `dinov2_vits14` + tamaño de imagen `224` para ahorrar VRAM).
7. Entrena con normalidad. CREPA añade una pérdida auxiliar y registra `crepa_loss` / `crepa_similarity`.

## Configuración rápida (config JSON / CLI)

Agrega lo siguiente a tu `config.json` o args de CLI:

```json
{
  "crepa_enabled": true,
  "crepa_block_index": 8,
  "crepa_lambda": 0.5,
  "crepa_adjacent_distance": 1,
  "crepa_adjacent_tau": 1.0,
  "crepa_encoder": "dinov2_vitg14",
  "crepa_encoder_image_size": 518
}
```

## Controles de ajuste

- `crepa_spatial_align`: conserva la estructura por parches (por defecto). Configura `false` para promediar si la memoria es limitada.
- `crepa_normalize_by_frames`: mantiene estable la escala de la pérdida en distintos largos de clip (por defecto). Desactívalo si quieres que clips más largos aporten más.
- `crepa_drop_vae_encoder`: libera memoria si solo vas a **decodificar** latentes (no seguro si necesitas codificar píxeles).
- `crepa_adjacent_distance=0`: se comporta como REPA* por frame (sin ayuda de vecinos); combina con `crepa_adjacent_tau` para el decaimiento por distancia.
- `crepa_cumulative_neighbors=true` (solo config): usa todos los offsets `1..d` en lugar de solo los vecinos más cercanos.
- `crepa_use_backbone_features=true`: omite el encoder externo y alinea con un bloque transformer más profundo; configura `crepa_teacher_block_index` para elegir el maestro.
- Tamaño de encoder: baja a `dinov2_vits14` + `224` si la VRAM es ajustada; mantén `dinov2_vitg14` + `518` para la mejor calidad.

## Programación del coeficiente

CREPA soporta programar el coeficiente (`crepa_lambda`) durante el entrenamiento con calentamiento, decaimiento y corte automático basado en umbral de similitud. Esto es particularmente útil para entrenamiento text2video donde CREPA puede causar franjas horizontales/verticales o una apariencia deslavada si se aplica demasiado fuerte durante demasiado tiempo.

### Programación básica

```json
{
  "crepa_enabled": true,
  "crepa_lambda": 0.5,
  "crepa_scheduler": "cosine",
  "crepa_warmup_steps": 100,
  "crepa_decay_steps": 5000,
  "crepa_lambda_end": 0.0
}
```

Esta configuración:
1. Aumenta el peso de CREPA de 0 a 0.5 durante los primeros 100 pasos
2. Decae de 0.5 a 0.0 usando un programa coseno durante 5000 pasos
3. Después del paso 5100, CREPA está efectivamente desactivado

### Tipos de programador

- `constant`: Sin decaimiento, el peso se mantiene en `crepa_lambda` (por defecto)
- `linear`: Interpolación lineal de `crepa_lambda` a `crepa_lambda_end`
- `cosine`: Annealing coseno suave (recomendado para la mayoría de casos)
- `polynomial`: Decaimiento polinomial con potencia configurable mediante `crepa_power`

### Corte basado en pasos

Para un corte duro después de un paso específico:

```json
{
  "crepa_cutoff_step": 3000
}
```

CREPA se desactiva completamente después del paso 3000.

### Corte basado en similitud

Este es el enfoque más flexible: CREPA se desactiva automáticamente cuando la métrica de similitud se estabiliza, indicando que el modelo ha aprendido suficiente alineación temporal:

```json
{
  "crepa_similarity_threshold": 0.9,
  "crepa_similarity_ema_decay": 0.99,
  "crepa_threshold_mode": "permanent"
}
```

- `crepa_similarity_threshold`: Cuando la media móvil exponencial de similitud alcanza este valor, CREPA se corta
- `crepa_similarity_ema_decay`: Factor de suavizado (0.99 ≈ ventana de 100 pasos)
- `crepa_threshold_mode`: `permanent` (permanece apagado) o `recoverable` (puede reactivarse si la similitud baja)

### Configuraciones recomendadas

**Para image2video (i2v)**:
```json
{
  "crepa_scheduler": "constant",
  "crepa_lambda": 0.5
}
```
CREPA estándar funciona bien para i2v ya que el frame de referencia ancla la consistencia.

**Para text2video (t2v)**:
```json
{
  "crepa_scheduler": "cosine",
  "crepa_lambda": 0.5,
  "crepa_warmup_steps": 100,
  "crepa_decay_steps": 0,
  "crepa_lambda_end": 0.1,
  "crepa_similarity_threshold": 0.85,
  "crepa_threshold_mode": "permanent"
}
```
Decae CREPA durante el entrenamiento y corta cuando la similitud se satura para prevenir artefactos.

**Para fondos sólidos (t2v)**:
```json
{
  "crepa_cutoff_step": 2000
}
```
El corte temprano previene artefactos de franjas en fondos uniformes.

<details>
<summary>Cómo funciona (practicante)</summary>

- Captura estados ocultos de un bloque DiT elegido, los proyecta con una cabeza LayerNorm+Linear y los alinea con características de visión congeladas.
- Por defecto codifica frames de píxel con DINOv2; el modo backbone reutiliza un bloque transformer más profundo.
- Alinea cada frame con sus vecinos con un decaimiento exponencial por distancia (`crepa_adjacent_tau`); el modo acumulativo suma opcionalmente todos los offsets hasta `d`.
- La alineación espacial/temporal re-muestrea tokens para que los parches del DiT y del encoder se alineen antes de la similitud coseno; la pérdida se promedia sobre parches y frames.

</details>

<details>
<summary>Técnico (internos de SimpleTuner)</summary>

- Implementación: `simpletuner/helpers/training/crepa.py`; registrada desde `ModelFoundation._init_crepa_regularizer` y adjunta al modelo entrenable (el proyector vive en el modelo para cobertura del optimizador).
- Captura de estados ocultos: los transformers de video guardan `crepa_hidden_states` (y opcionalmente `crepa_frame_features`) cuando `crepa_enabled` es true; el modo backbone también puede tomar `layer_{idx}` del buffer compartido de estados ocultos.
- Ruta de pérdida: decodifica latentes con el VAE a píxeles salvo que `crepa_use_backbone_features` esté activado; normaliza estados ocultos proyectados y características del encoder, aplica similitud coseno ponderada por distancia, registra `crepa_loss` / `crepa_similarity`, y añade la pérdida escalada.
- Interacción: corre antes de LayerSync para que ambos reutilicen el buffer de estados ocultos; limpia el buffer después. Requiere un índice de bloque válido y un tamaño oculto inferido del config del transformer.

</details>

## Errores comunes

- Habilitar CREPA en familias no soportadas causa estados ocultos faltantes; usa `kandinsky5_video`, `ltxvideo`, `sanavideo` o `wan`.
- **Índice de bloque demasiado alto** → “hidden states not returned”. Baja el índice; es de base cero en los bloques transformer.
- **Picos de VRAM** → prueba `crepa_spatial_align=false`, un encoder más pequeño (`dinov2_vits14` + `224`) o un índice de bloque más bajo.
- **Errores en modo backbone** → configura tanto `crepa_block_index` (estudiante) como `crepa_teacher_block_index` (maestro) con capas existentes.
- **Sin memoria** → Si RamTorch no ayuda, tu única solución puede ser una GPU más grande; si H200 o B200 no funcionan, por favor abre un issue.

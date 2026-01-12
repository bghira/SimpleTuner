# LayerSync (SimpleTuner)

LayerSync es un empujón de “aprendizaje autónomo” para modelos transformer: una capa (el estudiante) aprende a alinearse con una capa más fuerte (el maestro). Es ligero, autocontenido y no requiere descargar modelos adicionales.

## Cuándo usarlo

- Estás entrenando familias transformer que exponen estados ocultos (p. ej., Flux/Flux Kontext/Flux.2, PixArt Sigma, SD3/SDXL, Sana, Wan, Qwen Image/Edit, Hunyuan Video, LTXVideo, Kandinsky5 Video, Chroma, ACE-Step, HiDream, Cosmos/LongCat/Z-Image/Auraflow).
- Quieres un regularizador integrado sin usar un checkpoint de maestro externo.
- Observas deriva en mitad del entrenamiento o cabezas inestables y quieres traer una capa intermedia de vuelta hacia un maestro más profundo.
- Tienes algo de margen de VRAM para mantener activaciones de estudiante/maestro del paso actual.

## Configuración rápida (WebUI)

1. Abre **Training → Loss functions**.
2. Habilita **LayerSync**.
3. Configura **Student Block** en una capa media y **Teacher Block** en una más profunda. En modelos tipo DiT de 24 capas (Flux, PixArt, SD3), empieza con `8` → `16`; en stacks más cortos, mantén el maestro unas pocas capas más profundo que el estudiante.
4. Deja **Weight** en `0.2` (por defecto cuando LayerSync está habilitado).
5. Entrena con normalidad; los logs incluirán `layersync_loss` y `layersync_similarity`.

## Configuración rápida (config JSON / CLI)

```json
{
  "layersync_enabled": true,
  "layersync_student_block": 8,
  "layersync_teacher_block": 16,
  "layersync_lambda": 0.2
}
```

## Controles de ajuste

- `layersync_student_block` / `layersync_teacher_block`: indexación amigable de base 1; probamos `idx-1` primero y luego `idx`.
- `layersync_lambda`: escala la pérdida de coseno; debe ser > 0 cuando está habilitado (por defecto `0.2`).
- El maestro por defecto es la misma capa que el estudiante cuando se omite, haciendo que la pérdida sea de auto-similitud.
- VRAM: las activaciones de ambas capas se mantienen hasta que corre la pérdida auxiliar; desactiva LayerSync (o CREPA) si necesitas liberar memoria.
- Funciona bien con CREPA/TwinFlow; comparten el mismo buffer de estados ocultos.

<details>
<summary>Cómo funciona (practicante)</summary>

- Calcula la similitud coseno negativa entre tokens aplanados del estudiante y del maestro; un peso más alto empuja al estudiante hacia las características del maestro.
- Los tokens del maestro siempre se separan para evitar que los gradientes fluyan hacia atrás.
- Maneja estados ocultos 3D `(B, S, D)` y 4D `(B, T, P, D)` tanto para transformers de imagen como de video.
- Mapeo de opciones upstream:
  - `--encoder-depth` → `--layersync_student_block`
  - `--gt-encoder-depth` → `--layersync_teacher_block`
  - `--reg-weight` → `--layersync_lambda`
- Defaults: desactivado por defecto; cuando se habilita y no se configura, `layersync_lambda=0.2`.

</details>

<details>
<summary>Técnico (internos de SimpleTuner)</summary>

- Implementación: `simpletuner/helpers/training/layersync.py`; invocado desde `ModelFoundation._apply_layersync_regularizer`.
- Captura de estados ocultos: se dispara cuando LayerSync o CREPA lo solicitan; los transformers almacenan estados como `layer_{idx}` vía `_store_hidden_state`.
- Resolución de capas: intenta índices base 1 y luego base 0; error si faltan las capas solicitadas.
- Ruta de pérdida: normaliza tokens de estudiante/maestro, calcula similitud coseno media, registra `layersync_loss` y `layersync_similarity`, y añade la pérdida escalada al objetivo principal.
- Interacción: corre después de CREPA para que ambos reutilicen el mismo buffer; limpia el buffer después.

</details>

## Errores comunes

- Falta bloque de estudiante → error al inicio; configura `layersync_student_block` explícitamente.
- Peso ≤ 0 → error al inicio; deja el valor por defecto `0.2` si no estás seguro.
- Solicitar bloques más profundos de los que expone el modelo → errores de “LayerSync could not find layer”; baja los índices.
- Habilitar en modelos que no exponen estados ocultos del transformer (Kolors, Lumina2, Stable Cascade C, Kandinsky5 Image, OmniGen) fallará; usa familias con transformer.
- Picos de VRAM: baja los índices de bloques o desactiva CREPA/LayerSync para liberar el buffer de estados ocultos.

Usa LayerSync cuando quieras un regularizador barato e integrado para guiar suavemente representaciones intermedias sin añadir maestros externos.

# Self-Flow (alineación interna)

Self-Flow es un modo de CREPA que reemplaza el encoder visual externo por una vista EMA más limpia del mismo modelo. Sigue bastante de cerca la idea del paper de Black Forest Labs: entrenar al estudiante con una programación tokenwise de ruido mezclado, ejecutar al profesor EMA sobre una vista más limpia y alinear estados ocultos internos mientras se mantiene la loss generativa normal.

> **¿Buscas alineación con encoder externo?** Consulta [IMAGE_REPA.es.md](IMAGE_REPA.es.md) para REPA / U-REPA y [VIDEO_CREPA.es.md](VIDEO_CREPA.es.md) para CREPA temporal.

## Cuándo usarlo

- Quieres el regularizador auto-supervisado estilo BFL en lugar de un encoder externo.
- Estás entrenando una familia transformer que ya expone hooks de Self-Flow en SimpleTuner.
- Quieres que el mismo regularizador ayude a generación estándar, edición y entrenamiento multimodal.
- Ya usas EMA o puedes activarlo. Self-Flow requiere un profesor EMA.

Las familias soportadas incluyen actualmente:

- Imagen / edición: `flux`, `flux2`, `sd3`, `pixart`, `sana`, `qwen_image`, `chroma`, `hidream`, `auraflow`, `lumina2`, `z_image`, `z_image_omni`, `kandinsky5_image`, `longcat_image`, `omnigen`, `ace_step`
- Vídeo / multimodal: `wan`, `wan_s2v`, `ltxvideo`, `ltxvideo2`, `sanavideo`, `kandinsky5_video`, `hunyuanvideo`, `longcat_video`, `cosmos`, `anima`

## Configuración rápida (WebUI)

1. Abre **Training → Loss functions**.
2. Activa **CREPA**.
3. Configura **CREPA Feature Source** en `self_flow`.
4. Usa **CREPA Block Index** como bloque estudiante temprano. Empieza con `8` en DiT de 24 capas y `10` en stacks más profundos.
5. Usa **CREPA Teacher Block Index** como bloque profesor más profundo. Buenos puntos de partida: `16` o `20`.
6. Deja **Weight** en `0.5`.
7. Usa **Self-Flow Mask Ratio**:
   - `0.25` para imagen
   - `0.10` para vídeo
   - `0.50` para audio como `ace_step`
8. Asegúrate de que **EMA** esté activado.
9. No lo combines con TwinFlow.

## Configuración rápida (config JSON / CLI)

```json
{
  "use_ema": true,
  "crepa_enabled": true,
  "crepa_feature_source": "self_flow",
  "crepa_block_index": 8,
  "crepa_teacher_block_index": 16,
  "crepa_lambda": 0.5,
  "crepa_self_flow_mask_ratio": 0.25
}
```

El alias legado `crepa_self_flow=true` sigue funcionando, pero `crepa_feature_source=self_flow` es la opción preferida.

## Ajustes importantes

- `crepa_block_index`: bloque estudiante.
- `crepa_teacher_block_index`: bloque profesor EMA. Es obligatorio.
- `crepa_lambda`: fuerza de alineación. Empieza en `0.5`.
- `crepa_self_flow_mask_ratio`: fracción de tokens con el timestep alternativo. Debe estar en `[0.0, 0.5]`.
- `crepa_scheduler`, `crepa_warmup_steps`, `crepa_decay_steps`, `crepa_lambda_end`, `crepa_cutoff_step`: mismos controles de scheduling que CREPA.
- `crepa_use_backbone_features`: es otro modo distinto. No lo mezcles con Self-Flow.

## Muestreo / validación

Self-Flow cambia el entrenamiento, no el algoritmo básico de inferencia.

- El entrenamiento usa ruido tokenwise mezclado para el estudiante y una vista EMA más limpia para el profesor.
- La loss de validación sigue evaluando el schedule homogéneo solicitado.
- El muestreo normal no cambia. No se usa enmascarado dual-timestep en inferencia.

<details>
<summary>Cómo funciona (práctico)</summary>

- Se muestrean dos timesteps y se asignan por token con una máscara aleatoria.
- Se construye una vista del estudiante con corrupción mezclada y una vista del profesor con el timestep más limpio.
- El estudiante corre normalmente y el profesor EMA bajo `no_grad`.
- Se alinea una capa estudiante más temprana con una capa profesor más profunda usando similitud coseno, manteniendo la loss generativa normal.

</details>

<details>
<summary>Técnico (internals de SimpleTuner)</summary>

- La selección del modo vive en `simpletuner/helpers/training/crepa.py` como `CrepaFeatureSource.SELF_FLOW`.
- Los batch builders compartidos están en `_prepare_image_crepa_self_flow_batch` y `_prepare_video_crepa_self_flow_batch`.
- El forward del profesor EMA se ejecuta desde `auxiliary_loss` mediante `_run_crepa_teacher_forward`.
- La validación reconstruye batches homogéneos cuando se piden `custom_timesteps`, para que la loss de evaluación no use el batch mezclado de entrenamiento.

</details>

## Errores comunes

- **EMA desactivado**: Self-Flow requiere `use_ema=true`.
- **Teacher block sin definir**: configura `crepa_teacher_block_index`.
- **TwinFlow activado**: no es compatible.
- **Familia no soportada**: solo funciona en familias que implementan `supports_crepa_self_flow()`.
- **Mask ratio demasiado alto**: mantente en `0.5` o menos.
- **Esperar un sampler especial**: la inferencia sigue siendo normal.

## Referencias

- [Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis](https://bfl.ai/research/self-flow)

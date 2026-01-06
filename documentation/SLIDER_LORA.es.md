# Targeting de Slider LoRA

En esta guía entrenaremos un adaptador estilo slider en SimpleTuner. Usaremos Z-Image Turbo porque entrena rápido, tiene licencia Apache 2.0 y da grandes resultados para su tamaño, incluso con pesos destilados.

Para la matriz completa de compatibilidad (LoRA, LyCORIS, full-rank), consulta la columna Sliders en [documentation/QUICKSTART.md](QUICKSTART.md); esta guía aplica a todas las arquitecturas.

El targeting de sliders funciona con LoRA estándar, LyCORIS (incluyendo `full`) y ControlNet. El toggle está disponible en CLI y WebUI; todo viene en SimpleTuner, no se requieren instalaciones extra.

## Paso 1 — Sigue la configuración base

- **CLI**: Sigue `documentation/quickstart/ZIMAGE.md` para entorno, instalación, notas de hardware y el `config.json` inicial.
- **WebUI**: Usa `documentation/webui/TUTORIAL.md` para ejecutar el wizard del trainer; elige Z-Image Turbo como de costumbre.

Todo de esas guías se puede seguir hasta el punto de configurar un dataset porque los sliders solo cambian dónde se colocan los adaptadores y cómo se muestrean los datos.

## Paso 2 — Habilita targets de slider

- CLI: añade `"slider_lora_target": true` (o pasa `--slider_lora_target true`).
- WebUI: Model → LoRA Config → Advanced → marca “Use slider LoRA targets”.

Para LyCORIS, mantén `lora_type: "lycoris"` y para `lycoris_config.json`, usa los presets en la sección de detalles más abajo.

## Paso 3 — Construye datasets amigables para slider

Los sliders de concepto aprenden de un dataset contrastivo de "opuestos". Crea pares antes/después pequeños (4–6 pares es suficiente para empezar, más si los tienes):

- **Bucket positivo**: “más del concepto” (p. ej., ojos más brillantes, sonrisa más fuerte, más arena). Configura `"slider_strength": 0.5` (cualquier valor positivo).
- **Bucket negativo**: “menos del concepto” (p. ej., ojos más apagados, expresión neutral). Configura `"slider_strength": -0.5` (cualquier valor negativo).
- **Bucket neutral (opcional)**: ejemplos normales. Omite `slider_strength` o establécelo en `0`.

No es necesario mantener nombres de archivo emparejados entre carpetas positivas/negativas; solo asegura el mismo número de muestras en cada bucket.

## Paso 4 — Apunta el dataloader a tus buckets

- Usa el mismo patrón JSON del dataloader del quickstart de Z-Image.
- Añade `slider_strength` a cada entrada del backend. SimpleTuner:
  - Rotará batches **positivo → negativo → neutral** para mantener frescas ambas direcciones.
  - Seguirá respetando la probabilidad de cada backend, así que tus knobs de ponderación siguen funcionando.

No necesitas flags extra: solo los campos `slider_strength`.

## Paso 5 — Entrena

Usa el comando habitual (`simpletuner train ...`) o inicia desde el WebUI. El targeting de slider es automático una vez que el flag está activo.

## Paso 6 — Validar (ajustes opcionales del slider)

Las bibliotecas de prompts pueden llevar escalas de adaptador por prompt para pruebas A/B:

```json
{
  "plain": "regular prompt",
  "slider_plus": { "prompt": "same prompt", "adapter_strength": 1.2 },
  "slider_minus": { "prompt": "same prompt", "adapter_strength": 0.5 }
}
```

Si se omite, la validación usa tu fuerza global.

---

## Referencias y detalles

<details>
<summary>¿Por qué estos targets? (técnico)</summary>

SimpleTuner envía los LoRAs de slider a capas de self-attention, conv/proj y time-embedding para imitar la regla de Concept Sliders de “dejar el texto intacto”. Las ejecuciones con ControlNet también respetan el targeting de sliders. Los adaptadores assistant se mantienen congelados.
</details>

<details>
<summary>Listas de targets de slider por defecto (por arquitectura)</summary>

- General (SD1.x, SDXL, SD3, Lumina2, Wan, HiDream, LTXVideo, Qwen-Image, Cosmos, Stable Cascade, etc.):

  ```json
  [
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    "attn1.to_qkv", "to_qkv",
    "proj_in", "proj_out",
    "conv_in", "conv_out",
    "time_embedding.linear_1", "time_embedding.linear_2"
  ]
  ```

- Flux / Flux2 / Chroma / AuraFlow (solo stream visual):

  ```json
  ["to_q", "to_k", "to_v", "to_out.0", "to_qkv"]
  ```

  Las variantes Flux2 incluyen `attn.to_q`, `attn.to_k`, `attn.to_v`, `attn.to_out.0`, `attn.to_qkv_mlp_proj`.

- Kandinsky 5 (imagen/video):

  ```json
  ["attn1.to_query", "attn1.to_key", "attn1.to_value", "conv_in", "conv_out", "time_embedding.linear_1", "time_embedding.linear_2"]
  ```

</details>

<details>
<summary>Presets de LyCORIS (ejemplo LoKr)</summary>

La mayoría de modelos:

```json
{
  "algo": "lokr",
  "multiplier": 1.0,
  "linear_dim": 4,
  "linear_alpha": 1,
  "apply_preset": {
    "target_module": [
      "attn1.to_q",
      "attn1.to_k",
      "attn1.to_v",
      "attn1.to_out.0",
      "conv_in",
      "conv_out",
      "time_embedding.linear_1",
      "time_embedding.linear_2"
    ]
  }
}
```

Flux/Chroma/AuraFlow: cambia los targets a `{"attn.to_q","attn.to_k","attn.to_v","attn.to_out.0","attn.to_qkv_mlp_proj"}` (elimina `attn.` cuando los checkpoints no lo incluyen). Evita proyecciones `add_*` para mantener texto/contexto intacto.

Kandinsky 5: usa `attn1.to_query/key/value` más `conv_*` y `time_embedding.linear_*`.
</details>

<details>
<summary>Cómo funciona el muestreo (técnico)</summary>

Los backends marcados con `slider_strength` se agrupan por signo y se muestrean en un ciclo fijo: positivo → negativo → neutral. Dentro de cada grupo, aplican las probabilidades habituales del backend. Los backends agotados se eliminan y el ciclo continúa con lo que queda.
</details>

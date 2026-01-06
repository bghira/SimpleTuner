# Inicio rápido de Chroma 1

![image](https://github.com/user-attachments/assets/3c8a12c6-9d45-4dd4-9fc8-6b7cd3ed51dd)

Chroma 1 es una variante recortada de 8.9B parámetros de Flux.1 Schnell publicada por Lodestone Labs. Esta guía recorre la configuración de SimpleTuner para entrenamiento LoRA.

## Requisitos de hardware

A pesar del menor número de parámetros, el uso de memoria es cercano a Flux Schnell:

- Cuantizar el transformer base aún puede usar **≈40–50 GB** de RAM del sistema.
- El entrenamiento LoRA de rango 16 normalmente consume:
  - ~28 GB de VRAM sin cuantización base
  - ~16 GB de VRAM con int8 + bf16
  - ~11 GB de VRAM con int4 + bf16
  - ~8 GB de VRAM con NF4 + bf16
- Mínimo realista de GPU: tarjetas clase **RTX 3090 / RTX 4090 / L40S** o mejores.
- Funciona bien en **Apple M-series (MPS)** para entrenamiento LoRA y en AMD ROCm.
- Aceleradores de clase 80 GB o configuraciones multi-GPU se recomiendan para fine-tuning de rango completo.

## Requisitos previos

Chroma comparte las mismas expectativas de runtime que la guía de Flux:

- Python **3.10 – 3.12**
- Un backend de aceleración compatible (CUDA, ROCm o MPS)

Comprueba tu versión de Python:

```bash
python3 --version
```

Instala SimpleTuner (ejemplo CUDA):

```bash
pip install simpletuner[cuda]
```

Para detalles de configuración específicos del backend (CUDA, ROCm, Apple), consulta la [guía de instalación](../INSTALL.md).

## Lanzar la web UI

```bash
simpletuner server
```

La UI estará disponible en http://localhost:8001.

## Configuración vía CLI

`simpletuner configure` te guía por los ajustes centrales. Los valores clave para Chroma son:

- `model_type`: `lora`
- `model_family`: `chroma`
- `model_flavour`: uno de
  - `base` (predeterminado, calidad equilibrada)
  - `hd` (más fidelidad, más exigente en cómputo)
  - `flash` (rápido pero inestable – no recomendado para producción)
- `pretrained_model_name_or_path`: déjalo vacío para usar el mapeo de flavour de arriba
- `model_precision`: mantiene el predeterminado `bf16`
- `flux_fast_schedule`: déjalo **deshabilitado**; Chroma tiene su propio muestreo adaptativo

### Fragmento de configuración manual de ejemplo

<details>
<summary>Ver ejemplo de config</summary>

```jsonc
{
  "model_type": "lora",
  "model_family": "chroma",
  "model_flavour": "base",
  "output_dir": "/workspace/chroma-output",
  "network_rank": 16,
  "learning_rate": 2.0e-4,
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "pretrained_model_name_or_path": null
}
```
</details>

> ⚠️ Si el acceso a Hugging Face es lento en tu región, exporta `HF_ENDPOINT=https://hf-mirror.com` antes de iniciar.

### Funciones experimentales avanzadas

<details>
<summary>Mostrar detalles experimentales avanzados</summary>


SimpleTuner incluye funciones experimentales que pueden mejorar significativamente la estabilidad y el rendimiento del entrenamiento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduce el sesgo de exposición y mejora la calidad de salida dejando que el modelo genere sus propias entradas durante el entrenamiento.

> ⚠️ Estas funciones aumentan la sobrecarga computacional del entrenamiento.

</details>

## Dataset y dataloader

Chroma usa el mismo formato de dataloader que Flux. Consulta el [tutorial general](../TUTORIAL.md) o el [tutorial de la web UI](../webui/TUTORIAL.md) para preparar datasets y librerías de prompts.

## Opciones de entrenamiento específicas de Chroma

- `flux_lora_target`: controla qué módulos del transformer reciben adaptadores LoRA (`all`, `all+ffs`, `context`, `tiny`, etc.). Los valores predeterminados reflejan Flux y funcionan bien en la mayoría de casos.
- `flux_guidance_mode`: `constant` funciona bien; Chroma no expone un rango de guidance.
- El enmascarado de atención está siempre habilitado: asegúrate de que tu caché de embeddings de texto se haya generado con máscaras de padding (comportamiento predeterminado en versiones actuales de SimpleTuner).
- Las opciones de schedule shift (`flow_schedule_shift` / `flow_schedule_auto_shift`) no son necesarias para Chroma: el helper ya refuerza automáticamente los timesteps de cola.
- `flux_t5_padding`: configura `zero` si prefieres poner en cero los tokens con padding antes del enmascarado.

## Muestreo automático de timesteps de cola

Flux usaba un schedule log-normal que submuestreaba los extremos de alto ruido / bajo ruido. El helper de entrenamiento de Chroma aplica un remapeo cuadrático (`σ ↦ σ²` / `1-(1-σ)²`) a las sigmas muestreadas para visitar con más frecuencia las regiones de cola. Esto **no requiere configuración extra**: está integrado en la familia de modelos `chroma`.

## Validación y consejos de muestreo

- `validation_guidance_real` se asigna directamente al `guidance_scale` de la pipeline. Déjalo en `1.0` para muestreo de un solo pase, o súbelo a `2.0`–`3.0` si quieres classifier-free guidance durante renders de validación.
- Usa 20 pasos de inferencia para previews rápidos; 28–32 para mayor calidad.
- Los prompts negativos siguen siendo opcionales; el modelo base ya está de-destilado.
- El modelo solo soporta texto-a-imagen por ahora; el soporte img2img llegará en una actualización posterior.

## Solución de problemas

- **OOM al iniciar**: habilita `offload_during_startup` o cuantiza el modelo base (`base_model_precision: int8-quanto`).
- **El entrenamiento diverge temprano**: asegúrate de que gradient checkpointing esté activado, baja `learning_rate` a `1e-4` y verifica que los captions sean diversos.
- **La validación repite la misma pose**: alarga los prompts; los modelos flow-matching colapsan cuando la variedad de prompts es baja.

Para temas avanzados—DeepSpeed, FSDP2, métricas de evaluación—consulta las guías compartidas enlazadas en el README.

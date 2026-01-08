# Guía de FlexAttention

**FlexAttention requiere dispositivos CUDA.**

FlexAttention es el kernel de atención a nivel de bloques de PyTorch que llegó en PyTorch 2.5.0. Reescribe el cómputo SDPA como un bucle programable para que puedas expresar estrategias de máscara sin escribir CUDA. Diffusers lo expone a través del nuevo dispatcher `attention_backend`, y SimpleTuner conecta ese dispatcher a `--attention_mechanism=flex`.

> ⚠️ FlexAttention todavía está etiquetado como “prototype” upstream. Espera recompilar cuando cambies drivers, versiones de CUDA o builds de PyTorch.

## Requisitos previos

1. **GPU Ampere+** – Se admiten NVIDIA SM80 (A100), Ada (4090/L40S) o Hopper (H100/H200). Las tarjetas más antiguas fallan la comprobación de capacidad durante el registro del kernel.
2. **Toolchain de compilación** – Los kernels se compilan en tiempo de ejecución con `nvcc`. Instala `cuda-nvcc` que coincida con el wheel (CUDA 12.8 en las versiones actuales) y asegúrate de que `nvcc` esté en `$PATH`.

## Compilar los kernels

La primera importación de `torch.nn.attention.flex_attention` compila la extensión CUDA en la caché diferida de PyTorch. Puedes hacerlo con antelación para exponer errores de compilación temprano:

```bash
python - <<'PY'
import torch
from torch.nn.attention import flex_attention

assert torch.__version__ >= "2.5.0", torch.__version__
flex_attention.build_flex_attention_kernels()  # no-op when already compiled
print("FlexAttention kernels installed at", flex_attention.kernel_root)
PY
```

- Si ves `AttributeError: flex_attention has no attribute build_flex_attention_kernels`, actualiza PyTorch: el helper viene en 2.5.0+.
- La caché vive bajo `~/.cache/torch/kernels`. Elimínala si actualizas CUDA y necesitas forzar una recompilación.

## Habilitar FlexAttention en SimpleTuner

Una vez que existan los kernels, selecciona el backend vía `config.json`:

```json
{
  "attention_mechanism": "flex"
}
```

Qué esperar:

- Solo los bloques de transformer con dispatch habilitado (Flux, Wan 2.2, LTXVideo, QwenImage, etc.) pasan por `attention_backend` de Diffusers. Los UNet clásicos de SD/SDXL siguen llamando a PyTorch SDPA directamente, por lo que FlexAttention no tiene efecto allí.
- FlexAttention actualmente soporta tensores BF16/FP16. Si ejecutas pesos FP32 o FP8 verás `ValueError: Query, key, and value must be either bfloat16 or float16`.
- El backend respeta solo `is_causal=False`. Proporcionar una máscara la convierte en la máscara por bloques que el kernel espera, pero las máscaras irregulares no están soportadas todavía (refleja el comportamiento upstream).

## Checklist de solución de problemas

| Síntoma | Solución |
| --- | --- |
| `RuntimeError: Flex Attention backend 'flex' is not usable because of missing package` | El build de PyTorch es < 2.5 o no incluye CUDA. Instala un wheel CUDA más reciente. |
| `Could not compile flex_attention kernels` | Asegúrate de que `nvcc` coincida con la versión de CUDA que espera tu wheel de torch (12.1+). Define `export CUDA_HOME=/usr/local/cuda-12.4` si el instalador no encuentra headers. |
| `ValueError: Query, key, and value must be on a CUDA device` | FlexAttention es solo CUDA. Quita la configuración de backend en ejecuciones Apple/ROCm. |
| El entrenamiento nunca cambia al backend | Asegúrate de usar una familia de modelos que ya use `dispatch_attention_fn` de Diffusers (Flux/Wan/LTXVideo). Los UNet estándar de SD seguirán usando PyTorch SDPA sin importar el backend seleccionado. |

Consulta la documentación upstream para más detalles internos y flags de API: [docs de PyTorch FlexAttention](https://pytorch.org/docs/stable/nn.attention.html#flexattention).

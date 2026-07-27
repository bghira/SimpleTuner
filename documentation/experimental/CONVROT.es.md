# ConvRot / Hadamard SDNQ

SimpleTuner expone una rotación estilo ConvRot mediante la ruta Hadamard de SDNQ. Es útil para trabajos PEFT grandes donde el modelo base congelado debe ejecutarse en int8 mientras los adaptadores LoRA o LyCORIS siguen entrenándose en bf16.

SimpleTuner no consume buffers sidecar ConvRot arbitrarios como una funcion separada. Para la ruta comun, carga los pesos originales del modelo y deja que SimpleTuner cuantice el componente entrenado con SDNQ despues de cargar el modelo. Los loaders que soportan pesos transformer cuantizados de archivo unico tambien pueden cargar safetensors transformer INT8 ConvRot compatibles y ejecutarlos mediante SDNQ Hadamard.

## Configuración rápida

```json
{
  "base_model_precision": "int8-sdnq",
  "gradient_checkpointing": true,
  "sdnq_use_hadamard": true,
  "sdnq_hadamard_group_size": 256,
  "sdnq_group_size": -1,
  "sdnq_use_quantized_matmul": true,
  "sdnq_compile_mode": "compile"
}
```

Para modelos grandes, mantén `quantize_via` en `cpu` salvo que la guía del modelo indique otra cosa. La cuantización en CPU reduce el pico de memoria del acelerador durante la preparación.

## Qué hacen las opciones

- `base_model_precision: int8-sdnq` selecciona cuantización SDNQ int8 posterior a la carga para el componente base entrenado.
- `sdnq_use_hadamard: true` activa la ruta de rotación Hadamard.
- `sdnq_hadamard_group_size: 256` define el tamaño de bloque de rotación usado por SDNQ. Usa `256` para ConvRot; bloques mas pequenos seleccionan una ruta estilo QuaRot.
- `sdnq_group_size: -1` usa escalas estáticas por fila de pesos. Esto evita la ruta dinámica agrupada, orientada principalmente a full fine-tuning, que puede recuantizar pesos durante el entrenamiento.
- `sdnq_use_quantized_matmul: true` mantiene activa la ruta SDNQ int8 matmul.
- `sdnq_compile_mode: compile` compila helpers y kernels de cuantización donde SDNQ lo soporta.
- `gradient_checkpointing: true` permite que SDNQ use la ruta de entrenamiento de menor overhead para cargas PEFT. SimpleTuner pasa esto a SDNQ como `use_grad_ckpt=True`; con gradient checkpointing habilitado, poner ese flag de SDNQ en false solo agrega trabajo para guardar entradas backward cuantizadas que checkpointing descarta de inmediato.

## Comportamiento PEFT

El transformer base se cuantiza con SDNQ. Los pesos del adaptador siguen siendo entrenables y usan el dtype normal de precisión mixta, normalmente bf16.

Algunos modelos cargan adaptadores auxiliares fijos antes del entrenamiento. Z-Image Turbo, por ejemplo, tiene una assistant LoRA. SimpleTuner retrasa ese adaptador hasta después de la cuantización SDNQ para que SDNQ vea los módulos transformer originales en lugar de pesos proxy del wrapper PEFT.

## Requisitos y límites

- SimpleTuner instala y configura la dependencia de entrenamiento SDNQ para los targets de instalacion soportados.
- Este preset está pensado para entrenamiento LoRA y LyCORIS de modelos grandes. Full fine-tuning con SDNQ Hadamard necesita validación separada.
- Los primeros steps pueden ser lentos porque SDNQ y Torch compilan kernels durante la preparación y el inicio del entrenamiento.
- Validación e inferencia usan el modelo base cuantizado más el adaptador activo, igual que el entrenamiento.
- ConvRot puede reducir el dano de cuantizacion, pero no garantiza que INT8 iguale a BF16 o FP8 en todos los modelos. Valida tanto la curva de loss como las muestras generadas antes de comprometerte con un run largo.
- La inferencia standalone con SDNQ ConvRot queda fuera de esta guia de entrenamiento. Para APIs directas de inferencia SDNQ, sigue la [documentacion upstream de SDNQ](https://github.com/Disty0/sdnq) porque esa API cambia con mas frecuencia que la configuracion de entrenamiento de SimpleTuner.

## Resultados medidos

Estas son mediciones del trainer real de SimpleTuner por modelo, no resultados sinteticos solo de GEMM. `Loop s/step` es el tiempo de pared del loop de entrenamiento por paso. `Paso medio` excluye los primeros cinco pasos de warmup.

| Modelo | GPU | Pasos | Ruta de pesos | Loop s/step | Paso medio | p50 | p95 | VRAM maxima asignada |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Z-Image Turbo LoRA | H100 80GB | 1000 | cuantizacion post-load SDNQ Hadamard | 1.107 | 1.087 | 1.071 | 1.109 | 9.70 GiB |
| Z-Image Turbo LoRA | L40S | 1000 | cuantizacion post-load SDNQ Hadamard | 1.026 | 1.018 | 1.002 | 1.040 | 9.66 GiB |
| Z-Image Turbo LoRA | L40S | 1000 | baseline SDNQ Hadamard | 1.131 | 1.072 | 1.055 | 1.102 | 9.66 GiB |
| Krea 2 Raw LoRA | H100 80GB | 100 | pesos transformer `lilcheaty/Krea2-INT8-ConvRot`, atencion diffusers | 0.787 | 0.399 | 0.397 | 0.411 | 32.15 GiB |
| Krea 2 Raw LoRA | L40S | 100 | pesos transformer `lilcheaty/Krea2-INT8-ConvRot`, atencion cuDNN | 0.945 | 0.794 | 0.793 | 0.799 | 31.89 GiB |
| Mage-Flow LoRA, crop cuadrado | H100 80GB | 100 | cuantizacion post-load SDNQ INT8 vanilla | 1.113 | 0.277 | 0.276 | 0.286 | 20.12 GiB |
| Mage-Flow LoRA, crop cuadrado | H100 80GB | 100 | cuantizacion post-load SDNQ ConvRot 256 | 0.436 | 0.299 | 0.297 | 0.308 | 20.15 GiB |

En la comparacion Z-Image con caches calientes en L40S, la ruta actual fue 10.3% mas rapida por tiempo de loop y 5.2% mas rapida por media de paso medida que el baseline SDNQ Hadamard. Las filas de Krea 2 verifican la ruta de pesos transformer INT8 ConvRot de Hugging Face en ejecuciones reales de entrenamiento de 100 pasos. Las filas de Mage-Flow muestran por que la validacion por modelo importa: el crop cuadrado elimino la mayor parte del churn de compilacion por formas, ConvRot redujo el tiempo total del loop frente a INT8 vanilla, pero el paso medido ya caliente fue un poco mas lento que INT8 vanilla.

## Modelos de ejemplo

SimpleTuner incluye ejemplos SDNQ Hadamard para Z-Image Turbo, Krea 2, FLUX.2, Cosmos 3 y LTXVideo 2.3. Estos ejemplos usan `sdnq_group_size: -1` porque esa configuración encajó mejor con PEFT que el default dinámico agrupado de entrenamiento.

# ConvRot / Hadamard SDNQ

SimpleTuner expone una rotación estilo ConvRot mediante la ruta Hadamard de SDNQ. Es útil para trabajos PEFT grandes donde el modelo base congelado debe ejecutarse en int8 mientras los adaptadores LoRA o LyCORIS siguen entrenándose en bf16.

Esto no carga directamente buffers de checkpoints ConvRot externos. Carga los pesos originales del modelo y deja que SimpleTuner cuantice el componente entrenado con SDNQ después de cargar el modelo.

## Configuración rápida

```json
{
  "base_model_precision": "int8-sdnq",
  "gradient_checkpointing": true,
  "sdnq_use_hadamard": true,
  "sdnq_hadamard_group_size": 128,
  "sdnq_group_size": -1,
  "sdnq_use_quantized_matmul": true,
  "sdnq_compile_mode": "compile"
}
```

Para modelos grandes, mantén `quantize_via` en `cpu` salvo que la guía del modelo indique otra cosa. La cuantización en CPU reduce el pico de memoria del acelerador durante la preparación.

## Qué hacen las opciones

- `base_model_precision: int8-sdnq` selecciona cuantización SDNQ int8 posterior a la carga para el componente base entrenado.
- `sdnq_use_hadamard: true` activa la ruta de rotación Hadamard.
- `sdnq_hadamard_group_size: 128` define el tamaño de bloque de rotación usado por SDNQ.
- `sdnq_group_size: -1` usa escalas estáticas por fila de pesos. Esto evita la ruta dinámica agrupada, orientada principalmente a full fine-tuning, que puede recuantizar pesos durante el entrenamiento.
- `sdnq_use_quantized_matmul: true` mantiene activa la ruta SDNQ int8 matmul.
- `sdnq_compile_mode: compile` compila helpers y kernels de cuantización donde SDNQ lo soporta.
- `gradient_checkpointing: true` permite que SDNQ use la ruta de entrenamiento de menor overhead para cargas PEFT. SimpleTuner pasa esto a SDNQ como `use_grad_ckpt=True`; con gradient checkpointing habilitado, poner ese flag de SDNQ en false solo agrega trabajo para guardar entradas backward cuantizadas que checkpointing descarta de inmediato.

## Comportamiento PEFT

El transformer base se cuantiza con SDNQ. Los pesos del adaptador siguen siendo entrenables y usan el dtype normal de precisión mixta, normalmente bf16.

Algunos modelos cargan adaptadores auxiliares fijos antes del entrenamiento. Z-Image Turbo, por ejemplo, tiene una assistant LoRA. SimpleTuner retrasa ese adaptador hasta después de la cuantización SDNQ para que SDNQ vea los módulos transformer originales en lugar de pesos proxy del wrapper PEFT.

## Requisitos y límites

- Usa un build de SDNQ con soporte Hadamard. La verificación en H100 usó SDNQ upstream `0.2.3`; PyPI `0.2.2` no incluye la misma corrección Hadamard para bf16.
- Este preset está pensado para entrenamiento LoRA y LyCORIS de modelos grandes. Full fine-tuning con SDNQ Hadamard necesita validación separada.
- Los primeros steps pueden ser lentos porque SDNQ y Torch compilan kernels durante la preparación y el inicio del entrenamiento.
- Validación e inferencia usan el modelo base cuantizado más el adaptador activo, igual que el entrenamiento.

## Modelos de ejemplo

SimpleTuner incluye ejemplos SDNQ Hadamard para Z-Image Turbo, Krea 2, FLUX.2, Cosmos 3 y LTXVideo 2.3. Estos ejemplos usan `sdnq_group_size: -1` porque esa configuración encajó mejor con PEFT que el default dinámico agrupado de entrenamiento.

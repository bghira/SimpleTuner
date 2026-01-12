# Atención dispersa-lineal (SLA) en SimpleTuner

La atención dispersa-lineal (SLA) fusiona FlashAttention dispersa y un compensador de atención lineal dentro de un único kernel CUDA. Los bloques críticos de query/key toman la ruta dispersa costosa, mientras que los bloques marginales usan atención lineal ligera más una proyección aprendible. Esto mantiene la calidad cerca de la atención completa mientras reduce drásticamente los FLOPs.

SimpleTuner expone SLA mediante el flag normal `--attention_mechanism`, por lo que puedes ajustar modelos con SLA y luego ejecutar inferencia con el mismo kernel.

## Requisitos

1. Instala la implementación de referencia:

   ```bash
   git clone https://github.com/thu-ml/SLA.git ~/src/SLA
   pip install -e ~/src/SLA
   ```

2. Usa una compilación de PyTorch con CUDA (los kernels de SLA hoy son solo CUDA).

## Habilitar SLA

- Pasa `--attention_mechanism=sla` (o configura `attention_mechanism: "sla"` en los configs).
- No se requieren flags adicionales; SimpleTuner inyecta SLA envolviendo el punto de entrada SDPA de PyTorch.
- Sobrescribe la configuración de SLA (proporción top-k, tamaños de bloque, tipo de feature map, si los feature maps de query/key están atados) vía `--sla_config` / `sla_config` en forma de dict JSON/Python. Ejemplo: `--sla_config '{"topk":0.15,"blkq":32,"tie_feature_map_qk":false}'`. Valores por defecto: top 20 %, tamaño de bloque 64, feature maps atados.

## Comportamiento de entrenamiento

- SLA es entrenable. El controlador mantiene la cabeza de proyección lineal (`proj_l`) en `float32` incluso cuando el resto de SLA se ejecuta en BF16/FP16, de modo que AMP/GradScaler se mantengan estables.
- Como el backbone se ajusta para esperar el comportamiento mixto disperso/lineal de SLA, deberías seguir usando SLA durante la inferencia. Volver a Diffusers SDPA/XFormers después del entrenamiento probablemente perjudique la calidad.
- Durante los guardados de checkpoint, SimpleTuner escribe `sla_attention.pt` junto al estado normal del acelerador. Este archivo contiene los pesos de proyección de SLA y los buffers relacionados para cada par único de dimensión de cabeza/tipo de dato que se materializó. Mantén este archivo con el resto del checkpoint; eliminarlo significa que el siguiente reanudado/inferencia reinicializará la capa de proyección de SLA.

## Inferencia

- Mantén `--attention_mechanism=sla` habilitado siempre que reanudes entrenamiento o vuelvas a ejecutar pasos de validación para que el checkpoint siga usando el kernel SLA con el que se ajustó.
- El loader reproduce automáticamente `sla_attention.pt` si existe dentro del directorio del checkpoint, por lo que no se requieren flags adicionales.
- Si intencionalmente quieres comparar pesos entrenados con SLA con SDPA estándar, espera una caída de calidad. El paper de SLA muestra que se requieren algunos miles de pasos de ajuste para adaptar el backbone, por lo que la inferencia sin SLA debe tratarse como no soportada.

## Solución de problemas y notas

- **Falta `sla_attention.pt`:** Esto significa que el checkpoint se creó antes de que existiera el guardado del estado de SLA o que el archivo fue eliminado. Reejecuta una sesión corta de entrenamiento (incluso un solo paso) con SLA habilitado para regenerar el archivo.
- **Errores de AMP/GradScaler:** Asegúrate de no estar casteando manualmente módulos de SLA a BF16/FP16. SimpleTuner fuerza la cabeza de proyección a FP32 automáticamente; casts adicionales pueden desestabilizar el entrenamiento.
- **Subidas al Hub:** Al publicar checkpoints en Hugging Face Hub (o cualquier almacén de artefactos), incluye `sla_attention.pt`. Quienes descarguen tu checkpoint heredarán los pesos entrenados de SLA sin pasos extra.

Para más contexto sobre el diseño de SLA y el algoritmo completo, consulta [SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse–Linear Attention](https://www.arxiv.org/abs/2509.24006).

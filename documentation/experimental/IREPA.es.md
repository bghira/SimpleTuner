# iREPA

iREPA mejora la alineación de representaciones al conservar la estructura espacial en la ruta de alineación. Sustituye el proyector lineal por token por una convolución espacial y normaliza cada canal de las features del teacher con z-score sobre los patches de la imagen.

SimpleTuner usa el motor existente según el backbone: los modelos Transformer de imagen usan REPA/CREPA; los modelos Transformer de video aplican iREPA por frame y conservan la loss temporal de CREPA; los modelos UNet usan el mid-block y la manifold loss de U-REPA. La cuadrícula rectangular se deriva de la forma de los latents limpios.

```json
{
  "irepa_enabled": true,
  "irepa_spatial_norm_alpha": 0.6,
  "irepa_projector_kernel_size": 3,
  "crepa_enabled": true,
  "crepa_block_index": 8,
  "crepa_lambda": 1.0
}
```

Activa `crepa_enabled` con iREPA para Transformer o `urepa_enabled` con iREPA para UNet. Las opciones `crepa_*` o `urepa_*` controlan teacher, peso, capa y schedule. `0.6` corresponde a la receta de latent diffusion; kernel `3` es la arquitectura publicada.

iREPA requiere hidden states con patch tokens espaciales y latents limpios para recuperar la cuadrícula. En video, la convolución no mezcla frames.

Usa entrenamiento de modelo completo o LoRA PEFT estándar. LyCORIS no puede guardar el proyector auxiliar y no es compatible.

Referencia: [What Matters for Representation Alignment: Global Information or Spatial Structure?](https://arxiv.org/abs/2512.10794)

# iREPA

iREPA melhora o alinhamento de representacoes preservando a estrutura espacial no caminho de alinhamento. Ele troca o projetor linear por token por uma convolucao espacial e normaliza cada canal das features do teacher com z-score sobre os patches da imagem.

O SimpleTuner usa o mecanismo existente conforme o backbone: modelos Transformer de imagem usam REPA/CREPA; modelos Transformer de video aplicam iREPA por frame e mantem a loss temporal do CREPA; modelos UNet usam o mid-block e a manifold loss do U-REPA. A grade retangular de tokens e derivada do shape dos latents limpos.

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

Habilite `crepa_enabled` com iREPA para Transformer ou `urepa_enabled` com iREPA para UNet. As opcoes `crepa_*` ou `urepa_*` controlam teacher, peso, camada e schedule. `0.6` corresponde a receita de latent diffusion; kernel `3` e a arquitetura publicada.

iREPA requer hidden states com patch tokens espaciais e latents limpos para recuperar a grade. Em video, a convolucao nao mistura frames.

Use treinamento de modelo completo ou LoRA PEFT padrao. LyCORIS nao consegue salvar o projetor auxiliar e nao e suportado.

Referencia: [What Matters for Representation Alignment: Global Information or Spatial Structure?](https://arxiv.org/abs/2512.10794)

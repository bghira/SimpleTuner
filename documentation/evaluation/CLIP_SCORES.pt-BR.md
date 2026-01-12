# Acompanhamento de CLIP score

CLIP scores estao vagamente relacionados a medicao da capacidade do modelo de seguir prompts; nao tem relacao com qualidade/fidelidade de imagem.

O score `clip/mean` do seu modelo indica o quao proximos os recursos extraidos da imagem estao dos recursos extraidos do prompt. Atualmente e uma metrica popular para determinar aderencia geral ao prompt, embora normalmente seja avaliada em um numero muito grande (~5.000) de prompts de teste (ex.: Parti Prompts).

Gerar CLIP scores durante o pretraining do modelo pode ajudar a demonstrar que ele esta se aproximando do objetivo, mas quando um valor `clip/mean` em torno de `.30` a `.39` e alcancado, a comparacao parece ficar menos significativa. Modelos com CLIP score medio em torno de `.33` podem superar um modelo com `.36` em analise humana. No entanto, um modelo com CLIP score medio muito baixo, em torno de `0.18` a `0.22`, provavelmente tera desempenho ruim.

Dentro de uma unica execucao de teste, alguns prompts resultarao em CLIP score muito baixo em torno de `0.14` (valor `clip/min` nos graficos do tracker) mesmo que suas imagens estejam bem alinhadas ao prompt do usuario e tenham alta qualidade; por outro lado, scores tao altos quanto `0.39` (valor `clip/max` nos graficos do tracker) podem aparecer em imagens com qualidade questionavel, ja que este teste nao foi feito para capturar essa informacao. Por isso, costuma-se usar um numero tao grande de prompts para medir desempenho - _e mesmo assim_.

Por si so, CLIP scores nao demoram muito para calcular; porem, o numero de prompts exigido para uma avaliacao significativa pode fazer isso demorar bastante.

Como nao custa muito rodar, nao faz mal incluir avaliacao de CLIP em treinos pequenos. Talvez voce descubra um padrao nas saidas onde faz sentido abandonar um treino ou ajustar outros hiperparametros como taxa de aprendizado.

Para incluir uma biblioteca de prompts padrao para avaliacao, `--validation_prompt_library` pode ser fornecido e entao geraremos um benchmark relativo entre execucoes de treinamento.

No `config.json`:

```json
{
  ...
  "evaluation_type": "clip",
  "pretrained_evaluation_model_name_or_path": "openai/clip-vit-large-patch14-336",
  "report_to": "tensorboard", # ou wandb
  ...
}
```

## Compatibilidade

SageAttention nao e compatível com acompanhamento de CLIP score no momento. Um ou outro deve ser desativado.

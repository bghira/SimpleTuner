# Sparse-Linear Attention (SLA) no SimpleTuner

Sparse-Linear Attention (SLA) combina FlashAttention esparso e um compensador de atencao linear em um unico kernel CUDA. Blocos criticos de query/key seguem o caminho esparso caro, enquanto blocos marginais usam atencao linear leve mais uma projecao aprendivel. Isso mantem a qualidade proxima da atencao completa enquanto reduz drasticamente FLOPs.

O SimpleTuner expoe SLA pela flag regular `--attention_mechanism`, entao voce pode fazer fine-tuning com SLA e depois rodar inferencia com o mesmo kernel.

## Requisitos

1. Instale a implementacao de referencia:

   ```bash
   git clone https://github.com/thu-ml/SLA.git ~/src/SLA
   pip install -e ~/src/SLA
   ```

2. Use uma build CUDA do PyTorch (os kernels SLA sao somente CUDA hoje).

## Habilitando SLA

- Passe `--attention_mechanism=sla` (ou defina `attention_mechanism: "sla"` nos configs).
- Nenhuma flag extra e necessaria; o SimpleTuner injeta SLA ao envolver o entrypoint SDPA do PyTorch.
- Sobrescreva configuracoes do SLA (top-k ratio, tamanhos de bloco, tipo de feature map, se os feature maps de query/key sao amarrados) via `--sla_config` / `sla_config` em formato JSON/dict Python. Exemplo: `--sla_config '{"topk":0.15,"blkq":32,"tie_feature_map_qk":false}'`. Padroes: top 20%, block size 64, feature maps amarrados.

## Comportamento no treinamento

- SLA e treinavel. O controlador mantem a cabeca de projecao linear (`proj_l`) em `float32` mesmo quando o resto do SLA executa em BF16/FP16 para manter AMP/GradScaler estaveis.
- Como o backbone e ajustado para o comportamento misto esparso/linear do SLA, voce deve continuar a usar SLA durante a inferencia. Voltar para SDPA/XFormers do Diffusers apos o treinamento provavelmente vai reduzir a qualidade.
- Durante o salvamento de checkpoints, o SimpleTuner grava `sla_attention.pt` junto com o estado normal do acelerador. Esse arquivo contem os pesos de projecao do SLA e buffers relacionados para cada par unico de head dimension/dtype que foi materializado. Mantenha esse arquivo com o restante do checkpoint; removê-lo significa que a proxima retomada/inferencia reinicializara a camada de projecao do SLA.

## Inferencia

- Mantenha `--attention_mechanism=sla` habilitado sempre que voce retomar treino ou rerodar etapas de validacao para que o checkpoint continue usando o kernel SLA com o qual foi ajustado.
- O loader reexecuta automaticamente `sla_attention.pt` se ele existir dentro do diretorio do checkpoint, entao nenhuma flag extra e necessaria.
- Se voce intencionalmente quiser comparar pesos treinados com SLA usando SDPA padrao, espere uma queda de qualidade. O paper do SLA mostra que alguns milhares de steps de ajuste sao necessarios para adaptar o backbone, entao inferencia sem SLA deve ser tratada como nao suportada.

## Troubleshooting e notas

- **`sla_attention.pt` ausente:** Isso significa que o checkpoint foi criado antes de existir salvamento de estado do SLA ou o arquivo foi removido. Rode uma sessao curta de treino (mesmo um unico step) com SLA habilitado para regenerar o arquivo.
- **Erros de AMP/GradScaler:** Garanta que voce nao esta convertendo manualmente modulos do SLA para BF16/FP16. O SimpleTuner forca a cabeca de projecao para FP32 automaticamente; conversoes adicionais podem desestabilizar o treinamento.
- **Uploads para o Hub:** Ao enviar checkpoints para o Hugging Face Hub (ou qualquer store de artefatos), inclua `sla_attention.pt`. Quem baixar seu checkpoint herdara os pesos treinados do SLA sem passos extras.

Para mais contexto sobre o design do SLA e o algoritmo completo, veja [SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention](https://www.arxiv.org/abs/2509.24006).

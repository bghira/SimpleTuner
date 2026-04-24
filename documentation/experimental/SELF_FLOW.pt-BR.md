# Self-Flow (alinhamento interno)

Self-Flow e um modo de CREPA que substitui o encoder visual externo por uma visao EMA mais limpa do proprio modelo. Ele segue de perto a ideia do paper da Black Forest Labs: treinar o estudante com ruido tokenwise misto, rodar o professor EMA numa visao mais limpa e alinhar hidden states internos enquanto mantem a loss generativa normal.

Comparado com metodos proximos do SimpleTuner:

| Metodo | Fonte do professor | Assimetria de ruido | Modelo professor extra | Ideia principal |
| --- | --- | --- | --- | --- |
| REPA / VIDEO_CREPA | Encoder externo congelado, geralmente DINOv2 | Nao | Sim | Alinhar hidden states do modelo a features semanticas externas |
| LayerSync | Camada mais profunda do mesmo forward pass | Nao | Nao | Alinhar uma camada mais cedo a uma camada posterior mais forte |
| TwinFlow | Professor EMA e alvos recursivos de trajetoria | Sem divisao tokenwise entre visoes mais limpa e mais ruidosa | Sem modelo externo | Matching de trajetoria em poucos passos, opcionalmente com semantica de sinal de tempo negativo |
| Self-Flow | Professor EMA do mesmo modelo em uma visao mais limpa | Sim | Sem modelo externo | Aprender representacoes internas mais fortes via dual-timestep scheduling |

> **Quer alinhamento com encoder externo?** Veja [IMAGE_REPA.pt-BR.md](IMAGE_REPA.pt-BR.md) para REPA / U-REPA e [VIDEO_CREPA.pt-BR.md](VIDEO_CREPA.pt-BR.md) para CREPA temporal.

## Quando usar

- Voce quer o regularizador auto-supervisionado no estilo BFL em vez de um encoder externo.
- Voce esta treinando uma familia transformer que ja expoe hooks de Self-Flow no SimpleTuner.
- Voce quer que o mesmo regularizador ajude geracao normal, edicao e treino multimodal.
- Voce ja usa EMA ou pode ativar. Self-Flow exige professor EMA.

Familias suportadas atualmente:

- Imagem / edicao: `flux`, `flux2`, `sd3`, `pixart`, `sana`, `qwen_image`, `chroma`, `hidream`, `auraflow`, `lumina2`, `z_image`, `z_image_omni`, `kandinsky5_image`, `longcat_image`, `omnigen`, `ace_step`
- Video / multimodal: `wan`, `wan_s2v`, `ltxvideo`, `ltxvideo2`, `sanavideo`, `kandinsky5_video`, `hunyuanvideo`, `longcat_video`, `cosmos`, `anima`

## Setup rapido (WebUI)

1. Abra **Training → Loss functions**.
2. Ative **CREPA**.
3. Defina **CREPA Feature Source** como `self_flow`.
4. Use **CREPA Block Index** como bloco estudante mais cedo. Comece com `8` em DiTs de 24 camadas e `10` em stacks mais profundos.
5. Use **CREPA Teacher Block Index** como bloco professor mais profundo. `16` ou `20` sao bons pontos de partida.
6. Deixe **Weight** em `0.5`.
7. Use **Self-Flow Mask Ratio**:
   - `0.25` para imagem
   - `0.10` para video
   - `0.50` para audio como `ace_step`
8. Garanta que **EMA** esteja ativado.
9. Nao combine com TwinFlow.

## Setup rapido (config JSON / CLI)

```json
{
  "use_ema": true,
  "crepa_enabled": true,
  "crepa_feature_source": "self_flow",
  "crepa_block_index": 8,
  "crepa_teacher_block_index": 16,
  "crepa_lambda": 0.5,
  "crepa_self_flow_mask_ratio": 0.25
}
```

O alias legado `crepa_self_flow=true` ainda funciona, mas `crepa_feature_source=self_flow` e a opcao preferida.

## Ajustes importantes

- `crepa_block_index`: bloco estudante
- `crepa_teacher_block_index`: bloco professor EMA. Obrigatorio
- `crepa_lambda`: forca do alinhamento. Comece em `0.5`
- `crepa_self_flow_mask_ratio`: fracao de tokens com timestep alternativo. Deve ficar em `[0.0, 0.5]`
- `crepa_scheduler`, `crepa_warmup_steps`, `crepa_decay_steps`, `crepa_lambda_end`, `crepa_cutoff_step`: mesmos controles de agendamento do CREPA
- `crepa_use_backbone_features`: e outro modo. Nao combine com Self-Flow

## Sampling / validacao

Self-Flow muda o treinamento, nao o algoritmo basico de inferencia.

- O treinamento usa ruido tokenwise misto no estudante e uma visao EMA mais limpa no professor.
- A loss de validacao continua avaliando o schedule homogeneo solicitado.
- O sampling normal nao muda. Nao ha dual-timestep masking na inferencia.

<details>
<summary>Como funciona (pratico)</summary>

- Amostra dois timesteps e os distribui pelos tokens com uma mascara aleatoria.
- Constroi uma visao do estudante com corrupcao mista e uma visao do professor com o timestep mais limpo.
- O estudante roda normalmente e o professor EMA roda em `no_grad`.
- Alinha uma camada estudante mais cedo a uma camada professor mais profunda com similaridade de cosseno, mantendo a loss generativa normal.

</details>

<details>
<summary>Tecnico (internals do SimpleTuner)</summary>

- A selecao do modo vive em `simpletuner/helpers/training/crepa.py` como `CrepaFeatureSource.SELF_FLOW`
- Os batch builders compartilhados ficam em `_prepare_image_crepa_self_flow_batch` e `_prepare_video_crepa_self_flow_batch`
- O forward do professor EMA roda a partir de `auxiliary_loss` via `_run_crepa_teacher_forward`
- A validacao reconstrui batches homogeneos quando `custom_timesteps` sao pedidos, evitando contaminar a eval loss com o batch misto de treino

</details>

## Erros comuns

- **EMA desligado**: Self-Flow exige `use_ema=true`
- **Teacher block nao definido**: configure `crepa_teacher_block_index`
- **TwinFlow ligado**: nao e compativel
- **Familia nao suportada**: so funciona em familias com `supports_crepa_self_flow()`
- **Mask ratio alto demais**: mantenha em `0.5` ou menos
- **Esperar sampler especial**: a inferencia continua normal

## Referencias

- [Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis](https://bfl.ai/research/self-flow)

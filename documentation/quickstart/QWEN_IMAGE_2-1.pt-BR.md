# Guia rápido do Qwen Image 2.1

[Qwen Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) usa um transformer de imagem de 7B com 32 blocos, um encoder de texto Qwen3-VL e um VAE de 64 canais com compressão espacial de 16×. O padrão do SimpleTuner é `model_family: "qwen_image"` e `model_flavour: "v2.1"`. Este guia cobre treino de LoRA para texto para imagem.

Para os modelos de 20B `v1.0` / `v2.0`, consulte o [guia anterior do Qwen Image](QWEN_IMAGE.pt-BR.md). O treino com referências pareadas das variantes antigas `edit-*` está em [Qwen Edit](QWEN_EDIT.pt-BR.md). Adaptadores, embeddings de texto e caches de latentes dessas versões não são intercambiáveis com 2.1.

[Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)

## Instalação

Use Python 3.12–3.14 e siga o [guia de instalação](../INSTALL.pt-BR.md) da sua plataforma. Os presets abaixo foram testados em GPUs NVIDIA. `webshart` já faz parte das dependências do SimpleTuner; os dados de regularização precisam de acesso à rede e espaço de cache local.

```bash
pip install 'simpletuner[cuda]'
```

<a id="vram-presets"></a>

## Escolha um preset de VRAM

Os exemplos combinam [assistente v3](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v3), regularização sintética, REPA e deslocamento automático do fluxo. O padrão `qwen_image.peft-lora` usa a mesma receita de 512px + 1024px do preset de 48 GB. Os presets de 24/32 GB usam apenas 512px, inclusive na validação.

| VRAM | Exemplo | Resoluções base | Atualizações | Intervalo de checkpointing de gradientes |
| --- | --- | --- | --- | --- |
| 24 GB | `qwen_image-2.1-24g.peft-lora` | 512px | 2000 | 1 |
| 32 GB | `qwen_image-2.1-32g.peft-lora` | 512px | 2000 | 2 |
| 48 GB | `qwen_image-2.1-48g.peft-lora` | 512px + 1024px | 4000 | 2 |
| 80 GB | `qwen_image-2.1-80g.peft-lora` | 512px + 1024px | 4000 | 2 |
| 144 GB | `qwen_image-2.1-144g.peft-lora` | 512px + 1024px | 4000 | 2 |

```bash
simpletuner train example=qwen_image-2.1-48g.peft-lora
```

Todos usam BF16, rank/alpha 32, lote 1, AdamW BF16 a `1e-4`, 25 passos de aquecimento e corte de norma em 1.0. REPA usa `dinov2_vitg14`, bloco 8, peso 0.5, tamanho 518, alinhamento espacial e distância temporal 0. O deslocamento automático está ativo e o estático é 0. O assistente fica congelado e é desativado na validação. A regularização usa a previsão do modelo base com ambos os adaptadores desativados como alvo.

O amostrador probabilístico padrão dá metade do peso a `RareConcepts/Domokun`, com gatilho `🟫`, e metade a `webshart/qwen-image-2.1-generated-images`, com `is_regularisation_data: true`. No treino multiescala, cada metade é dividida igualmente entre 512px e 1024px. Buckets por área preservam proporções: 0.262144 e 1.048576 megapixels. Os subconjuntos sintéticos quadrados, verticais e horizontais dividem igualmente o peso da resolução. Não há alternância estrita. Validação e checkpoints ocorrem a cada 250 passos.

Os dataloaders limitam cada subconjunto sintético de proporção a 1.024 imagens e usam caches de latentes distintos por resolução e fonte. VAE tiling fica desativado. Os limites de atualizações são pontos de partida; examine as imagens dos checkpoints antes de prolongar o treino.

<details markdown="1">
<summary>Medições de memória</summary>

Os testes de memória na L40S usaram 16 passos, quatro imagens por backend, validação e salvamento. A receita 512px com intervalo 1 passou com limite de 24 GiB: pico de 20.06 GiB alocados / 21.10 GiB reservados pelo PyTorch. A receita multiescala com intervalo 2 passou na L40S com 32.49 / 41.21 GiB, mas esgotou a memória sob limite de 32 GiB. São testes de memória com pequenos subconjuntos, não de desempenho ou convergência; os limites menores foram simulados na L40S. O preset final de 32 GB, com 512px e intervalo 2, também passou: 22.12 GiB alocados / 23.68 GiB reservados.

</details>

## Use seus próprios dados

Copie `config.json` e `dataloader.json` do exemplo escolhido para o ambiente de treino e forneça sua biblioteca de prompts. Ajuste `data_backend_config`, `user_prompt_library` e `output_dir` para esses arquivos. Consulte o [tutorial de treino](../TUTORIAL.pt-BR.md) e a [referência de dataloaders](../DATALOADER.pt-BR.md) para a organização.

Substitua os backends Domokun pelos dados do seu conceito ou de fotos. Em multiescala, use backends separados para 512px e 1024px, com IDs e caches VAE distintos. `resolution_type: "area"` usa megapixels: **0.262144** e **1.048576**. `crop: false` preserva a proporção. Mantenha `aspect_bucket_alignment: 32` na configuração de treino.

Na receita combinada, mantenha os backends sintéticos com `is_regularisation_data: true` e marque os de treino como false. Reserve metade do peso para treino e metade para regularização, dividindo cada metade entre as resoluções. O sampler é probabilístico, sem alternância rígida. Repeats controlam a disponibilidade do dataset, não sua probabilidade. Inicie um treino novo ao alterar dados, resolução, lotes ou topologia distribuída.

## Escolha o assistente de treino

Main e os exemplos usam atualmente **assistente v3**. Ele fica congelado com força 1 durante o treino e desativado na validação. Nos lotes de regularização, o alvo do modelo pai é a **previsão da base sem adaptadores**, com LoRA treinável e assistente desativados.

O [assistente v3](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v3) é o adaptador rank-64 padrão, treinado por 30k atualizações em 512/1024/1536/2048. As medições de memória dos presets usaram v2; confira a VRAM disponível ao usar v3. Para reproduzir treinos anteriores, selecione explicitamente o [assistente v2](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v2) com a configuração abaixo:

```json
{
  "assistant_lora_path": "SimpleTuner/Qwen-Image-2.1-training-assistant-v2",
  "assistant_lora_strength": 1.0,
  "assistant_lora_inference_strength": 0.0
}
```

Defina `disable_assistant_lora: true` para treinar sem assistente. Use a mesma opção ao criar um novo assistente. Na inferência downstream, carregue apenas sua LoRA; o assistente e o projetor REPA são componentes de treino.

## Valide o aprendizado e a qualidade

Os exemplos validam e salvam a cada 250 atualizações, com 40 passos de inferência, true CFG 1 e seed 42. Validam em 512px nos presets de 24/32 GB, e em 512px e 1024px nos maiores. Mantenha estas opções na configuração personalizada:

```json
{
  "validation_guidance": 1.0,
  "validation_guidance_real": 1.0,
  "validation_num_inference_steps": 40,
  "validation_seed": 42,
  "validation_step_interval": 250,
  "checkpoint_step_interval": 250
}
```

Inclua o gatilho nos prompts do conceito e sujeitos sem relação para verificar coerência e qualidade. Compare checkpoints com os mesmos prompts, seed, decoder, resolução e guidance. Resolução de treino e de saída são distintas: teste também em 1024px uma LoRA treinada em 512px. Estas receitas não exigem dados do conceito em 2048px.

O [livro de experimentos](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-LoRA-experiments) compara assistentes, regularização, REPA, auto shift e treino prolongado. Alguns treinos pioram e depois se recuperam; um checkpoint fraco não define a trajetória. Na [comparação photo-aesthetics v3](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-LoRA-photo-aesthetics-v3), 50k atualizações em 512px produzem cenas mais detalhadas do que 10k em 1024px, mesmo com saída de 1MP; os orçamentos de atualizações diferem.

## VAE e inferência

O Qwen Image 2.1 usa o [VAE com correção de textura de Ollin](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1) por padrão na validação e em outras operações de decodificação VAE. Apenas o decodificador foi ajustado; o codificador permanece inalterado, portanto os latentes de treinamento existentes da versão 2.1 continuam compatíveis. A revisão do VAE é fixada independentemente do modelo base. Para reproduzir saídas com o VAE original, defina `pretrained_vae_model_name_or_path: "Qwen/Qwen-Image-2.1"`. Substituições explícitas do VAE e variantes anteriores mantêm o comportamento existente.

Para adaptadores PEFT padrão, o pipeline incluído no SimpleTuner dispensa instalar Diffusers via Git. Este exemplo lê a LoRA exportada e exclui os tensores do projetor REPA usados só no treino. Usa decodificação completa e nenhum assistente:

O offload de modelos para CPU evita que o encoder de texto, o transformer e o VAE ocupem a GPU ao mesmo tempo. Com VRAM suficiente, substitua `pipe.enable_model_cpu_offload()` por `pipe.to("cuda")` para acelerar inferências repetidas.

```python
import torch
from safetensors.torch import load_file
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from simpletuner.helpers.models.qwen_image.autoencoder_21 import AutoencoderKLQwenImage21
from simpletuner.helpers.models.qwen_image.pipeline_21 import QwenImage21Pipeline
from simpletuner.helpers.models.qwen_image.transformer_21 import QwenImage21Transformer2DModel

vae = AutoencoderKLQwenImage21.from_pretrained(
    "madebyollin/texture-fix-vae-for-qwen-image-2.1",
    revision="e9f84623d22c47f8bc9fb799bc54201fa53cf80b",
    torch_dtype=torch.bfloat16,
)
model_id = "Qwen/Qwen-Image-2.1"
pipe = QwenImage21Pipeline(
    transformer=QwenImage21Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
    ),
    vae=vae,
    text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, subfolder="text_encoder", dtype=torch.bfloat16,
    ),
    processor=Qwen3VLProcessor.from_pretrained(model_id, subfolder="processor"),
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler"),
)
pipe.enable_model_cpu_offload()
pipe.vae.disable_tiling()
pipe.vae.enable_slicing()
weights = load_file("output/my-qwen21/pytorch_lora_weights.safetensors")
weights = {
    key: value for key, value in weights.items()
    if ".lora_A." in key or ".lora_B." in key
}
pipe.load_lora_weights(weights, adapter_name="trained")
pipe.set_adapters("trained", adapter_weights=1.0)
image = pipe(
    prompt="🟫 wearing a red scarf, standing in a snowy forest",
    width=1024, height=1024, num_inference_steps=40, true_cfg_scale=1.0,
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]
image.save("output.png")
```

## Memória e artefatos nas imagens

Comece com BF16 (`base_model_precision: "no_change"`). Reduza o lote ou escolha um preset menor antes de quantizar; quantização economiza memória, mas não garante treino mais rápido. Os exemplos ativam compilação regional. Captions longas, mais resoluções, validação e REPA podem mudar o pico de VRAM. Meça a folga antes de aumentar o lote.

Mantenha `vae_enable_tiling: false` quando possível. O decoder corrigido trata a textura de tela; emendas de cor no tiling são outro problema, relacionado ao contexto espacial. Clipping de norma 1.0 usa `grad_clip_method: "norm"` e `max_grad_norm: 1.0`, diferente do clipping por elemento 0.01 do antigo piloto AnyFlow.

<details markdown="1">
<summary>Medições de memória: VAE</summary>

Qwen Image 2.1 decodifica imagens individuais sem manter caches temporais de características que não são utilizados. Na decodificação isolada de uma imagem de 2048×2048 em H200 com BF16, isso reduziu o pico de memória alocada de 26.87 para 15.28 GiB com saída idêntica. A decodificação em blocos economiza mais memória, mas pode introduzir linhas de cor ao limitar o contexto espacial; remover os caches sem uso não corrige essas linhas.

</details>

## Receitas históricas e experimentais

As receitas abaixo criam assistentes ou testam destilação. São distintas da receita downstream recomendada acima; configurações e resultados antigos foram mantidos para referência.

<details markdown="1">
<summary>Piloto AnyFlow e histórico do otimizador</summary>

<a id="experimental-anyflow-pilot"></a>

### Piloto experimental de AnyFlow

Os três exemplos `qwen_image-2.1-anyflow-stage*.peft-lora` testam se a destilação de intervalos pode preservar o comportamento da base ao introduzir `🟫`. São experimentais; o model card do Qwen não confirma que a destilação de guidance causou a deterioração anterior.

Todas as etapas usam 1024px, BF16, AdamW, rank 32 e batch 4. A etapa 1 desativa o checkpointing de gradientes no H200; as etapas 2 e 3 usam grupos contíguos de dois blocos. É um treino de destilação diferente dos presets com REPA acima.

Estes exemplos de AnyFlow usam explicitamente `grad_clip_method: "value"` com `max_grad_norm: 0.01`: cada elemento do gradiente é limitado a ±0.01. Isso não limita a norma global. Para testar o clipping por norma em 1.0, defina tanto `grad_clip_method: "norm"` quanto `max_grad_norm: 1.0`.

1. **Etapa 1:** 10.000 atualizações forward de AnyFlow a `1e-5`. CC12M usa `webshart/cc12m-structured-captions` com `caption_key: "long_caption"`; e621 usa `webshart/e621-2024-webp-4Mpixel-webshart-indices`. Cada dataset de conhecimentos prévios fica limitado a 4.096 imagens aceitas, com peso 0.49 cada. `RareConcepts/Domokun` usa o gatilho `🟫`, peso 0.02 e `repeats: 0`.
2. **Etapa 2:** 2.000 atualizações DMD on-policy a `2e-6`, co-treinando o objetivo forward com a mesma mistura. É DMD de AnyFlow, não DPO com pares de preferência. Incluir o personagem nas duas etapas fornece exemplos reais; o professor congelado sozinho não pode ensiná-lo.
3. **Etapa 3:** 100 atualizações a `5e-7`, com peso 0.5 para Domokun e 0.25 para cada um dos dois datasets de regularização, limitados a 64 imagens cada. Todos os intervalos usam `r=t` e o alvo flow bruto, mantendo o embedder de intervalos durante um breve ajuste supervisionado. Os batches de regularização usam a previsão da base com o adapter desativado.

Revise os checkpoints de 2.000, 5.000 e 10.000 atualizações antes de estender a etapa 1 para 20.000. A etapa 2 tem um orçamento de 2.000 atualizações; pare antes se as imagens de validação comparáveis piorarem. A contagem de passos, sozinha, não determina um modelo final.

A compilação dinâmica está habilitada para captions de comprimentos variados. `schedule_shift: 2.000802574061872` corresponde ao scheduler publicado a 1024px (4.096 tokens latentes); recalcule ao alterar a resolução.

```bash
simpletuner train example=qwen_image-2.1-anyflow-stage1.peft-lora
simpletuner train example=qwen_image-2.1-anyflow-stage2.peft-lora
simpletuner train example=qwen_image-2.1-anyflow-stage3.peft-lora
```

As etapas 2 e 3 carregam o adapter final anterior com `init_lora` e iniciam novos estados do otimizador e dataloader. Os pesos controlam a seleção entre datasets ainda disponíveis, sem garantir porcentagens finais. O piloto limitado não cobre os corpora completos. O campo textual `long_caption` funciona com o seletor de captions Webshart existente e não exige suporte a captions como objetos JSON nativos.

As etapas 1 e 2 usam `diffusion_target: "base_prediction"` e `fuse_guidance_scale: 1.0`: o ramo diffusion preserva o campo condicional congelado, enquanto os outros ramos aprendem com os dados. É uma hipótese de preservação, não uma garantia de aprendizado do personagem. Compare os prompts de personagem e de conhecimentos prévios em 4, 16 e 40 passos de inferência com uma referência da base em 40 passos; a validação automática usa 4. Consulte [AnyFlow](../experimental/ANYFLOW.pt-BR.md) para os objetivos e requisitos de checkpoints.

Piloto concluído: a etapa 1 chegou a 10.000 atualizações e a etapa 2 a 2.000. Após recarregar os adaptadores, as imagens de 40 passos da raposa e dos prompts do personagem continuaram coerentes, mas estes últimos produziram pessoas em vez de Domokun. As imagens de quatro passos permaneceram borradas ou ruidosas. Outra comparação na L40S usou os dois checkpoints em 1024px, seed 42 e CFG 1/2/4/6, com prompt negativo vazio. Asserções de chamadas confirmaram duas passagens por passo acima de CFG 1. As amostras revisadas da raposa e da praia não foram recuperadas: aumentar CFG intensificou saturação e artefatos. Esses resultados não identificam a causa; a etapa 3 ainda não foi validada.

Duas continuações da etapa 1 a partir do mesmo checkpoint adicionaram 1.000 atualizações cada, comparando limites de clipping por valor de 0.01 e 1.0. Após recarregar, as imagens da raposa com quatro passos continuaram ruidosas em ambas; as imagens de praia com 40 passos ainda mostraram pessoas. O clipping ocorreu em 65/1.000 atualizações com 0.01 e em 0/1.000 com 1.0. Ambas usaram o otimizador sem correção, e os gradientes iniciais diferiram antes de o clipping ser aplicado; pequenas diferenças não podem ser atribuídas apenas ao limite.

<a id="qwen21-optimizer-correction"></a>

Os pilotos da etapa 1 e do assistente descritos aqui foram executados antes da correção do helper de soma estocástica do AdamW BF16: ele calculava `other + alpha * input` em vez de `input + alpha * other`. Com β₁ = 0,9, o primeiro momento seguia `m = 0.09 * m + g` em vez de `m = 0.9 * m + 0.1 * g`. Testes de regressão com aritmética exata cobrem CPU, MPS e CUDA. As observações das imagens continuam válidas para esses checkpoints, mas não são um teste isolado da arquitetura ou do objetivo de destilação; o treinamento precisa ser verificado novamente com o otimizador corrigido.

Repetir a continuação de 1.000 atualizações com o otimizador corrigido, o mesmo checkpoint inicial e clipping por valor de 1.0 não recuperou as imagens revisadas da raposa ou da praia com quatro passos. Com 40 passos a raposa permaneceu coerente, enquanto o prompt de praia ainda gerou uma pessoa. Isso testa a recuperação do checkpoint existente, não o treinamento do zero com o otimizador corrigido; a causa da falha geral continua sem solução.

</details>

<details markdown="1">
<summary>Criar um assistente: receitas online, multirresolução e offline</summary>

<a id="assistant-lora"></a>

### Treinar uma LoRA auxiliar a partir de legendas

`qwen_image-2.1-assistant-lora.peft-lora` é um ponto de partida experimental para L40S: BF16, lote 1, checkpointing com intervalo 2, rank 32 e AdamW BF16 a `1e-4`. O orçamento é de 1.000 atualizações, com validação e salvamento a cada 50. Substitua as doze legendas de teste por textos diversos antes de um treino efetivo; a convergência ainda não foi validada.

`grad_clip_method: "norm"`, `max_grad_norm: 1.0`.

Defina `distillation_method: assistant_lora`. O backend pré-calcula os embeddings de texto. Cada lote gera novos latentes do modelo base com o adaptador desativado, 40 passos nativos e CFG 1. O pipeline privado compartilha o transformer sem carregar VAE, processador ou codificador de texto. Depois restaura o adaptador e realiza o treino normal de remoção de ruído. Os latentes finais não são armazenados. Atualmente, somente texto para imagem com Qwen Image 2.1 é compatível.

`distillation_config.assistant_lora` aceita `num_inference_steps` (padrão 40), `resolutions` (lista não vazia de `[largura, altura]`, padrão `[[1024, 1024]]`) e `seed` (42). As dimensões devem ser múltiplas de 32. As resoluções alternam por lote e as sementes avançam por amostra, incluindo lotes incompletos. Os checkpoints preservam esses contadores. Retome sem mudar dados, lote, acumulação ou topologia distribuída. Cache de texto sob demanda não é compatível.

Para testar a execução, use 8 atualizações e 2 passos do professor; volte a 40 antes de avaliar imagens. Revise nas atualizações 100, 250, 500 e 1.000, comparando os mesmos prompts e sementes do modelo base. A utilidade da LoRA auxiliar precisa de outro teste de treino de conceitos.

O treino LoRA do Qwen Image 2.1 carrega por padrão o [assistente v3](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v3), congelado durante o treino e desativado na validação. Use `disable_assistant_lora: true` para desativá-lo, ou `assistant_lora_path` para escolher outro adaptador. As versões anteriores mantêm seu comportamento. Ao treinar um novo assistente, mantenha `disable_assistant_lora: true`, como nos exemplos correspondentes.

Use este como único método de destilação; não há suporte para combiná-lo com outros destiladores.

Datasets de legendas exigem `dataloader_prefetch: false` para que o cursor do checkpoint corresponda às legendas consumidas. A retomada rejeita mudanças nos identificadores/textos, lote, repetições, embaralhamento, semente, acumulação ou configuração distribuída. Os checkpoints do auxiliar também rejeitam mudanças na semente de geração, na lista de resoluções ou no número de passos de inferência do professor.

<a id="assistant-lora-multires"></a>

#### Experimento de assistente com quatro resoluções base e buckets de proporção

`qwen_image-2.1-assistant-lora-multires.peft-lora` percorre 12 buckets de proporção nas resoluções base 512, 1024, 1536 e 2048. Cada base inclui um bucket quadrado, um retrato 4:7 e uma paisagem 7:4; cada lote usa um bucket, com exposição igual por base e proporção durante um ciclo completo. Mantém os 40 passos do professor, BF16 com lote 1, checkpointing a cada 2 blocos, rank 32, AdamW BF16 e 1.000 atualizações do exemplo base, compartilhando seu backend de captions e seus prompts de validação. Substitua as captions de teste pelo mesmo conjunto diverso usado na referência. Comece com um adaptador e um otimizador novos no novo diretório de saída; `resume_from_checkpoint: ""` desativa a retomada. Preserve os pesos da primeira execução para comparação.

O experimento testa se a exposição a vários tamanhos melhora o assistente; não estabelece uma exigência de resolução nativa nem um ganho de qualidade. A execução de 1.000 atualizações na L40S completou os 12 buckets, com validação em 1024×1024 e 2048×2048. As imagens finais da raposa e do retrato permaneceram coerentes, com as conhecidas linhas de cor da decodificação do VAE com tiling. O professor gera alvos latentes sem decodificação pelo VAE. O benefício no treinamento posterior de conceitos continua sem verificação.

Assistant LoRA usa o mesmo mínimo de 32 entradas de cache Dynamo, restrito à execução, que AnyFlow para as variantes do professor, aluno e validação. Limites maiores definidos pelo usuário são preservados, e o limite original é restaurado ao sair. Monitore recompilações ao adicionar resoluções ou captions mais longas.

Um teste posterior de Domokun treinou dois adaptadores novos por 250 atualizações cada a 2048px, batch 1, LR `1e-5` e clipping por valor de 1.0. A intensidade do assistente no treinamento foi 0 no controle e 1 na outra execução; ambas desativaram o assistente na inferência. Os pesos iniciais e as imagens iniciais de validação eram idênticos. Com 40 passos de inferência a 1024px, ambos os resultados finais ainda geraram pessoas para os prompts do personagem e mantiveram raposas/retratos coerentes; nenhum benefício do assistente foi demonstrado. As duas execuções e a preparação do assistente usaram o otimizador sem correção descrito [acima](#qwen21-optimizer-correction).

<a id="assistant-lora-offline"></a>

#### LoRA auxiliar com imagens geradas reutilizáveis

`qwen_image-2.1-assistant-lora-offline.peft-lora` treina com [10.000 imagens geradas](https://huggingface.co/datasets/webshart/qwen-image-2.1-generated-images) via Webshart. O dataset usa prompts `long_caption` do CC12M, 40 passos nativos do professor, CFG 1 e decodificação VAE da imagem inteira. Doze backends cobrem buckets quadrados, verticais e horizontais nas resoluções base 512, 1024, 1536 e 2048, com pesos de amostragem iguais e sem repetições.

Esta receita inicia um treinamento novo com `adamw_bf16` corrigido, LR `1e-4`, `grad_clip_method: "norm"`, `max_grad_norm: 1.0`, BF16 com lote 1, rank 32 e checkpointing a cada 2 blocos. São 1.000 atualizações, salvamento a cada 50 e validação a cada 100 em 1024. Gere prévias adicionais em 2048px antes de publicar. O tiling do VAE fica desativado e a codificação usa lote 1. `vae_cache_ondemand: true` codifica e armazena as imagens conforme são amostradas, evitando codificar todas as 10.000 imagens antes de 1.000 atualizações. O treinamento comum com imagens substitui a geração online do professor; omita `distillation_method` e mantenha `disable_assistant_lora: true` ao criar o auxiliar.

O dataset pode ser reutilizado. Os PNGs exigem nova codificação VAE, portanto os alvos não são idênticos aos latentes finais do professor. Inspecione as validações antes de publicar e teste o benefício do auxiliar em outro treinamento de conceito. Ao trocar a receita de captions por esta, comece do zero sem retomar estados do otimizador ou do dataset.

O [assistente v1 substituto](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v1) concluiu esta receita de 1.000 atualizações na L40S, com revisão final das imagens em 1024 e 2048. Uma comparação equivalente de Domokun com 1.000 atualizações em 2048, LR `1e-4` e clipping de norma 1.0 manteve o assistente congelado no treinamento e desativado na validação. Tanto o controle quanto o treino assistido ainda geraram pessoas para os dois prompts do personagem. O controle introduziu traços fortes de Domokun em um prompt não relacionado de raposa; o treino assistido preservou uma raposa reconhecível. Ambos preservaram um retrato coerente. Isso é evidência limitada de menor contaminação entre conceitos, não de aprendizado bem-sucedido do personagem ou de benefício geral de qualidade; a avaliação usa uma semente de treinamento e quatro prompts.

```bash
simpletuner train example=qwen_image-2.1-assistant-lora-offline.peft-lora
```

</details>

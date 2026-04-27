# Integracao com CaptionFlow

SimpleTuner pode usar [CaptionFlow](https://github.com/bghira/CaptionFlow) para legendar datasets de imagem pela Web UI. CaptionFlow e um sistema escalavel de captioning com vLLM, orquestrador, workers GPU, storage com checkpoints e configuracao YAML. No SimpleTuner ele aparece como a subaba **Captioning** na pagina Datasets, usando a mesma fila local de GPU que os jobs de treino e cache.

Use esta integracao para gerar ou atualizar captions antes do treino sem sair do fluxo do SimpleTuner.

## Instalacao

CaptionFlow e opcional. Instale o target de captioning no mesmo virtualenv usado pelo SimpleTuner:

```bash
pip install "simpletuner[captioning]"
```

Em ambientes CUDA 13, use o target CUDA 13 mostrado pela Web UI. Ele inclui a wheel do vLLM esperada por esse runtime.

## O que o SimpleTuner gerencia

Ao iniciar um job de Captioning, o SimpleTuner:

- mapeia o dataset selecionado para um processador do CaptionFlow;
- inicia um orquestrador CaptionFlow local em `127.0.0.1`;
- inicia um ou mais workers GPU locais pela fila de jobs;
- captura logs do orquestrador e dos workers no workspace do job;
- faz checkpoint gracioso do storage do CaptionFlow antes da exportacao;
- grava captions `.txt` no diretorio do dataset para datasets locais;
- grava exportacoes JSONL no workspace do CaptionFlow para datasets Hugging Face.

As dependencias do CaptionFlow nao sao necessarias para a aba aparecer. Se elas estiverem ausentes, a aba mostra o comando de instalacao no lugar do builder.

## Modo Builder

A tela **Builder** cobre o fluxo comum de captioning em uma unica etapa:

- selecao de dataset da configuracao ativa do dataloader;
- modelo, prompt, sampling, batch size, chunk size e memoria GPU;
- quantidade de workers e comportamento da fila;
- exportacao de arquivos de texto para datasets locais.

O modelo padrao e `Qwen/Qwen2.5-VL-3B-Instruct`. Datasets locais exportam arquivos de texto ao lado das imagens usando o campo de saida selecionado. Datasets Hugging Face nao sao escritos de volta no dataset remoto; eles exportam JSONL para o workspace do CaptionFlow.

## Modo Raw Config

Use **Raw Config** quando precisar de recursos do CaptionFlow que o builder nao modela, como captioning multiestagio, modelos por estagio, sampling por estagio ou cadeias de prompts em que um estagio consome a saida de outro.

Raw config aceita YAML ou JSON. Voce pode colar a config completa com raiz `orchestrator:` ou apenas o objeto do orquestrador.

O SimpleTuner sobrescreve estes campos em runtime:

- `host`, `port` e `ssl`;
- `dataset`, baseado no dataset selecionado no SimpleTuner;
- `storage.data_dir` e `storage.checkpoint_dir`, dentro do workspace do job;
- `auth.worker_tokens` e `auth.admin_tokens`.

Outros ajustes, incluindo `chunk_size`, `chunks_per_request`, `storage.caption_buffer_size`, `vllm.sampling`, `vllm.inference_prompts` e `vllm.stages`, sao preservados salvo quando o SimpleTuner precisa aplicar um default.

## Exemplo multiestagio

Este raw config executa uma etapa de caption detalhado e passa `{caption}` para uma etapa de encurtamento. O SimpleTuner preenche dataset, paths de storage, portas e tokens no lancamento do job.

```yaml
orchestrator:
  chunk_size: 1000
  chunks_per_request: 1
  chunk_buffer_multiplier: 2
  min_chunk_buffer: 10
  vllm:
    model: "Qwen/Qwen2.5-VL-3B-Instruct"
    tensor_parallel_size: 1
    max_model_len: 16384
    dtype: "float16"
    gpu_memory_utilization: 0.92
    enforce_eager: true
    disable_mm_preprocessor_cache: true
    limit_mm_per_prompt:
      image: 1
    batch_size: 8
    sampling:
      temperature: 0.7
      top_p: 0.95
      max_tokens: 256
    stages:
      - name: "base_caption"
        prompts:
          - "describe this image in detail"
        output_field: "caption"
      - name: "caption_shortening"
        model: "Qwen/Qwen2.5-VL-7B-Instruct"
        prompts:
          - "Please condense this elaborate caption to only the important details: {caption}"
        output_field: "captions"
        requires: ["base_caption"]
        gpu_memory_utilization: 0.35
```

## Documentacao externa

- [Repositorio CaptionFlow](https://github.com/bghira/CaptionFlow)
- [README do CaptionFlow](https://github.com/bghira/CaptionFlow#readme)
- [Exemplos de orquestrador do CaptionFlow](https://github.com/bghira/CaptionFlow/tree/main/examples/orchestrator)

Use os exemplos upstream como referencia para campos avancados. Ao executar pelo SimpleTuner, lembre-se de que o SimpleTuner controla o dataset, portas locais, paths do workspace de storage e tokens de autenticacao.

# Self-Transcendence

Self-Transcendence treina blocos rasos de um Transformer de difusão com alvos internos, sem encoder visual externo. A implementação segue o método em duas etapas de [Sun et al.](https://arxiv.org/abs/2601.07773).

Aplica-se a famílias de difusão para imagem, vídeo e áudio que expõem estados de tokens latentes. Não suporta UNets, modelos autorregressivos ou LyCORIS. Suporta treino completo e PEFT LoRA padrão.

## Etapa 1: orientação estrutural VAE

A etapa 1 projeta um bloco raso para o alvo de difusão da família do modelo no espaço latente VAE: velocidade de fluxo, epsilon, predição v ou amostra limpa. O alvo é dividido em patches na grade de tokens do modelo sem descartar valores.

```json
{
  "distillation_method": "self_transcendence",
  "distillation_config": {"self_transcendence": {
    "stage": "vae", "student_block": 8, "weight": 0.5,
    "timestep_min": 0.4, "timestep_max": 0.7,
    "projector_hidden_dim": 2048
  }}
}
```

Salve o adaptador ou checkpoint para usá-lo como professor fixo na etapa 2.

## Etapa 2: representação auto-orientada

O professor fixo processa a mesma entrada ruidosa com a legenda e com o prompt vazio em cache. O CFG no espaço de características combina os estados profundos e supervisiona o bloco raso de um novo aluno.

Em PEFT LoRA, crie um adaptador aluno novo e configure `teacher_adapter_path` com o safetensors da etapa 1:

```json
{
  "distillation_method": "self_transcendence",
  "distillation_config": {"self_transcendence": {
    "stage": "self", "student_block": 8, "teacher_block": 16,
    "teacher_adapter_path": "output/stage1/pytorch_lora_weights.safetensors",
    "cfg_scale": 30.0, "weight": 0.5,
    "timestep_min": 0.4, "timestep_max": 0.7,
    "stop_step": 5000, "projector_hidden_dim": 2048
  }}
}
```

Professor e aluno devem usar o mesmo modelo base, rank PEFT e módulos-alvo. Sem `teacher_adapter_path`, a etapa 2 captura os parâmetros treináveis presentes após o resume. Isso permite treino completo e testes de uma etapa, mas não reproduz o aluno novo do artigo.

Os índices de bloco começam em zero. Comece com o aluno em cerca de 1/3 da profundidade e o professor em 2/3. Após `stop_step`, os forwards do professor param; uma rota de projeção com peso zero permanece para o DDP. Os embeddings do prompt vazio são armazenados automaticamente.

As métricas são `self_transcendence/loss`, `self_transcendence/weight` e `self_transcendence/teacher_cfg_scale` na etapa 2. O método não pode ser combinado com outro destilador ou com treino do encoder de texto.

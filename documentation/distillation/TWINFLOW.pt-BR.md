# TwinFlow (RCGM) Treinamento em poucos steps

TwinFlow e uma receita leve e independente de poucos steps baseada em **recursive consistency gradient matching (RCGM)**. Ele **nao faz parte das opcoes principais de `distillation_method`** - voce ativa diretamente via flags `twinflow_*`. O loader define `twinflow_enabled` como `false` por padrao em configs vindos do hub para que configs de transformer vanilla permaneçam intocadas.

TwinFlow no SimpleTuner:
* Apenas flow-matching, a menos que voce conecte modelos de difusao com `diff2flow_enabled` + `twinflow_allow_diff2flow`.
* Teacher EMA por padrao; captura/restaura RNG esta **sempre ligada** em torno de passagens do teacher/CFG para espelhar o run de referencia TwinFlow.
* Embeddings de sinal opcionais para semantica de tempo negativo sao conectados nos transformers, mas so usados quando `twinflow_enabled` e true; configs HF sem a flag evitam qualquer mudanca de comportamento.
* Losses padrao usam RCGM + real-velocity; opcionalmente habilite treinamento auto-adversarial completo com `twinflow_adversarial_enabled: true` para losses L_adv e L_rectify. Espera geracao em 1-4 steps com guidance `0.0`.
* Logs W&B podem emitir um scatter experimental de trajetoria TwinFlow (teoria nao verificada) para debug.

---

## Config rapido (modelo flow-matching)

Adicione os trechos TwinFlow ao seu config normal (deixe `distillation_method` sem definir/null):

```json
{
  "model_family": "sd3",
  "model_type": "lora",
  "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3.5-large",
  "output_dir": "output/sd3-twinflow",

  "distillation_method": null,
  "use_ema": true,

  "twinflow_enabled": true,
  "twinflow_target_step_count": 2,
  "twinflow_estimate_order": 2,
  "twinflow_enhanced_ratio": 0.5,
  "twinflow_delta_t": 0.01,
  "twinflow_target_clamp": 1.0,

  "learning_rate": 1e-4,
  "train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "mixed_precision": "bf16",
  "validation_guidance": 0.0,
  "validation_num_inference_steps": 2
}
```

Para modelos de difusao (epsilon/v prediction), ative explicitamente:

```json
{
  "prediction_type": "epsilon",
  "diff2flow_enabled": true,
  "twinflow_allow_diff2flow": true
}
```

> Por padrao, TwinFlow usa losses RCGM + real-velocity. Habilite `twinflow_adversarial_enabled: true` para treinamento auto-adversarial completo com losses L_adv e L_rectify (sem necessidade de discriminador externo).

---

## O que esperar (dados do paper)

Do arXiv:2512.05150 (texto do PDF):
* Benchmarks de inferencia foram medidos em **uma unica A100 (BF16)** com throughput (batch=10) e latencia (batch=1) em 1024x1024. Numeros exatos nao estavam no texto, apenas o hardware.
* Uma **comparacao de memoria GPU** (1024x1024) para Qwen-Image-20B (LoRA) e SANA-1.6B mostra TwinFlow cabendo onde DMD2 / SANA-Sprint podem dar OOM.
* Configs de treino (Tabela 6) listam **batch sizes 128/64/32/24** e **training steps 30k-60k (ou runs mais curtos de 7k-10k)**; LR constante, decay EMA geralmente 0.99.
* O PDF **nao** informa contagem total de GPUs, layout de nos ou tempo total.

Trate como expectativas direcionais, nao garantias. Para hardware/runtime exatos, seria necessario confirmacao dos autores.

---

## Opcoes chave

* `twinflow_enabled`: Ativa a loss auxiliar RCGM; mantenha `distillation_method` vazio e scheduled sampling desabilitado. Padrao `false` se ausente no config.
* `twinflow_target_step_count` (1-4 recomendado): Guia o treino e e reutilizado para validacao/inferencia. Guidance e forcado para `0.0` porque CFG esta embutido.
* `twinflow_estimate_order`: Ordem de integracao para o rollout RCGM (padrao 2). Valores maiores adicionam passagens do teacher.
* `twinflow_enhanced_ratio`: Refinamento opcional estilo CFG do alvo a partir de predicoes cond/uncond do teacher (padrao 0.5; defina 0.0 para desativar). Usa RNG capturado para manter cond/uncond alinhados.
* `twinflow_delta_t` / `twinflow_target_clamp`: Moldam o alvo recursivo; padroes espelham as configuracoes estaveis do paper.
* `use_ema` + `twinflow_require_ema` (padrao true): Pesos EMA sao usados como teacher. Defina `twinflow_allow_no_ema_teacher: true` apenas se aceitar qualidade student-as-teacher.
* `twinflow_allow_diff2flow`: Habilita ponte para modelos epsilon/v-prediction quando `diff2flow_enabled` tambem e true.
* Captura/restauracao de RNG: Sempre habilitada para espelhar a implementacao TwinFlow de referencia para passagens teacher/CFG consistentes. Nao ha opcao para desligar.
* Embeddings de sinal: Quando `twinflow_enabled` e true, modelos passam `twinflow_time_sign` para transformers que suportam `timestep_sign`; caso contrario nenhum embedding extra e usado.

### Branch adversarial (TwinFlow completo)

Habilite o treinamento auto-adversarial do paper original para melhor qualidade:

* `twinflow_adversarial_enabled` (padrao false): Habilita losses L_adv e L_rectify. Usam tempo negativo para treinar uma trajetoria "fake", permitindo correspondencia de distribuicao sem discriminadores externos.
* `twinflow_adversarial_weight` (padrao 1.0): Multiplicador de peso para a loss adversarial (L_adv).
* `twinflow_rectify_weight` (padrao 1.0): Multiplicador de peso para a loss de retificacao (L_rectify).

Quando habilitado, o treinamento gera amostras fake via geracao de um passo, entao treina ambas:
- **L_adv**: Loss de velocidade fake com tempo negativo—ensina o modelo a mapear amostras fake de volta para ruido.
- **L_rectify**: Loss de correspondencia de distribuicao—alinha predicoes de trajetoria real e fake para caminhos mais retos.

---

## Fluxo de treino e validacao

1. Treine como um run flow-matching normal (sem distiller). EMA deve existir a menos que voce opte explicitamente por nao usar; alinhamento de RNG e automatico.
2. A validacao troca automaticamente para o **scheduler TwinFlow/UCGM** e usa `twinflow_target_step_count` steps com `guidance_scale=0.0`.
3. Para pipelines exportados, conecte o scheduler manualmente:

```python
from simpletuner.helpers.training.custom_schedule import TwinFlowScheduler

pipe = ...  # seu pipeline diffusers carregado
pipe.scheduler = TwinFlowScheduler(num_train_timesteps=1000, prediction_type="flow_matching", shift=1.0)
pipe.scheduler.set_timesteps(num_inference_steps=2, device=pipe.device)
result = pipe(prompt="A cinematic portrait, 35mm", guidance_scale=0.0, num_inference_steps=2).images
```

---

## Logging

* Quando `report_to=wandb` e `twinflow_enabled=true`, o trainer pode registrar um scatter experimental de trajetoria TwinFlow (σ vs tt vs sign). O visual e apenas para debug e marcado na UI como “experimental/theory unverified”.

---

## Solucao de problemas

* **Erro sobre flow-matching**: TwinFlow exige `prediction_type=flow_matching` a menos que voce habilite `diff2flow_enabled` + `twinflow_allow_diff2flow`.
* **EMA obrigatorio**: Habilite `use_ema` ou defina `twinflow_allow_no_ema_teacher: true` / `twinflow_require_ema: false` se aceitar fallback de student-teacher.
* **Qualidade plana em 1 step**: Tente `twinflow_target_step_count: 2`–`4`, mantenha guidance em `0.0`, e reduza `twinflow_enhanced_ratio` se estiver overfitting.
* **Drift teacher/student**: Alinhamento de RNG sempre habilitado; drift deve vir de incompatibilidade de modelo, nao de diferencas estocasticas. Se seu transformer nao tiver `timestep_sign`, deixe `twinflow_enabled` desligado ou atualize o modelo para consumi-lo antes de habilitar TwinFlow.

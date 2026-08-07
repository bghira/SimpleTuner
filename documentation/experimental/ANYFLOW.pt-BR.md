# AnyFlow

O SimpleTuner implementa o NVIDIA AnyFlow como duas etapas explícitas de treinamento para modelos de flow matching. As
duas etapas treinam um modelo que recebe o tempo de fluxo atual `t` e um endpoint de intervalo `r`.

- `stage=forward` implementa o objetivo forward MeanFlow da NVIDIA.
- `stage=onpolicy` implementa Flow Map Backward Simulation e DMD on-policy enquanto co-treina o objetivo forward.

Os modos removidos `online_teacher` e `linear` eram objetivos específicos do SimpleTuner e não são mais aceitos.

Para um exemplo de continuação Wan usando os checkpoints publicados pela NVIDIA, veja
[Quickstart de continuação AnyFlow](/documentation/quickstart/ANYFLOW.pt-BR.md).

## Etapa forward

```json
{
  "model_type": "lora",
  "distillation_method": "anyflow",
  "distillation_config": {
    "anyflow": {
      "stage": "forward",
      "diffusion_ratio": 0.5,
      "consistency_ratio": 0.25,
      "central_difference_epsilon": 0.005,
      "meanflow_weight_type": "beta08",
      "meanflow_adaptive_weighting": true,
      "gate_value": 0.25,
      "deltatime_type": "r",
      "loss_weight": 1.0
    }
  }
}
```

Para cada batch global, a etapa forward:

1. Amostra dois tempos de fluxo uniformes e os ordena como `t >= r`.
2. Atribui 50% das amostras a intervalos de difusão (`r=t`), 25% a intervalos de endpoint (`r=0`) e o restante a intervalos arbitrários.
3. Aplica o flow shift do scheduler do modelo aos dois endpoints.
4. Avalia uma diferença central ao longo da trajetória latente reta.
5. Constrói o target tangente MeanFlow e aplica o weighting normalizado `beta08` da NVIDIA.
6. Balanceia cada amostra não-diffusion contra a média global de loss da ramificação diffusion.

## Etapa on-policy

Inicie esta etapa a partir de um adapter AnyFlow da etapa forward usando `init_lora` ou retomando seu checkpoint:

```json
{
  "model_type": "lora",
  "lora_type": "standard",
  "init_lora": "path-or-repo-to-forward-anyflow-adapter",
  "learning_rate": 0.000002,
  "optimizer_beta1": 0.0,
  "optimizer_beta2": 0.999,
  "optimizer_weight_decay": 0.0,
  "distillation_method": "anyflow",
  "distillation_config": {
    "anyflow": {
      "stage": "onpolicy",
      "cotrain_forward": true,
      "rollout_step_counts": [2, 4, 8, 16, 50],
      "dmd_weight": 1.0,
      "dmd_batch_size": 1,
      "real_score_guidance_scale": 0.0,
      "discriminator_lr": 0.000002,
      "discriminator_betas": [0.0, 0.999],
      "discriminator_weight_decay": 0.0,
      "discriminator_grad_clip": 1.0
    }
  }
}
```

A etapa on-policy usa três papéis de score. O treinamento LoRA padrão compartilha um transformer base congelado entre eles:

- O adapter AnyFlow carregado é o gerador.
- O modelo base com adapters desativados é o score real congelado.
- Um adapter `anyflow_discriminator` otimizado separadamente é o score fake.

Cada atualização do gerador escolhe um orçamento de rollout de `rollout_step_counts`, executa um rollout FlowMap
diferenciável, adiciona ruído ao latent gerado em um tempo uniforme deslocado e aplica o gradiente DMD normalizado da
NVIDIA. Cada atualização do discriminador executa um rollout do student sem gradientes, amostra um tempo deslocado
logit-normal e treina o score fake no target flow normal. O adapter e o otimizador do discriminador são salvos ao lado
de cada checkpoint do SimpleTuner como `anyflow_discriminator.safetensors` e `anyflow_discriminator_optim.pt`.

MiniMax-H3 já contém destilação CFG, então suas execuções on-policy normalmente devem manter
`real_score_guidance_scale=0`. Modelos que exigem uma passada CFG externa para o score real precisam cachear embeddings
de texto negativos e podem configurar a escala explicitamente.

## Configuração compartilhada

- `stage`: `forward` ou `onpolicy`. Padrão: `forward`.
- `diffusion_ratio`: fração do batch global usando `r=t`. Padrão: `0.5`.
- `consistency_ratio`: fração do batch global usando `r=0`. Padrão: `0.25`.
- `central_difference_epsilon`: offset normalizado no tempo deslocado. Padrão: `0.005`, igual ao `5/1000` da NVIDIA.
- `meanflow_weight_type`: `beta08` ou `uniform`. Padrão: `beta08`.
- `meanflow_adaptive_weighting`: balanceia amostras não-diffusion contra a ramificação diffusion. Padrão: `true`.
- `gate_value`: mistura do embedding delta-timestep FlowMap. Padrão: `0.25`.
- `deltatime_type`: `r` ou `t-r`. Padrão: `r`.
- `loss_weight`: multiplicador da loss forward MeanFlow. Padrão: `1.0`.

## Limites

- AnyFlow requer um modelo flow-matching com conditioning de intervalo FlowMap específico do modelo.
- O treinamento on-policy atualmente requer LoRA PEFT padrão. Compartilhar a base evita alocar cópias do gerador, score real e discriminador de um transformer grande em cada rank DDP.
- Treinamento conjunto MiniMax-H3 audio-video é rejeitado. Video usa schedule shift 12 e audio usa shift 3; targets MeanFlow e rollouts nativos de duplo schedule precisam ser implementados antes que treinamento AV seja válido.
- Treinamento do text encoder é desativado para todos os métodos de destilação do SimpleTuner.
- A validação usa `AnyFlowValidationScheduler`, que fornece o próximo endpoint de intervalo aos componentes FlowMap registrados.

## Logs

O treinamento forward adiciona `anyflow_forward_loss`, valores de timestep e intervalo, e frações globais de ramificação.
O treinamento on-policy também adiciona `anyflow_dmd_loss`, `anyflow_dmd_gradient_norm`, `anyflow_dmd_sigma` e
`anyflow_rollout_steps`.

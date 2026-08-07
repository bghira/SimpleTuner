# AnyFlow

SimpleTuner NVIDIA AnyFlow को flow-matching models के लिए दो स्पष्ट training stages के रूप में implement करता है। दोनों
stages ऐसे model को train करते हैं जो current flow time `t` और interval endpoint `r` प्राप्त करता है।

- `stage=forward` NVIDIA का forward MeanFlow objective implement करता है।
- `stage=onpolicy` forward objective को co-train करते हुए Flow Map Backward Simulation और on-policy DMD implement करता है।

हटाए गए `online_teacher` और `linear` target modes SimpleTuner-specific objectives थे और अब accept नहीं किए जाते।

NVIDIA के released checkpoints के साथ Wan continuation example के लिए
[AnyFlow Continuation Quickstart](/documentation/quickstart/ANYFLOW.hi.md) देखें।

## Forward Stage

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

हर global batch के लिए forward stage:

1. दो uniform flow times sample करता है और उन्हें `t >= r` में sort करता है।
2. 50% samples को diffusion intervals (`r=t`), 25% को endpoint intervals (`r=0`), और बाकी को arbitrary intervals देता है।
3. दोनों endpoints पर model scheduler का flow shift apply करता है।
4. straight latent flow path पर central difference evaluate करता है।
5. MeanFlow tangent target बनाता है और NVIDIA का normalized `beta08` timestep weighting apply करता है।
6. हर non-diffusion sample को global diffusion-branch loss mean के विरुद्ध balance करता है।

## On-Policy Stage

इस stage को forward-stage AnyFlow adapter से `init_lora` सेट करके या उसका checkpoint resume करके शुरू करें:

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

on-policy stage तीन score roles इस्तेमाल करता है। Standard LoRA training इनके बीच एक frozen base transformer share करती है:

- loaded AnyFlow adapter generator है।
- adapters disabled वाला base model frozen real score है।
- अलग से optimized `anyflow_discriminator` adapter fake score है।

हर generator update `rollout_step_counts` से rollout budget चुनता है, differentiable FlowMap rollout चलाता है, shifted uniform
time पर generated latent में noise जोड़ता है, और NVIDIA का normalized DMD gradient apply करता है। हर discriminator update
no-grad student rollout चलाता है, logit-normal shifted time sample करता है, और normal flow target पर fake score train करता
है। discriminator adapter और optimizer हर SimpleTuner checkpoint के साथ `anyflow_discriminator.safetensors` और
`anyflow_discriminator_optim.pt` के रूप में save होते हैं।

MiniMax-H3 में पहले से CFG distillation है, इसलिए इसके on-policy runs में आम तौर पर `real_score_guidance_scale=0` रखा जाना चाहिए।
जिन models को external real-score CFG pass चाहिए, उन्हें negative text embeddings cache करने होंगे और scale explicitly set किया जा सकता है।

## Shared Configuration

- `stage`: `forward` या `onpolicy`। Default: `forward`।
- `diffusion_ratio`: `r=t` use करने वाला global batch fraction। Default: `0.5`।
- `consistency_ratio`: `r=0` use करने वाला global batch fraction। Default: `0.25`।
- `central_difference_epsilon`: normalized shifted-time offset। Default: `0.005`, NVIDIA के `5/1000` से matching।
- `meanflow_weight_type`: `beta08` या `uniform`। Default: `beta08`।
- `meanflow_adaptive_weighting`: non-diffusion samples को diffusion branch के विरुद्ध balance करता है। Default: `true`।
- `gate_value`: FlowMap delta-timestep embedding blend। Default: `0.25`।
- `deltatime_type`: `r` या `t-r`। Default: `r`।
- `loss_weight`: forward MeanFlow loss multiplier। Default: `1.0`।

## Limits

- AnyFlow को model-specific FlowMap interval conditioning वाला flow-matching model चाहिए।
- on-policy training अभी standard PEFT LoRA मांगता है। Base share करने से हर DDP rank पर generator, real-score, और discriminator के बड़े transformer copies allocate नहीं होते।
- Joint MiniMax-H3 audio-video training reject की जाती है। Video schedule shift 12 use करता है और audio shift 3; AV training valid होने से पहले native dual-schedule MeanFlow targets और rollouts implement करने होंगे।
- Text encoder training SimpleTuner के सभी distillation methods में disabled है।
- Validation `AnyFlowValidationScheduler` use करता है, जो registered FlowMap model components को अगला interval endpoint देता है।

## Logs

Forward training `anyflow_forward_loss`, timestep और interval values, और global branch fractions जोड़ती है। On-policy
training `anyflow_dmd_loss`, `anyflow_dmd_gradient_norm`, `anyflow_dmd_sigma`, और `anyflow_rollout_steps` भी जोड़ती है।

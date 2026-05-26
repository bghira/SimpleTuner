# AnyFlow

AnyFlow flow-matching models के लिए experimental distillation mode है। यह model को दो flow times पर condition करता है: normal timestep `t` और उससे छोटा reference timestep `r`, ताकि model सिर्फ single rectified-flow velocity नहीं, बल्कि interval पर flow map सीखे।

SimpleTuner में:

- `--distillation_method=anyflow` `AnyFlowDistiller` चालू करता है।
- distiller startup पर trained component में `enable_flowmap_time_conditioning()` call करता है।
- हर prepared batch में `flowmap_r_timesteps` जोड़ा जाता है।
- normal target को model loss से पहले AnyFlow target से replace किया जाता है।

SimpleTuner में AnyFlow online है। इसे precomputed ODE cache की जरूरत नहीं है।

NVIDIA के released AnyFlow checkpoints के साथ Wan continuation example के लिए [AnyFlow continuation quickstart](/documentation/quickstart/ANYFLOW.hi.md) देखें।

## Quick Setup

```json
{
  "model_type": "lora",
  "distillation_method": "anyflow",
  "distillation_config": {
    "anyflow": {
      "target_mode": "online_teacher",
      "teacher_rollout_steps": 1,
      "r_timestep_sampler": "uniform",
      "min_interval_ratio": 0.02,
      "gate_value": 0.25,
      "deltatime_type": "r",
      "loss_weight": 1.0
    }
  }
}
```

SimpleTuner के सभी distillation methods की तरह AnyFlow के साथ text encoder training blocked है।

## कैसे काम करता है

हर flow-matching batch के लिए SimpleTuner:

1. normal `prepare_batch()` से `sigmas`, `timesteps`, `noisy_latents`, और base flow target बनाता है।
2. current interval से `r < t` sample करता है।
3. batch में `flowmap_r_timesteps` लिखता है ताकि model wrapper इसे `r_timestep` की तरह pass करे।
4. training target बनाता है।
5. normal model loss से prediction को target से compare करता है।

`target_mode=online_teacher` में target, `t` पर current noisy latent से `r` की ओर average velocity होता है। LoRA और LyCORIS training में distiller teacher rollout के लिए adapter temporarily disable करता है और बाद में enable करता है।

`target_mode=linear` में teacher rollout नहीं होता। target straight flow target `noise - latents` होता है। यह smoke tests और ablations के लिए उपयोगी है, लेकिन full AnyFlow teacher-map objective नहीं है।

## Options

- `target_mode`: `online_teacher` या `linear`। Default: `online_teacher`.
- `teacher_rollout_steps`: `t` और `r` के बीच online teacher Euler steps। Default: `1`.
- `r_timestep_sampler`: `uniform` या `zero`। Default: `uniform`.
- `min_interval_ratio`: `t` और `r` के बीच minimum normalized interval। Default: `0.02`.
- `gate_value`: FlowMap delta timestep embedding blend weight। Default: `0.25`.
- `deltatime_type`: `r` या `t-r`। Default: `r`.
- `loss_weight`: computed training loss multiplier। Default: `1.0`.
- `timestep_scale`: custom timestep scale वाले models के लिए override। Normal use में unset रखें।

## Limits

- flow-matching model चाहिए।
- scalar per-sample timesteps चाहिए। Tokenwise AnyFlow intervals अभी wired नहीं हैं।
- `r_timestep < timestep` चाहिए; timestep zero reject होता है।
- current online teacher mode LoRA/LyCORIS के लिए है। Full-rank online teacher के लिए अलग student/teacher wiring चाहिए।
- standard validation बिना `r_timestep` के चल सकती है, लेकिन AnyFlow-style few-step sampling के लिए sampler या pipeline को interval endpoint `r_timestep` के रूप में पास करना होगा। generation-time integration अभी follow-up है।

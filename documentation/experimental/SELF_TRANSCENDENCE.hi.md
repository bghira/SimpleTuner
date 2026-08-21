# Self-Transcendence

Self-Transcendence बाहरी vision encoder के बिना, आंतरिक targets से diffusion Transformer के शुरुआती blocks को train करता है। यह [Sun et al.](https://arxiv.org/abs/2601.07773) की दो-stage विधि पर आधारित है।

यह उन image, video और audio diffusion families पर लागू है जो latent-token hidden states देती हैं। UNet, autoregressive models और LyCORIS समर्थित नहीं हैं। Full-model और standard PEFT LoRA training समर्थित हैं।

## Stage 1: VAE structure guidance

Stage 1 शुरुआती block को VAE latent space में model family के diffusion target पर project करता है: flow velocity, epsilon, v-prediction या clean sample। Target को values हटाए बिना model की token grid पर patches में बदला जाता है।

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

इस stage का adapter या checkpoint save करें। Stage 2 इसे fixed teacher की तरह उपयोग करता है।

## Stage 2: self-guided representation

Fixed teacher एक ही noisy input को caption और cached empty prompt के साथ चलाता है। Feature-space CFG गहरे states को जोड़ता है और नए student के शुरुआती block को supervise करता है।

PEFT LoRA के लिए नया student adapter बनाएँ और `teacher_adapter_path` में stage-1 safetensors दें:

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

Teacher और student का base model, PEFT rank और target modules समान होने चाहिए। `teacher_adapter_path` के बिना stage 2 resume के बाद मौजूद trainable parameters का snapshot लेता है। यह full-model और one-stage प्रयोगों के लिए है, लेकिन paper के fresh-student setup के समान नहीं है।

Block indices zero-based हैं। Student को लगभग 1/3 depth और teacher को 2/3 depth से शुरू करें। `stop_step` के बाद teacher forwards बंद होते हैं; DDP के लिए zero-weight projector path चालू रहता है। Empty-prompt embeddings अपने-आप cache होते हैं।

Metrics: `self_transcendence/loss`, `self_transcendence/weight`, और stage 2 में `self_transcendence/teacher_cfg_scale`। इसे दूसरे distiller या text-encoder training के साथ उपयोग नहीं किया जा सकता।

# Self-Transcendence

Self-Transcendence trains shallow diffusion-transformer blocks with internal targets instead of an external vision encoder. It follows the two-stage method from [Sun et al.](https://arxiv.org/abs/2601.07773).

It applies to diffusion-transformer image, video, and audio families that expose latent-token block states. It does not apply to UNets or autoregressive models. Full-model and standard PEFT LoRA training are supported. LyCORIS is not supported because the projection head must be stored with the adapter.

## Stage 1: VAE structure guidance

Stage 1 projects a shallow block to the model family's diffusion target in VAE latent space. This is the flow velocity, epsilon, v-prediction, or clean sample used by the base loss. The target is patchified onto the model's latent-token grid without discarding values. The loss is active only inside the configured timestep range.

```json
{
  "distillation_method": "self_transcendence",
  "distillation_config": {
    "self_transcendence": {
      "stage": "vae",
      "student_block": 8,
      "weight": 0.5,
      "timestep_min": 0.4,
      "timestep_max": 0.7,
      "projector_hidden_dim": 2048
    }
  }
}
```

Train this stage until the selected intermediate representation is useful. Save the adapter or checkpoint. The paper used a partially trained stage-1 model as the fixed teacher.

## Stage 2: self-guided representation

Stage 2 runs the frozen teacher twice on the same noisy input: once with the caption and once with the cached empty prompt. It forms a feature-space CFG target from a deeper block and aligns the new student's shallow block to it.

For a LoRA run, start a fresh adapter and point `teacher_adapter_path` at the stage-1 safetensors file:

```json
{
  "distillation_method": "self_transcendence",
  "distillation_config": {
    "self_transcendence": {
      "stage": "self",
      "student_block": 8,
      "teacher_block": 16,
      "teacher_adapter_path": "output/stage1/pytorch_lora_weights.safetensors",
      "cfg_scale": 30.0,
      "weight": 0.5,
      "timestep_min": 0.4,
      "timestep_max": 0.7,
      "stop_step": 5000,
      "projector_hidden_dim": 2048
    }
  }
}
```

`teacher_adapter_path` requires the same PEFT rank, targets, and base model as the student. The teacher adapter is loaded only to create a fixed parameter snapshot; the fresh student parameters are restored before the first forward.

Without `teacher_adapter_path`, stage 2 snapshots the trainable parameters present after resume. This supports full-model training and one-stage experiments, but it is not the paper's fresh-student setup. Full-model snapshots require an additional copy of every trainable parameter.

## Layer selection

Block indices are zero-based. Start near one-third depth for `student_block` and two-thirds depth for `teacher_block`. The paper used approximately one-half and two-thirds depth. Both selected blocks must exist in the model's hidden-state buffer.

## Early stop and cost

`stop_step` disables teacher forwards after the selected optimization step. The projection head still executes behind a zero-weight path so DDP does not report unused parameters. A stage-2 step before cutoff adds two no-gradient transformer forwards. Empty-prompt embeddings are cached automatically.

Logged values:

- `self_transcendence/loss`
- `self_transcendence/weight`
- `self_transcendence/teacher_cfg_scale` in stage 2

Self-Transcendence occupies the distillation method slot and cannot be combined with another distiller or text-encoder training.

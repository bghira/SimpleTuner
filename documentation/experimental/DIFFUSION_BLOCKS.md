# DiffusionBlocks

DiffusionBlocks converts a compatible diffusion transformer into independently trained layer groups. Each group owns one noise range. A training forward executes only the group selected for that batch.

This is an experimental architecture conversion based on [DiffusionBlocks](https://arxiv.org/abs/2506.14202). It is not ordinary layer freezing. A trained adapter must use the same block routing during inference.

## Configuration

```json
{
  "diffusion_blocks_config": {
    "layers_per_block": 4,
    "overlap": 0.05
  },
  "find_unused_parameters": true
}
```

`find_unused_parameters` is enabled automatically for DDP. Setting it to `false` is an error.

| Key | Default | Meaning |
| --- | --- | --- |
| `layers_per_block` | required | Maximum number of consecutive transformer layers in one noise-range block. |
| `overlap` | `0.05` | Fractional expansion of adjacent training noise ranges. Valid range: `0.0` to `0.5`. |
| `blocks_to_train` | `"all"` | Block indices owned by this run. Other groups are frozen after adapter creation. |
| `block_paths` | auto | Explicit `ModuleList` paths when automatic discovery is insufficient. |
| `timestep_boundaries` | auto | Ascending normalized boundaries from `0.0` to `1.0`. Must contain `num_blocks + 1` values. |

Automatic boundaries split the configured timestep distribution into equal-probability ranges. Block `0` receives the highest-noise range and the earliest transformer layers. The last block receives the lowest-noise range and final layers.

## Model support

The shared implementation supports diffusion and flow-matching model families with homogeneous transformer block lists. It discovers common single-stage and multi-stage paths, including `transformer_blocks`, joint/single streams, double/single streams, `blocks`, and `layers`.

Unsupported configurations fail during setup:

- UNet families
- ControlNet
- Musubi block swap
- TwinFlow
- scheduled-sampling multi-timestep passes
- CREPA fixed-layer capture
- LayerSync fixed-layer capture

TREAD routes keep global model layer indices and are clipped to the active group's global range. Fixed-layer auxiliary losses, including CREPA and LayerSync, are not currently compatible because their requested layer may not execute for a given noise range.

Routing changes the denoiser architecture. Initial loss and output quality are not expected to match an ordinary full-depth run, and an existing ordinary LoRA does not become a trained DiffusionBlocks adapter by enabling this option.

Use `block_paths` only after confirming that every selected list is one sequential denoiser stage. Do not select text adapters, VAE blocks, or skip-connected UNet stages.
Skip-dependent encoder-decoder Transformer stacks, such as i1 `in_blocks`/`out_blocks`, are not discovered because an output group cannot run without activations from its paired input group.

## Memory behavior

Only the active layer group builds transformer activations. This is the main memory reduction in an all-block run.

An all-block run eventually allocates optimizer state for every trainable group. To reproduce independent block jobs, set `blocks_to_train` separately per job. Those jobs freeze unowned groups and allocate no optimizer state for them. Their checkpoints must be combined by parameter ownership before inference.

Model weights remain resident unless another compatible offload method is enabled. Group offload is compatible. Musubi block swap is not.

## Inference

SimpleTuner validation uses the training controller automatically. A standard Diffusers pipeline does not infer the architecture conversion from LoRA weights. Apply the controller before loading or using the adapter:

```python
from simpletuner.helpers.training.diffusion_blocks import DiffusionBlocksConfig, DiffusionBlocksController

config = DiffusionBlocksConfig.from_dict({"layers_per_block": 4, "overlap": 0.05})
controller = DiffusionBlocksController(pipe.transformer, config)
```

Keep the controller alive for the lifetime of the pipeline. It selects the layer group from each inference timestep. Use the exact training configuration stored in `simpletuner_config.json`.

## Anima example

The tuned example is `simpletuner/examples/anima.peft-lora+diffusion-blocks/config.json`. Anima v1.0 has 28 denoiser layers; `layers_per_block=4` creates seven blocks.

For a smoke test, use a short run first:

```bash
simpletuner train env=examples/anima.peft-lora+diffusion-blocks \
  max_train_steps=10 \
  validation_steps=10
```

Compare peak allocated VRAM and step time against the normal Anima example at the same resolution and batch size.

## Checkpoint rules

- Resume with the same block paths, layer count, boundaries, and `blocks_to_train`.
- Do not change model topology, world size, batch sampling, or timestep configuration on resume.
- A normal full-model or LoRA checkpoint is not a DiffusionBlocks checkpoint unless it was trained with this routing.
- Running all transformer layers at inference changes the architecture and invalidates the training objective.

# TREAD Training Documentation

> ⚠️ **Experimental Feature**: TREAD support in SimpleTuner is newly implemented. While functional, optimal configurations are still being explored and some behaviors may change in future releases.

## Overview

TREAD (Token Routing for Efficient Architecture-agnostic Diffusion Training) is a training acceleration method that speeds up diffusion model training by intelligently routing tokens through transformer layers. By selectively processing only the most important tokens during certain layers, TREAD can significantly reduce computational costs while maintaining model quality.

Based on the research by [Krause et al. (2025)](https://arxiv.org/abs/2501.04765), TREAD achieves training speedups by:
- Dynamically selecting which tokens to process in each transformer layer
- Maintaining gradient flow through all tokens via skip connections
- Using importance-based routing decisions

The speedup is directly proportional to the `selection_ratio` - the closer to 1.0, the more tokens are dropped and the faster training becomes.

## How TREAD Works

### Core Concept

During training, TREAD:
1. **Routes tokens** - For specified transformer layers, it selects a subset of tokens to process based on their importance
2. **Processes subset** - Only the selected tokens go through the expensive attention and MLP operations
3. **Restores full sequence** - After processing, the full token sequence is restored with gradients flowing to all tokens

### Token Selection

Tokens are selected based on their L1-norm (importance score), with optional randomization for exploration:
- Higher importance tokens are more likely to be kept
- A mix of importance-based and random selection prevents overfitting to specific patterns
- Force-keep masks can ensure certain tokens (like masked regions) are never dropped

## Configuration

### Basic Setup

To enable TREAD training in SimpleTuner, add the following to your configuration:

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 2,
        "end_layer_idx": 5
      }
    ]
  }
}
```

### Route Configuration

Each route defines a window where token routing is active:
- `selection_ratio`: Fraction of tokens to drop (0.5 = keep 50% of tokens)
- `start_layer_idx`: First layer where routing begins (0-indexed)
- `end_layer_idx`: Last layer where routing is active

Negative indices are supported: `-1` refers to the last layer.

### Advanced Example

Multiple routing windows with different selection ratios:

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.3,
        "start_layer_idx": 1,
        "end_layer_idx": 3
      },
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 4,
        "end_layer_idx": 8
      },
      {
        "selection_ratio": 0.7,
        "start_layer_idx": -4,
        "end_layer_idx": -1
      }
    ]
  }
}
```

## Compatibility

### Supported Models
- **FLUX Dev/Kontext, Wan, AuraFlow, PixArt, and SD3** - Currently the only supported model families
- Future support planned for other diffusion transformers

### Works Well With
- **Masked Loss Training** - TREAD automatically preserves masked regions when combined with mask/segmentation conditioning
- **Multi-GPU Training** - Compatible with distributed training setups
- **Quantized Training** - Can be used with int8/int4/NF4 quantization

### Limitations
- Only active during training (not inference)
- Requires gradient computation (won't work in eval mode)
- Currently FLUX and Wan-specific implementation, not available for Lumina2, other architectures yet

## Performance Considerations

### Speed Benefits
- Training speedup is proportional to `selection_ratio` (closer to 1.0 = more tokens dropped = faster training)
- **Biggest speedups occur with longer video inputs and higher resolutions** due to attention's O(n²) complexity
- Typically 20-40% speedup, but results vary based on configuration
- With masked loss training, speedup is reduced as masked tokens cannot be dropped

### Quality Trade-offs
- **Higher token dropping leads to higher initial loss** when starting LoRA/LoKr training
- The loss tends to correct fairly rapidly and images normalize quickly unless high selection ratio is in use
  - This may be the network adjusting to fewer tokens in intermediary layers
- Conservative ratios (0.1-0.25) typically maintain quality
- Aggressive ratios (>0.35) definitely will impact convergence

### LoRA-specific Considerations
- Performance may be data-dependent - optimal routing configs need more exploration
- Initial loss spike is more noticeable with LoRA/LoKr than full fine-tuning

### Recommended Settings

For balanced speed/quality:
```json
{
  "routes": [
    {"selection_ratio": 0.5, "start_layer_idx": 2, "end_layer_idx": -2}
  ]
}
```

For maximum speed (expect massive loss spike):
```json
{
  "routes": [
    {"selection_ratio": 0.7, "start_layer_idx": 1, "end_layer_idx": -1}
  ]
}
```

For high-resolution training (1024px+):
```json
{
  "routes": [
    {"selection_ratio": 0.6, "start_layer_idx": 2, "end_layer_idx": -3}
  ]
}
```

## Technical Details

### Router Implementation

The TREAD router (`TREADRouter` class) handles:
- Token importance calculation via L1-norm
- Permutation generation for efficient routing
- Gradient-preserving token restoration

### Integration with Attention

TREAD modifies the rotary position embeddings (RoPE) to match the routed sequence:
- Text tokens maintain original positions
- Image tokens use shuffled/sliced positions
- Ensures positional consistency during routing
- **Note**: The RoPE implementation for FLUX may not be 100% correct but appears functional in practice

### Masked Loss Compatibility

When using masked loss training:
- Tokens within the mask are automatically force-kept
- Prevents important training signal from being dropped
- Activated via `conditioning_type` in ["mask", "segmentation"]
- **Note**: This reduces speedup as more tokens must be processed

## Known Issues and Limitations

### Implementation Status
- **Experimental feature** - TREAD support is newly implemented and may have undiscovered issues
- **RoPE handling** - The rotary position embedding implementation for token routing may not be perfectly correct
- **Limited testing** - Optimal routing configurations haven't been extensively explored

### Training Behavior
- **Initial loss spike** - When starting LoRA/LoKr training with TREAD, expect higher initial loss that corrects rapidly
- **LoRA performance** - Some configurations may show slight slowdowns with LoRA training
- **Configuration sensitivity** - Performance highly depends on routing configuration choices

### Known Bugs (Fixed)
- Masked loss training was broken in earlier versions but has been fixed with proper model flavor checking (`kontext` guard)

## Troubleshooting

### Common Issues

**"TREAD training requires you to configure the routes"**
- Ensure `tread_config` includes a `routes` array
- Each route needs `selection_ratio`, `start_layer_idx`, and `end_layer_idx`

**Slower training than expected**
- Verify routes cover meaningful layer ranges
- Consider more aggressive selection ratios
- Check that gradient checkpointing isn't conflicting
- For LoRA training, some slowdown is expected - try different routing configs

**High initial loss with LoRA/LoKr**
- This is expected behavior - the network needs to adapt to fewer tokens
- Loss typically corrects within a few hundred steps
- If loss doesn't improve, reduce `selection_ratio` (keep more tokens)

**Quality degradation**
- Reduce selection ratios (keep more tokens)
- Avoid routing in early layers (0-2) or final layers
- Ensure sufficient training data for the increased efficiency

## Practical Examples

### High-Resolution Training (1024px+)
For maximum benefit when training at high resolutions:
```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.6, "start_layer_idx": 2, "end_layer_idx": -3}
    ]
  }
}
```

### LoRA Fine-tuning
Conservative config to minimize initial loss spike:
```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.4, "start_layer_idx": 3, "end_layer_idx": -4}
    ]
  }
}
```

### Masked Loss Training
When training with masks, tokens in masked regions are preserved:
```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.7, "start_layer_idx": 2, "end_layer_idx": -2}
    ]
  }
}
```
Note: Actual speedup will be less than 0.7 suggests due to forced token preservation.

## Future Work

As TREAD support in SimpleTuner is newly implemented, there are several areas for future improvement:

- **Configuration optimization** - More testing needed to find optimal routing configurations for different use cases
- **LoRA performance** - Investigation into why some LoRA configurations show slowdowns
- **RoPE implementation** - Refinement of the rotary position embedding handling for better correctness
- **Extended model support** - Implementation for other diffusion transformer architectures beyond Flux
- **Automated configuration** - Tools to automatically determine optimal routing based on model and dataset characteristics

Community contributions and testing results are welcome to help improve TREAD support.

## References

- [TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training](https://arxiv.org/abs/2501.04765)
- [SimpleTuner Flux Documentation](/documentation/quickstart/FLUX.md#tread-training)

## Citation

```bibtex
@misc{krause2025treadtokenroutingefficient,
      title={TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training},
      author={Felix Krause and Timy Phan and Vincent Tao Hu and Björn Ommer},
      year={2025},
      eprint={2501.04765},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.04765},
}
```

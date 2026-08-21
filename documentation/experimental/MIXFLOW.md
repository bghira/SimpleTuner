# MixFlow Training

MixFlow is a post-training method for flow-matching models. It trains the model at timestep $t$ using a noisier ground-truth interpolation. This reduces the gap between exact training interpolations and imperfect latents encountered during sampling.

## Configuration

```json
{
  "mixflow_enabled": true,
  "mixflow_gamma": 0.8
}
```

`mixflow_gamma` controls the slowed-interpolation range. `0.8` is the paper default. `0.0` preserves the standard interpolation while retaining MixFlow timestep sampling.

MixFlow samples the data-ward model timestep from $Beta(2,1)$. SimpleTuner stores flow sigmas in the opposite, noise-ward direction, so the implementation samples $sigma = 1 - sqrt(U)$ and then applies the model's configured flow schedule shift. The model receives the original timestep. Its latent input uses:

$$
sigma_{input} = sigma + U' gamma (1 - sigma)
$$

The velocity target is unchanged for a linear flow path. Inference is unchanged.

## Support

All SimpleTuner model families whose prediction type is `flow_matching` use the shared MixFlow path. Model-specific data-ward timestep conventions, nonlinear sigma transforms, and joint audio/video inputs are handled by their model wrappers.

MixFlow cannot be combined with another training-time trajectory replacement: custom/uniform/Beta/fast flow schedules, Self-Flow, TwinFlow, scheduled sampling, or distillation. Schedule shift remains supported.

MixFlow is intended for post-training an existing flow model. Start with the learning rate and optimizer used for a short conventional continuation run, then compare fixed-seed validation samples against the starting checkpoint.

## References

- [MixFlow paper](https://arxiv.org/abs/2512.19311)
- [Reference implementation](https://github.com/fudan-generative-vision/MixFlow)

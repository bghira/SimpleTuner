# CLIP score tracking

CLIP scores are loosely related to measurement of a model's ability to follow prompts; it is not at all related to image quality/fidelity.

The `clip/mean` score of your model indicates how closely the features extracted from the image align with the features extracted from the prompt. It is currently a popular metric for determining general prompt adherence, though is typically evaluated across a very large (~5,000) number of test prompts (eg. Parti Prompts).

CLIP score generation during model pretraining can help demonstrate that the model is approaching its objective, but once a `clip/mean` value around `.30` to `.39` is reached, the comparison seems to become less meaningful. Models that show an average CLIP score around `.33` can outperform a model with an average CLIP score of `.36` in human analysis. However, a model with a very low average CLIP score around `0.18` to `0.22` will probably be pretty poorly-performing.

Within a single test run, some prompts will result in a very low CLIP score of around `0.14` (`clip/min` value in the tracker charts) even though their images align fairly well with the user prompt and have high image quality; conversely, CLIP scores as high as `0.39` (`clip/max` value in the tracker charts) may appear from images with questionable quality, as this test is not meant to capture this information. This is why such a large number of prompts are typically used to measure model performance - _and even then_..

On its own, CLIP scores do not take long to calculate; however, the number of prompts required for meaningful evaluation can make it take an incredibly long time.

Since it doesn't take much to run, it doesn't hurt to include CLIP evaluation in small training runs. Perhaps you will discover a pattern of the outputs where it makes sense to abandon a training run or adjust other hyperparameters such as the learning rate.

To include a standard prompt library for evaluation, `--validation_prompt_library` can be provided and then we will generate a somewhat relative benchmark between training runs.

In `config.json`:

```json
{
  ...
  "evaluation_type": "clip",
  "pretrained_evaluation_model_name_or_path": "openai/clip-vit-large-patch14-336",
  "report_to": "tensorboard", # or wandb
  ...
}
```

## Compatibility

SageAttention is currently not compatible with CLIP score tracking. One or the other must be disabled.
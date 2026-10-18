# Training examples

Examples are organised as `model_family/model_flavour/vram/recipe`. Choose a model family, then its flavour and a memory tier such as `24g`, `48g`, or `80g`. The `default` tier has no specific VRAM target; a `default` flavour uses the model’s configured default. Multi-GPU tiers such as `8x80g` require eight GPUs with that memory per GPU.

Named recipe folders keep LoRA, LoKr, ControlNet, AnyFlow, and conditioning variants separate. Shared dataset plans, prompt libraries, and validation scripts live in `_shared/`.

## List examples and run a recipe

After installing SimpleTuner, list the available recipes and pass the full relative path to `example=`:

```bash
simpletuner examples list
simpletuner train example=flux3_image/klein/24g/peft-lora
```

## Copy and customise

The copy command preserves the catalog folders beneath the destination and copies referenced JSON assets into the recipe folder. Edit the copied configuration and dataset plan for your data, then train from that environment:

```bash
simpletuner examples copy flux3_image/klein/24g/peft-lora --dest config
# config/flux3_image/klein/24g/peft-lora/config.json
cd config/flux3_image/klein/24g/peft-lora
simpletuner train
```

In the WebUI environment creator and setup wizard, select the model family first. The example selector shows only that family’s recipes. Changing the family clears the previous example selection.

These recipes are starting points. Check the model guide and adjust batch size, resolution, and training length for your dataset and hardware.

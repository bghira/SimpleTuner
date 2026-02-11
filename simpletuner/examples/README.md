## Example configurations

These configurations are provided as an easy way to **immediately** run a training session with SimpleTuner across a large number of architectures.

The options are set up so that a 24G card (NVIDIA 4090) can run training out of the box. In order to do this, compromises were made for resolution, training batch size, or LoRA rank.

It's recommended to use these only as a basic starting point.

### Running an example

All examples can be easily run without modifying the configurations.

We'll assume you don't have any python dependencies installed yet, and that an NVIDIA device is in use.

To run `kontext.peft-lora` example:

**Option 1 (Recommended - pip install):**
```bash
pip install 'simpletuner[cuda]'
# or for CUDA 13 / Blackwell users: pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
simpletuner train example=kontext.peft-lora
```

**Option 2 (Git clone method):**
```bash
git clone https://github.com/bghira/simpletuner
cd simpletuner
python3.13 -m venv .venv
. .venv/bin/activate
pip install -e .
simpletuner train env=examples/kontext.peft-lora
```

**Option 3 (Legacy method - still works):**
```bash
git clone https://github.com/bghira/simpletuner
cd simpletuner
python3.13 -m venv .venv
. .venv/bin/activate
pip install -e .
ENV=examples/kontext.peft-lora ./train.sh
```

This will automatically download an example reference dataset, pre-cache embeds, and run 100 steps of training on a standard PEFT LoRA.

### Modifying and extending an example

You'll want to copy the folder from `simpletuner/examples` to `config` before modifying anything, otherwise your changes will conflict with newer example config updates.

```bash
cp -R simpletuner/examples/kontext.peft-lora config/kontext.peft-lora
```

Inside the file `config/kontext.peft-lora/config.json` you will need to update the locations of `output_dir` and `dataloader_config`

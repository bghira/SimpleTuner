# DiffusionBlocks

DiffusionBlocks किसी compatible diffusion Transformer को independently train होने वाले layer groups में बदलता है। हर group एक noise range संभालता है; एक forward केवल current batch का group चलाता है।

यह [DiffusionBlocks](https://arxiv.org/abs/2506.14202) पर आधारित experimental architecture conversion है, सामान्य layer freezing नहीं। Inference में वही routing आवश्यक है जो training में था।

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

DDP के लिए `find_unused_parameters` अपने आप enable होता है। इसे `false` करना error है।

| Key | Default | अर्थ |
| --- | --- | --- |
| `layers_per_block` | required | हर noise block में consecutive Transformer layers की अधिकतम संख्या। |
| `overlap` | `0.05` | पड़ोसी training noise ranges का fractional विस्तार; `0.0` से `0.5`। |
| `blocks_to_train` | `"all"` | इस job के block indices। Adapter बनने के बाद बाकी groups freeze होते हैं। |
| `block_paths` | auto | Auto discovery पर्याप्त न हो तो explicit `ModuleList` paths। |
| `timestep_boundaries` | auto | `0.0` से `1.0` तक ascending boundaries; `num_blocks + 1` values। |

Automatic boundaries configured timestep distribution को equal-probability ranges में बांटती हैं। Block `0` highest noise और शुरुआती layers संभालता है।

## Model support

Shared implementation homogeneous Transformer block lists वाले diffusion और flow-matching families को support करता है: single stage, joint/single stream, double/single stream, `blocks`, और `layers`।

UNet, ControlNet, Musubi block swap, TwinFlow, multi-timestep scheduled sampling, fixed-layer capture वाला CREPA, और LayerSync setup पर reject होते हैं। TREAD routes global model layer indices रखती हैं और active group की global range पर clip होती हैं।

Routing denoiser architecture को बदलता है। शुरुआती loss और output quality से normal full-depth run के समान होने की अपेक्षा नहीं करनी चाहिए। इस option को enable करने से मौजूदा normal LoRA trained DiffusionBlocks adapter नहीं बनता।

`block_paths` केवल sequential denoiser stages के लिए दें। Text adapters, VAE blocks, या skip-connected UNet stages न चुनें।
Skip-dependent encoder-decoder Transformer stacks, जैसे i1 के `in_blocks`/`out_blocks`, discover नहीं होते क्योंकि output group अपने paired input group की activations के बिना नहीं चल सकता।

## Memory

केवल active group Transformer activations बनाता है। सभी blocks वाले एक run में आखिरकार सभी trainable groups के optimizer states allocate होते हैं।

Independent jobs के लिए हर job में अलग `blocks_to_train` दें। Unowned groups freeze होते हैं और optimizer state नहीं लेते। Inference से पहले parameter ownership के अनुसार checkpoints combine करें।


## Inference

SimpleTuner validation controller को अपने आप इस्तेमाल करता है। Standard Diffusers pipeline LoRA weights से conversion नहीं पहचानता।

```python
from simpletuner.helpers.training.diffusion_blocks import DiffusionBlocksConfig, DiffusionBlocksController

config = DiffusionBlocksConfig.from_dict({"layers_per_block": 4, "overlap": 0.05})
controller = DiffusionBlocksController(pipe.transformer, config)
```

Pipeline के पूरे lifetime में `controller` रखें और `simpletuner_config.json` की exact configuration इस्तेमाल करें।

## Anima example

`simpletuner/examples/anima.peft-lora+diffusion-blocks/config.json` देखें। Anima v1.0 की 28 layers, `layers_per_block=4` पर 7 blocks बनाती हैं।

```bash
simpletuner train env=examples/anima.peft-lora+diffusion-blocks max_train_steps=10 validation_steps=10
```

Resume पर paths, layer count, boundaries, `blocks_to_train`, topology, world size, batch sampling, या timestep configuration न बदलें। Inference में सभी layers चलाना trained objective को invalid करता है।

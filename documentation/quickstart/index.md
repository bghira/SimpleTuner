# Model Guides

Step-by-step guides for training each supported model architecture.

## Image Models

### Flow Matching

| Model | Parameters | License | Allows commercial use | Guide |
| ------- | ------------ | --- | :---: | ------- |
| **Flux.1** | 12B | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) / [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Conditions apply<sup>3</sup> | [Flux.1 Guide](FLUX.md) |
| **Flux.2** | 32B | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) / [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Conditions apply<sup>4</sup> | [Flux.2 Guide](FLUX2.md) |
| **Flux Kontext** | 12B | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) | No<sup>5</sup> | [Kontext Guide](FLUX_KONTEXT.md) |
| **Chroma** | 8.9B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [Chroma Guide](CHROMA.md) |
| **Stable Diffusion 3** | 2-8B | [Stability AI Community](https://stability.ai/license) | Conditions apply<sup>2</sup> | [SD3 Guide](SD3.md) |
| **Auraflow** | 6.8B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) / [Pony License](https://huggingface.co/purplesmartai/pony-v7-base/blob/main/LICENSE) | Conditions apply<sup>8</sup> | [Auraflow Guide](AURAFLOW.md) |
| **Sana** | 0.6-4.8B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [Sana Guide](SANA.md) |
| **Lumina2** | 2B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [Lumina2 Guide](LUMINA2.md) |
| **HiDream** | 17B MoE | [MIT](https://opensource.org/license/mit) | Yes | [HiDream Guide](HIDREAM.md) |
| **Z-Image** | - | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [Z-Image Guide](ZIMAGE.md) |
| **Krea2** | - | [Krea 2 Community](https://www.krea.ai/krea-2-licensing) | Conditions apply<sup>6</sup> | [Krea2 Guide](KREA2.md) |
| **Mage-Flow** | 4B | [MIT](https://opensource.org/license/mit) | Yes | [Mage-Flow Guide](MAGEFLOW.md) |
| **Boogu-Image** | - | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [Boogu-Image Guide](BOOGU_IMAGE.md) |
| **zlab i1** | 3B | [Unspecified](https://huggingface.co/bghira/zlab-i1-diffusers) | Conditions apply<sup>12</sup> | [zlab i1 Guide](ZLAB_i1.md) |
| **Ideogram 4** | 9B | [Ideogram 4 Non-Commercial](https://huggingface.co/ideogram-ai/ideogram-4-nf4/blob/main/LICENSE.md) | No<sup>5</sup> | [Ideogram 4 Guide](IDEOGRAM4.md) |
| **ERNIE-Image** | - | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [ERNIE Guide](ERNIE.md) |

### DiT / Transformer

| Model | Parameters | License | Allows commercial use | Guide |
| ------- | ------------ | --- | :---: | ------- |
| **PixArt Sigma** | 0.6-0.9B | [OpenRAIL++](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | Conditions apply<sup>1</sup> | [Sigma Guide](SIGMA.md) |
| **Cosmos2** | 2-14B | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | Yes<sup>9</sup> | [Cosmos2 Guide](COSMOS2IMAGE.md) |
| **Cosmos3** | 4-65B | [OpenMDW 1.1](https://github.com/OpenMDW/openmdw/blob/main/1.1/LICENSE.OpenMDW-1.1) | Yes | [Cosmos3 Guide](COSMOS3.md) |
| **OmniGen** | 3.8B | [MIT](https://opensource.org/license/mit) | Yes | [OmniGen Guide](OMNIGEN.md) |
| **Qwen Image** | 20B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [Qwen Guide](QWEN_IMAGE.md) |
| **LongCat Image** | 6B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [LongCat Guide](LONGCAT_IMAGE.md) |
| **Kandinsky 5** | - | [MIT](https://opensource.org/license/mit) | Yes | [Kandinsky Guide](KANDINSKY5_IMAGE.md) |

### U-Net

| Model | Parameters | License | Allows commercial use | Guide |
| ------- | ------------ | --- | :---: | ------- |
| **Stable Diffusion XL** | 3.5B | [OpenRAIL++](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | Conditions apply<sup>1</sup> | [SDXL Guide](SDXL.md) |
| **Kolors** | 5B | [Kwai Kolors License](https://huggingface.co/terminusresearch/kwai-kolors-1.0/blob/main/MODEL_LICENSE) | Conditions apply<sup>7</sup> | [Kolors Guide](KOLORS.md) |
| **Stable Cascade** | - | [Stable Cascade NC Community](https://huggingface.co/stabilityai/stable-cascade/blob/main/LICENSE) | No<sup>5</sup> | [Cascade Guide](STABLE_CASCADE_C.md) |

### Image Editing

| Model | License | Allows commercial use | Guide |
| ------- | --- | :---: | ------- |
| **Qwen Edit** | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [Qwen Edit Guide](QWEN_EDIT.md) |
| **LongCat Edit** | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [LongCat Edit Guide](LONGCAT_EDIT.md) |

## Video Models

| Model | Parameters | License | Allows commercial use | Guide |
| ------- | ------------ | --- | :---: | ------- |
| **Wan Video** | 1.3-14B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [Wan Guide](WAN.md) |
| **LTX Video** | 5B | [LTX Video OpenRAIL-M](https://huggingface.co/Lightricks/LTX-Video-0.9.5/blob/main/ltx-video-2b-v0.9.5.license.txt) | Conditions apply<sup>10</sup> | [LTX Guide](LTXVIDEO.md) |
| **LTX Video 2** | 19B | [LTX-2 Community](https://ltx.io/model/license) | Conditions apply<sup>10</sup> | [LTX Video 2 Guide](LTXVIDEO2.md) |
| **Cosmos3** | 4-65B | [OpenMDW 1.1](https://github.com/OpenMDW/openmdw/blob/main/1.1/LICENSE.OpenMDW-1.1) | Yes | [Cosmos3 Guide](COSMOS3.md) |
| **Hunyuan Video** | 8.3B | [Tencent Hunyuan Community](https://huggingface.co/tencent/HunyuanVideo-1.5/blob/main/LICENSE) | Conditions apply<sup>11</sup> | [Hunyuan Guide](HUNYUANVIDEO.md) |
| **Sana Video** | - | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes | [Sana Video Guide](SANAVIDEO.md) |
| **Kandinsky 5 Video** | - | [MIT](https://opensource.org/license/mit) | Yes | [Kandinsky Video Guide](KANDINSKY5_VIDEO.md) |
| **LongCat Video** | - | [MIT](https://opensource.org/license/mit) | Yes | [LongCat Video Guide](LONGCAT_VIDEO.md) |
| **LongCat Video Edit** | - | [MIT](https://opensource.org/license/mit) | Yes | [LongCat Video Edit Guide](LONGCAT_VIDEO_EDIT.md) |

**License notes:** The commercial-use status covers model weights, derivative checkpoints, fine-tunes, and hosted model use. Generated-output rights can differ; read the linked license text before commercial deployment.

<sup>1</sup> OpenRAIL-style licenses generally permit commercial use with usage restrictions that remain attached to the model and derivatives.

<sup>2</sup> Stability AI Community License is available for qualifying users below the revenue threshold; larger commercial use needs Stability enterprise terms.

<sup>3</sup> Flux.1 varies by flavour: Schnell and LibreFlux are Apache-2.0, while Dev, Krea, and Kontext use BFL non-commercial terms; review FluxBooru upstream metadata before commercial use.

<sup>4</sup> Flux.2 varies by flavour: Klein 4B is Apache-2.0, while Dev and Klein 9B use BFL non-commercial terms.

<sup>5</sup> Public non-commercial model terms do not permit commercial use of weights, derivative checkpoints, or hosted model services without a separate license.

<sup>6</sup> Krea 2 Community License permits commercial use only under its revenue and safety/filtering requirements; otherwise an enterprise license is required.

<sup>7</sup> Kolors commercial model or derivative use requires applying for and receiving explicit permission from the licensor.

<sup>8</sup> AuraFlow supports Apache-2.0 upstream flavours and a Pony flavour with a separate custom license; check the selected flavour.

<sup>9</sup> NVIDIA Open Model License permits commercial use but includes agreement, acceptable-use, and export-control terms.

<sup>10</sup> LTX Video 0.9.5 uses OpenRAIL-M; LTX Video 2 uses LTX community terms with a revenue threshold for commercial use.

<sup>11</sup> Tencent Hunyuan Community License includes territorial exclusions and a commercial threshold for very large services.

<sup>12</sup> This mirror publishes `license: other` without a standard license text; review upstream terms before commercial use.

## Audio Models

| Model | Size / Version | Guide |
|-------|----------------|-------|
| **ACE-Step** | 3.5B / 1.5 | [ACE-Step Guide](ACE_STEP.md) |
| **HeartMuLa** | 3B | [HeartMuLa Guide](HEARTMULA.md) |

## Choosing a Model

**For beginners:**

- Start with **Flux.1** for high-quality image generation
- Use **LoRA** training to reduce memory requirements

**For production:**

- **SD3** or **SDXL** for broad compatibility
- **Flux.2** for maximum quality (requires more VRAM)

**For video:**

- **Wan Video** for best quality/resource balance
- **Hunyuan Video** for I2V with super-resolution

**For specific use cases:**

- **Flux Kontext** for image editing/conditioning
- **ACE-Step** for text-to-music LoRA training (v1 and v1.5)
- **HeartMuLa** for autoregressive text-to-audio

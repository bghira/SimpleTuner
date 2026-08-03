# क्विकस्टार्ट गाइड

**नोट**: अधिक उन्नत कॉन्फ़िगरेशनों के लिए, [ट्यूटोरियल](TUTORIAL.md) और [options reference](OPTIONS.md) देखें।

## फ़ीचर संगतता

पूरा और सबसे सटीक फीचर मैट्रिक्स देखने के लिए, [मुख्य README](https://github.com/bghira/SimpleTuner#model-architecture-support) देखें।

## मॉडल क्विकस्टार्ट गाइड

| मॉडल | पैरामीटर | PEFT LoRA | Lycoris | फुल-रैंक | क्वांटाइज़ेशन | मिक्स्ड प्रिसिजन | ग्रैड चेकपॉइंट | फ्लो शिफ्ट | TwinFlow | Self-Flow | LayerSync | Ref Inputs | ControlNet | Sliders† | लाइसेंस | व्यावसायिक उपयोग की अनुमति | गाइड |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- | :---: | --- |
| PixArt Sigma | 0.6B–0.9B | ✗ | ✓ | ✓ | int8 वैकल्पिक | bf16 | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | [OpenRAIL++](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | शर्तें लागू<sup>1</sup> | [SIGMA.md](quickstart/SIGMA.md) |
| NVLabs Sana | 1.6B–4.8B | ✗ | ✓ | ✓ | int8 वैकल्पिक | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | हाँ | [SANA.md](quickstart/SANA.md) |
| Kwai Kolors | 2.7B | ✓ | ✓ | ✓ | अनुशंसित नहीं | bf16 | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [Kwai Kolors License](https://huggingface.co/terminusresearch/kwai-kolors-1.0/blob/main/MODEL_LICENSE) | शर्तें लागू<sup>7</sup> | [KOLORS.md](quickstart/KOLORS.md) |
| Stable Diffusion 3 | 2B–8B | ✓ | ✓ | ✓ | int8/fp8/nf4 वैकल्पिक | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [Stability AI Community](https://stability.ai/license) | शर्तें लागू<sup>2</sup> | [SD3.md](quickstart/SD3.md) |
| Flux.1 | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 वैकल्पिक | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) / [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | शर्तें लागू<sup>3</sup> | [FLUX.md](quickstart/FLUX.md) |
| Flux.2 | 32B | ✓ | ✓ | ✓* | int8/fp8/nf4 वैकल्पिक | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ opt | ✗ | ✓ | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) / [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | शर्तें लागू<sup>4</sup> | [FLUX2.md](quickstart/FLUX2.md) |
| Flux Kontext | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 वैकल्पिक | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✓ | ✓ | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) | नहीं<sup>5</sup> | [FLUX_KONTEXT.md](quickstart/FLUX_KONTEXT.md) |
| Z-Image Turbo | 6B | ✓ | ✗ | ✓* | int8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | हाँ | [ZIMAGE.md](quickstart/ZIMAGE.md) |
| Krea2 | - | ✓ | ✗ | ✓* | int8 वैकल्पिक | bf16 | ✓+ | ✓ | ✗ | ✗ | ✗ | ✓ opt | ✗ | ✓ | [Krea 2 Community](https://www.krea.ai/krea-2-licensing) | शर्तें लागू<sup>6</sup> | [KREA2.md](quickstart/KREA2.hi.md) |
| Mage-Flow | 4B | ✓ | ✓ | ✓* | int8/fp8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ edit | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | हाँ | [MAGEFLOW.md](quickstart/MAGEFLOW.hi.md) |
| Boogu-Image 0.1 | - | ✓ | ✓ | ✓* | fp8 वैकल्पिक | bf16 | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ edit | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | हाँ | [BOOGU_IMAGE.md](quickstart/BOOGU_IMAGE.hi.md) |
| zlab i1 | 3B | ✓ | ✓ | ✓ | int8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Unspecified](https://huggingface.co/bghira/zlab-i1-diffusers) | शर्तें लागू<sup>12</sup> | [ZLAB_i1.md](quickstart/ZLAB_i1.hi.md) |
| Ideogram 4 | 9B | ✓ | ✓ | ✓* | fp8 डिफ़ॉल्ट, nf4 वैकल्पिक | bf16 | ✓+ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [Ideogram 4 Non-Commercial](https://huggingface.co/ideogram-ai/ideogram-4-nf4/blob/main/LICENSE.md) | नहीं<sup>5</sup> | [IDEOGRAM4.md](quickstart/IDEOGRAM4.hi.md) |
| ERNIE-Image | - | ✓ | ✓ | ✓* | int8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | हाँ | [ERNIE.md](quickstart/ERNIE.hi.md) |
| ACE-Step | 3.5B | ✓ | ✓ | ✓* | int8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B) / [MIT](https://huggingface.co/ACE-Step/Ace-Step1.5) | हाँ | [ACE_STEP.md](quickstart/ACE_STEP.md) |
| Chroma 1 | 8.9B | ✓ | ✓ | ✓* | int8/fp8/nf4 वैकल्पिक | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | हाँ | [CHROMA.md](quickstart/CHROMA.md) |
| Auraflow | 6B | ✓ | ✓ | ✓* | int8/fp8/nf4 वैकल्पिक | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) / [Pony License](https://huggingface.co/purplesmartai/pony-v7-base/blob/main/LICENSE) | शर्तें लागू<sup>8</sup> | [AURAFLOW.md](quickstart/AURAFLOW.md) |
| HiDream I1 | 17B (8.5B MoE) | ✓ | ✓ | ✓* | int8/fp8/nf4 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [MIT](https://opensource.org/license/mit) | हाँ | [HIDREAM.md](quickstart/HIDREAM.md) |
| OmniGen | 3.8B | ✓ | ✓ | ✓ | int8/fp8 वैकल्पिक | bf16 | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | हाँ | [OMNIGEN.md](quickstart/OMNIGEN.md) |
| Stable Diffusion XL | 2.6B | ✓ | ✓ | ✓ | अनुशंसित नहीं | bf16 | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ | [OpenRAIL++](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | शर्तें लागू<sup>1</sup> | [SDXL.md](quickstart/SDXL.md) |
| Lumina2 | 2B | ✓ | ✓ | ✓ | int8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | हाँ | [LUMINA2.md](quickstart/LUMINA2.md) |
| Cosmos2 | 2B | ✓ | ✓ | ✓ | अनुशंसित नहीं | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | हाँ<sup>9</sup> | [COSMOS2IMAGE.md](quickstart/COSMOS2IMAGE.md) |
| Cosmos3 | 16B-65B | ✓ | ✓ | ✓* | no_change first | bf16 | ✓ | ✓ | ✗ | ✗ | ✗ | audio opt | ✗ | ✓ | [OpenMDW 1.1](https://github.com/OpenMDW/openmdw/blob/main/1.1/LICENSE.OpenMDW-1.1) | हाँ | [COSMOS3.md](quickstart/COSMOS3.hi.md) |
| LTX Video | ~2.5B | ✓ | ✓ | ✓ | int8/fp8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ I2V | ✗ | ✓ | [LTX Video OpenRAIL-M](https://huggingface.co/Lightricks/LTX-Video-0.9.5/blob/main/ltx-video-2b-v0.9.5.license.txt) | शर्तें लागू<sup>10</sup> | [LTXVIDEO.md](quickstart/LTXVIDEO.md) |
| LTX Video 2 | 19B | ✓ | ✓ | ✓* | int8/fp8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ opt | ✗ | ✓ | [LTX-2 Community](https://ltx.io/model/license) | शर्तें लागू<sup>10</sup> | [LTXVIDEO2.md](quickstart/LTXVIDEO2.md) |
| Hunyuan Video 1.5 | 8.3B | ✓ | ✓ | ✓* | int8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ I2V | ✗ | ✓ | [Tencent Hunyuan Community](https://huggingface.co/tencent/HunyuanVideo-1.5/blob/main/LICENSE) | शर्तें लागू<sup>11</sup> | [HUNYUANVIDEO.md](quickstart/HUNYUANVIDEO.md) |
| SanaVideo | 2B | ✓ | ✓ | ✓* | int8/fp8 वैकल्पिक | bf16 | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | हाँ | [SANAVIDEO.md](quickstart/SANAVIDEO.hi.md) |
| Wan 2.x | 1.3B–14B | ✓ | ✓ | ✓* | int8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | हाँ | [WAN.md](quickstart/WAN.md) |
| Wan 2.2 S2V | 14B | ✓ | ✓ | ✓* | int8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | हाँ | [WAN_S2V.md](quickstart/WAN_S2V.md) |
| Qwen Image | 20B | ✓ | ✓ | ✓* | **आवश्यक** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | हाँ | [QWEN_IMAGE.md](quickstart/QWEN_IMAGE.md) |
| Qwen Image Edit | 20B | ✓ | ✓ | ✓* | **आवश्यक** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | हाँ | [QWEN_EDIT.md](quickstart/QWEN_EDIT.md) |
| Stable Cascade (C) | 1B, 3.6B prior | ✓ | ✓ | ✓* | समर्थित नहीं | fp32 (आवश्यक) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [Stable Cascade NC Community](https://huggingface.co/stabilityai/stable-cascade/blob/main/LICENSE) | नहीं<sup>5</sup> | [STABLE_CASCADE_C.md](quickstart/STABLE_CASCADE_C.md) |
| Kandinsky 5.0 Image | 6B (lite) | ✓ | ✓ | ✓* | int8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ I2I | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | हाँ | [KANDINSKY5_IMAGE.md](quickstart/KANDINSKY5_IMAGE.md) |
| Kandinsky 5.0 Video | 2B (lite), 19B (pro) | ✓ | ✓ | ✓* | int8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ I2V | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | हाँ | [KANDINSKY5_VIDEO.md](quickstart/KANDINSKY5_VIDEO.md) |
| LongCat-Video | 13.6B | ✓ | ✓ | ✓* | int8/fp8 वैकल्पिक | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ opt | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | हाँ | [LONGCAT_VIDEO.md](quickstart/LONGCAT_VIDEO.md) |
| LongCat-Video Edit | 13.6B | ✓ | ✓ | ✓* | int8/fp8 वैकल्पिक | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | हाँ | [LONGCAT_VIDEO_EDIT.md](quickstart/LONGCAT_VIDEO_EDIT.md) |
| LongCat-Image | 6B | ✓ | ✓ | ✓* | int8/fp8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | हाँ | [LONGCAT_IMAGE.md](quickstart/LONGCAT_IMAGE.md) |
| LongCat-Image Edit | 6B | ✓ | ✓ | ✓* | int8/fp8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | हाँ | [LONGCAT_EDIT.md](quickstart/LONGCAT_EDIT.md) |

*✓ = समर्थित, ✓* = फुल‑रैंक के लिए DeepSpeed/FSDP2 आवश्यक, ✗ = समर्थित नहीं, `✓+` VRAM दबाव के कारण checkpointing की सिफ़ारिश को दर्शाता है। Ref Inputs सिर्फ मौजूदा reference/edit/I2V conditioning paths को दिखाता है; `opt` वैकल्पिक है और `req` edit/I2V flavour के लिए आवश्यक है। TwinFlow ✓ का अर्थ है `twinflow_enabled=true` होने पर native support (diffusion मॉडल्स को `diff2flow_enabled+twinflow_allow_diff2flow` चाहिए)। Self-Flow ✓ का अर्थ है `crepa_enabled=true`, `crepa_feature_source=self_flow`, `use_ema=true`, और `crepa_teacher_block_index` सेट होने पर native support। LayerSync ✓ का अर्थ है कि backbone self‑alignment के लिए transformer hidden states उपलब्ध कराता है; ✗ UNet‑style backbones को दर्शाता है जिनमें वह buffer नहीं होता। †Sliders LoRA और LyCORIS (full‑rank LyCORIS “full” सहित) पर लागू होते हैं।*

**लाइसेंस नोट्स:** व्यावसायिक उपयोग की स्थिति model weights, derivative checkpoints, fine-tunes, और hosted model use को कवर करती है। Generated outputs के अधिकार अलग हो सकते हैं; commercial deployment से पहले linked license text पढ़ें।

<sup>1</sup> OpenRAIL-style licenses आम तौर पर commercial use की अनुमति देती हैं, लेकिन usage restrictions model और derivatives के साथ बनी रहती हैं।

<sup>2</sup> Stability AI Community License revenue threshold से नीचे qualify करने वाले users के लिए उपलब्ध है; बड़े commercial use के लिए Stability enterprise terms चाहिए।

<sup>3</sup> Flux.1 flavour के अनुसार बदलता है: Schnell और LibreFlux Apache-2.0 हैं, जबकि Dev, Krea, और Kontext BFL non-commercial terms का उपयोग करते हैं; commercial use से पहले FluxBooru upstream metadata देखें।

<sup>4</sup> Flux.2 flavour के अनुसार बदलता है: Klein 4B Apache-2.0 है, जबकि Dev और Klein 9B BFL non-commercial terms का उपयोग करते हैं।

<sup>5</sup> Public non-commercial model terms अलग license के बिना weights, derivative checkpoints, या hosted model services के commercial use की अनुमति नहीं देते।

<sup>6</sup> Krea 2 Community License केवल revenue और safety/filtering requirements के तहत commercial use की अनुमति देती है; अन्यथा enterprise license चाहिए।

<sup>7</sup> Kolors model या derivatives का commercial use करने के लिए licensor से explicit permission माँगनी और प्राप्त करनी होती है।

<sup>8</sup> AuraFlow Apache-2.0 upstream flavours और अलग custom license वाले Pony flavour को support करता है; selected flavour देखें।

<sup>9</sup> NVIDIA Open Model License commercial use की अनुमति देती है, लेकिन agreement, acceptable-use, और export-control terms शामिल हैं।

<sup>10</sup> LTX Video 0.9.5 OpenRAIL-M का उपयोग करता है; LTX Video 2 commercial use के लिए revenue threshold वाले LTX community terms का उपयोग करता है।

<sup>11</sup> Tencent Hunyuan Community License में territorial exclusions और बहुत बड़े services के लिए commercial threshold शामिल है।

<sup>12</sup> यह mirror standard license text के बिना `license: other` publish करता है; commercial use से पहले upstream terms देखें।

> ℹ️ Wan quickstart में 2.1 + 2.2 stage presets और time‑embedding toggle शामिल है। Flux Kontext में Flux.1 के ऊपर बने editing वर्कफ़्लो शामिल हैं।

> ⚠️ ये क्विकस्टार्ट living documents हैं। नए मॉडल आने या प्रशिक्षण रेसिपीज़ सुधरने के साथ समय‑समय पर अपडेट की उम्मीद करें।

### तेज़ रास्ते: Z-Image Turbo और Flux Schnell

- **Z-Image Turbo**: TREAD के साथ पूरी तरह समर्थित LoRA; NVIDIA और macOS पर quant के बिना भी तेज़ चलता है (int8 भी काम करता है)। अक्सर bottleneck केवल trainer setup होता है।
- **Flux Schnell**: क्विकस्टार्ट कॉन्फ़िग fast noise schedule और assistant LoRA stack को स्वतः संभालता है; Schnell LoRAs ट्रेन करने के लिए अतिरिक्त फ़्लैग्स की आवश्यकता नहीं है।

### उन्नत प्रायोगिक विशेषताएँ

- **Diff2Flow**: Flow Matching loss objective के साथ standard epsilon/v‑prediction मॉडल्स (SD1.5, SDXL, DeepFloyd, आदि) को ट्रेन करने की अनुमति देता है। यह पुराने आर्किटेक्चर और आधुनिक flow‑based प्रशिक्षण के बीच का अंतर भरता है।
- **Scheduled Sampling**: प्रशिक्षण के दौरान मॉडल को अपने ही intermediate noisy latents उत्पन्न करने देता है ("rollout"), जिससे exposure bias कम होता है। यह मॉडल को अपनी ही generation errors से उबरना सिखाता है।

## सामान्य समस्याएं

### Dataset में expected से कम samples हैं

यदि आपके dataset में expected से कम usable samples हैं, तो processing के दौरान files filter हो गई हो सकती हैं। सामान्य कारण:

- **Files बहुत छोटी हैं**: `minimum_image_size` से नीचे की images filter कर दी जाती हैं
- **Aspect ratio range से बाहर**: `minimum_aspect_ratio`/`maximum_aspect_ratio` bounds से बाहर की images exclude कर दी जाती हैं
- **Duration limits**: Duration limits से अधिक audio/video files skip कर दी जाती हैं

**Filtering statistics देखना:**
- WebUI में, अपने dataset directory पर browse करें और filtering statistics देखने के लिए इसे select करें
- Dataset processing के दौरान logs में इस तरह के statistics check करें: `Sample processing statistics: {'total_processed': 100, 'skipped': {'too_small': 15, ...}}`

विस्तृत troubleshooting के लिए, dataloader documentation में [Filtered datasets का Troubleshooting](DATALOADER.hi.md) देखें।

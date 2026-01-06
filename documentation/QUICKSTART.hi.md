# क्विकस्टार्ट गाइड

**नोट**: अधिक उन्नत कॉन्फ़िगरेशनों के लिए, [ट्यूटोरियल](TUTORIAL.md) और [options reference](OPTIONS.md) देखें।

## फ़ीचर संगतता

पूरा और सबसे सटीक फीचर मैट्रिक्स देखने के लिए, [मुख्य README](https://github.com/bghira/SimpleTuner#model-architecture-support) देखें।

## मॉडल क्विकस्टार्ट गाइड

| मॉडल | पैरामीटर | PEFT LoRA | Lycoris | फुल-रैंक | क्वांटाइज़ेशन | मिक्स्ड प्रिसिजन | ग्रैड चेकपॉइंट | फ्लो शिफ्ट | TwinFlow | LayerSync | ControlNet | Sliders† | गाइड |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| PixArt Sigma | 0.6B–0.9B | ✗ | ✓ | ✓ | int8 वैकल्पिक | bf16 | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | [SIGMA.md](quickstart/SIGMA.md) |
| NVLabs Sana | 1.6B–4.8B | ✗ | ✓ | ✓ | int8 वैकल्पिक | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [SANA.md](quickstart/SANA.md) |
| Kwai Kolors | 2.7B | ✓ | ✓ | ✓ | अनुशंसित नहीं | bf16 | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | [KOLORS.md](quickstart/KOLORS.md) |
| Stable Diffusion 3 | 2B–8B | ✓ | ✓ | ✓ | int8/fp8/nf4 वैकल्पिक | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✓ | [SD3.md](quickstart/SD3.md) |
| Flux.1 | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 वैकल्पिक | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ | [FLUX.md](quickstart/FLUX.md) |
| Flux.2 | 32B | ✓ | ✓ | ✓* | int8/fp8/nf4 वैकल्पिक | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [FLUX2.md](quickstart/FLUX2.md) |
| Flux Kontext | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 वैकल्पिक | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ | [FLUX_KONTEXT.md](quickstart/FLUX_KONTEXT.md) |
| Z-Image Turbo | 6B | ✓ | ✗ | ✓* | int8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [ZIMAGE.md](quickstart/ZIMAGE.md) |
| ACE-Step | 3.5B | ✓ | ✓ | ✓* | int8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [ACE_STEP.md](quickstart/ACE_STEP.md) |
| Chroma 1 | 8.9B | ✓ | ✓ | ✓* | int8/fp8/nf4 वैकल्पिक | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [CHROMA.md](quickstart/CHROMA.md) |
| Auraflow | 6B | ✓ | ✓ | ✓* | int8/fp8/nf4 वैकल्पिक | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✓ | [AURAFLOW.md](quickstart/AURAFLOW.md) |
| HiDream I1 | 17B (8.5B MoE) | ✓ | ✓ | ✓* | int8/fp8/nf4 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | [HIDREAM.md](quickstart/HIDREAM.md) |
| OmniGen | 3.8B | ✓ | ✓ | ✓ | int8/fp8 वैकल्पिक | bf16 | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | [OMNIGEN.md](quickstart/OMNIGEN.md) |
| Stable Diffusion XL | 2.6B | ✓ | ✓ | ✓ | अनुशंसित नहीं | bf16 | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | [SDXL.md](quickstart/SDXL.md) |
| Lumina2 | 2B | ✓ | ✓ | ✓ | int8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [LUMINA2.md](quickstart/LUMINA2.md) |
| Cosmos2 | 2B | ✓ | ✓ | ✓ | अनुशंसित नहीं | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [COSMOS2IMAGE.md](quickstart/COSMOS2IMAGE.md) |
| LTX Video | ~2.5B | ✓ | ✓ | ✓ | int8/fp8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [LTXVIDEO.md](quickstart/LTXVIDEO.md) |
| Hunyuan Video 1.5 | 8.3B | ✓ | ✓ | ✓* | int8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [HUNYUANVIDEO.md](quickstart/HUNYUANVIDEO.md) |
| Wan 2.x | 1.3B–14B | ✓ | ✓ | ✓* | int8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [WAN.md](quickstart/WAN.md) |
| Qwen Image | 20B | ✓ | ✓ | ✓* | **आवश्यक** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [QWEN_IMAGE.md](quickstart/QWEN_IMAGE.md) |
| Qwen Image Edit | 20B | ✓ | ✓ | ✓* | **आवश्यक** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [QWEN_EDIT.md](quickstart/QWEN_EDIT.md) |
| Stable Cascade (C) | 1B, 3.6B prior | ✓ | ✓ | ✓* | समर्थित नहीं | fp32 (आवश्यक) | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | [STABLE_CASCADE_C.md](quickstart/STABLE_CASCADE_C.md) |
| Kandinsky 5.0 Image | 6B (lite) | ✓ | ✓ | ✓* | int8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [KANDINSKY5_IMAGE.md](quickstart/KANDINSKY5_IMAGE.md) |
| Kandinsky 5.0 Video | 2B (lite), 19B (pro) | ✓ | ✓ | ✓* | int8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [KANDINSKY5_VIDEO.md](quickstart/KANDINSKY5_VIDEO.md) |
| LongCat-Video | 13.6B | ✓ | ✓ | ✓* | int8/fp8 वैकल्पिक | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_VIDEO.md](quickstart/LONGCAT_VIDEO.md) |
| LongCat-Video Edit | 13.6B | ✓ | ✓ | ✓* | int8/fp8 वैकल्पिक | bf16 | ✓+ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_VIDEO_EDIT.md](quickstart/LONGCAT_VIDEO_EDIT.md) |
| LongCat-Image | 6B | ✓ | ✓ | ✓* | int8/fp8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_IMAGE.md](quickstart/LONGCAT_IMAGE.md) |
| LongCat-Image Edit | 6B | ✓ | ✓ | ✓* | int8/fp8 वैकल्पिक | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_EDIT.md](quickstart/LONGCAT_EDIT.md) |

*✓ = समर्थित, ✓* = फुल‑रैंक के लिए DeepSpeed/FSDP2 आवश्यक, ✗ = समर्थित नहीं, `✓+` VRAM दबाव के कारण checkpointing की सिफ़ारिश को दर्शाता है। TwinFlow ✓ का अर्थ है `twinflow_enabled=true` होने पर native support (diffusion मॉडल्स को `diff2flow_enabled+twinflow_allow_diff2flow` चाहिए)। LayerSync ✓ का अर्थ है कि backbone self‑alignment के लिए transformer hidden states उपलब्ध कराता है; ✗ UNet‑style backbones को दर्शाता है जिनमें वह buffer नहीं होता। †Sliders LoRA और LyCORIS (full‑rank LyCORIS “full” सहित) पर लागू होते हैं।*

> ℹ️ Wan quickstart में 2.1 + 2.2 stage presets और time‑embedding toggle शामिल है। Flux Kontext में Flux.1 के ऊपर बने editing वर्कफ़्लो शामिल हैं।

> ⚠️ ये क्विकस्टार्ट living documents हैं। नए मॉडल आने या प्रशिक्षण रेसिपीज़ सुधरने के साथ समय‑समय पर अपडेट की उम्मीद करें।

### तेज़ रास्ते: Z-Image Turbo और Flux Schnell

- **Z-Image Turbo**: TREAD के साथ पूरी तरह समर्थित LoRA; NVIDIA और macOS पर quant के बिना भी तेज़ चलता है (int8 भी काम करता है)। अक्सर bottleneck केवल trainer setup होता है।
- **Flux Schnell**: क्विकस्टार्ट कॉन्फ़िग fast noise schedule और assistant LoRA stack को स्वतः संभालता है; Schnell LoRAs ट्रेन करने के लिए अतिरिक्त फ़्लैग्स की आवश्यकता नहीं है।

### उन्नत प्रायोगिक विशेषताएँ

- **Diff2Flow**: Flow Matching loss objective के साथ standard epsilon/v‑prediction मॉडल्स (SD1.5, SDXL, DeepFloyd, आदि) को ट्रेन करने की अनुमति देता है। यह पुराने आर्किटेक्चर और आधुनिक flow‑based प्रशिक्षण के बीच का अंतर भरता है।
- **Scheduled Sampling**: प्रशिक्षण के दौरान मॉडल को अपने ही intermediate noisy latents उत्पन्न करने देता है ("rollout"), जिससे exposure bias कम होता है। यह मॉडल को अपनी ही generation errors से उबरना सिखाता है।

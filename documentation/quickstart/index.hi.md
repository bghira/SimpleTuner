# मॉडल गाइड्स

प्रत्येक समर्थित मॉडल आर्किटेक्चर के प्रशिक्षण के लिए चरण‑दर‑चरण गाइड।

## छवि मॉडल

### फ्लो मैचिंग

| मॉडल | पैरामीटर | गाइड |
|-------|------------|-------|
| **Flux.1** | 12B | [Flux.1 गाइड](FLUX.md) |
| **Flux.2** | 32B | [Flux.2 गाइड](FLUX2.md) |
| **Flux Kontext** | 12B | [Kontext गाइड](FLUX_KONTEXT.md) |
| **Chroma** | 8.9B | [Chroma गाइड](CHROMA.md) |
| **Stable Diffusion 3** | 2-8B | [SD3 गाइड](SD3.md) |
| **Auraflow** | 6.8B | [Auraflow गाइड](AURAFLOW.md) |
| **Sana** | 0.6-4.8B | [Sana गाइड](SANA.md) |
| **Lumina2** | 2B | [Lumina2 गाइड](LUMINA2.md) |
| **HiDream** | 17B MoE | [HiDream गाइड](HIDREAM.md) |
| **Z-Image** | - | [Z-Image गाइड](ZIMAGE.md) |

### DiT / ट्रांसफ़ॉर्मर

| मॉडल | पैरामीटर | गाइड |
|-------|------------|-------|
| **PixArt Sigma** | 0.6-0.9B | [Sigma गाइड](SIGMA.md) |
| **Cosmos2** | 2-14B | [Cosmos2 गाइड](COSMOS2IMAGE.md) |
| **OmniGen** | 3.8B | [OmniGen गाइड](OMNIGEN.md) |
| **Qwen Image** | 20B | [Qwen गाइड](QWEN_IMAGE.md) |
| **LongCat Image** | 6B | [LongCat गाइड](LONGCAT_IMAGE.md) |
| **Kandinsky 5** | - | [Kandinsky गाइड](KANDINSKY5_IMAGE.md) |

### U‑Net

| मॉडल | पैरामीटर | गाइड |
|-------|------------|-------|
| **Stable Diffusion XL** | 3.5B | [SDXL गाइड](SDXL.md) |
| **Kolors** | 5B | [Kolors गाइड](KOLORS.md) |
| **Stable Cascade** | - | [Cascade गाइड](STABLE_CASCADE_C.md) |

### छवि संपादन

| मॉडल | गाइड |
|-------|-------|
| **Qwen Edit** | [Qwen Edit गाइड](QWEN_EDIT.md) |
| **LongCat Edit** | [LongCat Edit गाइड](LONGCAT_EDIT.md) |

## वीडियो मॉडल

| मॉडल | पैरामीटर | गाइड |
|-------|------------|-------|
| **Wan Video** | 1.3-14B | [Wan गाइड](WAN.md) |
| **LTX Video** | 5B | [LTX गाइड](LTXVIDEO.md) |
| **LTX Video 2** | 19B | [LTX Video 2 गाइड](LTXVIDEO2.md) |
| **Hunyuan Video** | 8.3B | [Hunyuan गाइड](HUNYUANVIDEO.md) |
| **Sana Video** | - | [Sana Video गाइड](SANAVIDEO.md) |
| **Kandinsky 5 Video** | - | [Kandinsky Video गाइड](KANDINSKY5_VIDEO.md) |
| **LongCat Video** | - | [LongCat Video गाइड](LONGCAT_VIDEO.md) |
| **LongCat Video Edit** | - | [LongCat Video Edit गाइड](LONGCAT_VIDEO_EDIT.md) |

## ऑडियो मॉडल

| मॉडल | पैरामीटर | गाइड |
|-------|------------|-------|
| **ACE-Step** | 3.5B | [ACE-Step गाइड](ACE_STEP.md) |

## मॉडल चुनना

**नए उपयोगकर्ताओं के लिए:**

- उच्च गुणवत्ता वाली इमेज जनरेशन के लिए **Flux.1** से शुरू करें
- मेमोरी आवश्यकता घटाने के लिए **LoRA** प्रशिक्षण का उपयोग करें

**प्रोडक्शन के लिए:**

- व्यापक संगतता के लिए **SD3** या **SDXL**
- अधिकतम गुणवत्ता के लिए **Flux.2** (अधिक VRAM आवश्यक)

**वीडियो के लिए:**

- सर्वश्रेष्ठ गुणवत्ता/संसाधन संतुलन के लिए **Wan Video**
- सुपर‑रेज़ोल्यूशन के साथ I2V के लिए **Hunyuan Video**

**विशिष्ट उपयोग मामलों के लिए:**

- इमेज एडिटिंग/कंडीशनिंग के लिए **Flux Kontext**
- टेक्स्ट‑टू‑म्यूज़िक के लिए **ACE-Step**

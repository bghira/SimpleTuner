# डेटासेट कॉन्फ़िगरेशन प्रीसेट्स

Hugging Face Hub पर विभिन्न बड़े पैमाने के डेटासेट्स के लिए यहाँ कॉन्फ़िगरेशन विवरण दिए गए हैं ताकि चीजें जल्दी काम करने लगें।

> **सुझाव:** बड़े regularization datasets के लिए, आप `max_num_samples` का उपयोग करके dataset को deterministic random subset तक सीमित कर सकते हैं। विवरण के लिए [DATALOADER.md](../DATALOADER.md#max_num_samples) देखें।

नया प्रीसेट जोड़ने के लिए, [इस टेम्पलेट](../data_presets/preset.md) का उपयोग करके नया pull-request सबमिट करें।

- [DALLE-3 1M](../data_presets/preset_dalle3.md)
- [bghira/photo-concept-bucket](../data_presets/preset_pexels.md)
- [Midjourney v6 520k](../data_presets/preset_midjourney.md)
- [Nijijourney v6 520k](../data_presets/preset_nijijourney.md)
- [Subjects200K](../data_presets/preset_subjects200k.md), `datasets` लाइब्रेरी के उपयोग का एक उदाहरण

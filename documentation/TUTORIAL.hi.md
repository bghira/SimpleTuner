# ट्यूटोरियल

इस ट्यूटोरियल को [इंस्टॉलेशन गाइड](INSTALL.md) में समेकित कर दिया गया है।

## क्विक स्टार्ट

1. **SimpleTuner इंस्टॉल करें**: `pip install 'simpletuner[cuda]'` (अन्य प्लेटफ़ॉर्म्स के लिए README देखें)
   - CUDA 13 / Blackwell users (NVIDIA B-series GPUs): `pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130`
2. **कॉन्फ़िगर करें**: `simpletuner configure` (इंटरैक्टिव सेटअप)
3. **ट्रेन करें**: `simpletuner train`

## विस्तृत गाइड्स

- **[इंस्टॉलेशन गाइड](INSTALL.md)** - प्रशिक्षण डेटा तैयारी सहित पूरा सेटअप
- **[क्विकस्टार्ट गाइड्स](QUICKSTART.md)** - मॉडल‑विशिष्ट प्रशिक्षण गाइड्स
- **[हार्डवेयर आवश्यकताएँ](https://github.com/bghira/SimpleTuner#hardware-requirements)** - VRAM और सिस्टम आवश्यकताएँ

अधिक जानकारी के लिए, देखें:

- **[इंस्टॉलेशन गाइड](INSTALL.md)** - प्रशिक्षण डेटा तैयारी सहित पूरा सेटअप
- **[Options Reference](OPTIONS.md)** - पैरामीटर्स की पूरी सूची
- **[Dataloader कॉन्फ़िगरेशन](DATALOADER.md)** - डेटासेट सेटअप

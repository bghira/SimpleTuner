# SimpleTuner के लिए Docker

यह Docker कॉन्फ़िगरेशन Runpod, Vast.ai, और अन्य Docker‑compatible hosts सहित विभिन्न प्लेटफ़ॉर्म्स पर SimpleTuner एप्लिकेशन चलाने के लिए एक व्यापक वातावरण प्रदान करता है। इसे उपयोग में आसानी और मजबूती के लिए ऑप्टिमाइज़ किया गया है, और इसमें मशीन‑लर्निंग प्रोजेक्ट्स के लिए आवश्यक टूल्स और लाइब्रेरीज़ शामिल हैं।

## कंटेनर फीचर्स

- **CUDA‑enabled Base Image**: GPU‑accelerated एप्लिकेशन समर्थन के लिए `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` पर आधारित।
- **डेवलपमेंट टूल्स**: Git, SSH, और `tmux`, `vim`, `htop` जैसी विभिन्न utilities शामिल हैं।
- **Python और लाइब्रेरीज़**: Python 3.10 के साथ आता है और SimpleTuner pip के जरिए pre‑installed है।
- **Huggingface और WandB इंटीग्रेशन**: Huggingface Hub और WandB के साथ सहज इंटीग्रेशन के लिए pre‑configured, जिससे मॉडल शेयरिंग और experiment tracking आसान हो जाता है।

## शुरू करें

### WSL के जरिए Windows OS समर्थन (प्रयोगात्मक)

निम्न गाइड को Dockerengine इंस्टॉल किए हुए WSL2 Distro में टेस्ट किया गया था।


### 1. कंटेनर बनाना

रिपॉज़िटरी क्लोन करें और Dockerfile वाली डायरेक्टरी में जाएँ। Docker इमेज इस तरह बनाएं:

```bash
docker build -t simpletuner .
```

### 2. कंटेनर चलाना

GPU समर्थन के साथ कंटेनर चलाने के लिए:

```bash
docker run --gpus all -it -p 22:22 simpletuner
```

यह कमांड GPU access सेटअप करता है और बाहरी कनेक्टिविटी के लिए SSH पोर्ट मैप करता है।

### 3. Environment variables

बाहरी टूल्स के साथ इंटीग्रेशन के लिए, कंटेनर Huggingface और WandB tokens के environment variables सपोर्ट करता है। रनटाइम पर इन्हें ऐसे पास करें:

```bash
docker run --gpus all -e HF_TOKEN='your_token' -e WANDB_API_KEY='your_token' -it -p 22:22 simpletuner
```

### 4. Data volumes

होस्ट और कंटेनर के बीच persistent storage और डेटा शेयरिंग के लिए, डेटा वॉल्यूम माउंट करें:

```bash
docker run --gpus all -v /path/on/host:/workspace -it -p 22:22 simpletuner
```

### 5. SSH एक्सेस

SSH by default कॉन्फ़िगर होता है। सुनिश्चित करें कि आप अपनी SSH public key उचित environment variable के जरिए दें (`Vast.ai` के लिए `SSH_PUBLIC_KEY` या `Runpod` के लिए `PUBLIC_KEY`)।

### 6. SimpleTuner का उपयोग

SimpleTuner pre‑installed और ready है। आप सीधे training commands चला सकते हैं:

```bash
simpletuner configure
simpletuner train
```

कॉन्फ़िगरेशन और सेटअप के लिए [इंस्टॉलेशन दस्तावेज़](INSTALL.md) और [क्विकस्टार्ट गाइड्स](QUICKSTART.md) देखें।

## अतिरिक्त कॉन्फ़िगरेशन

### कस्टम स्क्रिप्ट्स और कॉन्फ़िगरेशन्स

यदि आप custom startup scripts जोड़ना चाहते हैं या कॉन्फ़िगरेशन्स बदलना चाहते हैं, तो entry script (`docker-start.sh`) को अपने ज़रूरत के अनुसार बढ़ाएँ।

यदि कोई क्षमता इस सेटअप से हासिल नहीं हो सकती, तो कृपया नया issue खोलें।

### Docker Compose

जो उपयोगकर्ता `docker-compose.yaml` पसंद करते हैं, उनके लिए यह टेम्पलेट दिया गया है जिसे आप अपनी ज़रूरत के अनुसार extend और customize कर सकते हैं।

स्टैक डिप्लॉय होने के बाद आप कंटेनर से कनेक्ट कर सकते हैं और ऊपर बताए गए चरणों के अनुसार उसे ऑपरेट करना शुरू कर सकते हैं।

```bash
docker compose up -d

docker exec -it simpletuner /bin/bash
```

```docker-compose.yaml
services:
  simpletuner:
    container_name: simpletuner
    build:
      context: [Path to the repository]/SimpleTuner
      dockerfile: Dockerfile
    ports:
      - "[port to connect to the container]:22"
    volumes:
      - "[path to your datasets]:/datasets"
      - "[path to your configs]:/workspace/config"
    environment:
      HF_TOKEN: [your hugging face token]
      WANDB_API_KEY: [your wanddb token]
    command: ["tail", "-f", "/dev/null"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

> ⚠️ कृपया अपने WandB और Hugging Face tokens संभालते समय सावधान रहें! इन्हें निजी version‑control रिपॉज़िटरी में भी commit न करने की सलाह दी जाती है, ताकि वे लीक न हों। प्रोडक्शन उपयोग‑केस के लिए key management storage की सिफ़ारिश है, लेकिन यह इस गाइड के दायरे से बाहर है।
---

## समस्या समाधान

### CUDA Version Mismatch

**लक्षण**: एप्लिकेशन GPU का उपयोग नहीं कर पाता, या GPU‑accelerated कार्य चलाते समय CUDA libraries से जुड़े errors दिखाई देते हैं।

**कारण**: यह समस्या तब हो सकती है जब Docker कंटेनर के अंदर इंस्टॉल किया गया CUDA संस्करण होस्ट मशीन पर उपलब्ध CUDA driver संस्करण से मेल नहीं खाता।

**समाधान**:
1. **होस्ट पर CUDA Driver संस्करण जाँचें**: होस्ट मशीन पर इंस्टॉल CUDA driver का संस्करण जानने के लिए:
   ```bash
   nvidia-smi
   ```
   यह कमांड आउटपुट के top right पर CUDA संस्करण दिखाएगा।

2. **कंटेनर CUDA संस्करण मिलाएँ**: सुनिश्चित करें कि आपकी Docker इमेज में CUDA toolkit संस्करण होस्ट के CUDA driver के साथ संगत है। NVIDIA सामान्यतः forward compatibility देता है, लेकिन NVIDIA वेबसाइट पर विशिष्ट compatibility matrix देखें।

3. **इमेज रीबिल्ड करें**: यदि आवश्यक हो, Dockerfile में base image बदलकर होस्ट के CUDA driver से मेल कराएँ। उदाहरण के लिए, यदि आपका होस्ट CUDA 11.2 चला रहा है और कंटेनर CUDA 11.8 के लिए सेट है, तो आपको उपयुक्त base image पर स्विच करना पड़ सकता है:
   ```Dockerfile
   FROM nvidia/cuda:11.2.0-runtime-ubuntu22.04
   ```
   Dockerfile बदलने के बाद Docker इमेज फिर से बनाएं।

### SSH कनेक्शन समस्याएँ

**लक्षण**: SSH के जरिए कंटेनर से कनेक्ट नहीं हो पा रहा है।

**कारण**: SSH keys का गलत कॉन्फ़िगरेशन या SSH सेवा का ठीक से शुरू न होना।

**समाधान**:
1. **SSH कॉन्फ़िगरेशन जाँचें**: सुनिश्चित करें कि public SSH key कंटेनर के `~/.ssh/authorized_keys` में सही तरीके से जोड़ी गई है। साथ ही, कंटेनर में प्रवेश कर `service ssh status` चलाकर सुनिश्चित करें कि SSH सेवा चल रही है:
   ```bash
   service ssh status
   ```
2. **Exposed Ports**: सुनिश्चित करें कि SSH पोर्ट (22) ठीक से exposed और mapped है, जैसा कि रनिंग निर्देशों में दिखाया गया है:
   ```bash
   docker run --gpus all -it -p 22:22 simpletuner
   ```

### सामान्य सलाह

- **Logs और Output**: किसी भी error संदेश या warning के लिए कंटेनर लॉग्स और आउटपुट की समीक्षा करें।
- **दस्तावेज़ और फ़ोरम्स**: अधिक विस्तृत troubleshooting सलाह के लिए Docker और NVIDIA CUDA दस्तावेज़ देखें। आपके उपयोग किए जा रहे सॉफ़्टवेयर या dependencies से जुड़े community forums और issue trackers भी उपयोगी संसाधन हो सकते हैं।

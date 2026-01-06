# DeepSpeed offload / multi-GPU प्रशिक्षण

SimpleTuner v0.7 ने DeepSpeed ZeRO stages 1 से 3 तक का प्रारंभिक समर्थन SDXL प्रशिक्षण के लिए पेश किया।
v3.0 में यह समर्थन काफी बेहतर हो गया है, WebUI configuration builder, बेहतर optimizer समर्थन, और बेहतर offload management के साथ।

> ⚠️ DeepSpeed macOS (MPS) या ROCm सिस्टम्स पर उपलब्ध नहीं है।

**9237MiB VRAM पर SDXL 1.0 प्रशिक्षण**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:08:00.0 Off |                  Off |
|  0%   43C    P2   100W / 450W |   9237MiB / 24564MiB |    100%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     11500      C   ...uner/.venv/bin/python3.12     9232MiB |
+-----------------------------------------------------------------------------+
```

ये मेमोरी बचत DeepSpeed ZeRO Stage 2 offload के उपयोग से मिली है। इसके बिना SDXL U‑net 24G से अधिक VRAM लेगा और dreaded CUDA Out of Memory exception देगा।

## DeepSpeed क्या है?

ZeRO का अर्थ **Zero Redundancy Optimizer** है। यह तकनीक उपलब्ध devices (GPUs और CPUs) के बीच विभिन्न model training states (weights, gradients, और optimizer states) को बाँटकर हर GPU की मेमोरी खपत घटाती है।

ZeRO को optimization stages के रूप में लागू किया जाता है, जहाँ शुरुआती stages की optimizations बाद के stages में शामिल रहती हैं। ZeRO पर deep dive के लिए मूल [पेपर](https://arxiv.org/abs/1910.02054v3) (1910.02054v3) देखें।

## ज्ञात समस्याएँ

### LoRA समर्थन

DeepSpeed मॉडल saving routines को बदलता है, इसलिए अभी DeepSpeed के साथ LoRA प्रशिक्षण समर्थित नहीं है।

यह भविष्य के रिलीज़ में बदल सकता है।

### Existing checkpoints पर DeepSpeed enable/disable करना

SimpleTuner में, DeepSpeed को **enable** नहीं किया जा सकता यदि आप ऐसे checkpoint से resume कर रहे हैं जिसमें पहले DeepSpeed उपयोग नहीं हुआ।

इसके विपरीत, DeepSpeed को **disable** नहीं किया जा सकता यदि आप ऐसे checkpoint से resume कर रहे हैं जो DeepSpeed के साथ प्रशिक्षित था।

इस समस्या को workaround करने के लिए, DeepSpeed को enable/disable करने से पहले training pipeline को पूर्ण model weights के सेट में export करें।

यह समर्थन संभवतः कभी नहीं आएगा, क्योंकि DeepSpeed का optimiser सामान्य विकल्पों से काफी अलग है।

## DeepSpeed Stages

DeepSpeed मॉडल प्रशिक्षण के लिए तीन स्तर की optimization देता है, और हर स्तर के साथ overhead बढ़ता है।

विशेष रूप से multi‑GPU training के लिए, CPU transfers अभी DeepSpeed में बहुत अच्छी तरह optimized नहीं हैं।

इस overhead के कारण, जो **सबसे कम** स्तर काम करता है, वही चुनना अनुशंसित है।

### Stage 1

Optimizer states (जैसे Adam optimizer के लिए 32‑bit weights और first/second moment estimates) processes के बीच बाँटे जाते हैं, ताकि हर process केवल अपने partition को update करे।

### Stage 2

Model weights को update करने के लिए reduced 32‑bit gradients भी partition किए जाते हैं ताकि हर process केवल अपने optimizer states के हिस्से के gradients रखे।

### Stage 3

16‑bit model parameters processes के बीच partition किए जाते हैं। ZeRO‑3 forward और backward passes के दौरान इन्हें स्वतः collect और partition करता है।

## DeepSpeed सक्षम करना

[Official tutorial](https://www.deepspeed.ai/tutorials/zero/) बहुत अच्छी तरह structured है और यहाँ उल्लिखित न होने वाले कई scenarios शामिल करता है।

### Method 1: WebUI Configuration Builder (अनुशंसित)

SimpleTuner अब DeepSpeed कॉन्फ़िगरेशन के लिए user‑friendly WebUI प्रदान करता है:

1. SimpleTuner WebUI पर जाएँ
2. **Hardware** टैब पर जाएँ और **Accelerate & Distributed** सेक्शन खोलें
3. `DeepSpeed Config (JSON)` फ़ील्ड के पास **DeepSpeed Builder** बटन क्लिक करें
4. इंटरैक्टिव इंटरफ़ेस में:
   - ZeRO optimization stage चुनें (1, 2, या 3)
   - offload विकल्प कॉन्फ़िगर करें (CPU, NVMe)
   - optimizers और schedulers चुनें
   - gradient accumulation और clipping पैरामीटर्स सेट करें
5. जनरेटेड JSON कॉन्फ़िगरेशन प्रीव्यू करें
6. कॉन्फ़िगरेशन सेव और लागू करें

Builder JSON संरचना consistent रखता है और आवश्यक होने पर unsupported optimizers को safe defaults में बदल देता है, जिससे सामान्य कॉन्फ़िगरेशन गलतियों से बचा जा सके।

### Method 2: Manual JSON Configuration

जो उपयोगकर्ता सीधे कॉन्फ़िगरेशन एडिट करना पसंद करते हैं, वे `config.json` में सीधे DeepSpeed कॉन्फ़िगरेशन जोड़ सकते हैं:

```json
{
  "deepspeed_config": {
    "zero_optimization": {
      "stage": 2,
      "offload_param": {
        "device": "cpu",
        "pin_memory": true
      },
      "offload_optimizer": {
        "device": "cpu",
        "pin_memory": true
      }
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "optimizer": {
      "type": "AdamW",
      "params": {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01
      }
    },
    "scheduler": {
      "type": "WarmupLR",
      "params": {
        "warmup_min_lr": 0,
        "warmup_max_lr": 1e-4,
        "warmup_num_steps": 500
      }
    },
    "train_batch_size": 8,
    "train_micro_batch_size_per_gpu": 2
  }
}
```

**मुख्य कॉन्फ़िगरेशन विकल्प:**

- `zero_optimization.stage`: अलग ZeRO optimization स्तरों के लिए 1, 2, या 3 सेट करें
- `offload_param.device`: parameter offloading के लिए "cpu" या "nvme" उपयोग करें
- `offload_optimizer.device`: optimizer offloading के लिए "cpu" या "nvme" उपयोग करें
- `optimizer.type`: समर्थित optimizers चुनें (AdamW, Adam, Adagrad, Lamb, आदि)
- `gradient_accumulation_steps`: gradients accumulate करने के steps

**NVMe Offload उदाहरण:**
```json
{
  "deepspeed_config": {
    "zero_optimization": {
      "stage": 3,
      "offload_param": {
        "device": "nvme",
        "nvme_path": "/path/to/nvme/storage",
        "buffer_size": 100000000.0,
        "pin_memory": true
      }
    }
  }
}
```

### Method 3: accelerate config के जरिए manual configuration

Advanced users के लिए, DeepSpeed को `accelerate config` से सक्षम किया जा सकता है:

```
----------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?
No distributed training
Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)? [yes/NO]:NO
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO
Do you want to use DeepSpeed? [yes/NO]: yes
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: NO
----------------------------------------------------------------------------------------------------------------------------
What should be your DeepSpeed's ZeRO optimization stage?
1
How many gradient accumulation steps you're passing in your script? [1]: 4
Do you want to use gradient clipping? [yes/NO]:
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]:
How many GPU(s) should be used for distributed training? [1]:
----------------------------------------------------------------------------------------------------------------------------
Do you wish to use FP16 or BF16 (mixed precision)?bf16
accelerate configuration saved at /root/.cache/huggingface/accelerate/default_config.yaml
```

यह निम्न yaml फ़ाइल बनाता है:

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 4
  zero3_init_flag: false
  zero_stage: 1
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## SimpleTuner कॉन्फ़िगरेशन

SimpleTuner को DeepSpeed उपयोग के लिए किसी विशेष कॉन्फ़िगरेशन की आवश्यकता नहीं है।

यदि ZeRO stage 2 या 3 के साथ NVMe offload उपयोग कर रहे हैं, तो `--offload_param_path=/path/to/offload` दिया जा सकता है, ताकि parameter/optimiser offload फ़ाइलें dedicated partition पर स्टोर हों। यह storage ideally NVMe डिवाइस होनी चाहिए, लेकिन कोई भी storage चलेगी।

### हालिया सुधार (v0.7+)

#### WebUI Configuration Builder
SimpleTuner में अब WebUI पर एक व्यापक DeepSpeed configuration builder शामिल है, जिससे आप:
- सहज इंटरफ़ेस से custom DeepSpeed JSON कॉन्फ़िगरेशन बना सकते हैं
- उपलब्ध parameters auto‑discover कर सकते हैं
- लागू करने से पहले configuration impact देख सकते हैं
- configuration templates सेव और reuse कर सकते हैं

#### Enhanced Optimizer Support
सिस्टम में optimizer name normalization और validation बेहतर किया गया है:
- **समर्थित optimizers**: AdamW, Adam, Adagrad, Lamb, OneBitAdam, OneBitLamb, ZeroOneAdam, MuAdam, MuAdamW, MuSGD, Lion, Muon
- **Unsupported optimizers** (स्वतः AdamW से बदल दिए जाते हैं): cpuadam, fusedadam
- unsupported optimizers पर automatic fallback warnings

#### Improved Offload Management
- **Automatic cleanup**: stale DeepSpeed offload swap directories स्वतः हटाए जाते हैं ताकि corrupted resume states न बनें
- **Enhanced NVMe support**: NVMe offload paths के लिए बेहतर handling, automatic buffer size allocation सहित
- **Platform detection**: incompatible platforms (macOS/ROCm) पर DeepSpeed स्वतः disabled होता है

#### Configuration Validation
- बदलाव लागू करते समय optimizer names और configuration structure का automatic normalization
- unsupported optimizer selections और malformed JSON के लिए safety guards
- troubleshooting के लिए बेहतर error handling और logging

### DeepSpeed Optimizer / Learning rate scheduler

DeepSpeed अपना learning rate scheduler उपयोग करता है और डिफ़ॉल्ट रूप से AdamW का एक heavily‑optimized संस्करण उपयोग करता है — हालांकि 8bit नहीं। यह DeepSpeed के लिए कम महत्वपूर्ण लगता है, क्योंकि चीज़ें CPU के करीब रहती हैं।

यदि `default_config.yaml` में `scheduler` या `optimizer` कॉन्फ़िगर किया गया है, तो वही उपयोग होगा। यदि कोई `scheduler` या `optimizer` परिभाषित नहीं है, तो डिफ़ॉल्ट `AdamW` और `WarmUp` विकल्प क्रमशः optimiser और scheduler के रूप में उपयोग होंगे।

## कुछ त्वरित टेस्ट परिणाम

4090 24G GPU का उपयोग करते हुए:

* अब हम 1 megapixel (1024^2 pixel area) पर full U‑net ट्रेन कर सकते हैं और batch size 8 के लिए केवल **13102MiB VRAM** उपयोग होता है
  * यह प्रति iteration 8 सेकंड पर चला। इसका मतलब 1000 steps training लगभग 2.5 घंटे में हो सकती है।
  * DeepSpeed tutorial में बताया गया है कि batch size को कम करने से लाभ हो सकता है, ताकि उपलब्ध VRAM parameters और optimiser states के लिए उपयोग हो सके।
    * हालांकि SDXL अपेक्षाकृत छोटा मॉडल है, और यदि performance impact स्वीकार्य हो तो हम कुछ सिफ़ारिशों से बच सकते हैं।
* **128x128** image size पर batch size 8 के साथ training केवल **9237MiB VRAM** लेती है। यह pixel art training के लिए एक niche use case हो सकता है, जिसमें latent space के साथ 1:1 mapping चाहिए।

इन पैरामीटर्स में आप अलग‑अलग सफलता स्तर देखेंगे और संभव है कि full u‑net training को 1024x1024 पर batch size 1 के लिए केवल 8GiB VRAM में फिट कर सकें (अपरीक्षित)।

क्योंकि SDXL को लंबे समय तक विभिन्न image resolutions और aspect ratios पर ट्रेन किया गया था, आप pixel area को .75 megapixels (लगभग 768x768) तक घटाकर memory उपयोग और optimize कर सकते हैं।

# AMD device समर्थन

मेरे पास consumer या workstation‑grade AMD GPUs नहीं हैं, लेकिन कुछ रिपोर्ट्स में MI50 (अब support से बाहर) और अन्य उच्च‑स्तरीय Instinct कार्ड्स के साथ DeepSpeed **काम** करता है। AMD अपनी implementation के लिए एक repository रखता है।

## Troubleshooting

### सामान्य समस्याएँ और समाधान

#### "DeepSpeed crash on resume"
**समस्या**: DeepSpeed offload enabled होने पर checkpoint से resume करते समय training क्रैश हो जाती है।

**समाधान**: SimpleTuner अब stale DeepSpeed offload swap directories को स्वतः साफ़ करता है ताकि corrupted resume states न हों। यह समस्या नवीनतम अपडेट्स में हल हो गई है।

#### "Unsupported optimizer error"
**समस्या**: DeepSpeed कॉन्फ़िगरेशन में unsupported optimizer नाम हैं।

**समाधान**: सिस्टम अब optimizer नामों को स्वतः normalize करता है और unsupported optimizers (cpuadam, fusedadam) को AdamW से बदल देता है। fallback होने पर warnings लॉग होती हैं।

#### "DeepSpeed not available on this platform"
**समस्या**: DeepSpeed विकल्प disabled हैं या उपलब्ध नहीं हैं।

**समाधान**: DeepSpeed केवल CUDA सिस्टम्स पर समर्थित है। macOS (MPS) और ROCm पर यह स्वतः disabled होता है। यह compatibility समस्याओं से बचने के लिए by design है।

#### "NVMe offload path issues"
**समस्या**: NVMe offload path कॉन्फ़िगरेशन से संबंधित errors।

**समाधान**: सुनिश्चित करें कि `--offload_param_path` वैध directory की ओर संकेत करता है और पर्याप्त space उपलब्ध है। सिस्टम अब buffer size allocation और path validation स्वतः संभालता है।

#### "Configuration validation errors"
**समस्या**: DeepSpeed कॉन्फ़िगरेशन पैरामीटर्स invalid हैं।

**समाधान**: WebUI configuration builder का उपयोग करके JSON बनाएं; यह configuration लागू करने से पहले optimizer selection और structure normalize करता है।

### Debug जानकारी

DeepSpeed मुद्दों के लिए निम्न जाँचें:
- WebUI Hardware टैब (Hardware → Accelerate) या `nvidia-smi` से hardware compatibility
- training logs में DeepSpeed कॉन्फ़िगरेशन
- offload path permissions और उपलब्ध space
- platform detection logs

# EMA training (Exponential moving average)

EMA gradients को smooth करने और resulting weights की generalisation क्षमताओं को बेहतर बनाने का शानदार तरीका है, लेकिन यह memory‑heavy होता है।

EMA मॉडल parameters की shadow copy मेमोरी में रखता है, जिससे model footprint लगभग दोगुना हो जाता है। SimpleTuner में EMA को Accelerator module से नहीं गुजारा जाता, इसलिए यह DeepSpeed से प्रभावित नहीं होता। इसका मतलब है कि base U‑net के साथ मिली मेमोरी बचत EMA मॉडल के साथ नहीं मिलती।

हालाँकि, डिफ़ॉल्ट रूप से EMA मॉडल CPU पर रखा जाता है।

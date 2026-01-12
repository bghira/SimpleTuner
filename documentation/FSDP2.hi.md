# FSDP2 sharded / multi-GPU प्रशिक्षण

SimpleTuner अब PyTorch Fully Sharded Data Parallel v2 (DTensor‑backed FSDP) के लिए first‑class समर्थन के साथ आता है। WebUI full‑model runs के लिए v2 implementation को डिफ़ॉल्ट करता है और सबसे महत्वपूर्ण accelerate flags surface करता है ताकि आप custom launch scripts लिखे बिना multi‑GPU हार्डवेयर तक scale कर सकें।

> ⚠️ FSDP2 हाल की PyTorch 2.x releases को लक्षित करता है जिनमें distributed DTensor stack CUDA builds पर सक्षम है। WebUI केवल CUDA hosts पर context‑parallel controls दिखाता है; अन्य backends प्रयोगात्मक माने जाते हैं।

## FSDP2 क्या है?

FSDP2 PyTorch के sharded data‑parallel इंजन का अगला संस्करण है। FSDP v1 की legacy flat‑parameter लॉजिक की जगह v2 plugin DTensor पर बैठता है। यह model parameters, gradients, और optimizers को ranks के बीच shard करता है, जबकि प्रति‑rank working set छोटा रखता है। classic ZeRO‑style approaches की तुलना में यह Hugging Face accelerate launch flow को बनाए रखता है, ताकि checkpoints, optimizers, और inference paths SimpleTuner के बाकी हिस्सों के साथ संगत रहें।

## फीचर अवलोकन

- WebUI toggle (Hardware → Accelerate) जो sane defaults के साथ FullyShardedDataParallelPlugin बनाता है
- Automatic CLI normalisation (`--fsdp_version`, `--fsdp_state_dict_type`, `--fsdp_auto_wrap_policy`) ताकि manual flags forgiving रहें
- Optional context‑parallel sharding (`--context_parallel_size`, `--context_parallel_comm_strategy`) जो long‑sequence models के लिए FSDP2 के ऊपर layered है
- Built‑in transformer block detection modal जो base model inspect करता है और auto‑wrapping के लिए class names सुझाता है
- `~/.simpletuner/fsdp_block_cache.json` में cached detection metadata, WebUI settings में one‑click maintenance actions के साथ
- Checkpoint format switcher (sharded बनाम full) और tight host memory ceilings के लिए CPU‑RAM‑efficient resume mode

## ज्ञात सीमाएँ

- FSDP2 केवल `model_type` = `full` होने पर सक्षम हो सकता है। PEFT/LoRA style runs standard single‑device paths ही उपयोग करते हैं।
- DeepSpeed और FSDP परस्पर exclusive हैं। `--fsdp_enable` और DeepSpeed config दोनों देने पर CLI और WebUI में स्पष्ट error आता है।
- Context parallelism केवल CUDA सिस्टम्स पर सीमित है और `--context_parallel_size > 1` के साथ `--fsdp_version=2` की आवश्यकता होती है।
- Validation passes अब `--fsdp_reshard_after_forward=true` के साथ काम करते हैं — FSDP‑wrapped models सीधे pipelines को दिए जाते हैं, जो all‑gather/reshard को transparent तरीके से संभालते हैं।
- Block detection base model को locally instantiate करता है। बड़े checkpoints स्कैन करते समय थोड़ी देरी और elevated host memory usage की अपेक्षा करें।
- FSDP v1 backward compatibility के लिए मौजूद है, लेकिन UI और CLI logs में deprecated के रूप में चिह्नित है।

## FSDP2 सक्षम करना

### Method 1: WebUI (अनुशंसित)

1. SimpleTuner WebUI खोलें और वह training configuration लोड करें जिसे आप चलाना चाहते हैं।
2. **Hardware → Accelerate** पर जाएँ।
3. **Enable FSDP v2** toggle करें। version selector डिफ़ॉल्ट रूप से `2` होगा; जब तक v1 की खास जरूरत न हो, इसे छोड़ दें।
4. (वैकल्पिक) निम्न समायोजित करें:
   - **Reshard After Forward** ताकि VRAM बनाम communication trade‑off हो
   - **Checkpoint Format** `Sharded` और `Full` के बीच
   - **CPU RAM Efficient Loading** यदि tight host memory limits के साथ resume करना हो
   - **Auto Wrap Policy** और **Transformer Classes to Wrap** (नीचे detection workflow देखें)
   - **Context Parallel Size / Rotation** जब sequence sharding चाहिए
5. कॉन्फ़िगरेशन सेव करें। Trainer launch surface अब सही accelerate plugin पास करेगा।

### Method 2: CLI

WebUI में उपलब्ध वही flags `simpletuner-train` में उपयोग करें। दो GPUs पर SDXL full‑model run का उदाहरण:

```bash
simpletuner-train \
  --model_type=full \
  --model_family=sdxl \
  --output_dir=/data/experiments/sdxl-fsdp2 \
  --fsdp_enable \
  --fsdp_version=2 \
  --fsdp_state_dict_type=SHARDED_STATE_DICT \
  --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
  --num_processes=2
```

यदि आप पहले से accelerate config file रखते हैं, तो आप उसे उपयोग करते रह सकते हैं; SimpleTuner FSDP plugin को launch parameters में merge करता है, आपके पूरे configuration को override नहीं करता।

## Context parallelism

Context parallelism CUDA hosts के लिए FSDP2 के ऊपर एक वैकल्पिक layer के रूप में उपलब्ध है। `--context_parallel_size` (या WebUI field) को उन GPUs की संख्या पर सेट करें जो sequence dimension split करें। Communication इस प्रकार होती है:

- `allgather` (डिफ़ॉल्ट) – overlap को प्राथमिकता देता है और शुरुआत के लिए सबसे अच्छा है
- `alltoall` – बहुत बड़े attention windows वाले niche workloads में लाभदायक हो सकता है, लेकिन orchestration overhead बढ़ता है

Trainer context parallelism मांगने पर `fsdp_enable` और `fsdp_version=2` को enforce करता है। size को वापस `1` करने से feature साफ़ तरीके से disable हो जाता है और rotation string normalize हो जाती है ताकि saved configs consistent रहें।

## FSDP block detection workflow

SimpleTuner में एक detector शामिल है जो चुने गए base model को inspect करता है और FSDP auto wrapping के लिए सबसे उपयुक्त module classes दिखाता है:

1. Trainer form में **Model Family** चुनें (और वैकल्पिक रूप से **Model Flavour**)।
2. यदि आप custom weight directory से training कर रहे हैं तो checkpoint path दर्ज करें।
3. **Transformer Classes to Wrap** के पास **Detect Blocks** क्लिक करें। SimpleTuner मॉडल instantiate करेगा, modules traverse करेगा, और class‑wise parameter totals रिकॉर्ड करेगा।
4. modal analysis की समीक्षा करें:
   - **Select** करें कि कौन‑सी classes wrap हों (पहले कॉलम के checkboxes)
   - **Total Params** दिखाता है कि कौन‑से modules parameter budget dominate करते हैं
   - `_no_split_modules` (यदि मौजूद हों) badges के रूप में दिखते हैं और इन्हें exclusion lists में जोड़ना चाहिए
5. **Apply Selection** दबाकर `--fsdp_transformer_layer_cls_to_wrap` भरें।
6. बाद में खोलने पर cached result reuse होगा जब तक आप **Refresh Detection** न दबाएँ।

Detection results `~/.simpletuner/fsdp_block_cache.json` में model family, checkpoint path, और flavour के आधार पर key होते हैं। divergent checkpoints के बीच स्विच करते समय या weights update होने पर **Settings → WebUI Preferences → Cache Maintenance → Clear FSDP Detection Cache** उपयोग करें।

## Checkpoint handling

- **Sharded state dict** (`SHARDED_STATE_DICT`) rank‑local shards सेव करता है और बड़े मॉडलों पर अच्छी तरह scale होता है।
- **Full state dict** (`FULL_STATE_DICT`) external tooling compatibility के लिए parameters को rank 0 पर gather करता है; इसमें अधिक memory pressure होगा।
- **CPU RAM Efficient Loading** resume के दौरान all‑rank materialisation को delay करता है ताकि host memory spikes कम हों।
- **Reshard After Forward** forward passes के बीच parameter shards को lean रखता है। Validation अब FSDP‑wrapped models को सीधे diffusers pipelines में पास करके सही काम करता है।

अपने resume cadence और downstream tooling के अनुसार संयोजन चुनें। बहुत बड़े मॉडलों के लिए sharded checkpoints + RAM‑efficient loading सबसे सुरक्षित pairing है।

## Maintenance tooling

WebUI में **WebUI Preferences → Cache Maintenance** के तहत maintenance helpers उपलब्ध हैं:

- **Clear FSDP Detection Cache** सभी cached block scans हटाता है (`FSDP_SERVICE.clear_cache()` का wrapper)।
- **Clear DeepSpeed Offload Cache** ZeRO users के लिए बना रहता है; यह FSDP से स्वतंत्र रूप से काम करता है।

दोनों actions toast notifications दिखाते हैं और maintenance status area अपडेट करते हैं ताकि आप log files देखे बिना परिणाम की पुष्टि कर सकें।

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `"FSDP and DeepSpeed cannot be enabled simultaneously."` | दोनों plugins specify किए गए (उदा. DeepSpeed JSON + `--fsdp_enable`)। | DeepSpeed config हटाएँ या FSDP disable करें। |
| `"Context parallelism requires FSDP2."` | `context_parallel_size > 1` जबकि FSDP बंद है या v1 पर है। | FSDP enable करें, `--fsdp_version=2` रखें, या size वापस `1` करें। |
| Block detection fails with `Unknown model_family` | form में समर्थित family या flavour नहीं है। | dropdown से मॉडल चुनें; custom families को `model_families` में register करना होगा। |
| Detection shows stale classes | Cached result reuse हो गया। | **Refresh Detection** क्लिक करें या WebUI Preferences से cache साफ़ करें। |
| Resume exhausts host RAM | load के दौरान full state dict gather हो रहा है। | `SHARDED_STATE_DICT` पर स्विच करें और/या CPU RAM efficient loading सक्षम करें। |

## CLI flag reference

- `--fsdp_enable` – FullyShardedDataParallelPlugin चालू करें
- `--fsdp_version` – `1` और `2` के बीच चुनें (डिफ़ॉल्ट `2`, v1 deprecated)
- `--fsdp_reshard_after_forward` – forward के बाद parameter shards रिलीज़ करें (डिफ़ॉल्ट `true`)
- `--fsdp_state_dict_type` – `SHARDED_STATE_DICT` (डिफ़ॉल्ट) या `FULL_STATE_DICT`
- `--fsdp_cpu_ram_efficient_loading` – resume पर host memory spikes घटाएँ
- `--fsdp_auto_wrap_policy` – `TRANSFORMER_BASED_WRAP`, `SIZE_BASED_WRAP`, `NO_WRAP`, या dotted callable path
- `--fsdp_transformer_layer_cls_to_wrap` – detector द्वारा भरी गई comma‑separated class list
- `--context_parallel_size` – इतने ranks में attention shard करें (केवल CUDA + FSDP2)
- `--context_parallel_comm_strategy` – `allgather` (डिफ़ॉल्ट) या `alltoall` rotation strategy
- `--num_processes` – यदि config file न हो तो accelerate को दिए जाने वाले total ranks

ये Hardware → Accelerate के WebUI controls के साथ 1:1 मैप होते हैं, इसलिए interface से export की गई configuration CLI पर बिना अतिरिक्त बदलाव के चल सकती है।

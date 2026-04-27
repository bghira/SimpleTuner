# CaptionFlow integration

SimpleTuner [CaptionFlow](https://github.com/bghira/CaptionFlow) का उपयोग करके Web UI से image datasets के लिए captions बना सकता है। CaptionFlow एक scalable vLLM-powered captioning system है जिसमें orchestrator, GPU workers, checkpointed storage और YAML-based configuration है। SimpleTuner में यह Datasets page की **Captioning** sub-tab के रूप में उपलब्ध है, इसलिए captioning jobs उसी local GPU queue का उपयोग करते हैं जिसका उपयोग training और cache jobs करते हैं।

जब आप training से पहले captions generate या refresh करना चाहते हैं और SimpleTuner workflow में ही रहना चाहते हैं, तब इस integration का उपयोग करें।

## Installation

CaptionFlow optional dependency है। SimpleTuner वाले virtual environment में captioning target install करें:

```bash
pip install "simpletuner[captioning]"
```

CUDA 13 environments के लिए Web UI में दिखाया गया CUDA 13 target उपयोग करें। उसमें उस runtime के लिए अपेक्षित vLLM wheel शामिल होता है।

## SimpleTuner क्या manage करता है

Captioning job शुरू करने पर SimpleTuner:

- selected SimpleTuner dataset को CaptionFlow processor से map करता है;
- `127.0.0.1` पर local CaptionFlow orchestrator start करता है;
- job queue के जरिए एक या अधिक local GPU workers start करता है;
- orchestrator और worker logs को CaptionFlow job workspace में capture करता है;
- export से पहले CaptionFlow storage को gracefully checkpoint करता है;
- local datasets के लिए `.txt` captions dataset directory में लिखता है;
- Hugging Face datasets के लिए JSONL exports CaptionFlow job workspace में लिखता है।

CaptionFlow dependencies install न हों तब भी tab दिखाई देता है। उस स्थिति में job builder की जगह install command दिखाई जाती है।

## Builder mode

**Builder** view common single-stage captioning workflow के लिए है:

- active dataloader configuration से dataset selection;
- model, prompt, sampling, batch size, chunk size और GPU memory settings;
- worker count और queue behavior;
- local datasets के लिए text file export.

Default model `Qwen/Qwen2.5-VL-3B-Instruct` है। Local datasets selected output field का उपयोग करके images के पास text files export करते हैं। Hugging Face datasets remote dataset में वापस नहीं लिखे जाते; वे CaptionFlow workspace में JSONL export करते हैं।

## Raw Config mode

**Raw Config** तब उपयोग करें जब आपको ऐसे CaptionFlow features चाहिए जिन्हें builder model नहीं करता, जैसे multi-stage captioning, per-stage models, per-stage sampling, या prompt chains जहां एक stage दूसरे stage के output का उपयोग करता है।

Raw config YAML या JSON स्वीकार करता है। आप `orchestrator:` root वाली पूरी config paste कर सकते हैं या केवल orchestrator object।

SimpleTuner runtime पर इन fields को intentionally override करता है:

- `host`, `port`, और `ssl`;
- selected SimpleTuner dataset के आधार पर `dataset`;
- job workspace के अंदर `storage.data_dir` और `storage.checkpoint_dir`;
- `auth.worker_tokens` और `auth.admin_tokens`.

अन्य orchestrator settings, जैसे `chunk_size`, `chunks_per_request`, `storage.caption_buffer_size`, `vllm.sampling`, `vllm.inference_prompts`, और `vllm.stages`, preserve रहती हैं जब तक SimpleTuner को default भरने की जरूरत न हो।

## Multi-stage example

यह raw config पहले detailed caption stage चलाता है और फिर `{caption}` को shortening stage में पास करता है। Selected dataset, storage paths, ports और auth tokens job launch पर SimpleTuner भरता है।

```yaml
orchestrator:
  chunk_size: 1000
  chunks_per_request: 1
  chunk_buffer_multiplier: 2
  min_chunk_buffer: 10
  vllm:
    model: "Qwen/Qwen2.5-VL-3B-Instruct"
    tensor_parallel_size: 1
    max_model_len: 16384
    dtype: "float16"
    gpu_memory_utilization: 0.92
    enforce_eager: true
    disable_mm_preprocessor_cache: true
    limit_mm_per_prompt:
      image: 1
    batch_size: 8
    sampling:
      temperature: 0.7
      top_p: 0.95
      max_tokens: 256
    stages:
      - name: "base_caption"
        prompts:
          - "describe this image in detail"
        output_field: "caption"
      - name: "caption_shortening"
        model: "Qwen/Qwen2.5-VL-7B-Instruct"
        prompts:
          - "Please condense this elaborate caption to only the important details: {caption}"
        output_field: "captions"
        requires: ["base_caption"]
        gpu_memory_utilization: 0.35
```

## External CaptionFlow docs

- [CaptionFlow repository](https://github.com/bghira/CaptionFlow)
- [CaptionFlow README](https://github.com/bghira/CaptionFlow#readme)
- [CaptionFlow orchestrator examples](https://github.com/bghira/CaptionFlow/tree/main/examples/orchestrator)

Advanced CaptionFlow fields के लिए upstream examples को source of truth मानें। SimpleTuner के जरिए चलाते समय याद रखें कि dataset routing, local ports, storage workspace paths और auth tokens SimpleTuner manage करता है।

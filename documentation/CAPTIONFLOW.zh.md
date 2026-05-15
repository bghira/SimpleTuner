# CaptionFlow 集成

SimpleTuner 可以使用 [CaptionFlow](https://github.com/bghira/CaptionFlow) 在 Web UI 中为图像数据集生成 captions。CaptionFlow 是一个基于 vLLM 的可扩展 captioning 系统，包含 orchestrator、GPU worker、带 checkpoint 的存储以及 YAML 配置。SimpleTuner 将它集成到 Datasets 页面的 **Captioning** 子标签中，因此 captioning 作业可以使用和训练、缓存作业相同的本地 GPU 队列。

如果你想在训练前生成或刷新 captions，并希望留在 SimpleTuner 工作流中，可以使用此集成。

## 安装

CaptionFlow 是可选依赖。请在 SimpleTuner 使用的同一个虚拟环境中安装 captioning target：

```bash
pip install "simpletuner[captioning]"
```

对于 CUDA 13 环境，请使用 Web UI 安装提示中显示的 CUDA 13 target。它会包含该 runtime 需要的 vLLM wheel。

## SimpleTuner 管理的内容

启动 Captioning 作业时，SimpleTuner 会：

- 将选中的 SimpleTuner 数据集映射到 CaptionFlow processor；
- 在 `127.0.0.1` 启动本地 CaptionFlow orchestrator；
- 通过作业队列启动一个或多个本地 GPU worker；
- 将 orchestrator 和 worker 日志写入 CaptionFlow 作业 workspace；
- 在导出前安全 checkpoint CaptionFlow storage；
- 对本地数据集，将 `.txt` captions 写回数据集目录；
- 对 Hugging Face 数据集，将 JSONL 导出到 CaptionFlow 作业 workspace。

CaptionFlow 依赖未安装时，该标签仍会显示；此时会显示安装命令，而不是作业 builder。

## Builder 模式

**Builder** 视图覆盖常见的单阶段 captioning 工作流：

- 从当前 dataloader 配置中选择数据集；
- 设置 model、prompt、sampling、batch size、chunk size 和 GPU memory；
- 设置 worker 数量和队列行为；
- 为本地数据集导出 text files。

默认模型是 `Qwen/Qwen2.5-VL-3B-Instruct`。本地数据集会根据表单中的 output field，在图像旁边导出 text file。Hugging Face 数据集不会写回远程 dataset；它们会导出 JSONL 到 CaptionFlow workspace。

## Raw Config 模式

当你需要 builder 未覆盖的 CaptionFlow 功能时，请使用 **Raw Config**，例如 multi-stage captioning、每个 stage 使用不同 model、每个 stage 使用不同 sampling，或一个 stage 使用另一个 stage 输出的 prompt chain。

Raw config 接受 YAML 或 JSON。你可以粘贴带 `orchestrator:` 根节点的完整配置，也可以只粘贴 orchestrator 对象。

SimpleTuner 会在 runtime 有意覆盖这些字段：

- `host`、`port` 和 `ssl`；
- 基于所选 SimpleTuner 数据集生成的 `dataset`；
- 作业 workspace 下的 `storage.data_dir` 和 `storage.checkpoint_dir`；
- `auth.worker_tokens` 和 `auth.admin_tokens`。

其他 orchestrator 设置会保留，包括 `chunk_size`、`chunks_per_request`、`storage.caption_buffer_size`、`vllm.sampling`、`vllm.inference_prompts` 和 `vllm.stages`，除非 SimpleTuner 需要填入默认值。

## Multi-stage 示例

这个 raw config 先生成详细 caption，然后把 `{caption}` 传给缩短 stage。所选数据集、storage 路径、端口和认证 token 会由 SimpleTuner 在作业启动时填入。

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

## 外部文档

- [CaptionFlow 仓库](https://github.com/bghira/CaptionFlow)
- [CaptionFlow README](https://github.com/bghira/CaptionFlow#readme)
- [CaptionFlow orchestrator 示例](https://github.com/bghira/CaptionFlow/tree/main/examples/orchestrator)

高级 CaptionFlow 字段请以 upstream examples 为准。通过 SimpleTuner 运行时，请记住 dataset routing、本地端口、storage workspace 路径和认证 token 都由 SimpleTuner 管理。

# 语音克隆数据变换

语音克隆数据变换是一个计划中的实验性音频数据集功能。它会在主模型训练开始前，把一个目标人声身份迁移到更多歌曲、人声 stem 或表演上，从而扩展训练集。

它的目标不是把 SimpleTuner 变成独立的变声工作站，而是减少音频微调数据里的纠缠。如果某个歌手只出现在一种很窄的风格里，LoRA 可能学到的是“这个歌手在这种编曲里”，而不是歌手身份本身。通过语音克隆生成的扩展 split，可以让同一个人声身份出现在更多编曲、caption、歌词和歌曲结构中。

此功能仅面向音频数据集。

!!! warning "授权和权利"
    只应在你有权使用的声音和录音上使用此流程。声音身份是敏感的生物特征和创作数据。该变换可能生成听起来像真实人物的衍生音频，因此权限、授权和披露都很重要。

## ELI5

假设你有六首同一个歌手的录音，但每首都来自同一个乐队和同一种风格。只用这些歌训练时，模型可能会把歌手、吉他音色、鼓组感觉、速度范围和歌曲结构都当成一件事。

语音克隆数据变换试图拆开这些概念：

1. 从歌手示例中学习一个小型语音转换模型。
2. 读取更广泛的一组歌曲或人声 stem。
3. 把源人声的音色替换成目标歌手音色。
4. 让新的 captions 和歌词继续对齐生成后的音频。
5. 把生成音频作为另一个普通训练 split 加入。

这样主模型就能在更多上下文中看到目标声音，而不是只记住原始窄数据集。

## 适用场景

适合使用：

- 你有目标歌手的授权录音
- 目标身份和单一流派、乐队、制作风格或歌曲结构纠缠太强
- trigger 词只在原风格里有效，换流派就失效
- 一个数据集里有多个歌手，模型学成了平均声音
- 你想为不同人声身份训练不同 LoRA
- 你希望 SimpleTuner 在同一次训练配置里准备扩展 split

不适合使用：

- 你已经有大量、多样、干净的同一声音数据
- 扩展源音频质量差，或与 caption 对不齐
- 你需要公开发布且没有明确权利
- 基础生成模型即使用干净直接样本也学不会目标身份

## 它如何进入训练

语音克隆是数据准备变换，不是 conditioning 数据集。

`conditioning_data` 用于训练时一直和主样本绑定的辅助输入，例如参考图像或生成的条件图。

语音克隆应该放在数据集级别的 `data_transforms` 列表中。它会生成新的音频文件、caption 和可选歌词，然后把结果注册为另一个主 `audio` 数据集。之后普通 dataloader 会像读取其他训练 split 一样读取它。

伪配置形状：

```text
audio dataset:
    id: target-voice
    dataset_type: audio
    data_transforms:
        - task: identity_transfer
          source: expansion-audio-backend
          target: generated-audio-backend
          method: rvc
          audio_mode: separate_convert_remix
```

启动时的伪流程：

```text
for each audio dataset:
    for each data transform:
        if task is identity_transfer:
            prepare or reuse the target voice-conversion model
            prepare or reuse generated audio
            append generated audio as a normal train split

continue with normal metadata discovery, bucketing, caching, and training
```

## RVC 风格的身份迁移

第一个实现是 RVC 风格的语音转换，使用 HuBERT 内容特征、RMVPE 音高提取、NSF/VITS generator、multi-period discriminator、mel/adversarial losses，以及可选 retrieval index。

这里的“RVC 模型”是声音专用的。它从目标身份数据集中训练得到。检索索引也是声音专用的，并由同一目标声音的特征构建。内容特征、音高提取、分离模型等预训练组件是可复用基础设施；转换模型和索引才是歌手或说话人专用的工件。

SimpleTuner 应该能够：

1. 复用用户提供的语音转换模型和索引。
2. 如果没有提供模型，则训练语音转换模型。
3. 从目标声音数据构建检索索引。
4. 在训练输出目录下缓存模型、索引和生成音频。
5. 当源数据和变换设置未变化时，在启动时复用缓存工件。
6. 可选地通过 Hub 模型仓库复用或发布语音转换模型。

## 默认行为

默认值比较保守。在这个 workflow 中，音频 backend 是要被转换的扩展音乐，`model.identity_data_dir` 是目标声音数据集，`target.instance_data_dir` 只是生成输出 split 的路径。

| 设置 | 默认值 | 原因 |
| --- | --- | --- |
| `task` | `identity_transfer` | 明确标识该变换。 |
| `method` | `rvc` | 首个支持的声音迁移后端。 |
| `train_if_missing` | `true` | SimpleTuner 应能从目标数据集启动训练声音模型。 |
| `force_retrain` | `false` | 尽量复用有效缓存模型。 |
| `build_index` | `true` | 检索通常能提高身份稳定性并减少泄漏。 |
| `identity_data_dir` | 按需训练时必填 | 指向要迁移到扩展歌曲中的干净目标声音示例。 |
| `identity_audio_mode` | `separate` | 训练前对身份音频运行 Demucs。若身份数据集已是人声 stems，请用 `vocal_only`。 |
| `identity_stem_debug_dir` | 未设置 | 可选目录，用来保存 identity `vocals.wav` 和 `no_vocals.wav` 预览。用它确认 RVC 是否从隔离人声训练，而不是从乐器泄漏训练。 |
| `asset_hub_model_id` | `lj1995/VoiceConversionWebUI` | Default RVC asset repository for HuBERT, RMVPE, and v2 48k pretrained generator/discriminator checkpoints. |
| `model_name` | transform or Hub repo name | Human-readable name saved into the RVC artifact so downloaded caches are identifiable outside their folder name. |
| `sample_rate` | `48000` | Current implementation targets RVC v2 48k assets. Other rates need matching pretrained assets and configs. |
| `training_steps` | `1000` | Runs RVC generator/discriminator fine-tuning during startup. Increase for larger or more varied identity datasets. |
| `batch_size` | `4` | RVC training batch size before distributed sharding. Lower it for memory pressure. |
| `learning_rate` | `1e-4` | Standard RVC AdamW default. |
| `hub_model_id` | 未设置 | 用户未显式启用时，不使用远程声音模型缓存。 |
| `reuse_from_hub` | 设置 `hub_model_id` 时为 `true` | 在花时间按需训练前先检查 Hub。 |
| `push_to_hub` | `false` | 声音模型代表一个声音身份，上传必须显式开启。 |
| `public` | `false` | Hub uploads are private by default. Set this to `true` only when the voice artifact can be published publicly. |
| `audio_mode` | 完整歌曲默认 `separate_convert_remix`，人声 stem 默认 `vocal_only` | 完整混音需要分离；stem 不需要。 |
| `separation_method` | 需要分离时使用 `demucs` | Demucs 是预期默认 stem 分离器。 |
| `timbre_strength` | `1.0` | Controls how strongly the synthesized target voice replaces the source vocal. Lower values blend source and converted vocals. |
| `retrieval_strength` | `0.75` | Blends nearest target-voice content frames from the retrieval index into the generator input. |
| 生成 split 类型 | 主 `audio` 数据集 | 生成数据像普通音频一样训练，不作为 conditioning。 |
| 缓存位置 | `output_dir` 内 | 让生成工件绑定训练运行，并能重启复用。 |
| captions | 默认复制源 captions，除非另有配置 | 新 split 应保留歌词和编曲上下文。 |

如果提供了已有语音转换模型，SimpleTuner 应使用它；只有显式要求或必要工件缺失时才训练新模型。

## Hub 缓存

语音转换模型可能训练成本较高，重复按需训练会变成明显的资源浪费。因此该 transform 应支持可选的 Hub 后端缓存，用于保存声音模型和检索索引。

安全查找顺序：

```text
if local voice-conversion cache matches:
    reuse local model and index
else if hub_model_id is configured and reuse_from_hub is enabled:
    check the Hub repository
    download only if it has a SimpleTuner voice-transform manifest
    reuse only if the manifest matches this transform
else if train_if_missing is enabled:
    train the voice-conversion model
    build the retrieval index
    cache locally
    push to hub only when push_to_hub is true
else:
    stop and ask for a model path or a reusable cache
```

Hub 仓库应使用 SimpleTuner 专用布局，而不是松散文件集合：

```text
config.json
voice_transform/
    manifest.json
    model.safetensors
    features.safetensors
    index.index
```

Manifest 是契约。它应记录目标身份数据集指纹、RVC 训练设置、索引设置、预期采样率、工具版本和 SimpleTuner voice-transform 格式版本。缺少 manifest 或 manifest 与当前 transform 不匹配时，SimpleTuner 不应复用该 Hub 工件。这样可以避免把错误声音模型静默应用到新数据集。

发布应为 opt-in。合理伪配置：

```text
identity_transfer:
    method: rvc
    model:
        train_if_missing: true
        model_name: Target voice RVC
        hub_model_id: org/target-voice-rvc
        reuse_from_hub: true
        push_to_hub: true
        public: false
```

对于私有声音身份，除非有明确授权，否则 Hub 仓库应保持私有。生成音频和模型工件可能有不同共享权利，因此应分别处理上传设置。

## WebUI 配置

RVC 模型训练应该可以通过 WebUI 配置，而不只依赖原始 dataloader JSON。

预期 WebUI 形状是在音频数据集里提供 dataset transform 编辑器：

```text
Audio dataset
    Data transforms
        Add transform: Identity transfer
            Method: RVC
            Audio mode: vocal_only / separate_convert_remix / full_mix_convert
            Train RVC model if missing: on
            Force retrain: off
            Build retrieval index: on
            Hub model id: optional
            Reuse from Hub: on when Hub model id is set
            Push RVC model to Hub: off by default
            Hub repo privacy: private by default
            Caption 和歌词 sidecar：从源音频复制
```

WebUI 应让两种常见设置一眼可见：

- **已经有人声 stem：** 选择 `vocal_only`，保持 Demucs 关闭，并写出生成的人声 stem。
- **有完整歌曲：** 选择 `separate_convert_remix`，使用 Demucs 分离，只转换人声 stem，再和原始伴奏 stem 混回。

界面应明确显示生成音频会成为另一个主音频训练 split。不要把身份迁移展示成 `conditioning_data`，否则用户会以为它是训练期间的配对 conditioning 行为。

## 分布式启动行为

当 SimpleTuner 使用多个 data-parallel rank 启动时，语音克隆启动阶段应该利用可用 GPU，而不是让 rank 0 完成所有工作。

这里有两个不同的分布式阶段：

1. **RVC 模型训练：** 如果 `train_if_missing=true`，没有匹配的本地缓存，也没有匹配的 Hub 工件，则当 `world_size > 1` 时，RVC 训练循环应使用 DDP。每个 rank 应通过普通 distributed sampler 模式接收不同的目标声音 batch。
2. **生成音频准备：** 扩展源输入应按 rank 切分，类似 TextEmbedCache 和 VAECache。每个 rank 只分离、转换并写出自己负责的 shard，然后所有 rank 同步，再继续 metadata discovery。

伪行为：

```text
if world_size > 1:
    if RVC model must be trained:
        train RVC with DDP across all ranks
        save final model and index once

    split expansion inputs by global rank
    each rank generates its own audio shard
    barrier
    rank 0 writes or verifies the combined manifest
    barrier
else:
    train and generate serially
```

最终声音模型只应由一个进程发布到 Hub。最终 manifest 更新也一样。每个 rank 可以独立写出自己的生成结果，只要文件名是确定的且互不重叠。

这样可以避免在多 GPU 系统上浪费 GPU 时间，并让启动行为与 SimpleTuner 现有缓存准备模型保持一致。

## RVC 训练日志

启动阶段的 RVC 训练暂时不应创建 TensorBoard 或 WandB run。这些 logger 是给主 SimpleTuner 训练任务配置的，把它们复用于嵌套语音转换任务会需要额外 run 名称、路径、resume 规则和 artifact 策略。

RVC 阶段仍然可以通过 SimpleTuner 原生训练 logger 报告有用统计：

```text
output_dir/
    logs/
        rvc/
            training_stats.jsonl
            summary.json
```

有用的本地统计包括 generator loss、discriminator loss、mel loss、KL loss、已处理样本数、耗时、DDP world size、缓存命中或未命中原因，以及最终模型来自本地缓存、Hub 缓存还是按需训练。

除非未来实现明确为 RVC transforms 添加外部 logger 集成，否则这些统计仅保存在本地。

## 选择 `audio_mode`

### `vocal_only`

当扩展数据集已经预处理成干净人声 stem 时使用。

```text
source vocal stem -> RVC conversion -> generated vocal stem
```

注意事项：

- 不要对干净 stem 再跑 Demucs，除非你确实有理由。
- caption 应描述人声和歌词；除非之后会混回伴奏，否则不要描述完整乐队编曲。
- 如果主训练模型期望完整歌曲，纯人声生成数据可能会形成不同的数据分布。

### `separate_convert_remix`

当扩展数据集是完整混音歌曲时使用。

```text
source full song
    -> Demucs separates vocals and instrumental stems
    -> RVC converts the vocal stem
    -> converted vocal is remixed with the original instrumental stems
    -> generated full song is added to training
```

这是完整歌曲扩展的首选模式，因为它避免把鼓、贝斯、吉他、房间声和母带痕迹当成声音的一部分去转换。

注意事项：

- Stem 分离可能留下串音、伪影或相位问题。
- 人声 stem 太弱、混响太多或被埋住时，转换后声音可能不稳定。
- Remix 的响度很重要。生成 split 如果一直更响或更小，会影响训练偏向。
- caption 应描述最终混音结果，而不只是源歌曲。

### `full_mix_convert`

仅建议快速测试时使用。

```text
source full song -> RVC conversion over the whole mix -> generated full song
```

它更快，但质量通常更低，可能把乐器也拖进语音转换器，并把不需要的伪影教给最终 LoRA。

## Caption 和歌词策略

生成 split 的 caption 应匹配生成后的音频。

合理默认值：

```text
copy source caption
remove source-vocal identity words when configured
append target-vocal identity or style words when configured
copy lyrics sidecar when lyrics still match
```

当源人声和转换后人声唱的是同一段词时，复制歌词通常是正确的。如果变换改了歌曲、剪掉段落、删除人声或使用非歌词源，则不应复制。

Caption 不能总是盲目复制。如果源 caption 写着“female pop vocal”，但生成结果是男性摇滚音色，就应该调整 caption。变换应支持简单的追加和删除规则，更高级的 caption 重写可以之后叠加。

## 缓存和复用

变换应写入两类缓存：

```text
voice-conversion cache:
    model checkpoint
    retrieval index
    manifest

generated audio cache:
    generated audio files
    captions
    lyrics, when available
    manifest
```

Manifest 应记录身份数据集指纹、变换设置、扩展源数据指纹和工具版本。如果这些值匹配，启动时可以复用现有工件；如果变化，则只重新生成受影响阶段。

## 实用数据集建议

对于 `model.identity_data_dir` 里的目标声音，干净的有声覆盖比单纯时长更重要。

- **快速冒烟测试：** 30-60 秒干净人声可以证明 pipeline 能跑通，但转换声音通常会比较粗糙。
- **可用起点：** 对个人声音数据集来说，5-10 分钟干净、独立的人声是合理的第一目标。
- **歌唱身份：** 如果需要覆盖音高范围、元音、动态、咬字和表现性 phrasing，10-30 分钟会更好。

请使用许多短 clip，而不是一个很长的文件。5-20 秒左右的 clip 更容易检查、分离和复用。当前 RVC trainer 会把身份音频重采样到 48 kHz，并把每个身份文件截断到 `max_seconds_per_file`，默认值是 `180`。如果用户提供一个 30 分钟文件，默认只会使用前三分钟。拆分数据集可以避免意外丢掉有用的人声覆盖。

独立的 [`huggingface-hub-rvc`](https://github.com/SimpleTuner-io/huggingface-hub-rvc) 项目可以在不运行完整 SimpleTuner 训练任务的情况下训练、保存、加载和发布 RVC artifact。在 SimpleTuner 中，`scripts/run_rvc_model.py` 提供了一个直接入口，用于更深入地实验 pipeline 的 RVC 训练和转换部分。若想在主 LoRA 训练前调试身份数据集、Demucs 模式、retrieval strength、transfer strength 或 Hub artifact 复用，请先使用它。

- 身份控制重要时，每个 LoRA 保持一个目标歌手。
- 用干净、较干的人声样本训练语音转换模型。
- 除非目标就是学习合唱混合，否则避免二重唱。
- 扩展歌曲应覆盖不同速度、调性、流派、动态和歌词 phrasing。
- Caption 要足够多样，避免身份 token 和一种编曲永久绑定。
- 长训练前先抽查生成音频。
- 在合并所有内容之前，先分别比较直接训练数据、生成数据和混合训练运行。

## 常见失败模式

| 现象 | 可能原因 |
| --- | --- |
| LoRA 只在一种流派里有效 | 声音身份仍与编曲 caption 或源数据纠缠。 |
| 生成 split 听起来空、相位怪 | 完整歌曲处理中的分离或 remix 伪影。 |
| 乐器像被变声了一样 | 需要分离时使用了 `full_mix_convert`。 |
| 声音模型像学进了乐器 | Identity separation 的人声 stem 混入太多伴奏。设置 `model.identity_stem_debug_dir` 检查保存的 stems，或先预处理更干净的人声 stems 并使用 `identity_audio_mode=vocal_only`。 |
| 人声身份很弱 | 目标数据需要更干净、更多样，或检索索引更强。 |
| Caption 控制不了声音 | Caption 仍含源声音身份，或没有目标身份。 |
| 主模型学到伪影 | 生成音频质量太低或在训练 mix 中占比太高。 |
| 转换后的人声单调或像机器人 | RVC 路径缺少合适的 F0 提取、预训练 generator/discriminator 初始化、对抗训练，或足够干净的目标人声数据。 |

## 与正则化数据的关系

生成的身份迁移数据默认不是正则化数据。

正则化数据通常用于让 LoRA 保留基础模型行为。身份迁移数据用于让 LoRA 在更多上下文中学习目标声音。太多正则化配太少直接身份数据，会削弱身份 token；太多生成数据则可能教会转换伪影。

把它们看成三个独立旋钮：

- 直接目标数据集：最强身份信号
- 生成身份迁移数据集：更广的上下文和风格覆盖
- 正则化数据集：保留基础模型行为

## 状态

本文描述实验性的 `data_transforms` workflow。当前实现会训练或复用 SimpleTuner RVC v2 F0 artifact，从身份音频中提取 HuBERT content features 和 RMVPE pitch，微调预训练 RVC generator/discriminator，构建 retrieval index，生成扩展 split，缓存结果，然后直接进入正常训练，不要求用户手动执行第二个 preprocessing 阶段。

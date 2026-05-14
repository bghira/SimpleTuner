# NSFW 分类器检查

SimpleTuner 提供可选的分类器检查，可在 VAE 缓存预处理期间拒绝样本。此功能是本地过滤工具，不是法律建议、合规系统，也不能保证某个数据集在特定用途下合法或可接受。

## 你的责任

你需要自行判断你的数据集、训练运行、模型输出以及发布或分发计划是否符合适用于你的规则。

这些规则可能包括本地、地区、国家和平台要求，也可能取决于同意、年龄、肖像权、隐私权、公开权、淫秽规则、雇佣或机构政策，以及结果是否描绘或冒充真实人物。法律也会随时间变化，并且因司法管辖区而异。

SimpleTuner 不会替你做这些判断。它不会提醒你的策略不完整，不会检查你的阈值是否符合法律，也不会确认模型输出是否可以安全发布。如果你不确定，请针对你的司法管辖区和使用场景寻求合格法律意见。

## 隐私

NSFW 分类器检查在运行 SimpleTuner 的机器上本地执行。

- 此功能不会把数据集样本发送到第三方审核 API。
- 分类器结果不会转发给第三方。
- `--report_to` 训练遥测选项不会接收 NSFW 分类器结果。
- 报告会以 `nsfw_classifier_report_rank*.json` 存储在实例本地的 VAE 缓存目录中。

唯一需要预期的网络行为是：如果分类器权重尚未存在于本地模型缓存中，Hugging Face 模型加载可能会访问网络。模型本地可用后，分类本身在实例上运行。

## 选择启用

此功能默认关闭。使用以下参数启用：

```bash
--enable_nsfw_check=true
```

检查只应用于 VAE 缓存即将处理的未缓存样本。已有 VAE 缓存会被信任，`skip_file_discovery=vae` 会绕过强制检查，因为 SimpleTuner 会假定你已按自己的策略准备好缓存。

评估数据集不会被扫描。

## 支持的分类器

SimpleTuner 通过 `AutoImageProcessor` 和 `AutoModelForImageClassification` 支持标准 Hugging Face Transformers 图像分类模型。

默认模型为：

```text
Falconsai/nsfw_image_detection:threshold=0.5,AdamCodd/vit-base-nsfw-detector:threshold=0.5
```

你可以提供自己的 CSV 列表：

```bash
--nsfw_check_models="org/model-a:threshold=0.5,org/model-b:threshold=0.7"
```

SimpleTuner 不会为这些分类器启用 `trust_remote_code`，也不会为此功能添加 `timm` 依赖。需要自定义代码或非 Transformers 后端的模型不受此扫描器支持。

## 非 NSFW 用途

尽管选项名称包含 NSFW，此机制并不局限于性内容过滤。如果分类器输出的标签和分数能够清晰映射到 SimpleTuner 预期的不安全/安全标签提示，它也可以用于其他二分类或标签分数检查。

例如，可用于拒绝带有被禁止视觉类别、品牌敏感内容或其他本地数据集策略的样本。你仍然需要验证分类器标签、阈值和投票设置是否符合你的策略。

## 法律背景

成人性内容并非在所有地方都自动违法，SimpleTuner 也不会默认禁止 NSFW 模型训练。但这并不表示某个数据集、输出或部署一定合法。

高风险领域包括：

- 涉及未成年人或看似未成年人的内容。美国 FBI Internet Crime Complaint Center 表示，由生成式 AI 和类似工具创建的儿童性虐待材料是非法的。
- 未经同意的亲密图像、性剥削、骚扰、勒索或未经允许的分发。
- 冒充、再现或误导性描绘真实人物的输出，尤其是用于性、欺诈或损害名誉的场景。FTC 已强调 AI 冒充和 deepfake 欺诈风险。
- Deepfake 披露和透明度规则。例如，EU AI Act 第 50 条包含针对某些构成 deepfake 的 AI 生成或操纵图像、音频或视频内容的透明度义务。
- 合同或平台规则，包括数据集许可、托管服务政策、工作场所规则、支付处理方规则和模型分发条款。

请把分类器视为你自己的审核流程中的一个控制措施，而不是审核流程本身。

## 相关选项

- `--enable_nsfw_check`
- `--nsfw_check_models`
- `--nsfw_check_min_votes`
- `--nsfw_check_backend_types`
- `--nsfw_check_sample_types`
- `--delete_nsfw_images`
- `--nsfw_check_video_frame_count`
- `--nsfw_check_video_frame_selection`
- `--nsfw_check_video_min_flagged_frames`

VAE 缓存集成细节见 [DATALOADER.zh.md#nsfw-classifier-checks-during-vae-caching](DATALOADER.zh.md#nsfw-classifier-checks-during-vae-caching)。

## 参考资料

- [FBI IC3：由生成式 AI 和类似在线工具创建的儿童性虐待材料是非法的](https://www.ic3.gov/PSA/2024/PSA240329)
- [FTC：拟议保护措施以打击 AI 冒充个人](https://www.ftc.gov/news-events/news/press-releases/2024/02/ftc-proposes-new-protections-combat-ai-impersonation-individuals)
- [EU AI Act 第 50 条：透明度义务](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50)

# SimpleTuner Web 界面教程

## 简介

本教程将帮助您开始使用 SimpleTuner Web 界面。

## 安装依赖

对于 Ubuntu 系统，首先安装所需的软件包：

```bash
apt -y install python3.13-venv python3.13-dev
apt -y install libopenmpi-dev openmpi-bin cuda-toolkit-12-8 libaio-dev # 如果您使用 DeepSpeed
apt -y install ffmpeg # 如果训练视频模型
```

## 创建工作区目录

工作区包含您的配置、输出模型、验证图像，以及可能的数据集。

在 Vast 或类似的提供商上，您可以使用 `/workspace/simpletuner` 目录：

```bash
mkdir -p /workspace/simpletuner
export SIMPLETUNER_WORKSPACE=/workspace/simpletuner
cd $SIMPLETUNER_WORKSPACE
```

如果您想在主目录中创建：
```bash
mkdir ~/simpletuner-workspace
export SIMPLETUNER_WORKSPACE=~/simpletuner-workspace
cd $SIMPLETUNER_WORKSPACE
```

## 将 SimpleTuner 安装到您的工作区

创建虚拟环境以安装依赖：

```bash
python3.13 -m venv .venv
. .venv/bin/activate
```

### CUDA 特定依赖

NVIDIA 用户需要使用 CUDA extras 来获取所有正确的依赖：

```bash
pip install -e 'simpletuner[cuda]'
# CUDA 13 / Blackwell users (NVIDIA B-series GPUs):
# pip install -e 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
# 或者，如果您通过 git 克隆：
# pip install -e '.[cuda]'
```

Apple 和 ROCm 硬件用户还有其他 extras 可用，请参阅[安装说明](../INSTALL.md)。

## 启动服务器

使用 SSL 在端口 8080 上启动服务器：

```bash
# 对于 DeepSpeed，我们需要将 CUDA_HOME 指向正确的位置
export CUDA_HOME=/usr/local/cuda-12.8
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib/stubs:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH

simpletuner server --ssl --port 8080
```

现在，在您的网页浏览器中访问 https://localhost:8080。

您可能需要通过 SSH 转发端口，例如：

```bash
ssh -L 8080:localhost:8080 user@remote-server
```

> **提示：** 如果您有现有的配置环境（例如，来自之前的命令行使用），您可以使用 `--env` 启动服务器，以便在服务器准备就绪后自动开始训练：
>
> ```bash
> simpletuner server --ssl --port 8080 --env my-training-config
> ```
>
> 这相当于启动服务器然后在 Web 界面中手动点击"开始训练"，但允许无人值守启动。

## 首次设置：创建管理员账户

首次启动时，SimpleTuner 需要您创建一个管理员账户。当您首次访问 Web 界面时，您将看到一个设置屏幕，提示您创建第一个管理员用户。

输入您的电子邮件、用户名和安全密码。此账户将拥有完整的管理权限。

### 管理用户

设置完成后，您可以从**管理用户**页面管理用户（以管理员身份登录后可从侧边栏访问）：

- **用户**标签页：创建、编辑和删除用户账户。分配权限级别（查看者、研究员、负责人、管理员）。
- **级别**标签页：使用细粒度访问控制定义自定义权限级别。
- **认证提供者**标签页：配置外部认证（OIDC、LDAP）以实现单点登录。
- **注册**标签页：控制新用户是否可以自行注册（默认禁用）。

### 用于自动化的 API 密钥

用户可以从其个人资料或管理面板生成用于脚本访问的 API 密钥。API 密钥使用 `st_` 前缀，可以与 `X-API-Key` 标头一起使用：

```bash
curl -s http://localhost:8080/api/training/status \
  -H 'X-API-Key: st_your_key_here'
```

> **注意：** 对于私有/内部部署，请保持公开注册禁用，并通过管理面板手动创建用户账户。

## 使用 Web 界面

### 引导步骤

页面加载后，您将被询问引导问题以设置您的环境。

#### 配置目录

引入了特殊配置值 `configs_dir`，指向包含所有 SimpleTuner 配置的文件夹，建议将这些配置整理到子目录中 - **Web 界面将为您完成此操作**：

```
configs/
├── an-environment-named-something
│   ├── config.json
│   ├── lycoris_config.json
│   └── multidatabackend-DataBackend-Name.json
```

<img width="788" height="465" alt="image" src="https://github.com/user-attachments/assets/656aa287-3b59-476d-ac45-6ede325fe858" />

##### 从命令行使用迁移

如果您之前在没有 Web 界面的情况下使用过 SimpleTuner，您可以指向现有的 config/ 文件夹，所有环境都将被自动发现。

对于新用户，配置和数据集的默认位置将是 `~/.simpletuner/`，建议将您的数据集移动到有更多空间的地方：

<img width="775" height="454" alt="image" src="https://github.com/user-attachments/assets/39238810-da26-4bde-8fc9-1002251f778a" />


#### 多 GPU 选择和配置

配置默认路径后，您将到达可以配置多 GPU 的步骤（在 Macbook 上的截图）

<img width="755" height="646" alt="image" src="https://github.com/user-attachments/assets/de43a09d-06a7-45c0-8111-7a0b014499c8" />

如果您有多个 GPU，只想使用第二个，这里就是您可以进行设置的地方。

> **多 GPU 用户注意：** 使用多个 GPU 训练时，您的数据集大小要求会成比例增加。有效批次大小计算为 `train_batch_size × num_gpus × gradient_accumulation_steps`。如果您的数据集小于此值，您需要在数据集配置中增加 `repeats` 设置，或在高级设置中启用 `--allow_dataset_oversubscription` 选项。有关更多详细信息，请参阅下面的[批次大小部分](#多-gpu-批次大小考虑)。

#### 创建您的第一个训练环境

如果在您的 `configs_dir` 中没有找到任何现有配置，您将被要求创建**您的第一个训练环境**：

<img width="750" height="1381" alt="image" src="https://github.com/user-attachments/assets/4a3ee88f-c70f-416c-ae5d-6593deb9ca35" />

使用**从示例引导**选择一个示例配置作为起点，或者只需输入一个描述性名称并创建一个随机环境，如果您更喜欢使用设置向导。

### 切换训练环境

如果您有任何现有的配置环境，它们将显示在此下拉菜单中。

否则，我们在引导过程中刚创建的选项将已被选中并激活。

<img width="965" height="449" alt="image" src="https://github.com/user-attachments/assets/d8c73cef-ecbb-4229-ad54-9ccd55f8175a" />

使用**管理配置**进入 `环境` 标签页，在那里可以找到您的环境、数据加载器和其他配置的列表。

### 配置向导

我努力提供了一个全面的设置向导，帮助您以简洁的方式配置一些最重要的设置以开始使用。

<img width="394" height="286" alt="image" src="https://github.com/user-attachments/assets/21e99854-1d75-4ba9-8be6-15e715d77f4e" />

在左上角导航菜单中，向导按钮将带您进入选择对话框：

<img width="1186" height="1756" alt="image" src="https://github.com/user-attachments/assets/f6d4ac57-e3f6-4060-a4d3-b7f0829d7350" />

然后提供所有内置模型变体。每个变体将预先启用所需的设置，如注意力遮罩或扩展标记限制。

#### LoRA 模型选项

如果您希望训练 LoRA，您可以在这里设置模型量化选项。

一般来说，除非您正在训练 Stable Diffusion 类型的模型，否则建议使用 int8-quanto，因为它不会损害质量，并允许更高的批次大小。

一些小型模型如 Cosmos2、Sana 和 PixArt 真的不喜欢被量化。

<img width="1106" height="1464" alt="image" src="https://github.com/user-attachments/assets/0284d987-6060-4692-934a-0905ef2d5ca1" />

#### 全秩训练

不鼓励全秩训练，因为对于相同的数据集，它通常比 LoRA/LyCORIS 花费更长时间且消耗更多资源。

但是，如果您确实希望训练完整检查点，您可以在这里配置 DeepSpeed ZeRO 阶段，这对于 Auraflow、Flux 等更大的模型是必需的。

支持 FSDP2，但在此向导中不可配置。只需保持 DeepSpeed 禁用，如果您希望使用它，稍后手动配置 FSDP2

<img width="1097" height="1278" alt="image" src="https://github.com/user-attachments/assets/60475318-facd-4da1-a2a1-67cecff18e04" />


#### 您想训练多长时间？

您需要决定是以 epochs 还是 steps 来衡量训练时间。最终结果都差不多，尽管有些人会对其中一种方式产生偏好。

<img width="1136" height="1091" alt="image" src="https://github.com/user-attachments/assets/9146cdcd-f277-45e5-92cb-f74f23039d51" />

#### 通过 Hugging Face Hub 分享您的模型

可选地，您可以将最终*和*中间检查点发布到 [Hugging Face Hub](https://hf.co)，但您需要一个账户 - 您可以通过向导或发布标签页登录到 Hub。无论如何，您随时可以改变主意并启用或禁用它。

如果您选择发布模型，请注意选择 `私有仓库`，如果您不希望您的模型被公众访问。

<img width="1090" height="859" alt="image" src="https://github.com/user-attachments/assets/d1f86b6b-b0d5-4caa-b3ff-6bd106928094" />

#### 模型验证

如果您希望训练器定期生成图像，您可以在向导的这一步配置单个验证提示。向导完成后，可以在 `验证与输出` 标签页中配置多个提示库。

想要将验证外包给您自己的脚本或服务？在向导完成后，在验证标签页中将**验证方法**切换为 `external-script` 并提供 `--validation_external_script`。您可以使用占位符将训练上下文传递到脚本中，如 `{local_checkpoint_path}`、`{global_step}`、`{tracker_run_name}`、`{tracker_project_name}`、`{model_family}`、`{huggingface_path}`，以及任何 `validation_*` 配置值（例如 `validation_num_inference_steps`、`validation_guidance`、`validation_noise_scheduler`）。启用 `--validation_external_background` 以即发即忘而不阻塞训练。

需要在检查点写入磁盘时立即触发钩子？使用 `--post_checkpoint_script` 在每次保存后立即触发脚本（在上传开始之前）。它接受相同的占位符，`{remote_checkpoint_path}` 留空。

如果您想保留 SimpleTuner 的内置发布提供者（或 Hugging Face Hub 上传），但仍想使用远程 URL 触发您自己的自动化，请改用 `--post_upload_script`。它在每次上传后运行一次，使用占位符 `{remote_checkpoint_path}`、`{local_checkpoint_path}`、`{global_step}`、`{tracker_run_name}`、`{tracker_project_name}`、`{model_family}`、`{huggingface_path}`。SimpleTuner 不捕获脚本的输出 - 直接从您的脚本发出任何跟踪器更新。

示例钩子：

```bash
--post_upload_script='/opt/hooks/notify.sh {remote_checkpoint_path} {tracker_project_name} {tracker_run_name}'
```

其中 `notify.sh` 将 URL 发布到您的跟踪器 Web API。随意适配到 Slack、自定义仪表板或任何其他集成。

工作示例：`simpletuner/examples/external-validation/replicate_post_upload.py` 演示了使用 `{remote_checkpoint_path}`、`{model_family}`、`{model_type}`、`{lora_type}` 和 `{huggingface_path}` 在上传后触发 Replicate 推理。

另一个示例：`simpletuner/examples/external-validation/wavespeed_post_upload.py` 调用 WaveSpeed API 并轮询结果，使用相同的占位符。

Flux 专用示例：`simpletuner/examples/external-validation/fal_post_upload.py` 调用 fal.ai Flux LoRA 端点；它需要 `FAL_KEY` 并且仅在 `model_family` 包含 `flux` 时运行。

本地 GPU 示例：`simpletuner/examples/external-validation/use_second_gpu.py` 在另一个 GPU 上运行 Flux LoRA 推理（默认为 `cuda:1`），即使没有上传发生也可以使用。

<img width="1101" height="1357" alt="image" src="https://github.com/user-attachments/assets/97bdd3f1-b54c-4087-b4d5-05da8b271751" />

#### 记录训练统计

如果您希望将训练统计发送到目标 API，SimpleTuner 支持多个目标 API。

注意：您的任何个人数据、训练日志、标题或数据都**永远不会**发送给 SimpleTuner 项目开发者。您的数据控制权在**您**手中。

<img width="1099" height="1067" alt="image" src="https://github.com/user-attachments/assets/c9be9a20-12ad-402a-9605-66ba5771e630" />

#### 数据集配置

此时，您可以决定是保留任何现有数据集，还是通过数据集创建向导创建新配置（保留任何其他配置不变），点击后将出现向导。

<img width="1103" height="877" alt="image" src="https://github.com/user-attachments/assets/3d3cc391-52ed-422e-a4a1-676ca342df10" />

##### 数据集向导

如果您选择创建新数据集，您将看到以下向导，它将引导您完成添加本地或云数据集的过程。

<img width="1110" height="857" alt="image" src="https://github.com/user-attachments/assets/3719e0f5-774e-461d-be02-902e08a679f6" />

<img width="1082" height="1255" alt="image" src="https://github.com/user-attachments/assets/ac38a3de-364a-447f-a734-cab2bdd5338d" />

对于本地数据集，您可以使用**浏览目录**按钮访问数据集浏览器模态框。

<img width="1201" height="1160" alt="image" src="https://github.com/user-attachments/assets/66a333d0-30fa-45d1-a5b2-1e859d789677" />

如果您在引导过程中正确指向了数据集目录，您将在这里看到您的内容。

点击您希望添加的目录，然后点击**选择目录**。

<img width="907" height="709" alt="image" src="https://github.com/user-attachments/assets/1d482655-158a-4e3f-93b7-ef158396813c" />

之后，您将被引导配置分辨率值和裁剪。

**注意**：SimpleTuner 不会*放大*图像，因此请确保它们至少与您配置的分辨率一样大。

当您到达配置标题的步骤时，请**仔细考虑**哪个选项是正确的。

如果您只想使用单个触发词，那就是**实例提示**选项。

<img width="1146" height="896" alt="image" src="https://github.com/user-attachments/assets/6252bf9a-5e68-41c6-8a95-906993f2f546" />

##### 可选：从浏览器上传数据集

如果您的图像和标题还没有在服务器上，数据集向导现在在**浏览目录**旁边包含一个**上传**按钮。您可以：

- 在配置的数据集目录下创建一个新的子文件夹，然后上传单个文件或 ZIP（接受图像以及 .txt/.jsonl/.csv 元数据）。
- 让 SimpleTuner 将 ZIP 解压到该文件夹中（适用于本地后端的大小；非常大的档案将被拒绝）。
- 立即在浏览器中选择刚上传的文件夹，无需离开界面即可继续向导。

#### 学习率、批次大小和优化器

完成数据集向导后（或如果您选择保留现有数据集），您将获得优化器/学习率和批次大小的预设。

这些只是帮助新手在前几次训练运行中做出更好选择的起点 - 对于有经验的用户，使用**手动配置**获得完全控制。

**注意**：如果您计划稍后使用 DeepSpeed，这里的优化器选择并不重要。

##### 多 GPU 批次大小考虑 {#多-gpu-批次大小考虑}

使用多个 GPU 训练时，请注意您的数据集必须适应**有效批次大小**：

```
effective_batch_size = train_batch_size × num_gpus × gradient_accumulation_steps
```

如果您的数据集小于此值，SimpleTuner 将引发错误并提供具体指导。您可以：
- 减少批次大小
- 增加数据集配置中的 `repeats` 值
- 在高级设置中启用**允许数据集超额订阅**以自动调整重复次数

有关数据集大小的更多详细信息，请参阅 [DATALOADER.md](../DATALOADER.md#multi-gpu-training-and-dataset-sizing)。

<img width="1118" height="1015" alt="image" src="https://github.com/user-attachments/assets/25d5650d-e77b-42fe-b749-06c0ec92b1e2" />

#### 内存优化预设

为了在消费级硬件上更容易设置，每个模型都包含自定义预设，允许选择轻度、平衡或激进的内存节省。

在**训练**标签页的**内存优化**部分，您将找到**加载预设**按钮：

<img width="1048" height="940" alt="image" src="https://github.com/user-attachments/assets/804e84f6-7eb8-493e-95d2-a89d930bafa5" />

这将打开此界面：

<img width="1048" height="940" alt="image" src="https://github.com/user-attachments/assets/775aaee5-c3c0-4659-bbea-ebb39e3eb098" />


#### 审查并保存

如果您对所有选定的值感到满意，请继续**完成**向导。

然后您将看到您的新环境已激活并准备好进行训练！

在大多数情况下，这些设置将是您需要配置的全部内容。您可能想要添加额外的数据集或调整其他设置。

<img width="1096" height="1403" alt="image" src="https://github.com/user-attachments/assets/29fd0bb3-aab2-4455-9612-583ed949ce64" />

在**环境**页面上，您将看到新配置的训练任务，以及下载或复制配置的按钮，如果您希望将其用作模板。

<img width="1881" height="874" alt="image" src="https://github.com/user-attachments/assets/33c0cafa-3fd8-40ee-b6fa-3704b6e698da" />

**注意**：**默认**环境是特殊的，不建议用作通用训练环境；它的设置可以自动合并到任何启用了**使用环境默认值**选项的环境中：

<img width="1521" height="991" alt="image" src="https://github.com/user-attachments/assets/9d18b0c1-608e-4ab2-be14-65b98907ec69" />

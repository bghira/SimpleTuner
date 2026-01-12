# 分布式训练（多节点）

本文记录了用于 SimpleTuner 的 4 路 8xH100 集群配置注意事项*。

> *本指南不包含完整的端到端安装流程，而是作为参考，在遵循 [INSTALL](INSTALL.md) 文档或 [quickstart](QUICKSTART.md) 指南时提供考虑点。

## 存储后端

多节点训练默认需要在节点间共享 `output_dir`。


### Ubuntu NFS 示例

这是一个可用于入门的基础存储示例。

#### 在写入检查点的 “master” 节点上

**1. 安装 NFS 服务器包**

```bash
sudo apt update
sudo apt install nfs-kernel-server
```

**2. 配置 NFS 导出**

编辑 NFS 导出文件以共享目录：

```bash
sudo nano /etc/exports
```

在文件末尾添加以下行（将 `slave_ip` 替换为从节点的实际 IP）：

```
/home/ubuntu/simpletuner/output slave_ip(rw,sync,no_subtree_check)
```

*若允许多个从节点或整个子网，可使用：*

```
/home/ubuntu/simpletuner/output subnet_ip/24(rw,sync,no_subtree_check)
```

**3. 导出共享目录**

```bash
sudo exportfs -a
```

**4. 重启 NFS 服务**

```bash
sudo systemctl restart nfs-kernel-server
```

**5. 验证 NFS 服务状态**

```bash
sudo systemctl status nfs-kernel-server
```

---

#### 在发送优化器与其他状态的从节点上

**1. 安装 NFS 客户端包**

```bash
sudo apt update
sudo apt install nfs-common
```

**2. 创建挂载点目录**

确保目录存在（按你的设置通常已存在）：

```bash
sudo mkdir -p /home/ubuntu/simpletuner/output
```

*注记:* 如果目录已有数据请先备份，因为挂载后原内容会被隐藏。

**3. 挂载 NFS 共享**

将 master 的共享目录挂载到 slave 的本地目录（将 `master_ip` 替换为 master 的 IP）：

```bash
sudo mount master_ip:/home/ubuntu/simpletuner/output /home/ubuntu/simpletuner/output
```

**4. 验证挂载**

```bash
mount | grep /home/ubuntu/simpletuner/output
```

**5. 测试写入权限**

创建测试文件以确认写权限：

```bash
touch /home/ubuntu/simpletuner/output/test_file_from_slave.txt
```

然后在 master 上检查 `/home/ubuntu/simpletuner/output` 是否出现该文件。

**6. 设置开机自动挂载**

为了重启后仍保持挂载，将其加入 `/etc/fstab`：

```bash
sudo nano /etc/fstab
```

在末尾添加以下行：

```
master_ip:/home/ubuntu/simpletuner/output /home/ubuntu/simpletuner/output nfs defaults 0 0
```

---

#### **其他注意事项：**

- **用户权限：** 确保 `ubuntu` 用户在两台机器上的 UID/GID 一致，保证文件权限一致。可用 `id ubuntu` 查看。

- **防火墙设置：** 若启用防火墙，需允许 NFS 流量。在 master 上运行：

  ```bash
  sudo ufw allow from slave_ip to any port nfs
  ```

- **时钟同步：** 分布式环境应保持系统时钟同步，可使用 `ntp` 或 `systemd-timesyncd`。

- **测试 DeepSpeed 检查点：** 运行一个小型 DeepSpeed 作业确认检查点正确写入 master 目录。


## 数据加载器配置

超大数据集的高效管理是一个挑战。SimpleTuner 会自动在各节点间对数据集分片，并将预处理分发到集群中每张 GPU，同时使用异步队列与线程保持吞吐。

### 多 GPU 训练的数据集规模

当跨多个 GPU 或节点训练时，数据集必须包含足够样本以满足**有效批大小**：

```
effective_batch_size = train_batch_size × num_gpus × gradient_accumulation_steps
```

**计算示例：**

| 配置 | 计算 | 有效批大小 |
|--------------|-------------|---------------------|
| 1 节点、8 GPU、batch_size=4、grad_accum=1 | 4 × 8 × 1 | 32 样本 |
| 2 节点、16 GPU、batch_size=8、grad_accum=2 | 8 × 16 × 2 | 256 样本 |
| 4 节点、32 GPU、batch_size=8、grad_accum=1 | 8 × 32 × 1 | 256 样本 |

每个纵横比桶都必须包含至少这么多样本（考虑 `repeats`），否则训练会失败并给出详细错误信息。

#### 小数据集解决方案

若数据集小于有效批大小：

1. **降低批大小** - 减小 `train_batch_size` 以降低内存需求
2. **减少 GPU 数量** - 使用更少 GPU（训练会变慢）
3. **增加 repeats** - 在 [dataloader 配置](DATALOADER.md#repeats) 中设置 `repeats`
4. **启用自动超订** - 使用 `--allow_dataset_oversubscription` 自动调整 repeats

`--allow_dataset_oversubscription`（见 [OPTIONS.md](OPTIONS.md#--allow_dataset_oversubscription)）会自动计算并应用所需的最小 repeats，非常适合原型验证或小数据集实验。

### 图像扫描/发现过慢

**discovery** 后端目前限制纵横比桶数据采集为单节点。这在超大数据集中会非常慢，因为每张图像都必须从存储读取以获取几何信息。

为规避该问题，建议使用 [parquet metadata_backend](DATALOADER.md#parquet-caption-strategy-json-lines-datasets)。这样可按你的方式预处理数据。正如链接文档所示，parquet 表包含 `filename`、`width`、`height`、`caption` 列，可高效完成分桶。


### 存储空间

超大数据集（尤其使用 T5-XXL 文本编码器）会对原始数据、图像嵌入与文本嵌入占用巨量存储。

#### 云存储

使用 Cloudflare R2 等提供商可以以较低费用存储超大数据集。

关于在 `multidatabackend.json` 中配置 `aws` 类型，见 [dataloader 配置指南](DATALOADER.md#local-cache-with-cloud-dataset)。

- 图像数据可存于本地或 S3
  - 若在 S3，预处理速度受网络带宽影响
  - 若在本地，则**训练**时无法利用 NVMe 吞吐
- 图像嵌入与文本嵌入可分别存于本地或云端
  - 将嵌入放在云端对训练速率影响很小，因为会并行获取

理想情况下，所有图像与嵌入都存于云存储桶，可显著降低预处理与恢复训练的风险。

#### 按需 VAE 编码

当存储 VAE 潜变量缓存不现实（存储受限或共享存储访问慢）时，可使用 `--vae_cache_disable`。它会完全禁用 VAE 缓存，训练时按需编码图像。

这会增加 GPU 计算负载，但显著降低存储需求与缓存潜变量的网络 I/O。

#### 保留文件系统扫描缓存

如果数据集过大导致扫描新图像成为瓶颈，可在每个 dataloader 配置条目添加 `preserve_data_backend_cache=true`，避免重新扫描后端。

**注记**：此时应使用 `image_embeds` 数据后端类型（[详情](DATALOADER.md#local-cache-with-cloud-dataset)），以便在预处理被中断时将缓存列表独立保存。这将避免启动时重新扫描**图像列表**。

#### 数据压缩

在 `config.json` 中添加以下内容以启用数据压缩：

```json
{
    ...
    "--compress_disk_cache": true,
    ...
}
```

这会使用内联 gzip 减少大型文本/图像嵌入占用的磁盘空间。

## 通过 🤗 Accelerate 配置

使用 `accelerate config`（`/home/user/.cache/huggingface/accelerate/default_config.yaml`）部署 SimpleTuner 时，这些设置会优先于 `config/config.env`。

一个不包含 DeepSpeed 的 Accelerate default_config 示例：

```yaml
# this should be updated on EACH node.
machine_rank: 0
# Everything below here is the same on EACH node.
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
dynamo_config:
  dynamo_backend: NO
enable_cpu_affinity: false
main_process_ip: 10.0.0.100
main_process_port: 8080
main_training_function: main
mixed_precision: bf16
num_machines: 4
num_processes: 32
rdzv_backend: static
same_network: false
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### DeepSpeed

本文不展开太多细节，详见 [专页](DEEPSPEED.md)。

在多节点上使用 DeepSpeed 优化训练时，尽可能选择最低的 ZeRO 等级是**关键**。

例如，80G NVIDIA GPU 可使用 ZeRO 级别 1 的卸载来训练 Flux，从而显著降低开销。

添加如下行：

```yaml
# Update this from MULTI_GPU to DEEPSPEED
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_multinode_launcher: standard
  gradient_accumulation_steps: 1
  gradient_clipping: 0.01
  zero3_init_flag: false
  zero_stage: 1
```

### torch compile 优化

为了获得额外性能（但可能带来兼容性问题），可在每个节点的 YAML 配置中添加如下内容启用 torch compile：

```yaml
dynamo_config:
  # Update this from NO to INDUCTOR
  dynamo_backend: INDUCTOR
  dynamo_mode: max-autotune
  dynamo_use_dynamic: false
  dynamo_use_fullgraph: false
```

## 预期性能

- 4 台 H100 SXM5 节点通过本地网络连接
- 每节点 1TB 内存
- 训练缓存来自同区域 S3 兼容数据后端（Cloudflare R2）的共享流式读取
- 每个加速器批大小 **8**，不使用梯度累积
  - 总有效批大小为 **256**
- 分辨率 1024px，并启用分桶
- **速度**：使用 1024x1024 数据全秩训练 Flux.1-dev（12B）时，每步约 15 秒

降低批大小、降低分辨率并启用 torch compile 可使速度达到**每秒迭代数**级别：

- 分辨率降至 512px 且禁用分桶（仅方形裁剪）
- 将 DeepSpeed 优化器从 AdamW 换为 Lion fused
- 启用 torch compile 的 max-autotune
- **速度**：每秒 2 次迭代

## 分布式训练注意事项

- 每个节点必须拥有相同数量的加速器
- 仅 LoRA/LyCORIS 可量化，因此全量分布式模型训练需要 DeepSpeed
- 这是高成本操作，较大的批大小可能比预期更慢，需要增加 GPU 数量。应仔细平衡成本预算。
- （DeepSpeed）使用 ZeRO 3 训练时可能需要禁用验证
- （DeepSpeed）使用 ZeRO 3 保存时会生成分片检查点，但 ZeRO 1 和 2 的行为符合预期
- （DeepSpeed）需要使用 DeepSpeed 的 CPU 优化器，因为它负责处理优化器状态的分片与卸载

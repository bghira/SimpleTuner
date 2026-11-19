# 数据后端工厂架构

## 数据集构建

工厂验证标识符、展开生成的数据集，并在依赖方之前处理元数据源。构建器打开数据源和元数据提供方。`DatasetRuntime` 绑定提供方、解析后的配置、关联数据集和产物缓存。`StateTracker` 注册这些句柄，并为旧调用方保留兼容条目。

```text
FactoryRegistry → builders → DatasetRuntime
                              ├─ BaseDataBackend
                              ├─ MetadataBackend
                              ├─ TextEmbeddingCache / VAECache
                              └─ training dataloader
```

## 提供方契约

数据源提供方读取文件或对象。元数据提供方发布已接受的样本及其裁剪几何信息。`describe()` 返回可用的计数和版本，`read_page()` 返回有限的页面，`read_samples()` 按引用顺序读取指定字段并保留重复项。样本引用遵循提供方的标识规则。

使用 `iter_preparation_batches()` 和 `iter_preparation_text_records()` 进行预处理。提供方独立于训练重复与填充，将已接受样本分配给工作进程。产物可存放在独立存储中。在线文本编码使用当前批次。

本地发现、AWS 和 CSV 共用旧版元数据和调度实现。Hugging Face 可按 Arrow 分片存储预处理变换。Webshart 原生 Parquet 使用索引化成员视图、按分片保存的受支持裁剪变换，以及可重建的训练轮次游标。

## 训练与恢复

训练器从运行时请求已调度的批次。混合器选择数据集，预取器可提前读取批次。训练器在消耗批次后确认进度。检查点记录已消耗位置和混合器状态。原生恢复要求使用相同的成员视图、采样参数和分布式布局。

## 验证

添加提供方时，通过运行时检验引用稳定性、按请求顺序投影字段、预处理覆盖率、取消和重新打开。原生调度需要验证游标重放，也应测试旧视图仍被读取时发布新版本。添加产物能力时，检查预处理和训练采用相同的缓存键。

```bash
.venv/bin/python -m unittest -v -f tests.test_dataset_runtime tests.test_webshart_parquet_backend tests.test_vae_cache_traversal
```

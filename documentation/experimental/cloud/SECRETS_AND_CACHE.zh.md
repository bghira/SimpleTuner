# 密钥提供方与缓存系统

本文介绍 SimpleTuner 在云端训练中的密钥管理系统与缓存基础设施。

## 概览

SimpleTuner 使用分层的密钥管理系统（多种提供方后端）以及内存缓存层来优化性能。

## 密钥提供方

`SecretsManager` 类按优先级串联多个密钥提供方。检索密钥时，会按顺序检查各提供方，直到找到值为止。

### 优先级顺序

1. **环境变量**（最高优先级）
2. **文件密钥**
3. **AWS Secrets Manager**
4. **HashiCorp Vault**（最低优先级）

该顺序保证环境变量可以覆盖其他来源，适用于：
- 本地开发覆盖
- 容器部署中的密钥注入
- CI/CD 流水线

### 常用密钥键

<details>
<summary>密钥常量</summary>

```python
REPLICATE_API_TOKEN  # Replicate API 认证
HF_TOKEN             # HuggingFace Hub Token
CLOUD_WEBHOOK_SECRET # Webhook HMAC 校验密钥
```
</details>

---

## 提供方配置

### 1. 环境变量（默认）

环境变量提供方始终可用，且优先级最高。

**键规范化规则：**
- 键全部转为大写
- 将连字符（`-`）转换为下划线（`_`）
- 将点号（`.`）转换为下划线（`_`）

**可选前缀：**

<details>
<summary>Python 配置示例</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import EnvironmentSecretProvider

# 无前缀
provider = EnvironmentSecretProvider()
# 键 "replicate-api-token" -> REPLICATE_API_TOKEN

# 有前缀
provider = EnvironmentSecretProvider(prefix="SIMPLETUNER")
# 键 "replicate-api-token" -> SIMPLETUNER_REPLICATE_API_TOKEN
```
</details>

**用法：**

<details>
<summary>通过环境变量设置密钥</summary>

```bash
# 通过环境变量设置密钥
export REPLICATE_API_TOKEN="r8_your_token_here"
export HF_TOKEN="hf_your_token_here"
export CLOUD_WEBHOOK_SECRET="your_webhook_secret"
```
</details>

### 2. 文件密钥

密钥可以保存为权限受限的 JSON 或 YAML 文件。

**默认路径（按顺序检查）：**
- `~/.simpletuner/secrets.json`
- `~/.simpletuner/secrets.yaml`
- `~/.simpletuner/secrets.yml`

<details>
<summary>文件格式示例（JSON/YAML）</summary>

**文件格式（JSON）：**

```json
{
  "REPLICATE_API_TOKEN": "r8_your_token_here",
  "HF_TOKEN": "hf_your_token_here",
  "CLOUD_WEBHOOK_SECRET": "your_webhook_secret"
}
```

**文件格式（YAML）：**

```yaml
REPLICATE_API_TOKEN: r8_your_token_here
HF_TOKEN: hf_your_token_here
CLOUD_WEBHOOK_SECRET: your_webhook_secret
```
</details>

<details>
<summary>手动配置</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import FileSecretProvider

# 使用自定义文件路径
provider = FileSecretProvider(file_path="/path/to/secrets.json")
```
</details>

**安全提示：** 通过 API 保存密钥时，文件会以 `chmod 0o600`（仅所有者可读写）创建。

### 3. AWS Secrets Manager

在生产环境中，AWS Secrets Manager 提供安全、集中式的密钥存储。

<details>
<summary>环境变量</summary>

```bash
# 必填：AWS Secrets Manager 中的密钥名称
export SIMPLETUNER_AWS_SECRET_NAME="simpletuner/production"

# 可选：AWS 区域（未设置则使用默认值）
export AWS_DEFAULT_REGION="us-west-2"
```
</details>

<details>
<summary>AWS 密钥格式</summary>

AWS Secrets Manager 中的密钥应为 JSON 对象：

```json
{
  "REPLICATE_API_TOKEN": "r8_your_token_here",
  "HF_TOKEN": "hf_your_token_here",
  "CLOUD_WEBHOOK_SECRET": "your_webhook_secret"
}
```
</details>

<details>
<summary>手动配置</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import AWSSecretsManagerProvider

provider = AWSSecretsManagerProvider(
    secret_name="simpletuner/production",
    region_name="us-west-2"
)
```
</details>

<details>
<summary>所需 AWS IAM 权限</summary>

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "secretsmanager:GetSecretValue",
      "Resource": "arn:aws:secretsmanager:REGION:ACCOUNT:secret:simpletuner/*"
    }
  ]
}
```
</details>

### 4. HashiCorp Vault

适用于使用 HashiCorp Vault 进行密钥管理的企业环境。

**必需依赖：**

```bash
pip install hvac
```

<details>
<summary>环境变量</summary>

```bash
# 必填：Vault 服务器 URL
export VAULT_ADDR="https://vault.example.com:8200"

# 必填：Vault 认证 Token
export VAULT_TOKEN="s.your_vault_token"

# 可选：密钥路径（默认："simpletuner"）
export SIMPLETUNER_VAULT_PATH="simpletuner"
```
</details>

<details>
<summary>手动配置</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import HashiCorpVaultProvider

provider = HashiCorpVaultProvider(
    url="https://vault.example.com:8200",
    token="s.your_vault_token",
    path="simpletuner",
    mount_point="secret"  # KV 挂载点（默认："secret"）
)
```
</details>

<details>
<summary>Vault KV 设置</summary>

```bash
# KV v2（推荐）
vault kv put secret/simpletuner \
    REPLICATE_API_TOKEN="r8_your_token" \
    HF_TOKEN="hf_your_token"

# KV v1
vault write secret/simpletuner \
    REPLICATE_API_TOKEN="r8_your_token" \
    HF_TOKEN="hf_your_token"
```

提供方会先尝试 KV v2，失败后回退到 v1。
</details>

---

## 使用 Secrets Manager

### 基础用法

<details>
<summary>获取密钥</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import get_secrets_manager

# 获取单例实例
secrets = get_secrets_manager()

# 获取任意密钥
api_token = secrets.get("REPLICATE_API_TOKEN")
api_token = secrets.get("REPLICATE_API_TOKEN", default="fallback_value")

# 便捷方法
replicate_token = secrets.get_replicate_token()
hf_token = secrets.get_hf_token()
webhook_secret = secrets.get_webhook_secret()
```
</details>

### 保存密钥（仅文件提供方）

<details>
<summary>设置与删除密钥</summary>

```python
secrets = get_secrets_manager()

# 保存密钥（写入 ~/.simpletuner/secrets.json）
secrets.set_secret("MY_CUSTOM_KEY", "my_value")

# Replicate Token 便捷方法
secrets.set_replicate_token("r8_new_token")

# 删除密钥
secrets.delete_secret("MY_CUSTOM_KEY")
```
</details>

### 自定义提供方配置

<details>
<summary>自定义配置示例</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import SecretsManager

# 重置单例以应用自定义配置
SecretsManager.reset()

# 创建自定义配置
secrets = SecretsManager(config={
    "file_path": "/custom/path/secrets.json",
    "aws_secret_name": "myapp/secrets",
    "aws_region": "eu-west-1",
    "vault_url": "https://vault.internal:8200",
    "vault_token": "s.token",
    "vault_path": "myapp/simpletuner"
})
```
</details>

---

## 缓存系统

SimpleTuner 使用内存中的 TTL（Time-To-Live）缓存来减少数据库查询并提高对高频数据的响应速度。

### TTLCache 特性

- **使用 `RLock` 的线程安全操作**
- **每键 TTL**，并可配置默认值
- **自动清理**过期条目
- **容量到达上限时 LRU 驱逐**
- **基于前缀的失效**，用于批量清除缓存

### 缓存配置

<details>
<summary>TTLCache 初始化</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import TTLCache

cache = TTLCache[str](
    default_ttl=300.0,      # 默认 TTL 5 分钟
    max_size=1000,          # 超过后触发驱逐
    cleanup_interval=60.0   # 每 60 秒清理过期条目
)
```
</details>

### 全局缓存实例

SimpleTuner 维护两个全局缓存实例：

| 缓存 | 默认 TTL | 最大容量 | 目的 |
|-------|-------------|----------|---------|
| Provider Config Cache | 300 秒（5 分钟） | 100 | 提供方设置、Webhook URL、成本限制 |
| User Permission Cache | 60 秒（1 分钟） | 500 | 用户权限（TTL 更短以提高安全性） |

<details>
<summary>访问全局缓存</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import (
    get_provider_config_cache,
    get_user_permission_cache
)

# 访问全局缓存
provider_cache = get_provider_config_cache()
user_cache = get_user_permission_cache()
```
</details>

### 基本缓存操作

<details>
<summary>缓存操作参考</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import TTLCache

cache = TTLCache[dict](default_ttl=120.0)

# 设置值（使用默认 TTL）
cache.set("my_key", {"data": "value"})

# 设置自定义 TTL
cache.set("my_key", {"data": "value"}, ttl=60.0)

# 获取值（过期或不存在返回 None）
value = cache.get("my_key")

# 未命中则计算并写入
value = cache.get_or_set("my_key", lambda: compute_value())

# 异步版本
value = await cache.get_or_set_async("my_key", async_compute_value)

# 删除特定键
cache.delete("my_key")

# 清空所有条目
count = cache.clear()

# 获取缓存统计
stats = cache.stats()
# {"size": 42, "max_size": 1000, "expired_count": 3, "default_ttl": 120.0}
```
</details>

---

## 提供方元数据缓存

提供方配置（Webhook URL、成本限制、硬件信息）会被缓存以降低数据库负载。

### 缓存键格式

```
provider:{provider_name}:config
```

示例：`provider:replicate:config`

### 缓存行为

**读取（`get_provider_config`）：**
1. 查询缓存 `provider:{name}:config`
2. 命中且未过期则直接返回
3. 未命中/过期则从数据库加载
4. 以 5 分钟 TTL 写入缓存
5. 返回结果

**写入（`save_provider_config`）：**
1. 写入数据库
2. 失效 `provider:{name}:*`（前缀失效）

### 缓存失效

<details>
<summary>失效示例</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import (
    invalidate_provider_cache,
    invalidate_user_cache
)

# 失效某提供方的所有缓存
invalidate_provider_cache("replicate")

# 失效某用户的所有缓存
invalidate_user_cache(user_id=123)
```
</details>

### 基于前缀的失效

<details>
<summary>前缀失效示例</summary>

```python
cache = get_provider_config_cache()

# 失效所有以 "provider:replicate:" 开头的键
count = cache.invalidate_prefix("provider:replicate:")
```
</details>

---

## 硬件信息缓存

硬件价格信息单独缓存，可手动失效。

### 缓存行为

<details>
<summary>硬件信息缓存操作</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.replicate_client import (
    get_hardware_info,
    get_hardware_info_async,
    clear_hardware_info_cache,
    set_hardware_info
)

# 获取硬件信息（使用缓存或默认值）
info = get_hardware_info()
info = await get_hardware_info_async(store)

# 清空缓存以强制从配置重新加载
clear_hardware_info_cache()

# 手动设置硬件信息（更新缓存）
set_hardware_info({
    "gpu-l40s": {"name": "L40S (48GB)", "cost_per_second": 0.000975}
})
```
</details>

### 默认硬件价格

<details>
<summary>默认硬件信息</summary>

```python
DEFAULT_HARDWARE_INFO = {
    "gpu-l40s": {"name": "L40S (48GB)", "cost_per_second": 0.000975},
    "gpu-a100-large": {"name": "A100 (80GB)", "cost_per_second": 0.001400},
}
```
</details>

---

## 基于装饰器的缓存

缓存函数结果时使用 `@cached` 装饰器：

<details>
<summary>cached 装饰器示例</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import cached

@cached(ttl=120, key_prefix="mymodule")
def get_user(user_id: int) -> dict:
    # 高成本数据库查询
    return {"id": user_id, "name": "..."}

@cached(ttl=60)
async def fetch_external_data() -> dict:
    # 异步 API 调用
    return await api_client.get("/data")

# 缓存键由函数名和参数自动生成：
# get_user(123) -> "mymodule:get_user:123"

# 访问底层缓存以便手动失效
get_user._cache.clear()
```
</details>

---

## 凭据加密

SimpleTuner 使用 Fernet 对称加密对敏感凭据（如 API Token）进行静态加密，为文件权限之外提供额外安全层。

### 工作原理

1. **密钥派生**：使用 PBKDF2-HMAC-SHA256（100,000 次）派生主密钥
2. **加密**：使用 Fernet（AES-128-CBC + HMAC）加密
3. **存储**：加密值以 base64 编码存入数据库

### 密钥管理

加密密钥按以下优先级获取：

| 优先级 | 来源 | 用途 |
|----------|--------|----------|
| 1 | `SIMPLETUNER_CREDENTIAL_KEY` 环境变量 | 生产部署、容器环境 |
| 2 | `~/.simpletuner/credential.key` 文件 | 本地开发、持久化密钥 |
| 3 | 自动生成 | 首次设置（保存到文件） |

<details>
<summary>通过环境变量设置密钥</summary>

```bash
# 生成安全密钥
export SIMPLETUNER_CREDENTIAL_KEY=$(openssl rand -base64 32)
```
</details>

**密钥文件位置：**

```
~/.simpletuner/credential.key
```

密钥文件以 `chmod 0600`（仅所有者可读写）创建。

### 使用方式

当通过 UI 或 API 保存提供方 API Token 时，凭据会自动加密。无需直接调用，但可使用以下函数：

<details>
<summary>加密/解密函数</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.auth.credential_encryption import (
    encrypt_credential,
    decrypt_credential,
)

# 加密 Token
encrypted = encrypt_credential("r8_my_api_token")

# 解密
plaintext = decrypt_credential(encrypted)
```
</details>

### 依赖要求

凭据加密需要 `cryptography` 包：

```bash
pip install cryptography
```

未安装时，凭据将以明文保存（日志中会有警告）。

### 密钥轮换

轮换加密密钥步骤：

1. 导出现有凭据（会用旧密钥解密）
2. 在环境变量或密钥文件中设置新密钥
3. 重启服务器
4. 在 UI 中重新录入凭据（将用新密钥加密）

**警告：** 若更换密钥但未重新录入凭据，已有加密数据将无法解密。

### 备份注意事项

备份 SimpleTuner 时：

- **包含：** `~/.simpletuner/credential.key`（或记录 `SIMPLETUNER_CREDENTIAL_KEY`）
- **包含：** 存放加密凭据的数据库
- 恢复时两者缺一不可

---

## 最佳实践

### 密钥管理

1. **CI/CD 使用环境变量** - 始终可用且优先级最高
2. **本地开发使用文件密钥** - 易管理且可跨重启
3. **生产环境使用 AWS/Vault** - 集中管理、可审计、可轮换
4. **不要提交密钥到版本库** - 将 `secrets.json` 加入 `.gitignore`

### 缓存调优

1. **提供方配置（TTL 5 分钟）** - 变化少，TTL 可长一些
2. **用户权限（TTL 1 分钟）** - 变更需更快传播
3. **写入即失效** - 更新底层数据时务必失效缓存
4. **监控缓存统计** - 使用 `cache.stats()` 查看命中率与过期数

### 安全注意

1. **文件权限** - 密钥文件以 `0o600` 创建（仅所有者）
2. **内存缓存** - 密钥首次读取后会缓存于内存
3. **清理敏感缓存** - 需要时调用 `secrets.clear_cache()`
4. **审计日志** - 所有提供方配置变更都会记录到审计日志

---

## 故障排查

### 找不到密钥

<details>
<summary>检查提供方可用性</summary>

```python
secrets = get_secrets_manager()

# 检查可用的提供方
for provider in secrets._providers:
    print(f"{provider.__class__.__name__}: {provider.is_available()}")
```
</details>

### 缓存不更新

<details>
<summary>强制缓存失效</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import (
    get_provider_config_cache,
    invalidate_provider_cache
)

# 方案 1：失效指定提供方
invalidate_provider_cache("replicate")

# 方案 2：清空整个缓存
cache = get_provider_config_cache()
cache.clear()
```
</details>

### Vault 连接问题

<details>
<summary>验证 Vault 可访问性</summary>

```bash
# 查看 Vault 状态
curl -s $VAULT_ADDR/v1/sys/health

# 验证 Token 是否有效
curl -s -H "X-Vault-Token: $VAULT_TOKEN" $VAULT_ADDR/v1/auth/token/lookup-self
```
</details>

### AWS Secrets Manager 问题

<details>
<summary>验证 AWS 凭据与权限</summary>

```bash
# 查看当前身份
aws sts get-caller-identity

# 测试密钥访问
aws secretsmanager get-secret-value --secret-id $SIMPLETUNER_AWS_SECRET_NAME
```
</details>

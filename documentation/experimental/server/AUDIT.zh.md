# 审计日志

SimpleTuner 的审计日志系统提供可防篡改的安全事件记录。所有管理操作、认证事件和任务操作都会记录，并支持加密链式校验。

## 概览

审计日志会捕获：
- **认证事件**：登录尝试（成功/失败）、登出、会话过期
- **用户管理**：用户创建、更新、删除、权限变更
- **API Key 操作**：创建、吊销、使用
- **凭据管理**：提供方凭据变更
- **任务操作**：提交、取消、审批

## 访问审计日志

### Web UI

在管理面板的 **Audit** 选项卡中浏览审计条目，并可使用筛选功能。

### CLI

```bash
# 列出最近的审计条目
simpletuner auth audit list

# 按事件类型筛选
simpletuner auth audit list --event-type auth.login.failed

# 按用户筛选
simpletuner auth audit user 123

# 仅查看安全事件
simpletuner auth audit security

# 获取统计
simpletuner auth audit stats

# 校验链完整性
simpletuner auth audit verify
```

### API 端点

所有端点都需要 `admin.audit` 权限。

| 方法 | 端点 | 描述 |
|--------|----------|-------------|
| GET | `/api/audit` | 列出审计条目（支持过滤） |
| GET | `/api/audit/stats` | 获取审计统计 |
| GET | `/api/audit/types` | 列出可用事件类型 |
| GET | `/api/audit/verify` | 校验链完整性 |
| GET | `/api/audit/user/{user_id}` | 获取指定用户的条目 |
| GET | `/api/audit/security` | 获取安全相关事件 |

## 事件类型

### 认证事件

| 事件 | 描述 |
|-------|-------------|
| `auth.login.success` | 登录成功 |
| `auth.login.failed` | 登录失败 |
| `auth.logout` | 退出登录 |
| `auth.session.expired` | 会话过期 |
| `auth.api_key.used` | API Key 被使用 |

### 用户管理事件

| 事件 | 描述 |
|-------|-------------|
| `user.created` | 创建新用户 |
| `user.updated` | 更新用户信息 |
| `user.deleted` | 删除用户 |
| `user.password.changed` | 用户修改密码 |
| `user.level.changed` | 用户级别/角色变更 |
| `user.permission.changed` | 用户权限变更 |

### API Key 事件

| 事件 | 描述 |
|-------|-------------|
| `api_key.created` | 创建新的 API Key |
| `api_key.revoked` | 吊销 API Key |

### 凭据事件

| 事件 | 描述 |
|-------|-------------|
| `credential.created` | 添加提供方凭据 |
| `credential.deleted` | 移除提供方凭据 |
| `credential.used` | 凭据被使用 |

### 任务事件

| 事件 | 描述 |
|-------|-------------|
| `job.submitted` | 任务已提交到队列 |
| `job.cancelled` | 任务被取消 |
| `job.approved` | 任务审批通过 |
| `job.rejected` | 任务审批被拒绝 |

## 查询参数

列出审计条目时可使用以下参数筛选：

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `event_type` | string | 按事件类型过滤 |
| `actor_id` | int | 按执行用户过滤 |
| `target_type` | string | 按目标资源类型过滤 |
| `target_id` | string | 按目标资源 ID 过滤 |
| `since` | ISO date | 起始时间戳 |
| `until` | ISO date | 结束时间戳 |
| `limit` | int | 最大条数（1-500，默认 50） |
| `offset` | int | 分页偏移 |

## 链完整性

每条审计记录都包含：
- 内容的加密哈希
- 前一条记录哈希的引用
- 单调时钟时间戳

这形成哈希链，可检测篡改。可通过验证端点或 CLI 命令检查完整性：

```bash
# 校验整条链
simpletuner auth audit verify

# 校验指定范围
simpletuner auth audit verify --start-id 100 --end-id 200
```

验证检查：
1. 每条记录的哈希与内容匹配
2. 每条记录正确引用前一条记录的哈希
3. 序列无缺失

## 保留策略

审计日志存储在 SimpleTuner 数据库中。可按部署需求配置保留期：

```bash
# 保留期（天）环境变量
SIMPLETUNER_AUDIT_RETENTION_DAYS=365
```

旧记录可按合规要求归档或清理。

## 安全注意事项

- 审计日志为追加写入，无法通过 API 修改或删除
- 查看日志需要 `admin.audit` 权限
- 失败的登录尝试会记录 IP 地址，便于安全监控
- 生产环境可考虑转发到 SIEM

# 服务器与多用户功能

本目录包含 SimpleTuner 的服务器端功能文档，这些功能适用于本地与云端训练部署。

## 内容

- [Worker Orchestration](WORKERS.md) - 分布式 Worker 注册、任务分发与 GPU 资源管理
- [Enterprise Guide](ENTERPRISE.md) - 多用户部署、SSO、审批、配额与治理
- [External Authentication](EXTERNAL_AUTH.md) - OIDC 与 LDAP 身份提供方配置
- [Audit Logging](AUDIT.md) - 带链式校验的安全审计日志

## 何时使用这些文档

以下场景适用：

- 将训练分发到多台 GPU 机器（Worker 编排）
- 将 SimpleTuner 作为多用户共享服务运行
- 集成企业身份提供方（Okta、Azure AD、Keycloak、LDAP）
- 需要任务提交审批流程
- 为合规或安全追踪用户操作
- 管理团队配额与资源限制

云端相关文档（Replicate、任务队列、Webhook）请参见 [Cloud Training](../cloud/README.md)。

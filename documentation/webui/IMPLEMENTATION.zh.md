# SimpleTuner WebUI 实现细节

## 设计概览

SimpleTuner 的 API 以灵活性为核心设计。

在 `trainer` 模式下，仅开放一个端口用于将 FastAPI 集成到其他服务中。
在 `unified` 模式下，会额外开放一个端口供 WebUI 接收远程 `trainer` 进程的回调事件。

## Web 框架

WebUI 基于 FastAPI 构建与提供：

- Alpine.js 用于响应式组件
- HTMX 用于动态内容加载与交互
- FastAPI（Starlette 与 SSE-Starlette）用于提供以数据为中心的 API，并通过 SSE 实现实时更新
- Jinja2 用于 HTML 模板

选择 Alpine 是因为其简单且易于集成，不需要引入 NodeJS，部署和维护更轻量。

HTMX 语法简洁，和 Alpine 配合自然，可在不引入完整前端框架的情况下实现动态加载、表单处理与交互。

选择 Starlette 和 SSE-Starlette 是为了最小化重复代码；从零散的过程式代码迁移到更声明式的结构，需要不少重构。

### 数据流

历史上，FastAPI 应用同时承担集群中的“服务工作者”角色：trainer 启动后暴露有限的回调入口，远程编排器通过 HTTP 回传状态更新。WebUI 复用同一套回调总线。在 unified 模式中 trainer 与界面同进程运行；在仅 trainer 的部署中，事件仍会推送到 `/callbacks`，而单独的 WebUI 实例通过 SSE 消费这些事件。无需新增队列或 broker，继续利用无头部署中已有的基础设施。

## 后端架构

Trainer UI 构建在核心 SDK 之上，SDK 现在提供清晰的服务层，而不是零散的过程式辅助函数。FastAPI 依旧负责终止请求，但大多数路由只是把参数委托给服务对象。这让 HTTP 层保持简单，并便于 CLI、配置向导与未来 API 复用。

### 路由处理

`simpletuner/simpletuner_sdk/server/routes/web.py` 负责 `/web/trainer`。只有两个关键端点：

- `trainer_page` – 渲染外层框架（导航、配置选择器、标签列表）。它向 `TabService` 请求元数据，并传入 `trainer_htmx.html` 模板。
- `render_tab` – 通用的 HTMX 目标。每个标签按钮都会访问该端点，路由通过 `TabService.render_tab` 解析对应布局并返回 HTML 片段。

其它 HTTP 路由位于 `simpletuner/simpletuner_sdk/server/routes/`，遵循相同模式：业务逻辑放在服务模块，路由提取参数、调用服务，然后返回 JSON 或 HTML。

### TabService

`TabService` 是训练表单的核心编排器，定义了：

- 每个标签的静态元数据（标题、图标、模板、可选上下文钩子）
- `render_tab()`：
  1. 获取标签配置（模板、描述）
  2. 向 `FieldService` 获取该标签/分区的字段集合
  3. 注入标签特定上下文（数据集列表、GPU 库存、引导状态）
  4. 渲染 `form_tab.html`、`datasets_tab.html` 等模板并返回

将逻辑收敛到类中使 HTMX、CLI 向导与测试复用同一渲染。模板不再依赖全局状态，一切都由上下文显式传入。

### FieldService 与 FieldRegistry

`FieldService` 将注册表条目转换为模板可用的字典，职责包括：

- 根据平台/模型上下文过滤字段（例如在 MPS 上隐藏 CUDA 专用项）
- 评估依赖规则（`FieldDependency`），以在 UI 中禁用或隐藏控件（例如 Dynamo 选项会在后端未选中时保持灰置）
- 补充提示、动态选项、展示格式与列样式

原始字段目录由 `FieldRegistry` 提供，位于 `simpletuner/simpletuner_sdk/server/services/field_registry` 的声明式列表中。每个 `ConfigField` 描述 CLI 标志、校验规则、重要性排序、依赖元数据与默认文案。这让 CLI 解析器、API 与文档生成共用同一事实来源。

### 状态持久化与引导

WebUI 通过 `WebUIStateStore` 保存轻量偏好设置。它从 `$SIMPLETUNER_WEB_UI_CONFIG`（或 XDG 路径）读取默认值，并提供：

- 主题、数据集根目录、输出目录默认值
- 各功能的引导清单状态
- Accelerate 覆盖项缓存（仅白名单键，如 `--num_processes`、`--dynamo_backend`）

这些值在首次 `/web/trainer` 渲染时注入页面，使 Alpine store 能在无需额外请求的情况下初始化。

### HTMX + Alpine 协作

每个设置面板都是带 `x-data` 的 HTML 片段。标签按钮会向 `/web/trainer/tabs/{tab}` 发起 HTMX GET；服务器返回渲染后的表单，而 Alpine 保持已有组件状态。一个小型辅助脚本（`trainer-form.js`）会重放已保存的值变化，避免切换标签时丢失编辑内容。

服务器更新（训练状态、GPU 遥测）通过 SSE 端点（`sse_manager.js`）进入 Alpine store，驱动提示、进度条与系统横幅。

### 文件布局速查

- `templates/` – Jinja 模板；`partials/form_field.html` 渲染单个控件。`partials/form_field_htmx.html` 是向导使用的 HTMX 版本。
- `static/js/modules/` – Alpine 组件脚本（训练表单、硬件库存、数据集浏览）。
- `static/js/services/` – 共享工具（依赖评估、SSE 管理、事件总线）。
- `simpletuner/simpletuner_sdk/server/services/` – 后端服务层（fields、tabs、configs、datasets、maintenance、events）。

整体上，WebUI 在服务端保持无状态，状态性内容（表单数据、提示）保存在浏览器。后端专注于纯数据转换，利于测试，也避免在 trainer 与 Web 服务器共进程时的线程问题。

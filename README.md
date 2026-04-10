# Claude Code Proxy

让 **Claude Code** 使用任意 OpenAI 兼容 API 的 **Cloudflare Worker** 代理。将 Claude API 请求转换为 OpenAI API 调用，支持 OpenAI、Azure、DeepSeek、GLM、Qwen、Gemini 等多种模型。同时支持 Anthropic API 直接转发（passthrough）模式，适用于 MiniMax 等 Anthropic 兼容提供商。

## ✨ 特性

- 完整的 `/v1/messages` Claude API 兼容
- 支持流式 SSE 响应、函数调用 (tool use)、图片输入
- 自动将 `reasoning_content` 转为 Claude 思维块 (thinking blocks)
- **Passthrough 模式**：直接转发 Anthropic 格式请求（支持 MiniMax 等）
- **模型映射可选**：默认忠实转发 model-id，可通过开关启用 BIG/MIDDLE/SMALL 映射
- **API Key 透传**：客户端 key 直接转发给后端，无需在服务端配置密钥（推荐）
- 部署在 Cloudflare 全球边缘网络，低延迟
- API Key 常量时间比较，防止时序攻击

---

## 🚀 部署

### 方式一：Fork + Cloudflare 自动部署（推荐）

> 最简单的方式：Fork 仓库，在 Cloudflare Dashboard 关联 GitHub 仓库，即可自动构建和部署。

1. **Fork 本仓库** 到你的 GitHub 账号

2. 登录 [Cloudflare Dashboard](https://dash.cloudflare.com/) → **Workers & Pages** → **Create**

3. 选择 **Import a repository**，连接你的 GitHub 账号并选择 fork 的仓库

4. 构建配置保持默认即可：
   - **Build command**: `npm run build`
   - **Deploy command**: `npm run deploy`

5. 部署完成后，在 Worker 的 **Settings → Variables and Secrets** 中按需添加：
   - `OPENAI_API_KEY`（可选）— 托管模式下的后端 API Key。不设置则使用客户端 key 透传（推荐）
   - `ANTHROPIC_API_KEY`（可选）— 用于客户端身份验证的额外安全层

6. 根据你的模型提供商，按需修改 `wrangler.toml` 中的环境变量（见下方[配置](#-配置)），推送即自动重新部署

### 方式二：Wrangler CLI 手动部署

```bash
# 1. 克隆仓库
git clone https://github.com/ray5cc/claude-code-proxy.git
cd claude-code-proxy

# 2. 安装依赖
npm install

# 3. 设置 secrets
wrangler secret put OPENAI_API_KEY
wrangler secret put ANTHROPIC_API_KEY  # 可选

# 4. 部署
npm run deploy
```

### 方式三：本地开发

```bash
cp .env.example .dev.vars   # 编辑 .dev.vars 填入你的 API Key
npm install
npm run dev                  # 启动本地开发服务器
```

---

## 🔧 配置

在 `wrangler.toml` 的 `[vars]` 中设置非敏感变量，敏感信息通过 `wrangler secret put` 或 Cloudflare Dashboard 设置。

### 环境变量

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 可选 (Secret) 后端 API Key。不设置则透传客户端 key（推荐） | — |
| `ANTHROPIC_API_KEY` | 可选 (Secret) 客户端验证 Key（额外安全层） | 不设置则接受任意 Key |
| `OPENAI_BASE_URL` | API 基础 URL | `https://api.openai.com/v1` |
| `PROXY_MODE` | 代理模式：`openai`（转换）或 `passthrough`（直接转发） | `openai` |
| `ENABLE_MODEL_MAPPING` | 设为 `true` 启用 Claude→Provider 模型映射 | `false`（直接转发 model-id） |
| `BIG_MODEL` | Claude opus 请求映射（需启用模型映射） | `gpt-4o` |
| `MIDDLE_MODEL` | Claude sonnet 请求映射（需启用模型映射） | `gpt-4o` |
| `SMALL_MODEL` | Claude haiku 请求映射（需启用模型映射） | `gpt-4o-mini` |
| `MAX_TOKENS_LIMIT` | 最大 token 数 | `16384` |
| `MIN_TOKENS_LIMIT` | 最小 token 数 | `4096` |
| `REQUEST_TIMEOUT` | 请求超时 (秒) | `90` |
| `AZURE_API_VERSION` | Azure OpenAI API 版本 | — |
| `CUSTOM_HEADERS` | 自定义 HTTP 头 (JSON 字符串) | — |

### 代理模式

| 模式 | 说明 |
| --- | --- |
| `openai` (默认) | 将 Claude API 请求转换为 OpenAI API 格式，适用于 OpenAI、DeepSeek、GLM、Qwen 等 |
| `passthrough` | 直接转发 Anthropic 格式请求到后端 API，适用于 MiniMax 等 Anthropic 兼容提供商 |

### API Key 模式

代理支持两种 key 管理方式：

| 模式 | `OPENAI_API_KEY` | 行为 |
| --- | --- | --- |
| **透传模式（推荐）** | 不设置 | 客户端的 key 直接转发给后端 API |
| **托管模式** | 设置 | 使用服务端配置的 key，客户端 key 仅用于验证身份 |

**透传模式**更灵活 — 每个用户使用自己的 API key，代理只负责格式转换。
可选配置 `ANTHROPIC_API_KEY` 作为额外的安全层，限制谁可以访问代理。

### 模型映射

默认情况下，代理会**忠实转发**客户端发送的 model-id，不做任何映射。

设置 `ENABLE_MODEL_MAPPING=true` 后，将启用以下映射：

| Claude 请求 | 映射到 | 默认模型 |
| --- | --- | --- |
| 包含 "opus" | `BIG_MODEL` | `gpt-4o` |
| 包含 "sonnet" | `MIDDLE_MODEL` | `gpt-4o` |
| 包含 "haiku" | `SMALL_MODEL` | `gpt-4o-mini` |

---

## 📡 提供商配置示例

修改 `wrangler.toml` 中的 `[vars]` 部分：

<details>
<summary><b>OpenAI（需启用模型映射）</b></summary>

```toml
[vars]
OPENAI_BASE_URL = "https://api.openai.com/v1"
ENABLE_MODEL_MAPPING = "true"
BIG_MODEL = "gpt-4o"
MIDDLE_MODEL = "gpt-4o"
SMALL_MODEL = "gpt-4o-mini"
```
</details>

<details>
<summary><b>Azure OpenAI（需启用模型映射）</b></summary>

```toml
[vars]
OPENAI_BASE_URL = "https://your-resource.openai.azure.com/openai/deployments/your-deployment"
ENABLE_MODEL_MAPPING = "true"
BIG_MODEL = "gpt-4"
MIDDLE_MODEL = "gpt-4"
SMALL_MODEL = "gpt-35-turbo"
```

另需设置 secret：`AZURE_API_VERSION`（如 `2024-03-01-preview`）
</details>

<details>
<summary><b>DeepSeek（需启用模型映射）</b></summary>

```toml
[vars]
OPENAI_BASE_URL = "https://api.deepseek.com/v1"
ENABLE_MODEL_MAPPING = "true"
BIG_MODEL = "deepseek-chat"
MIDDLE_MODEL = "deepseek-chat"
SMALL_MODEL = "deepseek-chat"
```
</details>

<details>
<summary><b>OpenCode Go — GLM 5.1（透传模式）</b></summary>

```toml
[vars]
OPENAI_BASE_URL = "https://opencode.ai/zen/go/v1"
# 不设置 OPENAI_API_KEY — 客户端 key 直接透传给后端
# model-id 由客户端直接指定，如 glm-5.1
```

自动将 GLM 5.1 的 `reasoning_content` 转换为 Claude 思维块。
</details>

<details>
<summary><b>GLM 5.1 (智谱 Z.AI 直连，透传模式)</b></summary>

```toml
[vars]
OPENAI_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
# model-id 由客户端直接指定，如 glm-5.1
```

自动将 GLM 5.1 的 `reasoning_content` 转换为 Claude 思维块。
</details>

<details>
<summary><b>OpenCode Go — MiniMax（Passthrough + 透传模式）</b></summary>

```toml
[vars]
PROXY_MODE = "passthrough"
OPENAI_BASE_URL = "https://opencode.ai/zen/go/v1"
# MiniMax 使用 Anthropic API 格式，直接转发请求
# model-id 由客户端指定，如 minimax-m2.5、minimax-m2.7
```

Passthrough 模式下请求以 Anthropic 格式直接转发，无需转换。
</details>

<details>
<summary><b>Gemini</b></summary>

```toml
[vars]
OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
ENABLE_MODEL_MAPPING = "true"
BIG_MODEL = "gemini-2.5-pro"
MIDDLE_MODEL = "gemini-2.5-pro"
SMALL_MODEL = "gemini-2.0-flash"
```
</details>

<details>
<summary><b>Qwen (通义千问，需启用模型映射)</b></summary>

```toml
[vars]
OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ENABLE_MODEL_MAPPING = "true"
BIG_MODEL = "qwen-max"
MIDDLE_MODEL = "qwen-plus"
SMALL_MODEL = "qwen-turbo"
```
</details>

---

## 🖥️ 使用 Claude Code

部署完成后，使用以下方式启动 Claude Code：

```bash
# 透传模式（推荐）— 客户端 key 直接转发给后端
ANTHROPIC_BASE_URL=https://your-worker.your-subdomain.workers.dev \
ANTHROPIC_API_KEY="your-backend-api-key" \
claude

# 托管模式 — 服务端已配置 OPENAI_API_KEY
ANTHROPIC_BASE_URL=https://your-worker.your-subdomain.workers.dev \
ANTHROPIC_API_KEY="any-value" \
claude

# 如果服务端设置了 ANTHROPIC_API_KEY 验证
ANTHROPIC_BASE_URL=https://your-worker.your-subdomain.workers.dev \
ANTHROPIC_API_KEY="your-matching-key" \
claude
```

将 `your-worker.your-subdomain.workers.dev` 替换为你实际的 Worker URL。

---

## 📋 API 端点

| 方法 | 路径 | 说明 | 需要认证 |
| --- | --- | --- | --- |
| GET | `/` | 代理信息 | 否 |
| GET | `/health` | 健康检查 | 否 |
| POST | `/v1/messages` | 聊天补全（代理） | 是 |
| POST | `/v1/messages/count_tokens` | Token 计数估算 | 是 |

---

## 🛠️ 开发

```bash
npm install          # 安装依赖
npm run dev          # 本地开发服务器
npm run lint         # 类型检查
npm run test         # 运行测试
npm run build        # 构建 (dry-run)
npm run deploy       # 部署到 Cloudflare Workers
```

### 项目结构

```
src/
├── index.ts              # Worker 入口 & 路由
├── types.ts              # TypeScript 类型定义
├── config.ts             # 配置 & 模型映射
├── constants.ts          # 共享常量
├── client.ts             # OpenAI API 客户端
├── handlers.ts           # 请求处理器
└── conversion/
    ├── request.ts        # Claude → OpenAI 请求转换
    └── response.ts       # OpenAI → Claude 响应转换
```

---

## 📄 License

MIT

# Claude Code Proxy (CF Workers)

[English](#english) | [中文](#中文)

---

<a id="english"></a>

## English

A **Cloudflare Worker** proxy that enables **Claude Code** to use any OpenAI-compatible API. Converts Claude API requests into OpenAI API calls, supporting OpenAI, Azure, DeepSeek, GLM, Qwen, Gemini, and more. Also supports Anthropic API passthrough mode for Anthropic-compatible providers like MiniMax.

### ✨ Features

- Full `/v1/messages` Claude API compatibility
- Streaming SSE responses, function calling (tool use), image input
- Automatic `reasoning_content` → Claude thinking blocks conversion
- **Passthrough mode**: auto-route by model prefix, forwarding in native Anthropic format (MiniMax, etc.)
- **Optional model mapping**: faithfully forwards model-id by default; enable BIG/MIDDLE/SMALL mapping via toggle
- **API key passthrough**: client key forwarded directly to backend — no server-side key required (recommended)
- Deployed on Cloudflare's global edge network for low latency
- Constant-time API key comparison to prevent timing attacks

### 🚀 Deployment

#### Option 1: Fork + Cloudflare Auto-Deploy (Recommended)

> The simplest approach: fork the repo, link it in Cloudflare Dashboard, and it auto-builds and deploys.

1. **Fork this repository** to your GitHub account

2. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/) → **Workers & Pages** → **Create**

3. Select **Import a repository**, connect your GitHub account, and choose the forked repo

4. Keep the default build settings:
   - **Build command**: `npm run build`
   - **Deploy command**: `npm run deploy`

5. After deployment, configure your provider in Worker **Settings → Variables and Secrets** (see [Configuration](#configuration) below):
   - `OPENAI_BASE_URL` — your provider's API base URL (default: `https://api.openai.com/v1`)
   - `OPENAI_API_KEY` (optional) — backend API key for managed mode. Leave unset for client key passthrough (recommended)
   - `ANTHROPIC_API_KEY` (optional) — additional security layer for client authentication
   - Other variables as needed for your provider

6. All configuration is done via Dashboard — no need to edit `wrangler.toml`. Syncing upstream updates won't overwrite your settings.

#### Option 2: Wrangler CLI Manual Deploy

```bash
# 1. Clone the repo
git clone https://github.com/ray5cc/claude-code-proxy-cf-workers.git
cd claude-code-proxy-cf-workers

# 2. Install dependencies
npm install

# 3. Set secrets
wrangler secret put OPENAI_API_KEY
wrangler secret put ANTHROPIC_API_KEY  # optional

# 4. Deploy
npm run deploy
```

#### Option 3: Local Development

```bash
cp .env.example .dev.vars   # Edit .dev.vars with your API keys
npm install
npm run dev                  # Start local dev server
```

### <a id="configuration"></a>🔧 Configuration

All configuration is done via **Cloudflare Dashboard** (Settings → Variables and Secrets) or `wrangler secret put`. No `[vars]` in `wrangler.toml` — this ensures your settings are never overwritten by deployments. Defaults are handled in code (`src/config.ts`).

#### Environment Variables

| Variable | Description | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | Optional (Secret) backend API key. Leave unset for client key passthrough (recommended) | — |
| `ANTHROPIC_API_KEY` | Optional (Secret) client validation key (additional security layer) | Accepts any key if unset |
| `OPENAI_BASE_URL` | API base URL | `https://api.openai.com/v1` |
| `PASSTHROUGH_MODELS` | Comma-separated model prefixes for Anthropic passthrough | Empty (all via OpenAI conversion) |
| `ENABLE_MODEL_MAPPING` | Set to `true` to enable Claude→provider model mapping | `false` (forwards model-id as-is) |
| `BIG_MODEL` | Claude opus request mapping (requires model mapping) | `gpt-4o` |
| `MIDDLE_MODEL` | Claude sonnet request mapping (requires model mapping) | `gpt-4o` |
| `SMALL_MODEL` | Claude haiku request mapping (requires model mapping) | `gpt-4o-mini` |
| `MAX_TOKENS_LIMIT` | Maximum token limit | `16384` |
| `MIN_TOKENS_LIMIT` | Minimum token limit | `4096` |
| `REQUEST_TIMEOUT` | Request timeout (seconds) | `90` |
| `AZURE_API_VERSION` | Azure OpenAI API version | — |
| `CUSTOM_HEADERS` | Custom HTTP headers (JSON string) | — |
| `LOG_LEVEL` | Log verbosity (`WARNING` / `DEBUG`) | `WARNING` |

> **💡 Debug Logging:** Set `LOG_LEVEL=DEBUG` to enable detailed request/response logging (raw content, streaming deltas, finish reasons). Useful for troubleshooting model compatibility issues. Keep it at `WARNING` (default) in production to avoid log costs.

#### Request Routing

By default, all requests go through **OpenAI conversion** (Claude format → OpenAI format).

Use `PASSTHROUGH_MODELS` to specify model prefixes for **Anthropic passthrough**:

```toml
# Example: MiniMax models use Anthropic API format, others go through OpenAI conversion
PASSTHROUGH_MODELS = "minimax"
```

| Request Model | Route |
| --- | --- |
| `glm-5.1` | → OpenAI conversion (`/chat/completions`) |
| `minimax-m2.5` | → Anthropic passthrough (`/messages`) |
| `deepseek-chat` | → OpenAI conversion (`/chat/completions`) |

This allows a single proxy to serve backends with different API formats.

#### API Key Modes

The proxy supports two key management modes:

| Mode | `OPENAI_API_KEY` | Behavior |
| --- | --- | --- |
| **Passthrough (recommended)** | Not set | Client key forwarded directly to backend API |
| **Managed** | Set | Uses server-configured key; client key only for authentication |

**Passthrough mode** is more flexible — each user uses their own API key, and the proxy only handles format conversion.
Optionally set `ANTHROPIC_API_KEY` as an additional security layer to restrict proxy access.

#### Model Mapping

By default, the proxy **faithfully forwards** the client's model-id without any mapping.

Set `ENABLE_MODEL_MAPPING=true` to enable the following mapping:

| Claude Request | Maps To | Default Model |
| --- | --- | --- |
| Contains "opus" | `BIG_MODEL` | `gpt-4o` |
| Contains "sonnet" | `MIDDLE_MODEL` | `gpt-4o` |
| Contains "haiku" | `SMALL_MODEL` | `gpt-4o-mini` |

### 📡 Provider Examples

Set the following variables in **Cloudflare Dashboard** (Settings → Variables and Secrets):

<details>
<summary><b>OpenAI (requires model mapping)</b></summary>

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
<summary><b>Azure OpenAI (requires model mapping)</b></summary>

```toml
[vars]
OPENAI_BASE_URL = "https://your-resource.openai.azure.com/openai/deployments/your-deployment"
ENABLE_MODEL_MAPPING = "true"
BIG_MODEL = "gpt-4"
MIDDLE_MODEL = "gpt-4"
SMALL_MODEL = "gpt-35-turbo"
```

Also set secret: `AZURE_API_VERSION` (e.g. `2024-03-01-preview`)
</details>

<details>
<summary><b>DeepSeek (requires model mapping)</b></summary>

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
<summary><b>OpenCode Go — GLM + MiniMax Hybrid</b></summary>

```toml
[vars]
OPENAI_BASE_URL = "https://opencode.ai/zen/go/v1"
PASSTHROUGH_MODELS = "minimax"
# Don't set OPENAI_API_KEY — client key is forwarded directly
# GLM models (e.g. glm-5.1) go through OpenAI conversion
# MiniMax models (e.g. minimax-m2.5) use Anthropic passthrough
```

Automatically converts GLM 5.1 `reasoning_content` to Claude thinking blocks.
</details>

<details>
<summary><b>GLM 5.1 (Zhipu Z.AI Direct, Passthrough Mode)</b></summary>

```toml
[vars]
OPENAI_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
# model-id specified directly by client, e.g. glm-5.1
```

Automatically converts GLM 5.1 `reasoning_content` to Claude thinking blocks.
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
<summary><b>Qwen (requires model mapping)</b></summary>

```toml
[vars]
OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ENABLE_MODEL_MAPPING = "true"
BIG_MODEL = "qwen-max"
MIDDLE_MODEL = "qwen-plus"
SMALL_MODEL = "qwen-turbo"
```
</details>

### 🖥️ Using with Claude Code

After deployment, start Claude Code with:

```bash
# Passthrough mode (recommended) — client key forwarded to backend
ANTHROPIC_BASE_URL=https://your-worker.your-subdomain.workers.dev \
ANTHROPIC_API_KEY="your-backend-api-key" \
claude

# Managed mode — server has OPENAI_API_KEY configured
ANTHROPIC_BASE_URL=https://your-worker.your-subdomain.workers.dev \
ANTHROPIC_API_KEY="any-value" \
claude

# If server has ANTHROPIC_API_KEY validation enabled
ANTHROPIC_BASE_URL=https://your-worker.your-subdomain.workers.dev \
ANTHROPIC_API_KEY="your-matching-key" \
claude
```

Replace `your-worker.your-subdomain.workers.dev` with your actual Worker URL.

### 📋 API Endpoints

| Method | Path | Description | Auth Required |
| --- | --- | --- | --- |
| GET | `/` | Proxy info | No |
| GET | `/health` | Health check | No |
| POST | `/v1/messages` | Chat completions (proxy) | Yes |
| POST | `/v1/messages/count_tokens` | Token count estimation | Yes |

### 🛠️ Development

```bash
npm install          # Install dependencies
npm run dev          # Local dev server
npm run lint         # Type check
npm run test         # Run tests
npm run build        # Build (dry-run)
npm run deploy       # Deploy to Cloudflare Workers
```

#### Project Structure

```
src/
├── index.ts              # Worker entry & routing
├── types.ts              # TypeScript type definitions
├── config.ts             # Configuration & model mapping
├── constants.ts          # Shared constants
├── client.ts             # OpenAI API client
├── handlers.ts           # Request handlers
└── conversion/
    ├── request.ts        # Claude → OpenAI request conversion
    └── response.ts       # OpenAI → Claude response conversion
```

---

<a id="中文"></a>

## 中文

让 **Claude Code** 使用任意 OpenAI 兼容 API 的 **Cloudflare Worker** 代理。将 Claude API 请求转换为 OpenAI API 调用，支持 OpenAI、Azure、DeepSeek、GLM、Qwen、Gemini 等多种模型。同时支持 Anthropic API 直接转发（passthrough）模式，适用于 MiniMax 等 Anthropic 兼容提供商。

### ✨ 特性

- 完整的 `/v1/messages` Claude API 兼容
- 支持流式 SSE 响应、函数调用 (tool use)、图片输入
- 自动将 `reasoning_content` 转为 Claude 思维块 (thinking blocks)
- **Passthrough 模式**：按模型自动路由，指定模型前缀使用 Anthropic 格式直转（支持 MiniMax 等）
- **模型映射可选**：默认忠实转发 model-id，可通过开关启用 BIG/MIDDLE/SMALL 映射
- **API Key 透传**：客户端 key 直接转发给后端，无需在服务端配置密钥（推荐）
- 部署在 Cloudflare 全球边缘网络，低延迟
- API Key 常量时间比较，防止时序攻击

### 🚀 部署

#### 方式一：Fork + Cloudflare 自动部署（推荐）

> 最简单的方式：Fork 仓库，在 Cloudflare Dashboard 关联 GitHub 仓库，即可自动构建和部署。

1. **Fork 本仓库** 到你的 GitHub 账号

2. 登录 [Cloudflare Dashboard](https://dash.cloudflare.com/) → **Workers & Pages** → **Create**

3. 选择 **Import a repository**，连接你的 GitHub 账号并选择 fork 的仓库

4. 构建配置保持默认即可：
   - **Build command**: `npm run build`
   - **Deploy command**: `npm run deploy`

5. 部署完成后，在 Worker 的 **Settings → Variables and Secrets** 中配置你的模型提供商（见下方[配置](#-配置-1)）：
   - `OPENAI_BASE_URL` — 你的提供商 API 基础 URL（默认：`https://api.openai.com/v1`）
   - `OPENAI_API_KEY`（可选）— 托管模式下的后端 API Key。不设置则使用客户端 key 透传（推荐）
   - `ANTHROPIC_API_KEY`（可选）— 用于客户端身份验证的额外安全层
   - 其他按需配置的变量

6. 所有配置均通过 Dashboard 完成，无需修改 `wrangler.toml`。同步上游更新不会覆盖你的配置。

#### 方式二：Wrangler CLI 手动部署

```bash
# 1. 克隆仓库
git clone https://github.com/ray5cc/claude-code-proxy-cf-workers.git
cd claude-code-proxy-cf-workers

# 2. 安装依赖
npm install

# 3. 设置 secrets
wrangler secret put OPENAI_API_KEY
wrangler secret put ANTHROPIC_API_KEY  # 可选

# 4. 部署
npm run deploy
```

#### 方式三：本地开发

```bash
cp .env.example .dev.vars   # 编辑 .dev.vars 填入你的 API Key
npm install
npm run dev                  # 启动本地开发服务器
```

### 🔧 配置

所有配置均通过 **Cloudflare Dashboard**（Settings → Variables and Secrets）或 `wrangler secret put` 完成。`wrangler.toml` 中不包含 `[vars]`，确保你的配置不会被部署覆盖。默认值已在代码（`src/config.ts`）中处理。

#### 环境变量

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 可选 (Secret) 后端 API Key。不设置则透传客户端 key（推荐） | — |
| `ANTHROPIC_API_KEY` | 可选 (Secret) 客户端验证 Key（额外安全层） | 不设置则接受任意 Key |
| `OPENAI_BASE_URL` | API 基础 URL | `https://api.openai.com/v1` |
| `PASSTHROUGH_MODELS` | 逗号分隔的模型前缀，匹配的模型以 Anthropic 格式直转 | 空（全部走 OpenAI 转换） |
| `ENABLE_MODEL_MAPPING` | 设为 `true` 启用 Claude→Provider 模型映射 | `false`（直接转发 model-id） |
| `BIG_MODEL` | Claude opus 请求映射（需启用模型映射） | `gpt-4o` |
| `MIDDLE_MODEL` | Claude sonnet 请求映射（需启用模型映射） | `gpt-4o` |
| `SMALL_MODEL` | Claude haiku 请求映射（需启用模型映射） | `gpt-4o-mini` |
| `MAX_TOKENS_LIMIT` | 最大 token 数 | `16384` |
| `MIN_TOKENS_LIMIT` | 最小 token 数 | `4096` |
| `REQUEST_TIMEOUT` | 请求超时 (秒) | `90` |
| `AZURE_API_VERSION` | Azure OpenAI API 版本 | — |
| `CUSTOM_HEADERS` | 自定义 HTTP 头 (JSON 字符串) | — |
| `LOG_LEVEL` | 日志级别（`WARNING` / `DEBUG`） | `WARNING` |

> **💡 调试日志：** 设置 `LOG_LEVEL=DEBUG` 可启用详细的请求/响应日志（原始内容、流式增量数据、结束原因等）。适用于排查模型兼容性问题。生产环境建议保持 `WARNING`（默认值）以避免日志费用。

#### 请求路由

默认所有请求走 **OpenAI 转换**（Claude 格式 → OpenAI 格式）。

通过 `PASSTHROUGH_MODELS` 可指定需要 **Anthropic 直转**的模型前缀：

```toml
# 例如：MiniMax 模型使用 Anthropic API 格式，其他模型走 OpenAI 转换
PASSTHROUGH_MODELS = "minimax"
```

| 请求模型 | 路由 |
| --- | --- |
| `glm-5.1` | → OpenAI 转换 (`/chat/completions`) |
| `minimax-m2.5` | → Anthropic 直转 (`/messages`) |
| `deepseek-chat` | → OpenAI 转换 (`/chat/completions`) |

这样同一个代理可以同时服务不同格式的后端 API。

#### API Key 模式

代理支持两种 key 管理方式：

| 模式 | `OPENAI_API_KEY` | 行为 |
| --- | --- | --- |
| **透传模式（推荐）** | 不设置 | 客户端的 key 直接转发给后端 API |
| **托管模式** | 设置 | 使用服务端配置的 key，客户端 key 仅用于验证身份 |

**透传模式**更灵活 — 每个用户使用自己的 API key，代理只负责格式转换。
可选配置 `ANTHROPIC_API_KEY` 作为额外的安全层，限制谁可以访问代理。

#### 模型映射

默认情况下，代理会**忠实转发**客户端发送的 model-id，不做任何映射。

设置 `ENABLE_MODEL_MAPPING=true` 后，将启用以下映射：

| Claude 请求 | 映射到 | 默认模型 |
| --- | --- | --- |
| 包含 "opus" | `BIG_MODEL` | `gpt-4o` |
| 包含 "sonnet" | `MIDDLE_MODEL` | `gpt-4o` |
| 包含 "haiku" | `SMALL_MODEL` | `gpt-4o-mini` |

### 📡 提供商配置示例

在 **Cloudflare Dashboard**（Settings → Variables and Secrets）中设置以下变量：

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
<summary><b>OpenCode Go — GLM + MiniMax 混合</b></summary>

```toml
[vars]
OPENAI_BASE_URL = "https://opencode.ai/zen/go/v1"
PASSTHROUGH_MODELS = "minimax"
# 不设置 OPENAI_API_KEY — 客户端 key 直接透传给后端
# GLM 模型（如 glm-5.1）走 OpenAI 转换
# MiniMax 模型（如 minimax-m2.5）走 Anthropic 直转
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

### 🖥️ 使用 Claude Code

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

### 📋 API 端点

| 方法 | 路径 | 说明 | 需要认证 |
| --- | --- | --- | --- |
| GET | `/` | 代理信息 | 否 |
| GET | `/health` | 健康检查 | 否 |
| POST | `/v1/messages` | 聊天补全（代理） | 是 |
| POST | `/v1/messages/count_tokens` | Token 计数估算 | 是 |

### 🛠️ 开发

```bash
npm install          # 安装依赖
npm run dev          # 本地开发服务器
npm run lint         # 类型检查
npm run test         # 运行测试
npm run build        # 构建 (dry-run)
npm run deploy       # 部署到 Cloudflare Workers
```

#### 项目结构

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

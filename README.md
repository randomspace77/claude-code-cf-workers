# Claude Code Proxy (CF Workers)

[English](#english) | [中文](#中文)

---

<a id="english"></a>

## English

A **Cloudflare Worker** proxy that enables **Claude Code** to use any OpenAI-compatible API — with built-in **multi-provider routing**. A single deployment can route different models to different providers (OpenAI, DeepSeek, GLM, Gemini, Anthropic, etc.) based on glob patterns, while handling API format conversion automatically.

### ✨ Features

- **Multi-provider routing** — route models to different providers by glob patterns (`glm-*` → Zhipu, `gpt-*` → OpenAI, etc.)
- **14 built-in providers** — hardcoded URLs for OpenAI, OpenRouter, DeepSeek, GLM, Qwen, Gemini, Anthropic, MiniMax, OpenCode, Doubao, SiliconFlow, Groq, Mistral, Together
- Full `/v1/messages` Claude API compatibility
- Streaming SSE, function calling (tool use), image input
- Automatic `reasoning_content` → Claude thinking blocks
- **Dual protocol support**: OpenAI-compatible conversion + Anthropic passthrough
- **Per-provider API keys** — `PROVIDER_<NAME>_API_KEY` as secrets, or client key passthrough
- **Per-provider model mapping** — map Claude model names (opus/sonnet/haiku) to provider-specific models
- Deployed on Cloudflare's global edge network for low latency
- Constant-time API key comparison to prevent timing attacks
- **Full backward compatibility** — legacy single-provider config still works

### 🏗️ Architecture

```
Client Request (Claude API format)
       │
       ▼
  ┌─────────┐
  │  Auth    │  ← ANTHROPIC_API_KEY validation
  └────┬────┘
       ▼
  ┌─────────┐
  │  Router  │  ← model name → provider (glob matching)
  └────┬────┘
       ▼
  ┌──────────────┐
  │   Provider   │  ← resolved provider config (URL, key, protocol)
  │   Dispatch   │
  └──┬───────┬───┘
     │       │
     ▼       ▼
  OpenAI  Anthropic
  Provider Provider
     │       │
     ▼       ▼
  Claude→  Passthrough
  OpenAI   (native fmt)
     │       │
     ▼       ▼
  Backend  Backend
```

### 🚀 Deployment

#### Option 1: Fork + Cloudflare Auto-Deploy (Recommended)

1. **Fork this repository** to your GitHub account
2. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/) → **Workers & Pages** → **Create**
3. Select **Import a repository**, connect your GitHub account, choose the forked repo
4. Keep default build settings (`npm run build` / `npm run deploy`)
5. Configure in Worker **Settings → Variables and Secrets** (see [Configuration](#configuration))
6. All configuration is done via Dashboard — syncing upstream updates won't overwrite your settings

#### Option 2: Wrangler CLI

```bash
git clone https://github.com/randomspace77/claude-code-proxy-cf-workers.git
cd claude-code-proxy-cf-workers
npm install

# Set provider API keys as secrets
wrangler secret put PROVIDER_OPENAI_API_KEY
wrangler secret put PROVIDER_DEEPSEEK_API_KEY  # etc.
wrangler secret put ANTHROPIC_API_KEY          # optional, for proxy auth

npm run deploy
```

#### Option 3: Local Development

```bash
cp .env.example .dev.vars   # Edit with your API keys
npm install
npm run dev
```

### <a id="configuration"></a>🔧 Configuration

All configuration is done via **Cloudflare Dashboard** (Settings → Variables and Secrets). No `[vars]` in `wrangler.toml` — deployments never overwrite your settings.

#### Multi-Provider Mode (Recommended)

Set the `PROVIDERS` environment variable (plaintext JSON) to configure multiple providers and model routing:

```jsonc
{
  "default": "openai",
  "routing": {
    "glm-*": "glm",
    "deepseek-*": "deepseek",
    "claude-*": "anthropic",
    "gpt-*": "openai",
    "o1-*": "openai",
    "o3-*": "openai",
    "gemini-*": "gemini",
    "qwen-*": "qwen"
  },
  "providers": {
    "openai": {
      "modelMapping": { "opus": "gpt-4o", "sonnet": "gpt-4o", "haiku": "gpt-4o-mini" }
    },
    "glm": { "timeout": 120 },
    "deepseek": {},
    "anthropic": {},
    "gemini": {},
    "qwen": {}
  }
}
```

Then set each provider's API key as a **secret**:

```bash
wrangler secret put PROVIDER_OPENAI_API_KEY
wrangler secret put PROVIDER_GLM_API_KEY
wrangler secret put PROVIDER_DEEPSEEK_API_KEY
# ... etc.
```

> **Key naming convention**: `PROVIDER_<UPPERCASE_NAME>_API_KEY`. Hyphens in provider names become underscores (e.g., `my-custom` → `PROVIDER_MY_CUSTOM_API_KEY`).

#### Known Providers

These providers have hardcoded base URLs — you only need to add an API key:

| Name | Base URL | Protocol |
| --- | --- | --- |
| `openai` | `https://api.openai.com/v1` | openai |
| `openrouter` | `https://openrouter.ai/api/v1` | openai |
| `deepseek` | `https://api.deepseek.com/v1` | openai |
| `glm` | `https://open.bigmodel.cn/api/paas/v4` | openai |
| `qwen` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | openai |
| `gemini` | `https://generativelanguage.googleapis.com/v1beta/openai` | openai |
| `anthropic` | `https://api.anthropic.com/v1` | anthropic |
| `minimax` | `https://api.minimax.chat/v1` | anthropic |
| `opencode` | `https://opencode.ai/zen/go/v1` | openai |
| `doubao` | `https://ark.cn-beijing.volces.com/api/v3` | openai |
| `siliconflow` | `https://api.siliconflow.cn/v1` | openai |
| `groq` | `https://api.groq.com/openai/v1` | openai |
| `mistral` | `https://api.mistral.ai/v1` | openai |
| `together` | `https://api.together.xyz/v1` | openai |

You can also define **custom providers** with a `baseUrl`:

```jsonc
{
  "default": "my-custom",
  "providers": {
    "my-custom": {
      "baseUrl": "https://my-api.example.com/v1",
      "protocol": "openai",
      "headers": { "X-Custom-Header": "value" }
    }
  }
}
```

#### PROVIDERS JSON Reference

| Field | Type | Description |
| --- | --- | --- |
| `default` | `string` | **Required.** Provider name for unmatched models |
| `routing` | `Record<string, string>` | Glob pattern → provider name (first match wins) |
| `providers` | `Record<string, ProviderConfig>` | Per-provider configuration overrides |

**ProviderConfig fields** (all optional for known providers):

| Field | Type | Description |
| --- | --- | --- |
| `baseUrl` | `string` | Override base URL (required for custom providers) |
| `protocol` | `"openai" \| "anthropic"` | API protocol (defaults from registry) |
| `timeout` | `number` | Request timeout in seconds |
| `headers` | `Record<string, string>` | Additional HTTP headers |
| `modelMapping` | `Record<string, string>` | Model keyword mapping (e.g., `{"opus": "gpt-4o"}`) |

#### API Key Resolution

For each request, after routing to a provider:

1. **Provider key** (`PROVIDER_<NAME>_API_KEY`) → use it (managed mode)
2. **No provider key** → passthrough client's API key

This allows mixed setups: some providers with server-managed keys, others with client key passthrough.

#### Global Environment Variables

| Variable | Description | Default |
| --- | --- | --- |
| `PROVIDERS` | Multi-provider config (JSON string) | — (legacy mode) |
| `ANTHROPIC_API_KEY` | Optional (Secret) proxy-level auth key | Accepts any key |
| `MAX_TOKENS_LIMIT` | Maximum token limit | `16384` |
| `MIN_TOKENS_LIMIT` | Minimum token limit | `4096` |
| `REQUEST_TIMEOUT` | Default request timeout (seconds) | `90` |
| `LOG_LEVEL` | Log verbosity (`WARNING` / `DEBUG`) | `WARNING` |

> **💡 Debug Logging:** Set `LOG_LEVEL=DEBUG` to enable detailed request/response logging (raw content, streaming deltas, provider routing). Useful for troubleshooting. Keep `WARNING` in production to avoid log costs.

#### Legacy Mode (Single Provider)

If `PROVIDERS` is **not set**, the proxy runs in legacy single-provider mode using these variables:

| Variable | Description | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | Backend API key (leave unset for passthrough) | — |
| `OPENAI_BASE_URL` | API base URL | `https://api.openai.com/v1` |
| `PASSTHROUGH_MODELS` | Comma-separated model prefixes for Anthropic passthrough | — |
| `ENABLE_MODEL_MAPPING` | Enable Claude→provider model mapping | `false` |
| `BIG_MODEL` | Opus mapping target | `gpt-4o` |
| `MIDDLE_MODEL` | Sonnet mapping target | `gpt-4o` |
| `SMALL_MODEL` | Haiku mapping target | `gpt-4o-mini` |
| `AZURE_API_VERSION` | Azure OpenAI API version | — |
| `CUSTOM_HEADERS` | Custom HTTP headers (JSON string) | — |

### 📡 Provider Examples

<details>
<summary><b>Multi-Provider: OpenAI + DeepSeek + GLM</b></summary>

Set `PROVIDERS` (plaintext):

```json
{
  "default": "openai",
  "routing": {
    "glm-*": "glm",
    "deepseek-*": "deepseek"
  },
  "providers": {
    "openai": {
      "modelMapping": { "opus": "gpt-4o", "sonnet": "gpt-4o", "haiku": "gpt-4o-mini" }
    },
    "glm": {},
    "deepseek": {}
  }
}
```

Set secrets: `PROVIDER_OPENAI_API_KEY`, `PROVIDER_GLM_API_KEY`, `PROVIDER_DEEPSEEK_API_KEY`

</details>

<details>
<summary><b>Multi-Provider: OpenRouter as default + Anthropic passthrough</b></summary>

```json
{
  "default": "openrouter",
  "routing": { "claude-*": "anthropic" },
  "providers": {
    "openrouter": {},
    "anthropic": {}
  }
}
```

Set secrets: `PROVIDER_OPENROUTER_API_KEY`, `PROVIDER_ANTHROPIC_API_KEY`

</details>

<details>
<summary><b>Multi-Provider: OpenRouter with slash-style model routing</b></summary>

OpenRouter uses `vendor/model` format (e.g., `qwen/qwen-3.6`, `openai/gpt-4o`). The proxy handles slashes in model names seamlessly:

```json
{
  "default": "openrouter",
  "routing": {
    "qwen/*": "openrouter",
    "openai/*": "openrouter",
    "meta-llama/*": "openrouter",
    "qwen-*": "qwen",
    "deepseek-*": "deepseek"
  },
  "providers": {
    "openrouter": {},
    "qwen": {},
    "deepseek": {}
  }
}
```

- Models with `/` prefix (e.g., `qwen/qwen-3.6`) → OpenRouter
- Models without `/` prefix (e.g., `qwen-turbo`) → native provider
- To strip the vendor prefix for a native provider, use `modelMapping`:

```json
{
  "default": "openrouter",
  "routing": { "qwen/*": "qwen" },
  "providers": {
    "openrouter": {},
    "qwen": {
      "modelMapping": { "qwen/qwen-3.6": "qwen-3.6" }
    }
  }
}
```

</details>

<details>
<summary><b>Single Provider: GLM 5.1 (legacy mode)</b></summary>

No `PROVIDERS` needed — use legacy env vars:

```
OPENAI_BASE_URL = https://open.bigmodel.cn/api/paas/v4
```

Client key is forwarded to GLM. Automatically converts `reasoning_content` to Claude thinking blocks.

</details>

<details>
<summary><b>Single Provider: OpenAI with model mapping (legacy mode)</b></summary>

```
OPENAI_BASE_URL = https://api.openai.com/v1
ENABLE_MODEL_MAPPING = true
BIG_MODEL = gpt-4o
MIDDLE_MODEL = gpt-4o
SMALL_MODEL = gpt-4o-mini
```

Set secret: `OPENAI_API_KEY`

</details>

<details>
<summary><b>Single Provider: Azure OpenAI (legacy mode)</b></summary>

```
OPENAI_BASE_URL = https://your-resource.openai.azure.com/openai/deployments/your-deployment
ENABLE_MODEL_MAPPING = true
BIG_MODEL = gpt-4
MIDDLE_MODEL = gpt-4
SMALL_MODEL = gpt-35-turbo
```

Set secrets: `OPENAI_API_KEY`, `AZURE_API_VERSION`

</details>

<details>
<summary><b>Hybrid: OpenCode Go — GLM + MiniMax (legacy mode)</b></summary>

```
OPENAI_BASE_URL = https://opencode.ai/zen/go/v1
PASSTHROUGH_MODELS = minimax
```

GLM models → OpenAI conversion, MiniMax models → Anthropic passthrough.

</details>

### 🖥️ Using with Claude Code

```bash
# Multi-provider mode — client key forwarded to resolved provider
ANTHROPIC_BASE_URL=https://your-worker.workers.dev \
ANTHROPIC_API_KEY="your-api-key" \
claude

# If ANTHROPIC_API_KEY validation is enabled on the proxy
ANTHROPIC_BASE_URL=https://your-worker.workers.dev \
ANTHROPIC_API_KEY="your-matching-proxy-key" \
claude
```

### 📋 API Endpoints

| Method | Path | Description | Auth Required |
| --- | --- | --- | --- |
| GET | `/` | Proxy info | No |
| GET | `/health` | Health check | No |
| POST | `/v1/messages` | Chat completions (proxy) | Yes |
| POST | `/v1/messages/count_tokens` | Token count estimation | Yes |

### 🔄 Migration from Legacy to Multi-Provider

If you're upgrading from a single-provider setup, **no changes are required** — your existing config continues to work.

To opt into multi-provider mode, add a `PROVIDERS` JSON variable:

```jsonc
// Before (legacy):
// OPENAI_BASE_URL = https://open.bigmodel.cn/api/paas/v4
// OPENAI_API_KEY = sk-xxx

// After (multi-provider):
// PROVIDERS = {"default":"glm","providers":{"glm":{}}}
// PROVIDER_GLM_API_KEY = sk-xxx  (secret)
```

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
├── index.ts                    # Worker entry & CORS
├── types.ts                    # TypeScript type definitions
├── config.ts                   # Config loading (legacy + multi-provider)
├── constants.ts                # Shared constants
├── auth.ts                     # Proxy-level authentication
├── router.ts                   # Model → Provider routing (glob matching)
├── handlers.ts                 # Request handlers (parse → route → dispatch)
├── client.ts                   # HTTP utilities
├── providers/
│   ├── index.ts                # Provider dispatch
│   ├── registry.ts             # Known providers registry (14 providers)
│   ├── openai-provider.ts      # OpenAI-compatible provider
│   └── anthropic-provider.ts   # Anthropic passthrough provider
└── conversion/
    ├── request.ts              # Claude → OpenAI request conversion
    └── response.ts             # OpenAI → Claude response conversion
```

---

### 🔒 Security Notes

- **Authentication**: Proxy-level auth via `ANTHROPIC_API_KEY` uses constant-time string comparison to prevent timing attacks.
- **Error sanitization**: All error messages are classified into generic user-facing categories — raw backend errors are never exposed to clients.
- **Header blocklist**: Custom headers cannot override `Authorization`, `api-key`, `Host`, or `Content-Type`.
- **No SSRF**: Backend URLs are resolved from environment variables and the built-in provider registry only — never from user request input.
- **CORS**: The proxy returns `Access-Control-Allow-Origin: *` by default. This is required for browser-based clients but means **any website can call your proxy if they know the URL and API key**. For stricter security, add Cloudflare Access or IP allowlists.
- **Rate limiting**: No built-in rate limiting. Use [Cloudflare Rate Limiting Rules](https://developers.cloudflare.com/waf/rate-limiting-rules/) for production deployments.
- **DEBUG logging**: When `LOG_LEVEL=DEBUG`, full request/response content (including user messages) is logged. **Do not use DEBUG in production** with sensitive data.
- **Zero production dependencies**: The deployed worker bundle has no npm dependencies — pure TypeScript compiled to a single file.
- **Region bypass (Smart Placement)**: The Worker uses Cloudflare [Smart Placement](https://developers.cloudflare.com/workers/configuration/smart-placement/) to run near the backend API instead of the end-user. This routes outbound requests through non-restricted regions, bypassing upstream provider region blocks.

---

<a id="中文"></a>

## 中文

让 **Claude Code** 使用任意 OpenAI 兼容 API 的 **Cloudflare Worker** 代理 — 内置**多供应商路由**。一次部署即可将不同模型路由到不同供应商（OpenAI、DeepSeek、GLM、Gemini、Anthropic 等），自动处理 API 格式转换。

### ✨ 特性

- **多供应商路由** — 通过 glob 模式将模型路由到不同供应商（`glm-*` → 智谱、`gpt-*` → OpenAI 等）
- **14 个内置供应商** — 预置 OpenAI、OpenRouter、DeepSeek、GLM、Qwen、Gemini、Anthropic、MiniMax、OpenCode、豆包、硅基流动、Groq、Mistral、Together 的 URL
- 完整的 `/v1/messages` Claude API 兼容
- 流式 SSE 响应、函数调用 (tool use)、图片输入
- 自动将 `reasoning_content` 转为 Claude 思维块
- **双协议支持**：OpenAI 兼容转换 + Anthropic 直转
- **每个供应商独立 API Key** — `PROVIDER_<NAME>_API_KEY` 作为密钥存储，或客户端 key 透传
- **每个供应商独立模型映射** — 将 Claude 模型名 (opus/sonnet/haiku) 映射为供应商模型
- 部署在 Cloudflare 全球边缘网络，低延迟
- API Key 常量时间比较，防止时序攻击
- **完全向后兼容** — 旧的单供应商配置继续有效

### 🏗️ 架构

```
客户端请求 (Claude API 格式)
       │
       ▼
  ┌─────────┐
  │  认证    │  ← ANTHROPIC_API_KEY 验证
  └────┬────┘
       ▼
  ┌─────────┐
  │  路由    │  ← 模型名 → 供应商 (glob 匹配)
  └────┬────┘
       ▼
  ┌──────────────┐
  │  供应商分发   │  ← 解析后的供应商配置 (URL、密钥、协议)
  └──┬───────┬───┘
     │       │
     ▼       ▼
  OpenAI  Anthropic
  供应商   供应商
     │       │
     ▼       ▼
  Claude→  直转
  OpenAI   (原生格式)
     │       │
     ▼       ▼
  后端 API  后端 API
```

### 🚀 部署

#### 方式一：Fork + Cloudflare 自动部署（推荐）

1. **Fork 本仓库** 到你的 GitHub 账号
2. 登录 [Cloudflare Dashboard](https://dash.cloudflare.com/) → **Workers & Pages** → **Create**
3. 选择 **Import a repository**，连接 GitHub 并选择 fork 的仓库
4. 构建配置保持默认（`npm run build` / `npm run deploy`）
5. 在 Worker 的 **Settings → Variables and Secrets** 中配置（见下方[配置](#-配置-1)）
6. 所有配置通过 Dashboard 完成，同步上游更新不会覆盖你的配置

#### 方式二：Wrangler CLI

```bash
git clone https://github.com/randomspace77/claude-code-proxy-cf-workers.git
cd claude-code-proxy-cf-workers
npm install

# 设置供应商 API Key（密钥）
wrangler secret put PROVIDER_OPENAI_API_KEY
wrangler secret put PROVIDER_DEEPSEEK_API_KEY  # 等等
wrangler secret put ANTHROPIC_API_KEY          # 可选，代理级认证

npm run deploy
```

#### 方式三：本地开发

```bash
cp .env.example .dev.vars   # 编辑填入 API Key
npm install
npm run dev
```

### 🔧 配置

所有配置通过 **Cloudflare Dashboard**（Settings → Variables and Secrets）完成。`wrangler.toml` 中不包含 `[vars]`，确保部署不会覆盖你的配置。

#### 多供应商模式（推荐）

设置 `PROVIDERS` 环境变量（明文 JSON）来配置多供应商和模型路由：

```jsonc
{
  "default": "openai",
  "routing": {
    "glm-*": "glm",
    "deepseek-*": "deepseek",
    "claude-*": "anthropic",
    "gpt-*": "openai",
    "o1-*": "openai",
    "o3-*": "openai",
    "gemini-*": "gemini",
    "qwen-*": "qwen"
  },
  "providers": {
    "openai": {
      "modelMapping": { "opus": "gpt-4o", "sonnet": "gpt-4o", "haiku": "gpt-4o-mini" }
    },
    "glm": { "timeout": 120 },
    "deepseek": {},
    "anthropic": {},
    "gemini": {},
    "qwen": {}
  }
}
```

然后为每个供应商设置 API Key（**密钥**）：

```bash
wrangler secret put PROVIDER_OPENAI_API_KEY
wrangler secret put PROVIDER_GLM_API_KEY
wrangler secret put PROVIDER_DEEPSEEK_API_KEY
# ...
```

> **密钥命名规则**：`PROVIDER_<大写名称>_API_KEY`。供应商名中的连字符变为下划线（如 `my-custom` → `PROVIDER_MY_CUSTOM_API_KEY`）。

#### 内置供应商

以下供应商已预置 URL，只需添加 API Key：

| 名称 | Base URL | 协议 |
| --- | --- | --- |
| `openai` | `https://api.openai.com/v1` | openai |
| `openrouter` | `https://openrouter.ai/api/v1` | openai |
| `deepseek` | `https://api.deepseek.com/v1` | openai |
| `glm` | `https://open.bigmodel.cn/api/paas/v4` | openai |
| `qwen` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | openai |
| `gemini` | `https://generativelanguage.googleapis.com/v1beta/openai` | openai |
| `anthropic` | `https://api.anthropic.com/v1` | anthropic |
| `minimax` | `https://api.minimax.chat/v1` | anthropic |
| `opencode` | `https://opencode.ai/zen/go/v1` | openai |
| `doubao` | `https://ark.cn-beijing.volces.com/api/v3` | openai |
| `siliconflow` | `https://api.siliconflow.cn/v1` | openai |
| `groq` | `https://api.groq.com/openai/v1` | openai |
| `mistral` | `https://api.mistral.ai/v1` | openai |
| `together` | `https://api.together.xyz/v1` | openai |

也可定义**自定义供应商**：

```jsonc
{
  "default": "my-custom",
  "providers": {
    "my-custom": {
      "baseUrl": "https://my-api.example.com/v1",
      "protocol": "openai",
      "headers": { "X-Custom-Header": "value" }
    }
  }
}
```

#### PROVIDERS JSON 参考

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `default` | `string` | **必需。** 未匹配模型的默认供应商 |
| `routing` | `Record<string, string>` | glob 模式 → 供应商名（先匹配先命中） |
| `providers` | `Record<string, ProviderConfig>` | 每个供应商的配置覆盖 |

**ProviderConfig 字段**（内置供应商可全部省略）：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `baseUrl` | `string` | 覆盖 Base URL（自定义供应商必须提供） |
| `protocol` | `"openai" \| "anthropic"` | API 协议（内置供应商有默认值） |
| `timeout` | `number` | 请求超时（秒） |
| `headers` | `Record<string, string>` | 额外 HTTP 头 |
| `modelMapping` | `Record<string, string>` | 模型关键词映射（如 `{"opus": "gpt-4o"}`） |

#### API Key 解析

每次请求路由到供应商后：

1. **供应商密钥** (`PROVIDER_<NAME>_API_KEY`) → 使用该密钥（托管模式）
2. **无供应商密钥** → 透传客户端 API Key

支持混合配置：部分供应商使用服务端密钥，其他透传客户端密钥。

#### 全局环境变量

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `PROVIDERS` | 多供应商配置 (JSON) | — (旧模式) |
| `ANTHROPIC_API_KEY` | 可选 (Secret) 代理级认证密钥 | 接受任意 Key |
| `MAX_TOKENS_LIMIT` | 最大 token 数 | `16384` |
| `MIN_TOKENS_LIMIT` | 最小 token 数 | `4096` |
| `REQUEST_TIMEOUT` | 默认请求超时 (秒) | `90` |
| `LOG_LEVEL` | 日志级别（`WARNING` / `DEBUG`） | `WARNING` |

> **💡 调试日志：** 设置 `LOG_LEVEL=DEBUG` 可启用详细的请求/响应日志（原始内容、流式数据、供应商路由）。适用于排查问题。生产环境建议保持 `WARNING` 以避免日志费用。

#### 旧模式（单供应商）

如果未设置 `PROVIDERS`，代理以旧的单供应商模式运行：

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 后端 API Key（不设置则透传） | — |
| `OPENAI_BASE_URL` | API 基础 URL | `https://api.openai.com/v1` |
| `PASSTHROUGH_MODELS` | 逗号分隔的模型前缀，匹配的走 Anthropic 直转 | — |
| `ENABLE_MODEL_MAPPING` | 启用 Claude→供应商模型映射 | `false` |
| `BIG_MODEL` | Opus 映射目标 | `gpt-4o` |
| `MIDDLE_MODEL` | Sonnet 映射目标 | `gpt-4o` |
| `SMALL_MODEL` | Haiku 映射目标 | `gpt-4o-mini` |
| `AZURE_API_VERSION` | Azure OpenAI API 版本 | — |
| `CUSTOM_HEADERS` | 自定义 HTTP 头 (JSON) | — |

### 📡 供应商配置示例

<details>
<summary><b>多供应商：OpenAI + DeepSeek + GLM</b></summary>

设置 `PROVIDERS`（明文）：

```json
{
  "default": "openai",
  "routing": {
    "glm-*": "glm",
    "deepseek-*": "deepseek"
  },
  "providers": {
    "openai": {
      "modelMapping": { "opus": "gpt-4o", "sonnet": "gpt-4o", "haiku": "gpt-4o-mini" }
    },
    "glm": {},
    "deepseek": {}
  }
}
```

设置密钥：`PROVIDER_OPENAI_API_KEY`、`PROVIDER_GLM_API_KEY`、`PROVIDER_DEEPSEEK_API_KEY`

</details>

<details>
<summary><b>多供应商：OpenRouter 默认 + Anthropic 直转</b></summary>

```json
{
  "default": "openrouter",
  "routing": { "claude-*": "anthropic" },
  "providers": {
    "openrouter": {},
    "anthropic": {}
  }
}
```

设置密钥：`PROVIDER_OPENROUTER_API_KEY`、`PROVIDER_ANTHROPIC_API_KEY`

</details>

<details>
<summary><b>多供应商：OpenRouter 含斜杠格式模型路由</b></summary>

OpenRouter 使用 `厂商/模型` 格式（如 `qwen/qwen-3.6`、`openai/gpt-4o`）。代理可无缝处理含斜杠的模型名：

```json
{
  "default": "openrouter",
  "routing": {
    "qwen/*": "openrouter",
    "openai/*": "openrouter",
    "meta-llama/*": "openrouter",
    "qwen-*": "qwen",
    "deepseek-*": "deepseek"
  },
  "providers": {
    "openrouter": {},
    "qwen": {},
    "deepseek": {}
  }
}
```

- 带 `/` 前缀的模型（如 `qwen/qwen-3.6`）→ OpenRouter
- 不带 `/` 前缀的模型（如 `qwen-turbo`）→ 原生供应商
- 若需去除厂商前缀发送给原生供应商，可用 `modelMapping`：

```json
{
  "default": "openrouter",
  "routing": { "qwen/*": "qwen" },
  "providers": {
    "openrouter": {},
    "qwen": {
      "modelMapping": { "qwen/qwen-3.6": "qwen-3.6" }
    }
  }
}
```

</details>

<details>
<summary><b>单供应商：GLM 5.1（旧模式）</b></summary>

无需设置 `PROVIDERS`，使用旧变量：

```
OPENAI_BASE_URL = https://open.bigmodel.cn/api/paas/v4
```

客户端 key 透传给 GLM。自动将 `reasoning_content` 转换为 Claude 思维块。

</details>

<details>
<summary><b>单供应商：OpenAI + 模型映射（旧模式）</b></summary>

```
OPENAI_BASE_URL = https://api.openai.com/v1
ENABLE_MODEL_MAPPING = true
BIG_MODEL = gpt-4o
MIDDLE_MODEL = gpt-4o
SMALL_MODEL = gpt-4o-mini
```

设置密钥：`OPENAI_API_KEY`

</details>

<details>
<summary><b>混合：OpenCode Go — GLM + MiniMax（旧模式）</b></summary>

```
OPENAI_BASE_URL = https://opencode.ai/zen/go/v1
PASSTHROUGH_MODELS = minimax
```

GLM 模型走 OpenAI 转换，MiniMax 模型走 Anthropic 直转。

</details>

### 🖥️ 使用 Claude Code

```bash
# 多供应商模式 — 客户端 key 透传给路由到的供应商
ANTHROPIC_BASE_URL=https://your-worker.workers.dev \
ANTHROPIC_API_KEY="your-api-key" \
claude

# 如果代理启用了 ANTHROPIC_API_KEY 验证
ANTHROPIC_BASE_URL=https://your-worker.workers.dev \
ANTHROPIC_API_KEY="your-matching-proxy-key" \
claude
```

### 📋 API 端点

| 方法 | 路径 | 说明 | 需要认证 |
| --- | --- | --- | --- |
| GET | `/` | 代理信息 | 否 |
| GET | `/health` | 健康检查 | 否 |
| POST | `/v1/messages` | 聊天补全（代理） | 是 |
| POST | `/v1/messages/count_tokens` | Token 计数估算 | 是 |

### 🔄 从旧版迁移到多供应商

升级自单供应商配置时，**无需任何改动** — 现有配置继续有效。

要启用多供应商模式，添加 `PROVIDERS` JSON 变量：

```jsonc
// 之前（旧模式）：
// OPENAI_BASE_URL = https://open.bigmodel.cn/api/paas/v4
// OPENAI_API_KEY = sk-xxx

// 之后（多供应商）：
// PROVIDERS = {"default":"glm","providers":{"glm":{}}}
// PROVIDER_GLM_API_KEY = sk-xxx  (密钥)
```

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
├── index.ts                    # Worker 入口 & CORS
├── types.ts                    # TypeScript 类型定义
├── config.ts                   # 配置加载（旧模式 + 多供应商）
├── constants.ts                # 共享常量
├── auth.ts                     # 代理级认证
├── router.ts                   # 模型 → 供应商路由 (glob 匹配)
├── handlers.ts                 # 请求处理（解析 → 路由 → 分发）
├── client.ts                   # HTTP 工具
├── providers/
│   ├── index.ts                # 供应商分发
│   ├── registry.ts             # 内置供应商注册表（14 个供应商）
│   ├── openai-provider.ts      # OpenAI 兼容供应商
│   └── anthropic-provider.ts   # Anthropic 直转供应商
└── conversion/
    ├── request.ts              # Claude → OpenAI 请求转换
    └── response.ts             # OpenAI → Claude 响应转换
```

---

### 🔒 安全说明

- **身份认证**：通过 `ANTHROPIC_API_KEY` 进行代理级认证，使用恒定时间字符串比较防止时序攻击。
- **错误信息脱敏**：所有错误信息均分类为通用的用户友好消息——后端原始错误永远不会暴露给客户端。
- **请求头黑名单**：自定义请求头无法覆盖 `Authorization`、`api-key`、`Host` 或 `Content-Type`。
- **无 SSRF 风险**：后端 URL 仅从环境变量和内置供应商注册表解析——绝不来自用户请求输入。
- **CORS**：代理默认返回 `Access-Control-Allow-Origin: *`。这是浏览器客户端所需的，但意味着**任何网站都可以在知道 URL 和 API 密钥的情况下调用你的代理**。如需更严格的安全控制，请配置 Cloudflare Access 或 IP 白名单。
- **速率限制**：无内置速率限制。生产环境请使用 [Cloudflare 速率限制规则](https://developers.cloudflare.com/waf/rate-limiting-rules/)。
- **DEBUG 日志**：当 `LOG_LEVEL=DEBUG` 时，会记录完整的请求/响应内容（包括用户消息）。**生产环境请勿使用 DEBUG 模式**处理敏感数据。
- **零生产依赖**：部署的 Worker 包没有 npm 依赖——纯 TypeScript 编译为单一文件。
- **区域绕过（Smart Placement）**：Worker 启用了 Cloudflare [Smart Placement](https://developers.cloudflare.com/workers/configuration/smart-placement/)，在靠近后端 API 的节点运行而非靠近用户。出站请求的 IP 来自非受限区域，自动绕过上游供应商的区域限制。

---

## 📄 License

MIT

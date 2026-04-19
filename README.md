# Claude Code Proxy (CF Workers)

[English](#english) | [中文](#中文)

---

<a id="english"></a>

## English

A proxy that lets **Claude Code** use any OpenAI-compatible API — with built-in **multi-provider routing**. Deploy on **Cloudflare Workers** or as a **Docker / Node.js** service. One deployment routes different models to different providers (OpenAI, DeepSeek, GLM, Gemini, Anthropic, etc.) and handles API format conversion automatically.

### ✨ Features

- **Multi-provider routing** — route models to different providers by glob patterns (`glm-*` → Zhipu, `gpt-*` → OpenAI, etc.)
- **14 built-in providers** — hardcoded URLs for OpenAI, OpenRouter, DeepSeek, GLM, Qwen, Gemini, Anthropic, MiniMax, OpenCode, Doubao, SiliconFlow, Groq, Mistral, Together
- Full `/v1/messages` Claude API compatibility
- Streaming SSE, function calling (tool use), image input
- Automatic `reasoning_content` → Claude thinking blocks
- **Dual protocol support**: OpenAI-compatible conversion + Anthropic passthrough
- **Per-provider API keys** — `PROVIDER_<NAME>_API_KEY` as secrets, or client key passthrough
- **Per-provider model mapping** — map Claude model names (opus/sonnet/haiku) to provider-specific models
- Deployed on Cloudflare's global edge network with [Smart Placement](https://developers.cloudflare.com/workers/configuration/smart-placement/) for region bypass
- **Docker / Node.js deployment** — run as a native Node.js service without Cloudflare (single ~48 KB bundled file, zero runtime deps)
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

### 🚀 Quick Start

#### Step 1: Deploy

**Option A: Fork + Cloudflare Auto-Deploy (Recommended)**

1. **Fork this repository** to your GitHub account
2. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/) → **Workers & Pages** → **Create**
3. Select **Import a repository**, connect your GitHub account, choose the forked repo
4. Keep default build settings — click **Deploy**
5. Configure environment variables in Worker **Settings → Variables and Secrets** (see Step 2)

> ℹ️ All configuration lives in the Dashboard. Syncing upstream updates will **never** overwrite your settings (`keep_vars = true`).

**Option B: Wrangler CLI**

```bash
git clone https://github.com/randomspace77/claude-code-proxy-cf-workers.git
cd claude-code-proxy-cf-workers
npm install
npm run deploy
```

**Option C: Docker (Node.js)**

```bash
git clone https://github.com/randomspace77/claude-code-proxy-cf-workers.git
cd claude-code-proxy-cf-workers
cp .env.example .env   # Edit with your API keys
docker compose up -d
```

The server runs at `http://localhost:3000`. See [Docker Deployment](#-docker-deployment-nodejs) for details.

**Option D: Local Development**

```bash
cp .env.example .dev.vars   # Edit with your API keys
npm install
npm run dev
```

#### Step 2: Configure Environment Variables

After deploying, go to **Cloudflare Dashboard → Your Worker → Settings → Variables and Secrets**.

You need to set up to **3 things**:

| # | What | Where | Purpose |
|---|------|-------|---------|
| 1 | `PROVIDERS` | Plaintext Variable | Tells the proxy which providers to use and how to route models |
| 2 | `PROVIDER_<NAME>_API_KEY` | Secret | API key for each provider (one per provider) |
| 3 | `ANTHROPIC_API_KEY` | Secret | *(Optional)* Protects your proxy from unauthorized access |

**Example setup** (OpenAI as default + DeepSeek for deepseek models):

```
┌─────────────────────────────────────────────────────────────────┐
│ Cloudflare Dashboard → Settings → Variables and Secrets         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📄 Plaintext Variables:                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ PROVIDERS = {                                           │    │
│  │   "default": "openai",                                  │    │
│  │   "routing": { "deepseek-*": "deepseek" },              │    │
│  │   "providers": {                                        │    │
│  │     "openai": {},                                       │    │
│  │     "deepseek": {}                                      │    │
│  │   }                                                     │    │
│  │ }                                                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  🔐 Secrets:                                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ PROVIDER_OPENAI_API_KEY    = sk-xxxxx                   │    │
│  │ PROVIDER_DEEPSEEK_API_KEY  = sk-xxxxx                   │    │
│  │ ANTHROPIC_API_KEY          = my-proxy-password           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Or via CLI:
```bash
# Set PROVIDERS as plaintext variable in Dashboard, then:
wrangler secret put PROVIDER_OPENAI_API_KEY
wrangler secret put PROVIDER_DEEPSEEK_API_KEY
wrangler secret put ANTHROPIC_API_KEY   # optional
```

#### Step 3: Connect Claude Code

```bash
ANTHROPIC_BASE_URL=https://your-worker.your-subdomain.workers.dev \
ANTHROPIC_API_KEY="my-proxy-password" \
claude
```

> If you didn't set `ANTHROPIC_API_KEY` on the proxy, use any non-empty string as `ANTHROPIC_API_KEY`.

---

### 🔧 Configuration Reference

#### Understanding the Two Modes

| | Multi-Provider Mode | Legacy Mode |
|---|---|---|
| **When** | `PROVIDERS` env var is set | `PROVIDERS` is NOT set |
| **Capabilities** | Multiple providers, glob routing, per-provider keys | Single provider only |
| **Recommended** | ✅ Yes | For simple single-provider setups |

#### Multi-Provider Mode — `PROVIDERS` JSON

The `PROVIDERS` variable is a JSON string with three fields:

```jsonc
{
  // REQUIRED: which provider handles models that don't match any routing rule
  "default": "openai",

  // OPTIONAL: route specific models to specific providers (first match wins)
  "routing": {
    "glm-*": "glm",           // all models starting with "glm-" → glm provider
    "deepseek-*": "deepseek", // deepseek-chat, deepseek-coder, etc.
    "claude-*": "anthropic",  // passthrough to Anthropic
    "qwen/*": "openrouter"    // slash-style models (e.g., qwen/qwen-3.6)
  },

  // OPTIONAL: per-provider configuration overrides
  "providers": {
    "openai": {
      "modelMapping": { "opus": "gpt-4o", "sonnet": "gpt-4o", "haiku": "gpt-4o-mini" }
    },
    "glm": { "timeout": 120 },
    "deepseek": {},
    "anthropic": {},
    "openrouter": {}
  }
}
```

**How routing works:**
1. Client sends a request with `model: "deepseek-chat"`
2. Proxy checks each pattern in `routing` top-to-bottom
3. `"deepseek-*"` matches → routes to the `deepseek` provider
4. If nothing matches → uses the `default` provider

**ProviderConfig fields** (all optional for built-in providers):

| Field | Type | Description |
|-------|------|-------------|
| `baseUrl` | `string` | Override the provider's API URL (required for custom providers) |
| `protocol` | `"openai"` \| `"anthropic"` | API format — `openai` converts Claude↔OpenAI, `anthropic` passes through |
| `timeout` | `number` | Request timeout in seconds (overrides global `REQUEST_TIMEOUT`) |
| `headers` | `Record<string, string>` | Additional HTTP headers sent to this provider |
| `modelMapping` | `Record<string, string>` | Map model keywords to provider-specific names |
| `azureApiVersion` | `string` | Azure OpenAI API version (for Azure provider) |

#### API Key Setup (Important!)

Each provider needs its own API key, stored as a **Cloudflare Secret**:

```
Provider name in config  →  Secret name in Dashboard
─────────────────────────────────────────────────────
openai                   →  PROVIDER_OPENAI_API_KEY
deepseek                 →  PROVIDER_DEEPSEEK_API_KEY
glm                      →  PROVIDER_GLM_API_KEY
my-custom                →  PROVIDER_MY_CUSTOM_API_KEY  (hyphens → underscores)
opencode-passthrough     →  PROVIDER_OPENCODE_PASSTHROUGH_API_KEY
```

**Rule**: `PROVIDER_` + uppercase name (hyphens → underscores) + `_API_KEY`

**What happens if a provider key is missing?**
The proxy falls back to the client's `x-api-key` / `Authorization: Bearer` key. This is called **passthrough mode** — useful when users bring their own API keys.

#### Proxy Authentication — `ANTHROPIC_API_KEY`

| Scenario | Behavior |
|----------|----------|
| `ANTHROPIC_API_KEY` **is set** | Client must send matching key via `x-api-key` header. Unauthorized requests get `401`. |
| `ANTHROPIC_API_KEY` **not set** | Proxy accepts all requests (no auth). Only use in trusted networks! |

> ⚠️ **Always set `ANTHROPIC_API_KEY`** if your proxy is publicly accessible. Otherwise anyone who discovers your Worker URL can use your API keys.

#### Global Environment Variables

Set these in Dashboard as **Plaintext Variables** (not Secrets):

| Variable | Description | Default |
|----------|-------------|---------|
| `PROVIDERS` | Multi-provider config (JSON) — enables multi-provider mode | *(not set = legacy mode)* |
| `LOG_LEVEL` | `WARNING` or `DEBUG` | `WARNING` |
| `MAX_TOKENS_LIMIT` | Maximum tokens per response | `16384` |
| `MIN_TOKENS_LIMIT` | Minimum tokens per response | `4096` |
| `REQUEST_TIMEOUT` | Default request timeout in seconds | `90` |

Set these as **Secrets**:

| Variable | Description | Required? |
|----------|-------------|-----------|
| `ANTHROPIC_API_KEY` | Proxy-level auth key (clients must send this to access proxy) | Recommended |
| `PROVIDER_<NAME>_API_KEY` | Per-provider API keys (one for each provider in your config) | Yes, per provider |

> **💡 Log Levels:**
> - `WARNING` (default): Only logs errors and routing diagnostics. Safe for production.
> - `DEBUG`: Logs full request/response content including user prompts. **Never use in production with sensitive data.**

#### Built-in Providers

These 14 providers have hardcoded base URLs — you only need an API key:

| Name | Base URL | Protocol |
|------|----------|----------|
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

For any provider **not** in this list, specify `baseUrl` in the config:

```json
{
  "default": "my-llm",
  "providers": {
    "my-llm": {
      "baseUrl": "https://api.my-llm-service.com/v1",
      "protocol": "openai"
    }
  }
}
```

---

### 📡 Complete Examples

<details>
<summary><b>Example 1: OpenCode default + MiniMax passthrough + OpenRouter for Qwen</b></summary>

**Scenario**: Default models go to OpenCode, MiniMax models use Anthropic passthrough via OpenCode, Qwen 3.6+ goes to OpenRouter.

**PROVIDERS** (Plaintext Variable):
```json
{
  "default": "opencode",
  "routing": {
    "minimax*": "opencode-passthrough",
    "qwen/qwen3.6-plus": "openrouter"
  },
  "providers": {
    "opencode": {},
    "opencode-passthrough": {
      "baseUrl": "https://opencode.ai/zen/go/v1",
      "protocol": "anthropic"
    },
    "openrouter": {}
  }
}
```

**Secrets**:
```
PROVIDER_OPENCODE_API_KEY              = your-opencode-key
PROVIDER_OPENCODE_PASSTHROUGH_API_KEY  = your-opencode-key
PROVIDER_OPENROUTER_API_KEY            = your-openrouter-key
ANTHROPIC_API_KEY                      = my-proxy-password
```

**How it routes**:
- `minimax-abc` → matches `minimax*` → `opencode-passthrough` (Anthropic protocol)
- `qwen/qwen3.6-plus` → exact match → `openrouter`
- `gpt-4o`, `any-other-model` → no match → `opencode` (default)

</details>

<details>
<summary><b>Example 2: OpenAI + DeepSeek + GLM multi-provider</b></summary>

**PROVIDERS** (Plaintext Variable):
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
    "glm": { "timeout": 120 },
    "deepseek": {}
  }
}
```

**Secrets**: `PROVIDER_OPENAI_API_KEY`, `PROVIDER_GLM_API_KEY`, `PROVIDER_DEEPSEEK_API_KEY`

</details>

<details>
<summary><b>Example 3: OpenRouter as default + Anthropic passthrough for Claude models</b></summary>

**PROVIDERS** (Plaintext Variable):
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

**Secrets**: `PROVIDER_OPENROUTER_API_KEY`, `PROVIDER_ANTHROPIC_API_KEY`

</details>

<details>
<summary><b>Example 4: OpenRouter with slash-style models (qwen/qwen-3.6)</b></summary>

OpenRouter uses `vendor/model` format. The proxy handles slashes in model names seamlessly:

**PROVIDERS** (Plaintext Variable):
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

- `qwen/qwen-3.6` → matches `qwen/*` → OpenRouter
- `qwen-turbo` → matches `qwen-*` → native Qwen provider

To strip vendor prefix for native providers, use `modelMapping`:
```json
"qwen": {
  "modelMapping": { "qwen/qwen-3.6": "qwen-3.6" }
}
```

</details>

<details>
<summary><b>Example 5: Single provider — GLM (legacy mode, no PROVIDERS needed)</b></summary>

If you only need one provider, skip `PROVIDERS` entirely:

**Plaintext Variables**:
```
OPENAI_BASE_URL = https://open.bigmodel.cn/api/paas/v4
```

**Secrets**: None needed (client key is forwarded to GLM)

</details>

<details>
<summary><b>Example 6: Azure OpenAI (legacy mode)</b></summary>

**Plaintext Variables**:
```
OPENAI_BASE_URL      = https://your-resource.openai.azure.com/openai/deployments/your-deployment
ENABLE_MODEL_MAPPING = true
BIG_MODEL            = gpt-4
MIDDLE_MODEL         = gpt-4
SMALL_MODEL          = gpt-35-turbo
AZURE_API_VERSION    = 2024-02-15-preview
```

**Secrets**: `OPENAI_API_KEY`

</details>

---

### 📋 API Endpoints

| Method | Path | Description | Auth Required |
|--------|------|-------------|---------------|
| GET | `/` | Proxy info | No |
| GET | `/health` | Health check | No |
| POST | `/v1/messages` | Chat completions (proxy) | Yes |
| POST | `/v1/messages/count_tokens` | Token count estimation | Yes |

### 🔄 Migration from Legacy to Multi-Provider

Your existing config continues to work — **no changes required**. To opt into multi-provider:

```
Before (legacy):                          After (multi-provider):
─────────────────                         ─────────────────────
OPENAI_BASE_URL = https://xxx/v1          PROVIDERS = {"default":"glm","providers":{"glm":{}}}
OPENAI_API_KEY  = sk-xxx                  PROVIDER_GLM_API_KEY = sk-xxx  (secret)
```

#### Legacy Mode Variables (only when PROVIDERS is not set)

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Backend API key (leave unset for passthrough) | — |
| `OPENAI_BASE_URL` | API base URL | `https://api.openai.com/v1` |
| `PASSTHROUGH_MODELS` | Comma-separated model prefixes for Anthropic passthrough | — |
| `ENABLE_MODEL_MAPPING` | Enable Claude→provider model mapping | `false` |
| `BIG_MODEL` | Opus mapping target | `gpt-4o` |
| `MIDDLE_MODEL` | Sonnet mapping target | same as `BIG_MODEL` |
| `SMALL_MODEL` | Haiku mapping target | `gpt-4o-mini` |
| `AZURE_API_VERSION` | Azure OpenAI API version | — |
| `CUSTOM_HEADERS` | Custom HTTP headers (JSON string) | — |

### 🐳 Docker Deployment (Node.js)

In addition to Cloudflare Workers, you can deploy this proxy as a native Node.js service using Docker — no Wrangler or Cloudflare account required.

#### Quick Start

```bash
cp .env.example .env   # Edit with your API keys / PROVIDERS config
docker compose up -d
```

The proxy will be available at `http://localhost:3000`.

#### Environment Variables

All the same environment variables from [Configuration Reference](#-configuration-reference) apply. Set them in your `.env` file or pass them directly:

```bash
docker compose up -d -e PROVIDERS='{"default":"openai"}' -e PROVIDER_OPENAI_API_KEY=sk-xxx
```

Extra Docker-specific variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | HTTP listen port | `3000` |
| `HOST` | HTTP listen address | `0.0.0.0` |

#### Manual Build (without Docker)

```bash
npm install
npm run build:node       # Bundle to dist-node/server.mjs
npm run start:node       # Start the Node.js server
```

#### Custom Docker Compose

```yaml
services:
  claude-code-proxy:
    build: .
    ports:
      - "8080:3000"
    environment:
      - PORT=3000
      - PROVIDERS={"default":"openai","routing":{"deepseek-*":"deepseek"}}
      - PROVIDER_OPENAI_API_KEY=sk-xxx
      - PROVIDER_DEEPSEEK_API_KEY=sk-xxx
      - ANTHROPIC_API_KEY=my-proxy-password
    restart: unless-stopped
```

> **Note**: The Docker image is a multi-stage build — the final image contains only Node.js 20 Alpine and a single bundled JS file (~48 KB), with zero npm runtime dependencies.

---

### 🛠️ Development

```bash
npm install          # Install dependencies
npm run dev          # Local dev server (Wrangler)
npm run lint         # Type check
npm run test         # Run tests
npm run build        # Build Worker (dry-run)
npm run deploy       # Deploy to Cloudflare Workers
npm run build:node   # Bundle Node.js server
npm run start:node   # Start Node.js server
```

#### Project Structure

```
src/
├── app.ts                      # Core request handler (platform-agnostic)
├── index.ts                    # Cloudflare Worker entry point
├── node-server.ts              # Node.js HTTP server entry point
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

### ❓ Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `401` with generic error message | Provider API key missing or wrong | Check `PROVIDER_<NAME>_API_KEY` is set correctly as a Secret |
| All requests go to `api.openai.com` | `PROVIDERS` env var not set | Add `PROVIDERS` JSON in Dashboard (plaintext, not secret) |
| `Invalid PROVIDERS JSON` | BOM or extra whitespace in JSON | Re-paste the JSON cleanly — the proxy strips BOM but check for typos |
| `authentication_error` 401 | Proxy auth key mismatch | Client's `ANTHROPIC_API_KEY` must match the proxy's `ANTHROPIC_API_KEY` secret |
| Model routes to wrong provider | Routing order issue | Routing rules are checked top-to-bottom, first match wins |

> **Debug tip**: Set `LOG_LEVEL=DEBUG` temporarily, then use `wrangler tail` to see real-time logs including routing decisions and request/response content.

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

让 **Claude Code** 使用任意 OpenAI 兼容 API 的代理 — 内置**多供应商路由**。支持部署到 **Cloudflare Workers** 或作为 **Docker / Node.js** 服务运行。一次部署即可将不同模型路由到不同供应商（OpenAI、DeepSeek、GLM、Gemini、Anthropic 等），自动处理 API 格式转换。

### ✨ 特性

- **多供应商路由** — 通过 glob 模式将模型路由到不同供应商（`glm-*` → 智谱、`gpt-*` → OpenAI 等）
- **14 个内置供应商** — 预置 OpenAI、OpenRouter、DeepSeek、GLM、Qwen、Gemini、Anthropic、MiniMax、OpenCode、豆包、硅基流动、Groq、Mistral、Together 的 URL
- 完整的 `/v1/messages` Claude API 兼容
- 流式 SSE 响应、函数调用 (tool use)、图片输入
- 自动将 `reasoning_content` 转为 Claude 思维块
- **双协议支持**：OpenAI 兼容转换 + Anthropic 直转
- **每个供应商独立 API Key** — `PROVIDER_<NAME>_API_KEY` 作为密钥存储，或客户端 key 透传
- **每个供应商独立模型映射** — 将 Claude 模型名 (opus/sonnet/haiku) 映射为供应商模型
- 部署在 Cloudflare 全球边缘网络，启用 [Smart Placement](https://developers.cloudflare.com/workers/configuration/smart-placement/) 绕过区域限制
- **Docker / Node.js 部署** — 无需 Cloudflare，作为原生 Node.js 服务运行（单个约 48 KB 打包文件，零运行时依赖）
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

### 🚀 快速开始

#### 第一步：部署

**方式 A：Fork + Cloudflare 自动部署（推荐）**

1. **Fork 本仓库** 到你的 GitHub 账号
2. 登录 [Cloudflare Dashboard](https://dash.cloudflare.com/) → **Workers & Pages** → **Create**
3. 选择 **Import a repository**，连接 GitHub 并选择 fork 的仓库
4. 保持默认构建配置 — 点击 **Deploy**
5. 在 Worker 的 **Settings → Variables and Secrets** 中配置环境变量（见第二步）

> ℹ️ 所有配置都在 Dashboard 中完成。同步上游更新**不会**覆盖你的配置（`keep_vars = true`）。

**方式 B：Wrangler CLI**

```bash
git clone https://github.com/randomspace77/claude-code-proxy-cf-workers.git
cd claude-code-proxy-cf-workers
npm install
npm run deploy
```

**方式 C：Docker 部署（Node.js）**

```bash
git clone https://github.com/randomspace77/claude-code-proxy-cf-workers.git
cd claude-code-proxy-cf-workers
cp .env.example .env   # 编辑填入 API Key / PROVIDERS 配置
docker compose up -d
```

服务默认运行在 `http://localhost:3000`。详见 [Docker 部署](#-docker-部署nodejs)。

**方式 D：本地开发**

```bash
cp .env.example .dev.vars   # 编辑填入 API Key
npm install
npm run dev
```

#### 第二步：配置环境变量

部署完成后，进入 **Cloudflare Dashboard → 你的 Worker → Settings → Variables and Secrets**。

你需要设置最多 **3 样东西**：

| # | 内容 | 类型 | 用途 |
|---|------|------|------|
| 1 | `PROVIDERS` | 明文变量 | 告诉代理使用哪些供应商、如何路由模型 |
| 2 | `PROVIDER_<NAME>_API_KEY` | 密钥 (Secret) | 每个供应商的 API Key（每个供应商一个） |
| 3 | `ANTHROPIC_API_KEY` | 密钥 (Secret) | *（可选）* 保护代理不被未授权访问 |

**配置示例**（OpenAI 为默认 + DeepSeek 处理 deepseek 模型）：

```
┌─────────────────────────────────────────────────────────────────┐
│ Cloudflare Dashboard → Settings → Variables and Secrets         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📄 明文变量 (Plaintext):                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ PROVIDERS = {                                           │    │
│  │   "default": "openai",                                  │    │
│  │   "routing": { "deepseek-*": "deepseek" },              │    │
│  │   "providers": {                                        │    │
│  │     "openai": {},                                       │    │
│  │     "deepseek": {}                                      │    │
│  │   }                                                     │    │
│  │ }                                                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  🔐 密钥 (Secrets):                                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ PROVIDER_OPENAI_API_KEY    = sk-xxxxx                   │    │
│  │ PROVIDER_DEEPSEEK_API_KEY  = sk-xxxxx                   │    │
│  │ ANTHROPIC_API_KEY          = my-proxy-password           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

或使用命令行：
```bash
# 在 Dashboard 中设置 PROVIDERS 明文变量后：
wrangler secret put PROVIDER_OPENAI_API_KEY
wrangler secret put PROVIDER_DEEPSEEK_API_KEY
wrangler secret put ANTHROPIC_API_KEY   # 可选
```

#### 第三步：连接 Claude Code

```bash
ANTHROPIC_BASE_URL=https://your-worker.your-subdomain.workers.dev \
ANTHROPIC_API_KEY="my-proxy-password" \
claude
```

> 如果代理未设置 `ANTHROPIC_API_KEY`，使用任意非空字符串作为 `ANTHROPIC_API_KEY` 即可。

---

### 🔧 配置参考

#### 理解两种模式

| | 多供应商模式 | 旧模式 |
|---|---|---|
| **何时** | 设置了 `PROVIDERS` 环境变量 | 未设置 `PROVIDERS` |
| **能力** | 多个供应商、glob 路由、每个供应商独立密钥 | 仅单个供应商 |
| **推荐** | ✅ 是 | 适用于简单的单供应商场景 |

#### 多供应商模式 — `PROVIDERS` JSON

`PROVIDERS` 变量是一个 JSON 字符串，包含三个字段：

```jsonc
{
  // 必需：未匹配任何路由规则的模型使用哪个供应商
  "default": "openai",

  // 可选：将特定模型路由到特定供应商（从上到下，首次匹配生效）
  "routing": {
    "glm-*": "glm",           // 所有以 "glm-" 开头的模型 → glm 供应商
    "deepseek-*": "deepseek", // deepseek-chat, deepseek-coder 等
    "claude-*": "anthropic",  // 直转到 Anthropic
    "qwen/*": "openrouter"    // 斜杠格式模型（如 qwen/qwen-3.6）
  },

  // 可选：每个供应商的配置覆盖
  "providers": {
    "openai": {
      "modelMapping": { "opus": "gpt-4o", "sonnet": "gpt-4o", "haiku": "gpt-4o-mini" }
    },
    "glm": { "timeout": 120 },
    "deepseek": {},
    "anthropic": {},
    "openrouter": {}
  }
}
```

**路由工作原理：**
1. 客户端发送请求，`model: "deepseek-chat"`
2. 代理从上到下检查 `routing` 中的每条规则
3. `"deepseek-*"` 匹配 → 路由到 `deepseek` 供应商
4. 如果没有匹配 → 使用 `default` 供应商

**ProviderConfig 字段**（内置供应商可全部省略）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `baseUrl` | `string` | 覆盖供应商 API URL（自定义供应商必须提供） |
| `protocol` | `"openai"` \| `"anthropic"` | API 格式 — `openai` 做 Claude↔OpenAI 转换，`anthropic` 直转 |
| `timeout` | `number` | 请求超时秒数（覆盖全局 `REQUEST_TIMEOUT`） |
| `headers` | `Record<string, string>` | 发送给该供应商的额外 HTTP 头 |
| `modelMapping` | `Record<string, string>` | 模型关键词映射 |
| `azureApiVersion` | `string` | Azure OpenAI API 版本 |

#### API Key 配置（重要！）

每个供应商需要自己的 API Key，存储为 **Cloudflare Secret**：

```
配置中的供应商名  →  Dashboard 中的 Secret 名
─────────────────────────────────────────────
openai           →  PROVIDER_OPENAI_API_KEY
deepseek         →  PROVIDER_DEEPSEEK_API_KEY
glm              →  PROVIDER_GLM_API_KEY
my-custom        →  PROVIDER_MY_CUSTOM_API_KEY  (连字符 → 下划线)
opencode-pt      →  PROVIDER_OPENCODE_PT_API_KEY
```

**命名规则**：`PROVIDER_` + 大写名称（连字符变下划线）+ `_API_KEY`

**如果供应商 Key 未设置怎么办？**
代理会回退到客户端的 `x-api-key` / `Authorization: Bearer` 密钥。这叫**透传模式** — 适用于让用户自带 API Key 的场景。

#### 代理认证 — `ANTHROPIC_API_KEY`

| 场景 | 行为 |
|------|------|
| `ANTHROPIC_API_KEY` **已设置** | 客户端必须通过 `x-api-key` 发送匹配的密钥，否则返回 `401` |
| `ANTHROPIC_API_KEY` **未设置** | 代理接受所有请求（无认证）。仅在可信网络中使用！ |

> ⚠️ 如果你的代理是公网可访问的，**务必设置 `ANTHROPIC_API_KEY`**。否则任何发现你 Worker URL 的人都能使用你的 API Key。

#### 全局环境变量

在 Dashboard 中设为**明文变量**（不是 Secret）：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PROVIDERS` | 多供应商配置 (JSON) — 启用多供应商模式 | *（未设置 = 旧模式）* |
| `LOG_LEVEL` | `WARNING` 或 `DEBUG` | `WARNING` |
| `MAX_TOKENS_LIMIT` | 最大 token 数 | `16384` |
| `MIN_TOKENS_LIMIT` | 最小 token 数 | `4096` |
| `REQUEST_TIMEOUT` | 默认请求超时秒数 | `90` |

设为**密钥 (Secret)**：

| 变量 | 说明 | 是否必需 |
|------|------|----------|
| `ANTHROPIC_API_KEY` | 代理级认证密钥（客户端需发送此密钥访问代理） | 推荐设置 |
| `PROVIDER_<NAME>_API_KEY` | 每个供应商的 API Key | 每个供应商都需要 |

> **💡 日志级别：**
> - `WARNING`（默认）：仅记录错误和路由诊断信息。生产环境安全使用。
> - `DEBUG`：记录完整的请求/响应内容（包括用户提示词）。**生产环境切勿使用。**

#### 内置供应商

以下 14 个供应商已预置 URL，只需添加 API Key：

| 名称 | Base URL | 协议 |
|------|----------|------|
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

不在列表中的供应商需要指定 `baseUrl`：

```json
{
  "default": "my-llm",
  "providers": {
    "my-llm": {
      "baseUrl": "https://api.my-llm-service.com/v1",
      "protocol": "openai"
    }
  }
}
```

---

### 📡 完整配置示例

<details>
<summary><b>示例 1：OpenCode 默认 + MiniMax 直转 + OpenRouter 处理 Qwen</b></summary>

**场景**：默认模型走 OpenCode，MiniMax 模型通过 OpenCode 走 Anthropic 直转，Qwen 3.6+ 走 OpenRouter。

**PROVIDERS**（明文变量）：
```json
{
  "default": "opencode",
  "routing": {
    "minimax*": "opencode-passthrough",
    "qwen/qwen3.6-plus": "openrouter"
  },
  "providers": {
    "opencode": {},
    "opencode-passthrough": {
      "baseUrl": "https://opencode.ai/zen/go/v1",
      "protocol": "anthropic"
    },
    "openrouter": {}
  }
}
```

**密钥 (Secrets)**：
```
PROVIDER_OPENCODE_API_KEY              = 你的 OpenCode Key
PROVIDER_OPENCODE_PASSTHROUGH_API_KEY  = 你的 OpenCode Key
PROVIDER_OPENROUTER_API_KEY            = 你的 OpenRouter Key
ANTHROPIC_API_KEY                      = 你的代理密码
```

**路由说明**：
- `minimax-abc` → 匹配 `minimax*` → `opencode-passthrough`（Anthropic 协议）
- `qwen/qwen3.6-plus` → 精确匹配 → `openrouter`
- `gpt-4o` 或其他模型 → 无匹配 → `opencode`（默认）

</details>

<details>
<summary><b>示例 2：OpenAI + DeepSeek + GLM 多供应商</b></summary>

**PROVIDERS**（明文变量）：
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
    "glm": { "timeout": 120 },
    "deepseek": {}
  }
}
```

**密钥**：`PROVIDER_OPENAI_API_KEY`、`PROVIDER_GLM_API_KEY`、`PROVIDER_DEEPSEEK_API_KEY`

</details>

<details>
<summary><b>示例 3：OpenRouter 默认 + Anthropic 直转 Claude 模型</b></summary>

**PROVIDERS**（明文变量）：
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

**密钥**：`PROVIDER_OPENROUTER_API_KEY`、`PROVIDER_ANTHROPIC_API_KEY`

</details>

<details>
<summary><b>示例 4：OpenRouter 含斜杠格式模型（qwen/qwen-3.6）</b></summary>

OpenRouter 使用 `厂商/模型` 格式。代理可无缝处理含斜杠的模型名：

**PROVIDERS**（明文变量）：
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

- `qwen/qwen-3.6` → 匹配 `qwen/*` → OpenRouter
- `qwen-turbo` → 匹配 `qwen-*` → 原生 Qwen 供应商

若需去除厂商前缀，可用 `modelMapping`：
```json
"qwen": {
  "modelMapping": { "qwen/qwen-3.6": "qwen-3.6" }
}
```

</details>

<details>
<summary><b>示例 5：单供应商 — GLM（旧模式，无需 PROVIDERS）</b></summary>

只需一个供应商时，完全不用设置 `PROVIDERS`：

**明文变量**：
```
OPENAI_BASE_URL = https://open.bigmodel.cn/api/paas/v4
```

**密钥**：无需设置（客户端 key 透传给 GLM）

</details>

<details>
<summary><b>示例 6：Azure OpenAI（旧模式）</b></summary>

**明文变量**：
```
OPENAI_BASE_URL      = https://your-resource.openai.azure.com/openai/deployments/your-deployment
ENABLE_MODEL_MAPPING = true
BIG_MODEL            = gpt-4
MIDDLE_MODEL         = gpt-4
SMALL_MODEL          = gpt-35-turbo
AZURE_API_VERSION    = 2024-02-15-preview
```

**密钥**：`OPENAI_API_KEY`

</details>

---

### 📋 API 端点

| 方法 | 路径 | 说明 | 需要认证 |
|------|------|------|----------|
| GET | `/` | 代理信息 | 否 |
| GET | `/health` | 健康检查 | 否 |
| POST | `/v1/messages` | 聊天补全（代理） | 是 |
| POST | `/v1/messages/count_tokens` | Token 计数估算 | 是 |

### 🔄 从旧版迁移到多供应商

现有配置继续有效 — **无需任何改动**。要启用多供应商：

```
之前（旧模式）：                          之后（多供应商）：
─────────────                             ─────────────────
OPENAI_BASE_URL = https://xxx/v1          PROVIDERS = {"default":"glm","providers":{"glm":{}}}
OPENAI_API_KEY  = sk-xxx                  PROVIDER_GLM_API_KEY = sk-xxx（密钥）
```

#### 旧模式变量（仅在未设置 PROVIDERS 时有效）

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_API_KEY` | 后端 API Key（不设置则透传） | — |
| `OPENAI_BASE_URL` | API 基础 URL | `https://api.openai.com/v1` |
| `PASSTHROUGH_MODELS` | 逗号分隔的模型前缀，匹配的走 Anthropic 直转 | — |
| `ENABLE_MODEL_MAPPING` | 启用 Claude→供应商模型映射 | `false` |
| `BIG_MODEL` | Opus 映射目标 | `gpt-4o` |
| `MIDDLE_MODEL` | Sonnet 映射目标 | 同 `BIG_MODEL` |
| `SMALL_MODEL` | Haiku 映射目标 | `gpt-4o-mini` |
| `AZURE_API_VERSION` | Azure OpenAI API 版本 | — |
| `CUSTOM_HEADERS` | 自定义 HTTP 头 (JSON) | — |

### 🐳 Docker 部署（Node.js）

除了 Cloudflare Workers，你还可以使用 Docker 将此代理部署为原生 Node.js 服务 — 无需 Wrangler 或 Cloudflare 账号。

#### 快速开始

```bash
cp .env.example .env   # 编辑填入 API Key / PROVIDERS 配置
docker compose up -d
```

服务运行在 `http://localhost:3000`。

#### 环境变量

所有[配置参考](#-配置参考)中的环境变量均适用，通过 `.env` 文件或直接传入：

```bash
docker compose up -d -e PROVIDERS='{"default":"openai"}' -e PROVIDER_OPENAI_API_KEY=sk-xxx
```

Docker 额外变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PORT` | HTTP 监听端口 | `3000` |
| `HOST` | HTTP 监听地址 | `0.0.0.0` |

#### 手动构建（不使用 Docker）

```bash
npm install
npm run build:node       # 打包为 dist-node/server.mjs
npm run start:node       # 启动 Node.js 服务
```

#### 自定义 Docker Compose

```yaml
services:
  claude-code-proxy:
    build: .
    ports:
      - "8080:3000"
    environment:
      - PORT=3000
      - PROVIDERS={"default":"openai","routing":{"deepseek-*":"deepseek"}}
      - PROVIDER_OPENAI_API_KEY=sk-xxx
      - PROVIDER_DEEPSEEK_API_KEY=sk-xxx
      - ANTHROPIC_API_KEY=my-proxy-password
    restart: unless-stopped
```

> **提示**：Docker 镜像采用多阶段构建 — 最终镜像仅包含 Node.js 20 Alpine 和一个打包后的 JS 文件（约 48 KB），零 npm 运行时依赖。

---

### 🛠️ 开发

```bash
npm install          # 安装依赖
npm run dev          # 本地开发服务器（Wrangler）
npm run lint         # 类型检查
npm run test         # 运行测试
npm run build        # 构建 Worker (dry-run)
npm run deploy       # 部署到 Cloudflare Workers
npm run build:node   # 打包 Node.js 服务
npm run start:node   # 启动 Node.js 服务
```

#### 项目结构

```
src/
├── app.ts                      # 核心请求处理（平台无关）
├── index.ts                    # Cloudflare Worker 入口
├── node-server.ts              # Node.js HTTP 服务入口
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

### ❓ 常见问题排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `401` 返回通用错误信息 | 供应商 API Key 未设置或错误 | 检查 `PROVIDER_<NAME>_API_KEY` 是否正确设置为 Secret |
| 所有请求都走 `api.openai.com` | `PROVIDERS` 环境变量未设置 | 在 Dashboard 添加 `PROVIDERS` JSON（明文变量，不是 Secret） |
| `Invalid PROVIDERS JSON` | JSON 中有 BOM 或多余空白字符 | 重新粘贴 JSON — 代理会自动去除 BOM，但请检查是否有拼写错误 |
| `authentication_error` 401 | 代理认证密钥不匹配 | 客户端的 `ANTHROPIC_API_KEY` 必须与代理的 `ANTHROPIC_API_KEY` Secret 匹配 |
| 模型路由到错误的供应商 | 路由顺序问题 | 路由规则从上到下检查，首次匹配生效 |

> **调试技巧**：临时设置 `LOG_LEVEL=DEBUG`，然后用 `wrangler tail` 查看实时日志，包括路由决策和请求/响应内容。

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

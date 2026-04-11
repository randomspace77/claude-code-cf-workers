import type { KnownProvider } from "../types";

/**
 * Built-in known providers with hardcoded base URLs.
 * Users only need to supply API keys — no URL configuration required.
 */
export const KNOWN_PROVIDERS: Record<string, KnownProvider> = {
  openai: {
    baseUrl: "https://api.openai.com/v1",
    protocol: "openai",
  },
  openrouter: {
    baseUrl: "https://openrouter.ai/api/v1",
    protocol: "openai",
    defaultHeaders: {
      "HTTP-Referer": "https://github.com/randomspace77/claude-code-proxy-cf-workers",
      "X-Title": "claude-code-proxy-cf-workers",
    },
  },
  deepseek: {
    baseUrl: "https://api.deepseek.com/v1",
    protocol: "openai",
  },
  glm: {
    baseUrl: "https://open.bigmodel.cn/api/paas/v4",
    protocol: "openai",
  },
  qwen: {
    baseUrl: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    protocol: "openai",
  },
  gemini: {
    baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai",
    protocol: "openai",
  },
  anthropic: {
    baseUrl: "https://api.anthropic.com/v1",
    protocol: "anthropic",
  },
  minimax: {
    baseUrl: "https://api.minimax.chat/v1",
    protocol: "anthropic",
  },
  opencode: {
    baseUrl: "https://opencode.ai/zen/go/v1",
    protocol: "openai",
  },
  doubao: {
    baseUrl: "https://ark.cn-beijing.volces.com/api/v3",
    protocol: "openai",
  },
  siliconflow: {
    baseUrl: "https://api.siliconflow.cn/v1",
    protocol: "openai",
  },
  groq: {
    baseUrl: "https://api.groq.com/openai/v1",
    protocol: "openai",
  },
  mistral: {
    baseUrl: "https://api.mistral.ai/v1",
    protocol: "openai",
  },
  together: {
    baseUrl: "https://api.together.xyz/v1",
    protocol: "openai",
  },
};

/**
 * Look up a known provider by name.
 */
export function getKnownProvider(name: string): KnownProvider | undefined {
  return KNOWN_PROVIDERS[name.toLowerCase()];
}

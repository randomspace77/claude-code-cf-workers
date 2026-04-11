import { describe, it, expect } from "vitest";
import { loadConfig } from "../src/config";
import { extractApiKey, validateClientApiKey } from "../src/auth";
import { resolveProvider, globMatch, mapModelForProvider } from "../src/router";
import type { Env, AppConfig, ResolvedProvider } from "../src/types";

// Helper: build a minimal AppConfig for tests
function makeConfig(overrides?: Partial<AppConfig>): AppConfig {
  return {
    anthropicApiKey: undefined,
    logLevel: "WARNING",
    requestTimeout: 90,
    maxTokensLimit: 16384,
    minTokensLimit: 4096,
    defaultProvider: "default",
    routing: {},
    providers: {
      default: {
        name: "default",
        baseUrl: "https://api.openai.com/v1",
        protocol: "openai",
        apiKey: "test",
        timeout: 90,
        headers: {},
      },
    },
    openaiApiKey: "test",
    openaiBaseUrl: "https://api.openai.com/v1",
    bigModel: "gpt-4o",
    middleModel: "gpt-4o",
    smallModel: "gpt-4o-mini",
    customHeaders: {},
    passthroughModels: [],
    enableModelMapping: false,
    ...overrides,
  };
}

// ---- loadConfig (legacy mode) ----

describe("loadConfig (legacy mode)", () => {
  const minimalEnv: Env = {
    OPENAI_API_KEY: "sk-test-key",
    OPENAI_BASE_URL: "https://api.openai.com/v1",
    BIG_MODEL: "gpt-4o",
    MIDDLE_MODEL: "gpt-4o",
    SMALL_MODEL: "gpt-4o-mini",
    MAX_TOKENS_LIMIT: "16384",
    MIN_TOKENS_LIMIT: "4096",
    REQUEST_TIMEOUT: "90",
    LOG_LEVEL: "WARNING",
  };

  it("parses all environment variables correctly", () => {
    const config = loadConfig(minimalEnv);
    expect(config.openaiApiKey).toBe("sk-test-key");
    expect(config.openaiBaseUrl).toBe("https://api.openai.com/v1");
    expect(config.bigModel).toBe("gpt-4o");
    expect(config.middleModel).toBe("gpt-4o");
    expect(config.smallModel).toBe("gpt-4o-mini");
    expect(config.maxTokensLimit).toBe(16384);
    expect(config.minTokensLimit).toBe(4096);
    expect(config.requestTimeout).toBe(90);
    expect(config.logLevel).toBe("WARNING");
    expect(config.customHeaders).toEqual({});
    expect(config.anthropicApiKey).toBeUndefined();
    expect(config.passthroughModels).toEqual([]);
    expect(config.enableModelMapping).toBe(false);
  });

  it("uses defaults when env vars are missing", () => {
    const emptyEnv = {} as unknown as Env;
    const config = loadConfig(emptyEnv);
    expect(config.openaiApiKey).toBe("");
    expect(config.openaiBaseUrl).toBe("https://api.openai.com/v1");
    expect(config.bigModel).toBe("gpt-4o");
    expect(config.middleModel).toBe("gpt-4o");
    expect(config.smallModel).toBe("gpt-4o-mini");
    expect(config.maxTokensLimit).toBe(16384);
    expect(config.minTokensLimit).toBe(4096);
    expect(config.requestTimeout).toBe(90);
    expect(config.logLevel).toBe("WARNING");
  });

  it("parses CUSTOM_HEADERS JSON string", () => {
    const env: Env = {
      ...minimalEnv,
      CUSTOM_HEADERS: '{"X-Custom": "value1", "X-Another": "value2"}',
    };
    const config = loadConfig(env);
    expect(config.customHeaders).toEqual({
      "X-Custom": "value1",
      "X-Another": "value2",
    });
  });

  it("ignores invalid CUSTOM_HEADERS JSON", () => {
    const env: Env = { ...minimalEnv, CUSTOM_HEADERS: "not valid json" };
    const config = loadConfig(env);
    expect(config.customHeaders).toEqual({});
  });

  it("ignores non-string values in CUSTOM_HEADERS", () => {
    const env: Env = {
      ...minimalEnv,
      CUSTOM_HEADERS: '{"valid": "string", "invalid": 123, "also_invalid": true}',
    };
    const config = loadConfig(env);
    expect(config.customHeaders).toEqual({ valid: "string" });
  });

  it("ignores array CUSTOM_HEADERS", () => {
    const env: Env = { ...minimalEnv, CUSTOM_HEADERS: '["not", "an", "object"]' };
    const config = loadConfig(env);
    expect(config.customHeaders).toEqual({});
  });

  it("blocks security-sensitive headers in CUSTOM_HEADERS", () => {
    const env: Env = {
      ...minimalEnv,
      CUSTOM_HEADERS: JSON.stringify({
        Authorization: "Bearer evil",
        "api-key": "stolen",
        Host: "evil.com",
        "X-Safe-Header": "allowed",
      }),
    };
    const config = loadConfig(env);
    expect(config.customHeaders).toEqual({ "X-Safe-Header": "allowed" });
  });

  it("sets optional secrets when provided", () => {
    const env: Env = {
      ...minimalEnv,
      ANTHROPIC_API_KEY: "sk-ant-test",
      AZURE_API_VERSION: "2024-06-01",
    };
    const config = loadConfig(env);
    expect(config.anthropicApiKey).toBe("sk-ant-test");
    expect(config.azureApiVersion).toBe("2024-06-01");
  });

  it("MIDDLE_MODEL defaults to BIG_MODEL value", () => {
    const env = {
      ...minimalEnv,
      BIG_MODEL: "glm-5.1",
      MIDDLE_MODEL: undefined,
    } as unknown as Env;
    const config = loadConfig(env);
    expect(config.middleModel).toBe("glm-5.1");
  });

  it("parses PASSTHROUGH_MODELS as comma-separated prefixes", () => {
    const env: Env = {
      ...minimalEnv,
      PASSTHROUGH_MODELS: "minimax, some-model , another",
    };
    const config = loadConfig(env);
    expect(config.passthroughModels).toEqual(["minimax", "some-model", "another"]);
  });

  it("creates default provider from legacy env vars", () => {
    const config = loadConfig(minimalEnv);
    expect(config.defaultProvider).toBe("default");
    expect(config.providers.default).toBeDefined();
    expect(config.providers.default.baseUrl).toBe("https://api.openai.com/v1");
    expect(config.providers.default.protocol).toBe("openai");
    expect(config.providers.default.apiKey).toBe("sk-test-key");
  });

  it("creates passthrough provider from PASSTHROUGH_MODELS", () => {
    const env: Env = { ...minimalEnv, PASSTHROUGH_MODELS: "minimax" };
    const config = loadConfig(env);
    expect(config.providers.passthrough).toBeDefined();
    expect(config.providers.passthrough.protocol).toBe("anthropic");
    expect(config.routing["minimax*"]).toBe("passthrough");
  });

  it("creates model mapping on legacy provider when enabled", () => {
    const env: Env = { ...minimalEnv, ENABLE_MODEL_MAPPING: "true" };
    const config = loadConfig(env);
    expect(config.providers.default.modelMapping).toEqual({
      opus: "gpt-4o",
      sonnet: "gpt-4o",
      haiku: "gpt-4o-mini",
    });
  });
});

// ---- loadConfig (multi-provider mode) ----

describe("loadConfig (multi-provider mode)", () => {
  const baseEnv: Env = {
    OPENAI_API_KEY: "",
    OPENAI_BASE_URL: "",
    BIG_MODEL: "",
    MIDDLE_MODEL: "",
    SMALL_MODEL: "",
    MAX_TOKENS_LIMIT: "16384",
    MIN_TOKENS_LIMIT: "4096",
    REQUEST_TIMEOUT: "90",
    LOG_LEVEL: "WARNING",
  };

  it("parses PROVIDERS JSON and creates providers", () => {
    const env: Env = {
      ...baseEnv,
      PROVIDERS: JSON.stringify({
        default: "openai",
        routing: { "glm-*": "glm" },
        providers: {
          openai: {},
          glm: {},
        },
      }),
      PROVIDER_OPENAI_API_KEY: "sk-openai",
      PROVIDER_GLM_API_KEY: "sk-glm",
    };
    const config = loadConfig(env);
    expect(config.defaultProvider).toBe("openai");
    expect(config.providers.openai).toBeDefined();
    expect(config.providers.openai.baseUrl).toBe("https://api.openai.com/v1");
    expect(config.providers.openai.apiKey).toBe("sk-openai");
    expect(config.providers.glm).toBeDefined();
    expect(config.providers.glm.baseUrl).toBe("https://open.bigmodel.cn/api/paas/v4");
    expect(config.providers.glm.apiKey).toBe("sk-glm");
    expect(config.routing["glm-*"]).toBe("glm");
  });

  it("supports custom provider with baseUrl", () => {
    const env: Env = {
      ...baseEnv,
      PROVIDERS: JSON.stringify({
        default: "my-custom",
        providers: {
          "my-custom": {
            baseUrl: "https://my-api.example.com/v1",
            protocol: "openai",
          },
        },
      }),
      PROVIDER_MY_CUSTOM_API_KEY: "sk-custom",
    };
    const config = loadConfig(env);
    expect(config.providers["my-custom"]).toBeDefined();
    expect(config.providers["my-custom"].baseUrl).toBe("https://my-api.example.com/v1");
    expect(config.providers["my-custom"].apiKey).toBe("sk-custom");
  });

  it("auto-creates default provider entry if not in providers list", () => {
    const env: Env = {
      ...baseEnv,
      PROVIDERS: JSON.stringify({
        default: "openai",
      }),
      PROVIDER_OPENAI_API_KEY: "sk-test",
    };
    const config = loadConfig(env);
    expect(config.providers.openai).toBeDefined();
    expect(config.providers.openai.baseUrl).toBe("https://api.openai.com/v1");
  });

  it("merges known provider defaults with user overrides", () => {
    const env: Env = {
      ...baseEnv,
      PROVIDERS: JSON.stringify({
        default: "glm",
        providers: {
          glm: { timeout: 120, headers: { "X-Custom": "value" } },
        },
      }),
    };
    const config = loadConfig(env);
    expect(config.providers.glm.baseUrl).toBe("https://open.bigmodel.cn/api/paas/v4");
    expect(config.providers.glm.timeout).toBe(120);
    expect(config.providers.glm.headers["X-Custom"]).toBe("value");
  });

  it("throws on invalid PROVIDERS JSON", () => {
    const env: Env = { ...baseEnv, PROVIDERS: "not json" };
    expect(() => loadConfig(env)).toThrow("Invalid PROVIDERS JSON");
  });

  it("throws when PROVIDERS has no default field", () => {
    const env: Env = { ...baseEnv, PROVIDERS: "{}" };
    expect(() => loadConfig(env)).toThrow("must have a 'default' field");
  });
});

// ---- globMatch ----

describe("globMatch", () => {
  it("matches exact strings", () => {
    expect(globMatch("gpt-4o", "gpt-4o")).toBe(true);
    expect(globMatch("gpt-4o", "gpt-4")).toBe(false);
  });

  it("matches wildcard patterns", () => {
    expect(globMatch("glm-5.1", "glm-*")).toBe(true);
    expect(globMatch("glm-4", "glm-*")).toBe(true);
    expect(globMatch("gpt-4o", "glm-*")).toBe(false);
  });

  it("matches multiple wildcards", () => {
    expect(globMatch("meta-llama-3.1", "meta-*-*")).toBe(true);
  });

  it("matches question mark wildcard", () => {
    expect(globMatch("gpt-4o", "gpt-?o")).toBe(true);
    expect(globMatch("gpt-4o", "gpt-??")).toBe(true);
    expect(globMatch("gpt-4o", "gpt-?")).toBe(false);
  });

  it("is case-sensitive (caller should lowercase)", () => {
    expect(globMatch("glm-5.1", "GLM-*")).toBe(false);
    expect(globMatch("glm-5.1", "glm-*")).toBe(true);
  });

  it("matches models with slashes (OpenRouter-style)", () => {
    expect(globMatch("qwen/qwen-3.6", "qwen/*")).toBe(true);
    expect(globMatch("openai/gpt-4o", "openai/*")).toBe(true);
    expect(globMatch("meta-llama/llama-3.1-70b", "meta-llama/*")).toBe(true);
    expect(globMatch("qwen/qwen-3.6", "deepseek/*")).toBe(false);
  });

  it("matches exact model with slash", () => {
    expect(globMatch("qwen/qwen-3.6", "qwen/qwen-3.6")).toBe(true);
    expect(globMatch("qwen/qwen-3.6", "qwen/qwen-3.5")).toBe(false);
  });

  it("matches models with dots in name", () => {
    expect(globMatch("glm-5.1", "glm-5.1")).toBe(true);
    expect(globMatch("glm-5.1", "glm-5.2")).toBe(false);
    expect(globMatch("glm-5.1", "glm-*")).toBe(true);
  });

  it("matches models with nested slashes", () => {
    expect(globMatch("org/provider/model-v2", "org/provider/*")).toBe(true);
    expect(globMatch("org/provider/model-v2", "org/*/*")).toBe(true);
    expect(globMatch("org/provider/model-v2", "org/*")).toBe(true); // * matches /
  });
});

// ---- resolveProvider ----

describe("resolveProvider", () => {
  const providers: Record<string, ResolvedProvider> = {
    openai: {
      name: "openai",
      baseUrl: "https://api.openai.com/v1",
      protocol: "openai",
      apiKey: "sk-openai",
      timeout: 90,
      headers: {},
    },
    glm: {
      name: "glm",
      baseUrl: "https://open.bigmodel.cn/api/paas/v4",
      protocol: "openai",
      apiKey: "sk-glm",
      timeout: 90,
      headers: {},
    },
    anthropic: {
      name: "anthropic",
      baseUrl: "https://api.anthropic.com/v1",
      protocol: "anthropic",
      apiKey: "sk-ant",
      timeout: 90,
      headers: {},
    },
  };

  const routing = {
    "glm-*": "glm",
    "claude-*": "anthropic",
    "gpt-*": "openai",
  };

  it("routes glm-* models to glm provider", () => {
    const result = resolveProvider("glm-5.1", routing, "openai", providers);
    expect(result?.name).toBe("glm");
  });

  it("routes claude-* models to anthropic provider", () => {
    const result = resolveProvider("claude-3.5-sonnet", routing, "openai", providers);
    expect(result?.name).toBe("anthropic");
  });

  it("routes gpt-* models to openai provider", () => {
    const result = resolveProvider("gpt-4o", routing, "openai", providers);
    expect(result?.name).toBe("openai");
  });

  it("falls back to default provider for unmatched models", () => {
    const result = resolveProvider("some-unknown-model", routing, "openai", providers);
    expect(result?.name).toBe("openai");
  });

  it("is case-insensitive for model matching", () => {
    const result = resolveProvider("GLM-5.1", routing, "openai", providers);
    expect(result?.name).toBe("glm");
  });

  it("routes slash-style model names (OpenRouter format)", () => {
    const orProviders: Record<string, ResolvedProvider> = {
      ...providers,
      openrouter: {
        name: "openrouter",
        baseUrl: "https://openrouter.ai/api/v1",
        protocol: "openai",
        apiKey: "sk-or",
        timeout: 90,
        headers: {},
      },
      qwen: {
        name: "qwen",
        baseUrl: "https://dashscope.aliyuncs.com/compatible-mode/v1",
        protocol: "openai",
        apiKey: "sk-qwen",
        timeout: 90,
        headers: {},
      },
    };
    const orRouting = {
      ...routing,
      "qwen/*": "openrouter",
      "qwen-*": "qwen",
      "openai/*": "openrouter",
    };

    // qwen/qwen-3.6 (OpenRouter format) → openrouter
    const r1 = resolveProvider("qwen/qwen-3.6", orRouting, "openai", orProviders);
    expect(r1?.name).toBe("openrouter");

    // qwen-turbo (native Qwen format) → qwen provider
    const r2 = resolveProvider("qwen-turbo", orRouting, "openai", orProviders);
    expect(r2?.name).toBe("qwen");

    // openai/gpt-4o (OpenRouter format) → openrouter
    const r3 = resolveProvider("openai/gpt-4o", orRouting, "openai", orProviders);
    expect(r3?.name).toBe("openrouter");
  });

  it("first matching routing rule wins", () => {
    const strictRouting = {
      "gpt-4o": "openai",      // exact match first
      "gpt-*": "glm",          // wildcard second
    };
    const r1 = resolveProvider("gpt-4o", strictRouting, "openai", providers);
    expect(r1?.name).toBe("openai");

    const r2 = resolveProvider("gpt-4o-mini", strictRouting, "openai", providers);
    expect(r2?.name).toBe("glm");
  });
});

// ---- mapModelForProvider ----

describe("mapModelForProvider", () => {
  const provider: ResolvedProvider = {
    name: "openai",
    baseUrl: "https://api.openai.com/v1",
    protocol: "openai",
    apiKey: "test",
    timeout: 90,
    headers: {},
  };

  it("returns model as-is when no mapping configured", () => {
    expect(mapModelForProvider(provider, "glm-5.1")).toBe("glm-5.1");
    expect(mapModelForProvider(provider, "gpt-4o")).toBe("gpt-4o");
  });

  it("maps models when mapping is configured", () => {
    const mapped = {
      ...provider,
      modelMapping: { opus: "gpt-4o", sonnet: "gpt-4o", haiku: "gpt-4o-mini" },
    };
    expect(mapModelForProvider(mapped, "claude-3-opus-20240229")).toBe("gpt-4o");
    expect(mapModelForProvider(mapped, "claude-3-5-sonnet-20241022")).toBe("gpt-4o");
    expect(mapModelForProvider(mapped, "claude-3-5-haiku-20241022")).toBe("gpt-4o-mini");
  });

  it("returns model as-is when no mapping keyword matches", () => {
    const mapped = {
      ...provider,
      modelMapping: { opus: "gpt-4o" },
    };
    expect(mapModelForProvider(mapped, "some-other-model")).toBe("some-other-model");
  });

  it("maps exact slash-style model names", () => {
    const mapped = {
      ...provider,
      modelMapping: {
        "qwen/qwen-3.6": "qwen-3.6",
        "openai/gpt-4o": "gpt-4o",
      },
    };
    expect(mapModelForProvider(mapped, "qwen/qwen-3.6")).toBe("qwen-3.6");
    expect(mapModelForProvider(mapped, "openai/gpt-4o")).toBe("gpt-4o");
    expect(mapModelForProvider(mapped, "some-other-model")).toBe("some-other-model");
  });

  it("first matching keyword wins (order matters)", () => {
    const mapped = {
      ...provider,
      modelMapping: {
        "qwen/qwen": "specific-match",
        "qwen": "broad-match",
      },
    };
    // "qwen/qwen-3.6" includes both "qwen/qwen" and "qwen" — first entry wins
    expect(mapModelForProvider(mapped, "qwen/qwen-3.6")).toBe("specific-match");
  });
});

// ---- validateClientApiKey ----

describe("validateClientApiKey", () => {
  const baseConfig = makeConfig();

  it("returns true when no ANTHROPIC_API_KEY is configured", () => {
    expect(validateClientApiKey(baseConfig, null)).toBe(true);
    expect(validateClientApiKey(baseConfig, "any-key")).toBe(true);
  });

  it("returns false when configured but no client key provided", () => {
    const config = makeConfig({ anthropicApiKey: "sk-ant-secret" });
    expect(validateClientApiKey(config, null)).toBe(false);
  });

  it("returns true for matching key", () => {
    const config = makeConfig({ anthropicApiKey: "sk-ant-secret" });
    expect(validateClientApiKey(config, "sk-ant-secret")).toBe(true);
  });

  it("returns false for non-matching key", () => {
    const config = makeConfig({ anthropicApiKey: "sk-ant-secret" });
    expect(validateClientApiKey(config, "wrong-key")).toBe(false);
  });

  it("returns false for key with different length", () => {
    const config = makeConfig({ anthropicApiKey: "short" });
    expect(validateClientApiKey(config, "a-much-longer-key")).toBe(false);
  });

  it("uses constant-time comparison (same length different values)", () => {
    const config = makeConfig({ anthropicApiKey: "aaaa" });
    expect(validateClientApiKey(config, "aaab")).toBe(false);
    expect(validateClientApiKey(config, "baaa")).toBe(false);
  });
});

// ---- extractApiKey ----

describe("extractApiKey", () => {
  it("extracts from x-api-key header", () => {
    const headers = new Headers({ "x-api-key": "sk-test-123" });
    expect(extractApiKey(headers)).toBe("sk-test-123");
  });

  it("extracts from Authorization Bearer header", () => {
    const headers = new Headers({ Authorization: "Bearer sk-test-456" });
    expect(extractApiKey(headers)).toBe("sk-test-456");
  });

  it("prefers x-api-key over Authorization", () => {
    const headers = new Headers({
      "x-api-key": "from-x-api-key",
      Authorization: "Bearer from-auth",
    });
    expect(extractApiKey(headers)).toBe("from-x-api-key");
  });

  it("returns null when no key present", () => {
    const headers = new Headers({});
    expect(extractApiKey(headers)).toBeNull();
  });

  it("returns null for non-Bearer Authorization", () => {
    const headers = new Headers({ Authorization: "Basic abc123" });
    expect(extractApiKey(headers)).toBeNull();
  });
});

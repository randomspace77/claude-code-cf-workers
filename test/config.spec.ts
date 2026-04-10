import { describe, it, expect } from "vitest";
import {
  loadConfig,
  mapModel,
  validateClientApiKey,
  extractApiKey,
} from "../src/config";
import type { Env, AppConfig } from "../src/types";

// ---- loadConfig ----

describe("loadConfig", () => {
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
    expect(config.azureApiVersion).toBeUndefined();
    expect(config.proxyMode).toBe("openai");
    expect(config.enableModelMapping).toBe(false);
  });

  it("uses defaults when env vars are missing", () => {
    const emptyEnv = {} as unknown as Env;
    const config = loadConfig(emptyEnv);
    expect(config.openaiApiKey).toBe("");
    expect(config.openaiBaseUrl).toBe("https://api.openai.com/v1");
    expect(config.bigModel).toBe("gpt-4o");
    expect(config.middleModel).toBe("gpt-4o"); // defaults to bigModel
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
    const env: Env = {
      ...minimalEnv,
      CUSTOM_HEADERS: "not valid json",
    };
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
    const env: Env = {
      ...minimalEnv,
      CUSTOM_HEADERS: '["not", "an", "object"]',
    };
    const config = loadConfig(env);
    expect(config.customHeaders).toEqual({});
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
});

// ---- mapModel ----

describe("mapModel", () => {
  const config: AppConfig = {
    openaiApiKey: "test",
    openaiBaseUrl: "https://api.openai.com/v1",
    bigModel: "gpt-4o",
    middleModel: "gpt-4o",
    smallModel: "gpt-4o-mini",
    maxTokensLimit: 16384,
    minTokensLimit: 4096,
    requestTimeout: 90,
    logLevel: "WARNING",
    customHeaders: {},
    proxyMode: "openai",
    enableModelMapping: false,
  };

  const mappingConfig: AppConfig = { ...config, enableModelMapping: true };

  // Default behavior: pass through all model names as-is
  it("forwards Claude model names as-is when mapping disabled", () => {
    expect(mapModel(config, "claude-3-5-sonnet-20241022")).toBe("claude-3-5-sonnet-20241022");
    expect(mapModel(config, "claude-sonnet-4-20250514")).toBe("claude-sonnet-4-20250514");
    expect(mapModel(config, "claude-3-opus-20240229")).toBe("claude-3-opus-20240229");
    expect(mapModel(config, "claude-3-5-haiku-20241022")).toBe("claude-3-5-haiku-20241022");
    expect(mapModel(config, "some-unknown-model")).toBe("some-unknown-model");
  });

  // Pass-through models (works regardless of mapping setting)
  it("passes through gpt-* model names", () => {
    expect(mapModel(config, "gpt-4o")).toBe("gpt-4o");
    expect(mapModel(config, "gpt-4o-mini")).toBe("gpt-4o-mini");
    expect(mapModel(config, "gpt-3.5-turbo")).toBe("gpt-3.5-turbo");
  });

  it("passes through o1-* model names", () => {
    expect(mapModel(config, "o1-preview")).toBe("o1-preview");
    expect(mapModel(config, "o1-mini")).toBe("o1-mini");
  });

  it("passes through o3-* model names", () => {
    expect(mapModel(config, "o3-mini")).toBe("o3-mini");
  });

  it("passes through o4-* model names", () => {
    expect(mapModel(config, "o4-mini")).toBe("o4-mini");
  });

  it("passes through deepseek-* model names", () => {
    expect(mapModel(config, "deepseek-chat")).toBe("deepseek-chat");
    expect(mapModel(config, "deepseek-coder")).toBe("deepseek-coder");
  });

  it("passes through glm-* model names", () => {
    expect(mapModel(config, "glm-5.1")).toBe("glm-5.1");
    expect(mapModel(config, "glm-4")).toBe("glm-4");
    expect(mapModel(config, "glm-5")).toBe("glm-5");
  });

  it("passes through qwen-* model names", () => {
    expect(mapModel(config, "qwen-turbo")).toBe("qwen-turbo");
    expect(mapModel(config, "qwen-max")).toBe("qwen-max");
  });

  it("passes through gemini-* model names", () => {
    expect(mapModel(config, "gemini-2.5-pro")).toBe("gemini-2.5-pro");
    expect(mapModel(config, "gemini-2.0-flash")).toBe("gemini-2.0-flash");
  });

  it("passes through ep-* model names", () => {
    expect(mapModel(config, "ep-2024-custom")).toBe("ep-2024-custom");
  });

  it("passes through doubao-* model names", () => {
    expect(mapModel(config, "doubao-pro")).toBe("doubao-pro");
  });

  // Claude model mapping (when enabled)
  it("maps haiku to small model when enabled", () => {
    expect(mapModel(mappingConfig, "claude-3-5-haiku-20241022")).toBe("gpt-4o-mini");
    expect(mapModel(mappingConfig, "claude-3-haiku-20240307")).toBe("gpt-4o-mini");
  });

  it("maps sonnet to middle model when enabled", () => {
    expect(mapModel(mappingConfig, "claude-3-5-sonnet-20241022")).toBe("gpt-4o");
    expect(mapModel(mappingConfig, "claude-sonnet-4-20250514")).toBe("gpt-4o");
  });

  it("maps opus to big model when enabled", () => {
    expect(mapModel(mappingConfig, "claude-3-opus-20240229")).toBe("gpt-4o");
    expect(mapModel(mappingConfig, "claude-opus-4-20250514")).toBe("gpt-4o");
  });

  it("defaults unknown models to big model when enabled", () => {
    expect(mapModel(mappingConfig, "some-unknown-model")).toBe("gpt-4o");
  });

  // With custom config for GLM 5.1
  it("maps claude models to GLM models when configured and enabled", () => {
    const glmConfig: AppConfig = {
      ...mappingConfig,
      bigModel: "glm-5.1",
      middleModel: "glm-5.1",
      smallModel: "glm-5.1",
    };
    expect(mapModel(glmConfig, "claude-3-5-sonnet-20241022")).toBe("glm-5.1");
    expect(mapModel(glmConfig, "claude-3-5-haiku-20241022")).toBe("glm-5.1");
    expect(mapModel(glmConfig, "claude-3-opus-20240229")).toBe("glm-5.1");
  });
});

// ---- validateClientApiKey ----

describe("validateClientApiKey", () => {
  const baseConfig: AppConfig = {
    openaiApiKey: "test",
    openaiBaseUrl: "https://api.openai.com/v1",
    bigModel: "gpt-4o",
    middleModel: "gpt-4o",
    smallModel: "gpt-4o-mini",
    maxTokensLimit: 16384,
    minTokensLimit: 4096,
    requestTimeout: 90,
    logLevel: "WARNING",
    customHeaders: {},
    proxyMode: "openai",
    enableModelMapping: false,
  };

  it("returns true when no ANTHROPIC_API_KEY is configured", () => {
    expect(validateClientApiKey(baseConfig, null)).toBe(true);
    expect(validateClientApiKey(baseConfig, "any-key")).toBe(true);
  });

  it("returns false when configured but no client key provided", () => {
    const config = { ...baseConfig, anthropicApiKey: "sk-ant-secret" };
    expect(validateClientApiKey(config, null)).toBe(false);
  });

  it("returns true for matching key", () => {
    const config = { ...baseConfig, anthropicApiKey: "sk-ant-secret" };
    expect(validateClientApiKey(config, "sk-ant-secret")).toBe(true);
  });

  it("returns false for non-matching key", () => {
    const config = { ...baseConfig, anthropicApiKey: "sk-ant-secret" };
    expect(validateClientApiKey(config, "wrong-key")).toBe(false);
  });

  it("returns false for key with different length", () => {
    const config = { ...baseConfig, anthropicApiKey: "short" };
    expect(validateClientApiKey(config, "a-much-longer-key")).toBe(false);
  });

  it("uses constant-time comparison (same length different values)", () => {
    const config = { ...baseConfig, anthropicApiKey: "aaaa" };
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

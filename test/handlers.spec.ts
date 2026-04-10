import { describe, it, expect } from "vitest";
import {
  handleCountTokens,
  handleHealth,
  handleRoot,
  authenticate,
} from "../src/handlers";
import type { AppConfig } from "../src/types";

const defaultConfig: AppConfig = {
  openaiApiKey: "sk-test",
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

// ---- handleHealth ----

describe("handleHealth", () => {
  it("returns healthy status with config info", () => {
    const response = handleHealth(defaultConfig);
    expect(response.status).toBe(200);
  });

  it("includes openai_api_configured as true when key present", async () => {
    const response = handleHealth(defaultConfig);
    const body = (await response.json()) as Record<string, unknown>;
    expect(body.status).toBe("healthy");
    expect(body.openai_api_configured).toBe(true);
    expect(body.client_api_key_validation).toBe(false);
    expect(body).toHaveProperty("timestamp");
  });

  it("includes openai_api_configured as false when key missing", async () => {
    const config = { ...defaultConfig, openaiApiKey: "" };
    const response = handleHealth(config);
    const body = (await response.json()) as Record<string, unknown>;
    expect(body.openai_api_configured).toBe(false);
  });

  it("includes client_api_key_validation as true when anthropic key set", async () => {
    const config = { ...defaultConfig, anthropicApiKey: "sk-ant-test" };
    const response = handleHealth(config);
    const body = (await response.json()) as Record<string, unknown>;
    expect(body.client_api_key_validation).toBe(true);
  });
});

// ---- handleRoot ----

describe("handleRoot", () => {
  it("returns proxy information", async () => {
    const response = handleRoot(defaultConfig);
    expect(response.status).toBe(200);
    const body = (await response.json()) as Record<string, unknown>;
    expect(body.message).toBe("Claude-to-OpenAI API Proxy v1.0.0");
    expect(body.status).toBe("running");
  });

  it("includes config and endpoints", async () => {
    const response = handleRoot(defaultConfig);
    const body = (await response.json()) as Record<string, unknown>;

    const config = body.config as Record<string, unknown>;
    expect(config.openai_base_url).toBe("https://api.openai.com/v1");
    expect(config.max_tokens_limit).toBe(16384);
    expect(config.api_key_configured).toBe(true);
    expect(config.client_api_key_validation).toBe(false);
    expect(config.big_model).toBe("gpt-4o");
    expect(config.middle_model).toBe("gpt-4o");
    expect(config.small_model).toBe("gpt-4o-mini");

    const endpoints = body.endpoints as Record<string, unknown>;
    expect(endpoints.messages).toBe("/v1/messages");
    expect(endpoints.count_tokens).toBe("/v1/messages/count_tokens");
    expect(endpoints.health).toBe("/health");
  });

  it("shows GLM models when configured", async () => {
    const config = {
      ...defaultConfig,
      bigModel: "glm-5.1",
      middleModel: "glm-5.1",
      smallModel: "glm-5.1",
    };
    const response = handleRoot(config);
    const body = (await response.json()) as Record<string, unknown>;
    const cfg = body.config as Record<string, unknown>;
    expect(cfg.big_model).toBe("glm-5.1");
    expect(cfg.middle_model).toBe("glm-5.1");
    expect(cfg.small_model).toBe("glm-5.1");
  });
});

// ---- authenticate ----

describe("authenticate", () => {
  it("returns null (authorized) when no ANTHROPIC_API_KEY configured", () => {
    const request = new Request("http://localhost/v1/messages", {
      method: "POST",
    });
    const result = authenticate(request, defaultConfig);
    expect(result).toBeNull();
  });

  it("returns null when valid key provided via x-api-key", () => {
    const config = { ...defaultConfig, anthropicApiKey: "sk-ant-test" };
    const request = new Request("http://localhost/v1/messages", {
      method: "POST",
      headers: { "x-api-key": "sk-ant-test" },
    });
    const result = authenticate(request, config);
    expect(result).toBeNull();
  });

  it("returns null when valid key provided via Authorization Bearer", () => {
    const config = { ...defaultConfig, anthropicApiKey: "sk-ant-test" };
    const request = new Request("http://localhost/v1/messages", {
      method: "POST",
      headers: { Authorization: "Bearer sk-ant-test" },
    });
    const result = authenticate(request, config);
    expect(result).toBeNull();
  });

  it("returns 401 when ANTHROPIC_API_KEY set but no client key provided", async () => {
    const config = { ...defaultConfig, anthropicApiKey: "sk-ant-test" };
    const request = new Request("http://localhost/v1/messages", {
      method: "POST",
    });
    const result = authenticate(request, config);
    expect(result).not.toBeNull();
    expect(result!.status).toBe(401);
    const body = (await result!.json()) as Record<string, unknown>;
    expect(body.type).toBe("error");
    const error = body.error as Record<string, unknown>;
    expect(error.type).toBe("authentication_error");
  });

  it("returns 401 when wrong key provided", async () => {
    const config = { ...defaultConfig, anthropicApiKey: "sk-ant-test" };
    const request = new Request("http://localhost/v1/messages", {
      method: "POST",
      headers: { "x-api-key": "wrong-key" },
    });
    const result = authenticate(request, config);
    expect(result).not.toBeNull();
    expect(result!.status).toBe(401);
  });
});

// ---- handleCountTokens ----

describe("handleCountTokens", () => {
  it("counts tokens for simple string message", async () => {
    const request = new Request("http://localhost/v1/messages/count_tokens", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "glm-5.1",
        messages: [{ role: "user", content: "Hello world" }],
      }),
    });
    const response = await handleCountTokens(request);
    expect(response.status).toBe(200);
    const body = (await response.json()) as Record<string, number>;
    expect(body.input_tokens).toBeGreaterThan(0);
    // "Hello world" = 11 chars => ~2 tokens
    expect(body.input_tokens).toBe(Math.floor(11 / 4));
  });

  it("counts tokens for system message (string)", async () => {
    const request = new Request("http://localhost/v1/messages/count_tokens", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "glm-5.1",
        system: "You are a helpful assistant.",
        messages: [{ role: "user", content: "Hi" }],
      }),
    });
    const response = await handleCountTokens(request);
    const body = (await response.json()) as Record<string, number>;
    // "You are a helpful assistant." (28 chars) + "Hi" (2 chars) = 30 chars => 7 tokens
    expect(body.input_tokens).toBe(Math.floor(30 / 4));
  });

  it("counts tokens for system message (array)", async () => {
    const request = new Request("http://localhost/v1/messages/count_tokens", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "glm-5.1",
        system: [
          { type: "text", text: "First part." },
          { type: "text", text: "Second part." },
        ],
        messages: [{ role: "user", content: "Hi" }],
      }),
    });
    const response = await handleCountTokens(request);
    const body = (await response.json()) as Record<string, number>;
    // "First part." (11) + "Second part." (12) + "Hi" (2) = 25 chars => 6 tokens
    expect(body.input_tokens).toBe(Math.floor(25 / 4));
  });

  it("counts tokens for array content blocks", async () => {
    const request = new Request("http://localhost/v1/messages/count_tokens", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "glm-5.1",
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "What is this?" },
              {
                type: "image",
                source: { type: "base64", media_type: "image/png", data: "abc" },
              },
            ],
          },
        ],
      }),
    });
    const response = await handleCountTokens(request);
    const body = (await response.json()) as Record<string, number>;
    // "What is this?" (14 chars) => 3 tokens (images not counted)
    expect(body.input_tokens).toBe(Math.floor(14 / 4));
  });

  it("handles null content in messages", async () => {
    const request = new Request("http://localhost/v1/messages/count_tokens", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "glm-5.1",
        messages: [
          { role: "user", content: null },
          { role: "assistant", content: null },
        ],
      }),
    });
    const response = await handleCountTokens(request);
    const body = (await response.json()) as Record<string, number>;
    // No content => minimum 1 token
    expect(body.input_tokens).toBe(1);
  });

  it("handles empty messages array", async () => {
    const request = new Request("http://localhost/v1/messages/count_tokens", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "glm-5.1",
        messages: [],
      }),
    });
    const response = await handleCountTokens(request);
    const body = (await response.json()) as Record<string, number>;
    expect(body.input_tokens).toBe(1); // minimum 1
  });

  it("returns 400 for invalid JSON body", async () => {
    const request = new Request("http://localhost/v1/messages/count_tokens", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "not json",
    });
    const response = await handleCountTokens(request);
    expect(response.status).toBe(400);
    const body = (await response.json()) as Record<string, unknown>;
    expect(body.type).toBe("error");
  });
});

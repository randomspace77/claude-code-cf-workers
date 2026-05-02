import { describe, it, expect } from "vitest";
import { convertClaudeToOpenAI } from "../src/conversion/request";
import { setToolReasoning } from "../src/conversion/reasoning-cache";
import type { ClaudeMessagesRequest, AppConfig, ReasoningCacheNamespace } from "../src/types";

const defaultConfig: AppConfig = {
  openaiApiKey: "test-key",
  openaiBaseUrl: "https://api.openai.com/v1",
  bigModel: "gpt-4o",
  middleModel: "gpt-4o",
  smallModel: "gpt-4o-mini",
  maxTokensLimit: 16384,
  minTokensLimit: 4096,
  reasoningCacheTtlSeconds: 2592000,
  requestTimeout: 90,
  logLevel: "WARNING",
  customHeaders: {},
  passthroughModels: [],
  enableModelMapping: false,
  defaultProvider: "default",
  routing: {},
  providers: {
    default: {
      name: "default",
      baseUrl: "https://api.openai.com/v1",
      protocol: "openai",
      apiKey: "test-key",
      timeout: 90,
      headers: {},
    },
  },
};

function createTestKV(): ReasoningCacheNamespace {
  const values = new Map<string, string>();
  return {
    get: async (key: string) => values.get(key) ?? null,
    put: async (key: string, value: string) => {
      values.set(key, value);
    },
  } as unknown as ReasoningCacheNamespace;
}

describe("Request Conversion", () => {
  it("converts a basic text message", async () => {
    const claudeReq: ClaudeMessagesRequest = {
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 1000,
      messages: [{ role: "user", content: "Hello" }],
    };

    const result = await convertClaudeToOpenAI(claudeReq, defaultConfig);

    expect(result.model).toBe("claude-3-5-sonnet-20241022");
    expect(result.messages).toHaveLength(1);
    expect(result.messages[0].role).toBe("user");
    expect(result.messages[0].content).toBe("Hello");
  });

  it("passes haiku model unchanged (mapping now in provider layer)", async () => {
    const mappingConfig: AppConfig = { ...defaultConfig, enableModelMapping: true };
    const claudeReq: ClaudeMessagesRequest = {
      model: "claude-3-5-haiku-20241022",
      max_tokens: 100,
      messages: [{ role: "user", content: "Hi" }],
    };

    const result = await convertClaudeToOpenAI(claudeReq, mappingConfig);
    expect(result.model).toBe("claude-3-5-haiku-20241022");
  });

  it("passes opus model unchanged (mapping now in provider layer)", async () => {
    const mappingConfig: AppConfig = { ...defaultConfig, enableModelMapping: true };
    const claudeReq: ClaudeMessagesRequest = {
      model: "claude-3-opus-20240229",
      max_tokens: 100,
      messages: [{ role: "user", content: "Hi" }],
    };

    const result = await convertClaudeToOpenAI(claudeReq, mappingConfig);
    expect(result.model).toBe("claude-3-opus-20240229");
  });

  it("includes system message", async () => {
    const claudeReq: ClaudeMessagesRequest = {
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 100,
      system: "You are a helpful assistant.",
      messages: [{ role: "user", content: "Hello" }],
    };

    const result = await convertClaudeToOpenAI(claudeReq, defaultConfig);
    expect(result.messages).toHaveLength(2);
    expect(result.messages[0].role).toBe("system");
    expect(result.messages[0].content).toBe("You are a helpful assistant.");
  });

  it("converts tools", async () => {
    const claudeReq: ClaudeMessagesRequest = {
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 200,
      messages: [{ role: "user", content: "What is the weather?" }],
      tools: [
        {
          name: "get_weather",
          description: "Get weather for a location",
          input_schema: {
            type: "object",
            properties: { location: { type: "string" } },
            required: ["location"],
          },
        },
      ],
      tool_choice: { type: "auto" },
    };

    const result = await convertClaudeToOpenAI(claudeReq, defaultConfig);
    expect(result.tools).toHaveLength(1);
    expect(result.tools![0].function.name).toBe("get_weather");
    expect(result.tool_choice).toBe("auto");
  });

  it("passes through OpenAI model names unchanged", async () => {
    const claudeReq: ClaudeMessagesRequest = {
      model: "gpt-4o",
      max_tokens: 100,
      messages: [{ role: "user", content: "Hi" }],
    };

    const result = await convertClaudeToOpenAI(claudeReq, defaultConfig);
    expect(result.model).toBe("gpt-4o");
  });

  it("enforces min/max token limits", async () => {
    const claudeReq: ClaudeMessagesRequest = {
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 10,
      messages: [{ role: "user", content: "Hi" }],
    };

    const result = await convertClaudeToOpenAI(claudeReq, defaultConfig);
    // Should be clamped to minTokensLimit
    expect(result.max_tokens).toBe(defaultConfig.minTokensLimit);
  });

  it("clamps max_tokens to maxTokensLimit", async () => {
    const claudeReq: ClaudeMessagesRequest = {
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 999999,
      messages: [{ role: "user", content: "Hi" }],
    };

    const result = await convertClaudeToOpenAI(claudeReq, defaultConfig);
    expect(result.max_tokens).toBe(defaultConfig.maxTokensLimit);
  });

  it("passes through GLM model names unchanged", async () => {
    const claudeReq: ClaudeMessagesRequest = {
      model: "glm-5.1",
      max_tokens: 8000,
      messages: [{ role: "user", content: "Hello" }],
    };

    const result = await convertClaudeToOpenAI(claudeReq, defaultConfig);
    expect(result.model).toBe("glm-5.1");
  });

  it("converts system as array of text blocks", async () => {
    const claudeReq: ClaudeMessagesRequest = {
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 100,
      system: [
        { type: "text", text: "First instruction." },
        { type: "text", text: "Second instruction." },
      ],
      messages: [{ role: "user", content: "Hello" }],
    };

    const result = await convertClaudeToOpenAI(claudeReq, defaultConfig);
    expect(result.messages[0].role).toBe("system");
    expect(result.messages[0].content).toBe("First instruction.\n\nSecond instruction.");
  });

  it("includes stop_sequences as stop", async () => {
    const claudeReq: ClaudeMessagesRequest = {
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 100,
      messages: [{ role: "user", content: "Hi" }],
      stop_sequences: ["Human:", "Assistant:"],
    };

    const result = await convertClaudeToOpenAI(claudeReq, defaultConfig);
    expect(result.stop).toEqual(["Human:", "Assistant:"]);
  });

  it("includes top_p when provided", async () => {
    const claudeReq: ClaudeMessagesRequest = {
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 100,
      messages: [{ role: "user", content: "Hi" }],
      top_p: 0.9,
    };

    const result = await convertClaudeToOpenAI(claudeReq, defaultConfig);
    expect(result.top_p).toBe(0.9);
  });

  it("converts multimodal content with image", async () => {
    const claudeReq: ClaudeMessagesRequest = {
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 100,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "What is in this image?" },
            {
              type: "image",
              source: {
                type: "base64",
                media_type: "image/png",
                data: "iVBORw0KGgoAAAANS...",
              },
            },
          ],
        },
      ],
    };

    const result = await convertClaudeToOpenAI(claudeReq, defaultConfig);
    const content = result.messages[0].content as unknown as Array<Record<string, unknown>>;
    expect(content).toHaveLength(2);
    expect(content[0]).toEqual({ type: "text", text: "What is in this image?" });
    expect(content[1]).toEqual({
      type: "image_url",
      image_url: { url: "data:image/png;base64,iVBORw0KGgoAAAANS..." },
    });
  });

  it("converts assistant message with tool use", async () => {
    const claudeReq: ClaudeMessagesRequest = {
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 100,
      messages: [
        { role: "user", content: "What is the weather?" },
        {
          role: "assistant",
          content: [
            { type: "text", text: "Let me check." },
            {
              type: "tool_use",
              id: "tool_1",
              name: "get_weather",
              input: { location: "NYC" },
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "tool_1",
              content: "Sunny, 72°F",
            },
          ],
        },
      ],
    };

    const result = await convertClaudeToOpenAI(claudeReq, defaultConfig);
    // user, assistant (with tool_calls), tool result
    expect(result.messages).toHaveLength(3);

    const assistantMsg = result.messages[1];
    expect(assistantMsg.role).toBe("assistant");
    expect(assistantMsg.content).toBe("Let me check.");
    expect(assistantMsg.tool_calls).toHaveLength(1);
    expect(assistantMsg.tool_calls![0].function.name).toBe("get_weather");

    const toolMsg = result.messages[2];
    expect(toolMsg.role).toBe("tool");
    expect(toolMsg.tool_call_id).toBe("tool_1");
    expect(toolMsg.content).toBe("Sunny, 72°F");
  });

  it("converts tool_choice type 'any' to 'required' (for non-DeepSeek models)", async () => {
    const claudeReq: ClaudeMessagesRequest = {
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 100,
      messages: [{ role: "user", content: "Hi" }],
      tools: [
        {
          name: "test_tool",
          description: "A test tool",
          input_schema: { type: "object", properties: {} },
        },
      ],
      tool_choice: { type: "any" },
    };

    const result = await convertClaudeToOpenAI(claudeReq, defaultConfig);
    expect(result.tool_choice).toBe("required");
  });

  it("converts tool_choice type 'any' to undefined for DeepSeek models (softened)", async () => {
    const claudeReq: ClaudeMessagesRequest = {
      model: "deepseek-v4-pro[1m]",
      max_tokens: 100,
      messages: [{ role: "user", content: "Hi" }],
      tools: [
        {
          name: "test_tool",
          description: "A test tool",
          input_schema: { type: "object", properties: {} },
        },
      ],
      tool_choice: { type: "any" },
    };

    const result = await convertClaudeToOpenAI(claudeReq, defaultConfig);
    expect(result.tool_choice).toBeUndefined();
    // Should have injected a system instruction instead
    const sysMsg = result.messages.find((m) => m.role === "system");
    expect(sysMsg).toBeDefined();
    expect(sysMsg!.content).toContain("requires a tool call");
  });

  it("replays cached DeepSeek reasoning_content for historical tool calls", async () => {
    const config = { ...defaultConfig, reasoningCache: createTestKV() };
    await setToolReasoning(config, "deepseek-v4-pro[1m]", "tool_1", "cached reasoning for tool_1");

    const claudeReq: ClaudeMessagesRequest = {
      model: "deepseek-v4-pro[1m]",
      max_tokens: 100,
      messages: [
        { role: "user", content: "Read a file" },
        {
          role: "assistant",
          content: [
            {
              type: "tool_use",
              id: "tool_1",
              name: "Read",
              input: { file_path: "README.md" },
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "tool_1",
              content: "file contents",
            },
          ],
        },
      ],
    };

    const result = await convertClaudeToOpenAI(claudeReq, config);
    expect(result.messages[1].reasoning_content).toBe("cached reasoning for tool_1");
  });

  it("converts tool_choice with specific tool name", async () => {
    const claudeReq: ClaudeMessagesRequest = {
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 100,
      messages: [{ role: "user", content: "Hi" }],
      tools: [
        {
          name: "specific_tool",
          description: "A specific tool",
          input_schema: { type: "object", properties: {} },
        },
      ],
      tool_choice: { type: "tool", name: "specific_tool" },
    };

    const result = await convertClaudeToOpenAI(claudeReq, defaultConfig);
    expect(result.tool_choice).toEqual({
      type: "function",
      function: { name: "specific_tool" },
    });
  });

  it("handles null content in user message", async () => {
    const claudeReq: ClaudeMessagesRequest = {
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 100,
      messages: [{ role: "user", content: null }],
    };

    const result = await convertClaudeToOpenAI(claudeReq, defaultConfig);
    expect(result.messages[0].content).toBe("");
  });

  it("handles null content in assistant message", async () => {
    const claudeReq: ClaudeMessagesRequest = {
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 100,
      messages: [
        { role: "user", content: "Hi" },
        { role: "assistant", content: null },
      ],
    };

    const result = await convertClaudeToOpenAI(claudeReq, defaultConfig);
    expect(result.messages[1].content).toBeNull();
  });

  it("sets stream from request", async () => {
    const claudeReq: ClaudeMessagesRequest = {
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 100,
      messages: [{ role: "user", content: "Hi" }],
      stream: true,
    };

    const result = await convertClaudeToOpenAI(claudeReq, defaultConfig);
    expect(result.stream).toBe(true);
  });

  it("defaults stream to false", async () => {
    const claudeReq: ClaudeMessagesRequest = {
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 100,
      messages: [{ role: "user", content: "Hi" }],
    };

    const result = await convertClaudeToOpenAI(claudeReq, defaultConfig);
    expect(result.stream).toBe(false);
  });

  it("skips tools with empty names", async () => {
    const claudeReq: ClaudeMessagesRequest = {
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 100,
      messages: [{ role: "user", content: "Hi" }],
      tools: [
        {
          name: "",
          description: "Empty name tool",
          input_schema: { type: "object" },
        },
        {
          name: "valid_tool",
          description: "Valid tool",
          input_schema: { type: "object" },
        },
      ],
    };

    const result = await convertClaudeToOpenAI(claudeReq, defaultConfig);
    expect(result.tools).toHaveLength(1);
    expect(result.tools![0].function.name).toBe("valid_tool");
  });
});

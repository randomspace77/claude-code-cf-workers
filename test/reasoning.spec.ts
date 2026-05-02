import { describe, it, expect } from "vitest";
import { convertOpenAIToClaude } from "../src/conversion/response";
import type { OpenAIResponse, ClaudeMessagesRequest, AppConfig } from "../src/types";

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
  providers: {},
};

describe("Response Conversion - Reasoning Content", () => {
  const originalRequest: ClaudeMessagesRequest = {
    model: "glm-5.1",
    max_tokens: 1000,
    messages: [{ role: "user", content: "What is 2+2?" }],
  };

  it("converts response with reasoning_content to thinking block", async () => {
    const openaiResponse: OpenAIResponse = {
      id: "chatcmpl-reasoning-1",
      object: "chat.completion",
      created: 1234567890,
      model: "glm-5.1",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: "2+2 equals 4.",
            reasoning_content: "Let me think step by step: 2+2=4",
          },
          finish_reason: "stop",
        },
      ],
      usage: {
        prompt_tokens: 15,
        completion_tokens: 20,
        total_tokens: 35,
      },
    };

    const result = await convertOpenAIToClaude(openaiResponse, originalRequest, defaultConfig);

    expect(result.type).toBe("message");
    expect(result.role).toBe("assistant");
    expect(result.model).toBe("glm-5.1");

    const content = result.content as Array<Record<string, unknown>>;
    // Should have both thinking and text blocks
    expect(content).toHaveLength(2);

    // First block: thinking
    expect(content[0].type).toBe("thinking");
    expect(content[0].thinking).toBe("Let me think step by step: 2+2=4");

    // Second block: text
    expect(content[1].type).toBe("text");
    expect(content[1].text).toBe("2+2 equals 4.");
  });

  it("handles response without reasoning_content normally", async () => {
    const openaiResponse: OpenAIResponse = {
      id: "chatcmpl-no-reasoning",
      object: "chat.completion",
      created: 1234567890,
      model: "glm-5.1",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: "Hello!",
          },
          finish_reason: "stop",
        },
      ],
    };

    const result = await convertOpenAIToClaude(openaiResponse, originalRequest, defaultConfig);
    const content = result.content as Array<Record<string, unknown>>;
    expect(content).toHaveLength(1);
    expect(content[0].type).toBe("text");
    expect(content[0].text).toBe("Hello!");
  });

  it("handles response with reasoning_content but null content", async () => {
    const openaiResponse: OpenAIResponse = {
      id: "chatcmpl-reasoning-only",
      object: "chat.completion",
      created: 1234567890,
      model: "glm-5.1",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: null,
            reasoning_content: "Deep thinking...",
          },
          finish_reason: "stop",
        },
      ],
    };

    const result = await convertOpenAIToClaude(openaiResponse, originalRequest, defaultConfig);
    const content = result.content as Array<Record<string, unknown>>;
    // Should have thinking block + empty text block (required by protocol)
    expect(content.length).toBeGreaterThanOrEqual(1);
    expect(content[0].type).toBe("thinking");
    expect(content[0].thinking).toBe("Deep thinking...");
  });

  it("handles response with empty reasoning_content string", async () => {
    const openaiResponse: OpenAIResponse = {
      id: "chatcmpl-empty-reasoning",
      object: "chat.completion",
      created: 1234567890,
      model: "glm-5.1",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: "Answer",
            reasoning_content: "",
          },
          finish_reason: "stop",
        },
      ],
    };

    const result = await convertOpenAIToClaude(openaiResponse, originalRequest, defaultConfig);
    const content = result.content as Array<Record<string, unknown>>;
    // Empty reasoning_content should not create a thinking block
    expect(content).toHaveLength(1);
    expect(content[0].type).toBe("text");
    expect(content[0].text).toBe("Answer");
  });

  it("handles tool calls alongside reasoning_content", async () => {
    const openaiResponse: OpenAIResponse = {
      id: "chatcmpl-reasoning-tool",
      object: "chat.completion",
      created: 1234567890,
      model: "glm-5.1",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: null,
            reasoning_content: "I need to use a tool to get the answer",
            tool_calls: [
              {
                id: "call_1",
                type: "function",
                function: {
                  name: "get_info",
                  arguments: '{"query":"test"}',
                },
              },
            ],
          },
          finish_reason: "tool_calls",
        },
      ],
    };

    const result = await convertOpenAIToClaude(openaiResponse, originalRequest, defaultConfig);
    const content = result.content as Array<Record<string, unknown>>;

    // Should have thinking + tool_use blocks
    expect(content.length).toBeGreaterThanOrEqual(2);
    expect(content[0].type).toBe("thinking");
    expect(content[0].thinking).toBe("I need to use a tool to get the answer");

    const toolBlock = content.find((b) => b.type === "tool_use");
    expect(toolBlock).toBeDefined();
    expect(toolBlock!.name).toBe("get_info");
    expect(result.stop_reason).toBe("tool_use");
  });
});

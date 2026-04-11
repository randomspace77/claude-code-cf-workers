import { describe, it, expect } from "vitest";
import { convertOpenAIToClaude } from "../src/conversion/response";
import type { OpenAIResponse, ClaudeMessagesRequest } from "../src/types";

describe("Response Conversion - Reasoning Content", () => {
  const originalRequest: ClaudeMessagesRequest = {
    model: "glm-5.1",
    max_tokens: 1000,
    messages: [{ role: "user", content: "What is 2+2?" }],
  };

  it("converts response with reasoning_content to thinking block", () => {
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

    const result = convertOpenAIToClaude(openaiResponse, originalRequest);

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

  it("handles response without reasoning_content normally", () => {
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

    const result = convertOpenAIToClaude(openaiResponse, originalRequest);
    const content = result.content as Array<Record<string, unknown>>;
    expect(content).toHaveLength(1);
    expect(content[0].type).toBe("text");
    expect(content[0].text).toBe("Hello!");
  });

  it("promotes reasoning_content to text when content is null (no tool calls)", () => {
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

    const result = convertOpenAIToClaude(openaiResponse, originalRequest);
    const content = result.content as Array<Record<string, unknown>>;
    // reasoning_content should be promoted to text (not thinking)
    // so Claude Code receives valid text content
    expect(content).toHaveLength(1);
    expect(content[0].type).toBe("text");
    expect(content[0].text).toBe("Deep thinking...");
  });

  it("promotes reasoning_content to text when content is empty string", () => {
    const openaiResponse: OpenAIResponse = {
      id: "chatcmpl-reasoning-empty",
      object: "chat.completion",
      created: 1234567890,
      model: "glm-5.1",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: "",
            reasoning_content: "Summary of the conversation...",
          },
          finish_reason: "stop",
        },
      ],
    };

    const result = convertOpenAIToClaude(openaiResponse, originalRequest);
    const content = result.content as Array<Record<string, unknown>>;
    expect(content).toHaveLength(1);
    expect(content[0].type).toBe("text");
    expect(content[0].text).toBe("Summary of the conversation...");
  });

  it("handles response with empty reasoning_content string", () => {
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

    const result = convertOpenAIToClaude(openaiResponse, originalRequest);
    const content = result.content as Array<Record<string, unknown>>;
    // Empty reasoning_content should not create a thinking block
    expect(content).toHaveLength(1);
    expect(content[0].type).toBe("text");
    expect(content[0].text).toBe("Answer");
  });

  it("handles tool calls alongside reasoning_content", () => {
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

    const result = convertOpenAIToClaude(openaiResponse, originalRequest);
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

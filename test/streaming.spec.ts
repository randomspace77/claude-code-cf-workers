import { describe, it, expect } from "vitest";
import { convertOpenAIStreamToClaude } from "../src/conversion/response";
import type { ClaudeMessagesRequest, AppConfig } from "../src/types";

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

/**
 * Helper to create a ReadableStream of SSE lines from an array of strings.
 * Uses start() to immediately enqueue all lines, avoiding pull-based
 * deadlock issues in the workers runtime.
 */
function makeSSEStream(lines: string[]): ReadableStream<string> {
  return new ReadableStream<string>({
    start(controller) {
      for (const line of lines) {
        controller.enqueue(line);
      }
      controller.close();
    },
  });
}

/**
 * Helper to collect all emitted SSE frames from the Claude stream.
 */
async function collectFrames(stream: ReadableStream<string>): Promise<string[]> {
  const reader = stream.getReader();
  const frames: string[] = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    frames.push(value);
  }
  return frames;
}

/**
 * Parse SSE frames into structured event objects.
 */
function parseFrames(
  frames: string[],
): Array<{ event: string; data: Record<string, unknown> }> {
  return frames
    .filter((f) => f.startsWith("event:"))
    .map((f) => {
      const lines = f.split("\n");
      const event = lines[0].replace("event: ", "").trim();
      const dataLine = lines.find((l) => l.startsWith("data: "));
      const data = dataLine ? JSON.parse(dataLine.replace("data: ", "")) : {};
      return { event, data };
    });
}

const originalRequest: ClaudeMessagesRequest = {
  model: "glm-5.1",
  max_tokens: 1000,
  messages: [{ role: "user", content: "Hello" }],
};

describe("convertOpenAIStreamToClaude", () => {
  it("converts a basic text streaming response", async () => {
    const openaiLines = [
      'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"glm-5.1","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}',
      'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"glm-5.1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',
      'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"glm-5.1","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}',
      'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":123,"model":"glm-5.1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
      "data: [DONE]",
    ];

    const stream = makeSSEStream(openaiLines);
    const claudeStream = convertOpenAIStreamToClaude(stream, originalRequest, defaultConfig);
    const frames = await collectFrames(claudeStream);
    const events = parseFrames(frames);

    // Should start with message_start
    expect(events[0].event).toBe("message_start");
    expect(events[0].data.type).toBe("message_start");
    const message = events[0].data.message as Record<string, unknown>;
    expect(message.model).toBe("glm-5.1");

    // Should have ping
    expect(events[1].event).toBe("ping");

    // Should have content_block_start for text
    const textStart = events.find(
      (e) =>
        e.event === "content_block_start" &&
        (e.data.content_block as Record<string, unknown>)?.type === "text",
    );
    expect(textStart).toBeDefined();

    // Should have text deltas
    const textDeltas = events.filter(
      (e) =>
        e.event === "content_block_delta" &&
        (e.data.delta as Record<string, unknown>)?.type === "text_delta",
    );
    expect(textDeltas.length).toBeGreaterThanOrEqual(1);

    // Should have message_delta with stop_reason
    const messageDelta = events.find((e) => e.event === "message_delta");
    expect(messageDelta).toBeDefined();
    const delta = messageDelta!.data.delta as Record<string, unknown>;
    expect(delta.stop_reason).toBe("end_turn");

    // Should have message_stop
    const messageStop = events.find((e) => e.event === "message_stop");
    expect(messageStop).toBeDefined();
  });

  it("converts streaming response with reasoning_content (GLM 5.1 thinking mode)", async () => {
    const openaiLines = [
      'data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":123,"model":"glm-5.1","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Let me"},"finish_reason":null}]}',
      'data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":123,"model":"glm-5.1","choices":[{"index":0,"delta":{"reasoning_content":" think..."},"finish_reason":null}]}',
      'data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":123,"model":"glm-5.1","choices":[{"index":0,"delta":{"content":"The answer"},"finish_reason":null}]}',
      'data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":123,"model":"glm-5.1","choices":[{"index":0,"delta":{"content":" is 4."},"finish_reason":null}]}',
      'data: {"id":"chatcmpl-2","object":"chat.completion.chunk","created":123,"model":"glm-5.1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
      "data: [DONE]",
    ];

    const stream = makeSSEStream(openaiLines);
    const claudeStream = convertOpenAIStreamToClaude(stream, originalRequest, defaultConfig);
    const frames = await collectFrames(claudeStream);
    const events = parseFrames(frames);

    // Should have thinking block start
    const thinkingStart = events.find(
      (e) =>
        e.event === "content_block_start" &&
        (e.data.content_block as Record<string, unknown>)?.type === "thinking",
    );
    expect(thinkingStart).toBeDefined();

    // Should have thinking deltas
    const thinkingDeltas = events.filter(
      (e) =>
        e.event === "content_block_delta" &&
        (e.data.delta as Record<string, unknown>)?.type === "thinking_delta",
    );
    expect(thinkingDeltas.length).toBeGreaterThanOrEqual(1);

    // Should have text block start after thinking
    const textStart = events.find(
      (e) =>
        e.event === "content_block_start" &&
        (e.data.content_block as Record<string, unknown>)?.type === "text",
    );
    expect(textStart).toBeDefined();

    // Text block should appear after thinking block
    const thinkingStartIdx = events.indexOf(thinkingStart!);
    const textStartIdx = events.indexOf(textStart!);
    expect(textStartIdx).toBeGreaterThan(thinkingStartIdx);

    // Should have text deltas
    const textDeltas = events.filter(
      (e) =>
        e.event === "content_block_delta" &&
        (e.data.delta as Record<string, unknown>)?.type === "text_delta",
    );
    expect(textDeltas.length).toBeGreaterThanOrEqual(1);
  });

  it("converts streaming response with tool calls", async () => {
    const openaiLines = [
      'data: {"id":"chatcmpl-3","object":"chat.completion.chunk","created":123,"model":"glm-5.1","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}',
      'data: {"id":"chatcmpl-3","object":"chat.completion.chunk","created":123,"model":"glm-5.1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"loc"}}]},"finish_reason":null}]}',
      'data: {"id":"chatcmpl-3","object":"chat.completion.chunk","created":123,"model":"glm-5.1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\\":\\"NYC\\"}"}}]},"finish_reason":null}]}',
      'data: {"id":"chatcmpl-3","object":"chat.completion.chunk","created":123,"model":"glm-5.1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
      "data: [DONE]",
    ];

    const stream = makeSSEStream(openaiLines);
    const claudeStream = convertOpenAIStreamToClaude(stream, originalRequest, defaultConfig);
    const frames = await collectFrames(claudeStream);
    const events = parseFrames(frames);

    // Should have tool_use block start
    const toolStart = events.find(
      (e) =>
        e.event === "content_block_start" &&
        (e.data.content_block as Record<string, unknown>)?.type === "tool_use",
    );
    expect(toolStart).toBeDefined();
    const block = toolStart!.data.content_block as Record<string, unknown>;
    expect(block.name).toBe("get_weather");
    expect(block.id).toBe("call_1");

    // Should have input_json_delta
    const jsonDeltas = events.filter(
      (e) =>
        e.event === "content_block_delta" &&
        (e.data.delta as Record<string, unknown>)?.type === "input_json_delta",
    );
    expect(jsonDeltas.length).toBeGreaterThanOrEqual(1);

    // Should map finish_reason to tool_use
    const messageDelta = events.find((e) => e.event === "message_delta");
    expect(messageDelta).toBeDefined();
    const mDelta = messageDelta!.data.delta as Record<string, unknown>;
    expect(mDelta.stop_reason).toBe("tool_use");
  });

  it("handles usage data in streaming", async () => {
    const openaiLines = [
      'data: {"id":"chatcmpl-4","object":"chat.completion.chunk","created":123,"model":"glm-5.1","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}',
      'data: {"id":"chatcmpl-4","object":"chat.completion.chunk","created":123,"model":"glm-5.1","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":3}}}',
      'data: {"id":"chatcmpl-4","object":"chat.completion.chunk","created":123,"model":"glm-5.1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
      "data: [DONE]",
    ];

    const stream = makeSSEStream(openaiLines);
    const claudeStream = convertOpenAIStreamToClaude(stream, originalRequest, defaultConfig);
    const frames = await collectFrames(claudeStream);
    const events = parseFrames(frames);

    const messageDelta = events.find((e) => e.event === "message_delta");
    expect(messageDelta).toBeDefined();
    const usage = messageDelta!.data.usage as Record<string, unknown>;
    expect(usage.input_tokens).toBe(10);
    expect(usage.output_tokens).toBe(5);
    expect(usage.cache_read_input_tokens).toBe(3);
  });

  it("handles length finish reason as max_tokens", async () => {
    const openaiLines = [
      'data: {"id":"chatcmpl-5","object":"chat.completion.chunk","created":123,"model":"glm-5.1","choices":[{"index":0,"delta":{"content":"Truncated"},"finish_reason":null}]}',
      'data: {"id":"chatcmpl-5","object":"chat.completion.chunk","created":123,"model":"glm-5.1","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}',
      "data: [DONE]",
    ];

    const stream = makeSSEStream(openaiLines);
    const claudeStream = convertOpenAIStreamToClaude(stream, originalRequest, defaultConfig);
    const frames = await collectFrames(claudeStream);
    const events = parseFrames(frames);

    const messageDelta = events.find((e) => e.event === "message_delta");
    expect(messageDelta).toBeDefined();
    const delta = messageDelta!.data.delta as Record<string, unknown>;
    expect(delta.stop_reason).toBe("max_tokens");
  });

  it("handles empty stream (stream done immediately)", async () => {
    const openaiLines = ["data: [DONE]"];

    const stream = makeSSEStream(openaiLines);
    const claudeStream = convertOpenAIStreamToClaude(stream, originalRequest, defaultConfig);
    const frames = await collectFrames(claudeStream);
    const events = parseFrames(frames);

    // Should still produce valid message structure
    expect(events[0].event).toBe("message_start");
    expect(events.find((e) => e.event === "message_stop")).toBeDefined();

    // Should create an empty text block
    const textStart = events.find(
      (e) =>
        e.event === "content_block_start" &&
        (e.data.content_block as Record<string, unknown>)?.type === "text",
    );
    expect(textStart).toBeDefined();
  });

  it("handles thinking-only stream (no text content)", async () => {
    const openaiLines = [
      'data: {"id":"chatcmpl-6","object":"chat.completion.chunk","created":123,"model":"glm-5.1","choices":[{"index":0,"delta":{"reasoning_content":"Thinking only..."},"finish_reason":null}]}',
      'data: {"id":"chatcmpl-6","object":"chat.completion.chunk","created":123,"model":"glm-5.1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
      "data: [DONE]",
    ];

    const stream = makeSSEStream(openaiLines);
    const claudeStream = convertOpenAIStreamToClaude(stream, originalRequest, defaultConfig);
    const frames = await collectFrames(claudeStream);
    const events = parseFrames(frames);

    // Should have thinking block
    const thinkingStart = events.find(
      (e) =>
        e.event === "content_block_start" &&
        (e.data.content_block as Record<string, unknown>)?.type === "thinking",
    );
    expect(thinkingStart).toBeDefined();

    // Should close thinking block and still have a text block
    const textStart = events.find(
      (e) =>
        e.event === "content_block_start" &&
        (e.data.content_block as Record<string, unknown>)?.type === "text",
    );
    expect(textStart).toBeDefined();

    // Should have message_stop
    expect(events.find((e) => e.event === "message_stop")).toBeDefined();
  });

  it("ignores non-data SSE lines", async () => {
    const openaiLines = [
      ": this is a comment",
      "",
      'data: {"id":"chatcmpl-7","object":"chat.completion.chunk","created":123,"model":"glm-5.1","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}',
      'data: {"id":"chatcmpl-7","object":"chat.completion.chunk","created":123,"model":"glm-5.1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
      "data: [DONE]",
    ];

    const stream = makeSSEStream(openaiLines);
    const claudeStream = convertOpenAIStreamToClaude(stream, originalRequest, defaultConfig);
    const frames = await collectFrames(claudeStream);
    const events = parseFrames(frames);

    // Should still work correctly
    const textDeltas = events.filter(
      (e) =>
        e.event === "content_block_delta" &&
        (e.data.delta as Record<string, unknown>)?.type === "text_delta",
    );
    expect(textDeltas.length).toBeGreaterThanOrEqual(1);
  });
});

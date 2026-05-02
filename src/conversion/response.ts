import { Constants } from "../constants";
import type {
  AppConfig,
  ClaudeMessagesRequest,
  OpenAIResponse,
  OpenAIStreamChunk,
} from "../types";
import { parseSSEChunk } from "../client";
import { setAssistantReasoning, setToolReasoning } from "./reasoning-cache";

// ---- Non-streaming conversion ----

/**
 * Convert an OpenAI chat completion response into a Claude Messages response.
 */
export async function convertOpenAIToClaude(
  openaiResponse: OpenAIResponse,
  originalRequest: ClaudeMessagesRequest,
  config: AppConfig,
  logLevel?: string,
  cacheModel = openaiResponse.model || originalRequest.model,
): Promise<Record<string, unknown>> {
  const choices = openaiResponse.choices ?? [];
  if (choices.length === 0) {
    throw new Error("No choices in OpenAI response");
  }

  const choice = choices[0];
  const message = choice.message;
  const debug = logLevel === "DEBUG";

  const reasoning = reasoningFromMessage(message);

  if (debug) {
    // Full response metadata + raw content in DEBUG mode
    console.log({
      _tag: "non-stream-response",
      has_content: message?.content !== null && message?.content !== undefined,
      content_type: typeof message?.content,
      content_length: typeof message?.content === "string" ? message.content.length : 0,
      has_reasoning: Boolean(reasoning),
      reasoning_length: reasoning.length,
      tool_calls_count: message?.tool_calls?.length ?? 0,
      finish_reason: choice.finish_reason,
      usage: openaiResponse.usage,
      content: message?.content,
      reasoning_content: reasoning,
    });
  } else {
    // WARNING level: only log anomalies
    const hasContent = message?.content !== null && message?.content !== undefined && message?.content !== "";
    const hasReasoning = Boolean(reasoning);
    const hasToolCalls = (message?.tool_calls?.length ?? 0) > 0;
    if (!hasContent && !hasReasoning && !hasToolCalls) {
      console.warn({
        _tag: "non-stream-response-empty",
        finish_reason: choice.finish_reason,
        usage: openaiResponse.usage,
        message: "Response has no content, reasoning, or tool calls",
      });
    }
  }

  const contentBlocks: Record<string, unknown>[] = [];

  // Reasoning/thinking content (e.g. from GLM 5.1 thinking mode)
  if (reasoning) {
    contentBlocks.push({
      type: Constants.CONTENT_THINKING,
      thinking: reasoning,
    });
  }

  // Text content
  if (message?.content !== null && message?.content !== undefined) {
    if (reasoning) {
      await setAssistantReasoning(config, cacheModel, message.content, reasoning);
    }
    contentBlocks.push({
      type: Constants.CONTENT_TEXT,
      text: message.content,
    });
  }

  // Tool calls
  if (message?.tool_calls) {
    for (const toolCall of message.tool_calls) {
      if (toolCall.type === Constants.TOOL_FUNCTION) {
        if (reasoning) {
          await setToolReasoning(config, cacheModel, toolCall.id, reasoning);
        }
        let args: Record<string, unknown>;
        try {
          args = JSON.parse(toolCall.function.arguments ?? "{}");
        } catch {
          args = { raw_arguments: toolCall.function.arguments ?? "" };
        }
        contentBlocks.push({
          type: Constants.CONTENT_TOOL_USE,
          id: toolCall.id ?? `tool_${crypto.randomUUID()}`,
          name: toolCall.function.name ?? "",
          input: args,
        });
      }
    }
  }

  // Ensure at least one content block
  if (contentBlocks.length === 0) {
    contentBlocks.push({ type: Constants.CONTENT_TEXT, text: "" });
  }

  // Map finish reason
  const finishReason = choice.finish_reason ?? "stop";
  const stopReasonMap: Record<string, string> = {
    stop: Constants.STOP_END_TURN,
    length: Constants.STOP_MAX_TOKENS,
    tool_calls: Constants.STOP_TOOL_USE,
    function_call: Constants.STOP_TOOL_USE,
  };
  const stopReason = stopReasonMap[finishReason] ?? Constants.STOP_END_TURN;

  return {
    id: openaiResponse.id ?? `msg_${crypto.randomUUID()}`,
    type: "message",
    role: Constants.ROLE_ASSISTANT,
    model: originalRequest.model,
    content: contentBlocks,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: openaiResponse.usage?.prompt_tokens ?? 0,
      output_tokens: openaiResponse.usage?.completion_tokens ?? 0,
    },
  };
}

// ---- Streaming conversion ----

/**
 * Convert an OpenAI streaming response (ReadableStream of SSE lines) into a
 * Claude SSE streaming response.
 *
 * Returns a ReadableStream<string> where each item is a complete SSE frame
 * (including "event:" and "data:" prefixes with trailing double newlines).
 */
export function convertOpenAIStreamToClaude(
  openaiStream: ReadableStream<string>,
  originalRequest: ClaudeMessagesRequest,
  config: AppConfig,
  logLevel?: string,
  cacheModel = originalRequest.model,
): ReadableStream<string> {
  const debug = logLevel === "DEBUG";
  const messageId = `msg_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`;

  // Block tracking: thinking block (optional, index 0 if present),
  // then text block, then tool blocks
  let thinkingBlockStarted = false;
  let thinkingBlockIndex = -1;
  let textBlockStarted = false;
  let textBlockIndex = 0;
  let nextBlockIndex = 0;
  let toolBlockCounter = 0;
  const currentToolCalls: Map<
    number,
    {
      id: string | null;
      name: string | null;
      argsBuffer: string;
      jsonSent: boolean;
      claudeIndex: number | null;
      started: boolean;
    }
  > = new Map();
  let finalStopReason: string = Constants.STOP_END_TURN;
  let usageData: Record<string, unknown> = {
    input_tokens: 0,
    output_tokens: 0,
  };
  let reasoningContent = "";
  let textContent = "";

  const reader = openaiStream.getReader();

  return new ReadableStream<string>({
    async start(controller) {
      // Initial SSE events
      controller.enqueue(
        sseFrame(Constants.EVENT_MESSAGE_START, {
          type: Constants.EVENT_MESSAGE_START,
          message: {
            id: messageId,
            type: "message",
            role: Constants.ROLE_ASSISTANT,
            model: originalRequest.model,
            content: [],
            stop_reason: null,
            stop_sequence: null,
            usage: { input_tokens: 0, output_tokens: 0 },
          },
        }),
      );

      controller.enqueue(
        sseFrame(Constants.EVENT_PING, { type: Constants.EVENT_PING }),
      );
    },

    async pull(controller) {
      try {
        let enqueued = false;
        // eslint-disable-next-line no-constant-condition
        while (true) {
          const { done, value: line } = await reader.read();
          if (done) {
            await emitFinalEvents(controller);
            controller.close();
            return;
          }

          const chunk = parseSSEChunk(line);
          if (!chunk) {
            // Check for [DONE]
            if (line.includes("[DONE]")) {
              await emitFinalEvents(controller);
              controller.close();
              return;
            }
            continue;
          }

          // Handle usage data
          if (chunk.usage) {
            const cacheRead =
              chunk.usage.prompt_tokens_details?.cached_tokens ?? 0;
            usageData = {
              input_tokens: chunk.usage.prompt_tokens ?? 0,
              output_tokens: chunk.usage.completion_tokens ?? 0,
              cache_read_input_tokens: cacheRead,
            };
          }

          const choices = chunk.choices ?? [];
          if (choices.length === 0) continue;

          const choice = choices[0];
          const delta = choice.delta ?? {};
          const finishReason = choice.finish_reason;

          // Log streaming deltas when DEBUG is enabled
          if (debug) {
            const reasoningDelta = reasoningFromMessage(delta);
            if (reasoningDelta) {
              console.log({
                _tag: "stream-delta",
                type: "reasoning",
                length: reasoningDelta.length,
                content: reasoningDelta,
              });
            }
            if (delta.content !== null && delta.content !== undefined) {
              console.log({
                _tag: "stream-delta",
                type: "content",
                length: delta.content.length,
                content: delta.content,
              });
            }
            if (finishReason) {
              console.log({
                _tag: "stream-finish",
                finish_reason: finishReason,
                usage: chunk.usage,
              });
            }
          }

          // Reasoning/thinking delta (e.g. from GLM 5.1)
          const reasoningDelta = reasoningFromMessage(delta);
          if (reasoningDelta) {
            reasoningContent += reasoningDelta;
            if (!thinkingBlockStarted) {
              thinkingBlockIndex = nextBlockIndex++;
              thinkingBlockStarted = true;
              controller.enqueue(
                sseFrame(Constants.EVENT_CONTENT_BLOCK_START, {
                  type: Constants.EVENT_CONTENT_BLOCK_START,
                  index: thinkingBlockIndex,
                  content_block: { type: Constants.CONTENT_THINKING, thinking: "" },
                }),
              );
            }
            controller.enqueue(
              sseFrame(Constants.EVENT_CONTENT_BLOCK_DELTA, {
                type: Constants.EVENT_CONTENT_BLOCK_DELTA,
                index: thinkingBlockIndex,
                delta: {
                  type: Constants.DELTA_THINKING,
                  thinking: reasoningDelta,
                },
              }),
            );
            enqueued = true;
          }

          // Text delta
          if (delta.content !== null && delta.content !== undefined) {
            textContent += delta.content;
            if (!textBlockStarted) {
              // Close thinking block first if it was open
              if (thinkingBlockStarted) {
                controller.enqueue(
                  sseFrame(Constants.EVENT_CONTENT_BLOCK_STOP, {
                    type: Constants.EVENT_CONTENT_BLOCK_STOP,
                    index: thinkingBlockIndex,
                  }),
                );
              }
              textBlockIndex = nextBlockIndex++;
              textBlockStarted = true;
              controller.enqueue(
                sseFrame(Constants.EVENT_CONTENT_BLOCK_START, {
                  type: Constants.EVENT_CONTENT_BLOCK_START,
                  index: textBlockIndex,
                  content_block: { type: Constants.CONTENT_TEXT, text: "" },
                }),
              );
            }
            controller.enqueue(
              sseFrame(Constants.EVENT_CONTENT_BLOCK_DELTA, {
                type: Constants.EVENT_CONTENT_BLOCK_DELTA,
                index: textBlockIndex,
                delta: {
                  type: Constants.DELTA_TEXT,
                  text: delta.content,
                },
              }),
            );
            enqueued = true;
          }

          // Tool call deltas
          if (delta.tool_calls) {
            for (const tcDelta of delta.tool_calls) {
              const tcIndex = tcDelta.index ?? 0;

              if (!currentToolCalls.has(tcIndex)) {
                currentToolCalls.set(tcIndex, {
                  id: null,
                  name: null,
                  argsBuffer: "",
                  jsonSent: false,
                  claudeIndex: null,
                  started: false,
                });
              }

              const toolCall = currentToolCalls.get(tcIndex)!;

              if (tcDelta.id) toolCall.id = tcDelta.id;

              const fnData = tcDelta.function ?? {};
              if (fnData.name) toolCall.name = fnData.name;

              // Start content block when we have complete initial data
              if (toolCall.id && toolCall.name && !toolCall.started) {
                toolBlockCounter += 1;
                const claudeIndex = nextBlockIndex++;
                toolCall.claudeIndex = claudeIndex;
                toolCall.started = true;

                controller.enqueue(
                  sseFrame(Constants.EVENT_CONTENT_BLOCK_START, {
                    type: Constants.EVENT_CONTENT_BLOCK_START,
                    index: claudeIndex,
                    content_block: {
                      type: Constants.CONTENT_TOOL_USE,
                      id: toolCall.id,
                      name: toolCall.name,
                      input: {},
                    },
                  }),
                );
                enqueued = true;
              }

              // Handle function arguments
              if (
                fnData.arguments !== undefined &&
                fnData.arguments !== null &&
                toolCall.started
              ) {
                toolCall.argsBuffer += fnData.arguments;

                // Try to parse complete JSON
                try {
                  JSON.parse(toolCall.argsBuffer);
                  if (!toolCall.jsonSent) {
                    controller.enqueue(
                      sseFrame(Constants.EVENT_CONTENT_BLOCK_DELTA, {
                        type: Constants.EVENT_CONTENT_BLOCK_DELTA,
                        index: toolCall.claudeIndex,
                        delta: {
                          type: Constants.DELTA_INPUT_JSON,
                          partial_json: toolCall.argsBuffer,
                        },
                      }),
                    );
                    toolCall.jsonSent = true;
                    enqueued = true;
                  }
                } catch {
                  // JSON incomplete, continue accumulating
                }
              }
            }
          }

          // Handle finish reason
          if (finishReason) {
            if (finishReason === "length") {
              finalStopReason = Constants.STOP_MAX_TOKENS;
            } else if (
              finishReason === "tool_calls" ||
              finishReason === "function_call"
            ) {
              finalStopReason = Constants.STOP_TOOL_USE;
            } else {
              finalStopReason = Constants.STOP_END_TURN;
            }
          }

          // Only yield back to the consumer when we have enqueued data;
          // otherwise continue processing the next line to avoid stalling.
          if (enqueued) {
            return;
          }
        }
      } catch (err) {
        console.error("Streaming conversion error:", err);
        const errorEvent = {
          type: "error",
          error: {
            type: "api_error",
            message: "An error occurred while processing the streaming response.",
          },
        };
        controller.enqueue(`event: error\ndata: ${JSON.stringify(errorEvent)}\n\n`);
        controller.close();
      }
    },

    cancel() {
      reader.cancel().catch(() => {});
    },
  });

  async function emitFinalEvents(controller: ReadableStreamDefaultController<string>) {
    if (debug) {
      console.log({
        _tag: "stream-final-state",
        thinkingBlockStarted,
        textBlockStarted,
        toolCallsCount: currentToolCalls.size,
        finalStopReason,
        usageData,
      });
    }

    // Close thinking block if it wasn't closed (no text content followed)
    if (thinkingBlockStarted && !textBlockStarted) {
      controller.enqueue(
        sseFrame(Constants.EVENT_CONTENT_BLOCK_STOP, {
          type: Constants.EVENT_CONTENT_BLOCK_STOP,
          index: thinkingBlockIndex,
        }),
      );
    }

    // Ensure text block exists and close it
    if (!textBlockStarted) {
      // Start an empty text block if none existed
      textBlockIndex = nextBlockIndex++;
      textBlockStarted = true;
      controller.enqueue(
        sseFrame(Constants.EVENT_CONTENT_BLOCK_START, {
          type: Constants.EVENT_CONTENT_BLOCK_START,
          index: textBlockIndex,
          content_block: { type: Constants.CONTENT_TEXT, text: "" },
        }),
      );
    }

    // Close text block
    controller.enqueue(
      sseFrame(Constants.EVENT_CONTENT_BLOCK_STOP, {
        type: Constants.EVENT_CONTENT_BLOCK_STOP,
        index: textBlockIndex,
      }),
    );

    // Close tool blocks
    for (const toolData of currentToolCalls.values()) {
      if (toolData.started && toolData.claudeIndex !== null) {
        controller.enqueue(
          sseFrame(Constants.EVENT_CONTENT_BLOCK_STOP, {
            type: Constants.EVENT_CONTENT_BLOCK_STOP,
            index: toolData.claudeIndex,
          }),
        );
      }
    }

    // Message delta with final stop reason
    controller.enqueue(
      sseFrame(Constants.EVENT_MESSAGE_DELTA, {
        type: Constants.EVENT_MESSAGE_DELTA,
        delta: {
          stop_reason: finalStopReason,
          stop_sequence: null,
        },
        usage: usageData,
      }),
    );

    // Message stop
    controller.enqueue(
      sseFrame(Constants.EVENT_MESSAGE_STOP, {
        type: Constants.EVENT_MESSAGE_STOP,
      }),
    );

    if (reasoningContent) {
      if (textContent) {
        await setAssistantReasoning(config, cacheModel, textContent, reasoningContent);
      }
      await Promise.all(
        [...currentToolCalls.values()].map((toolData) =>
          setToolReasoning(config, cacheModel, toolData.id, reasoningContent),
        ),
      );
    }
  }
}

/** Format a single SSE frame. */
function sseFrame(event: string, data: Record<string, unknown>): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function reasoningFromMessage(message: unknown): string {
  if (!message || typeof message !== "object") return "";
  const record = message as Record<string, unknown>;
  if (typeof record.reasoning_content === "string") return record.reasoning_content;
  if (typeof record.reasoning === "string") return record.reasoning;
  if (
    record.reasoning &&
    typeof record.reasoning === "object" &&
    typeof (record.reasoning as Record<string, unknown>).content === "string"
  ) {
    return (record.reasoning as Record<string, unknown>).content as string;
  }
  if (typeof record.thinking === "string") return record.thinking;
  if (
    record.thinking &&
    typeof record.thinking === "object" &&
    typeof (record.thinking as Record<string, unknown>).content === "string"
  ) {
    return (record.thinking as Record<string, unknown>).content as string;
  }
  return "";
}

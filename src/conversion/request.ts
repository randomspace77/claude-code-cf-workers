import { Constants } from "../constants";
import type {
  AppConfig,
  ClaudeMessagesRequest,
  ClaudeMessage,
  ClaudeContentBlock,
  OpenAIRequest,
  OpenAIMessage,
  OpenAIContentPart,
  OpenAIToolCall,
  OpenAITool,
} from "../types";
import { getAssistantReasoning, getToolReasoning } from "./reasoning-cache";

const PLACEHOLDER_REASONING =
  "Compatibility bridge placeholder reasoning for prior assistant history.";

function isDeepSeekModel(model: string): boolean {
  return /(^|[-_/])deepseek/i.test(model);
}

function thinkingFromAnthropicContent(content: unknown): string {
  if (!Array.isArray(content)) return "";
  return content
    .filter(
      (block) =>
        block &&
        block.type === Constants.CONTENT_THINKING &&
        typeof (block as Record<string, unknown>).thinking === "string",
    )
    .map((block) => (block as Record<string, unknown>).thinking as string)
    .filter(Boolean)
    .join("\n");
}

function reasoningEffortFromOutputConfig(
  outputConfig: { effort?: string } | undefined,
): string | undefined {
  const effort = outputConfig && typeof outputConfig === "object" ? outputConfig.effort : undefined;
  if (typeof effort !== "string") return undefined;
  const normalized = effort.toLowerCase();
  if (normalized === "max" || normalized === "xhigh") return "max";
  if (normalized === "high" || normalized === "medium" || normalized === "low") return "high";
  return undefined;
}

function thinkingToOpenAi(
  thinking: { type: string; budget_tokens?: number } | undefined,
): { type: string } | undefined {
  if (!thinking || typeof thinking !== "object") return undefined;
  if (thinking.type === "enabled" || thinking.type === "disabled") {
    return { type: thinking.type };
  }
  return undefined;
}

/**
 * DeepSeek reasoner rejects forced tool_choice. Convert "any" / "tool"
 * to a system instruction instead, so the model still knows it must call.
 */
function buildToolChoiceInstruction(
  toolChoice: Record<string, unknown> | undefined,
  model: string,
): string | null {
  if (!toolChoice || typeof toolChoice !== "object") return null;
  if (!isDeepSeekModel(model)) return null;
  if (toolChoice.type === "any") {
    return "The caller requires a tool call for this turn. Call one of the available tools instead of answering directly.";
  }
  if (toolChoice.type === "tool" && typeof toolChoice.name === "string") {
    return `The caller requires a tool call for this turn. Call the available tool named ${JSON.stringify(toolChoice.name)} instead of answering directly.`;
  }
  return null;
}

/**
 * Convert a Claude Messages API request into an OpenAI Chat Completions
 * request. Model mapping is handled by the provider layer before this is called.
 */
export async function convertClaudeToOpenAI(
  claudeRequest: ClaudeMessagesRequest,
  config: AppConfig,
): Promise<OpenAIRequest> {
  const openaiModel = claudeRequest.model;

  const openaiMessages: OpenAIMessage[] = [];

  // System message
  if (claudeRequest.system) {
    const systemText = extractSystemText(claudeRequest.system);
    if (systemText.trim()) {
      openaiMessages.push({
        role: Constants.ROLE_SYSTEM as "system",
        content: systemText.trim(),
      });
    }
  }

  // Build tool_choice system instruction for DeepSeek models (which reject forced tool_choice)
  const toolChoiceInstruction = buildToolChoiceInstruction(claudeRequest.tool_choice, openaiModel);

  // Process messages
  let i = 0;
  let prevAssistantHadToolCall = false;
  while (i < claudeRequest.messages.length) {
    const msg = claudeRequest.messages[i];

    if (msg.role === Constants.ROLE_USER) {
      const blocks = Array.isArray(msg.content) ? msg.content : [];
      const toolResults = blocks.filter(
        (block) => "type" in block && block.type === Constants.CONTENT_TOOL_RESULT,
      );

      if (!toolResults.length) {
        prevAssistantHadToolCall = false;
        // Normal user message (text, multimodal, or null content)
        openaiMessages.push(convertUserMessage(msg));
      } else {
        // User message with tool_results: push text portion (if any) as user,
        // then tool_results as tool-role messages
        const textBlocks = blocks.filter(
          (block) => block.type === Constants.CONTENT_TEXT,
        );
        if (textBlocks.length > 0) {
          openaiMessages.push(convertUserMessage(msg));
        }
        if (prevAssistantHadToolCall) {
          const toolMessages = convertToolResultsInternal(toolResults);
          openaiMessages.push(...toolMessages);
        }
      }
    } else if (msg.role === Constants.ROLE_ASSISTANT) {
      const hasToolUses =
        Array.isArray(msg.content) &&
        msg.content.some((block) => "type" in block && block.type === Constants.CONTENT_TOOL_USE);
      const assistant = await convertAssistantMessage(msg, openaiModel, config, prevAssistantHadToolCall);
      openaiMessages.push(assistant);
      prevAssistantHadToolCall = hasToolUses;
    }

    i += 1;
  }

  // Build request
  const openaiRequest: OpenAIRequest = {
    model: openaiModel,
    messages: openaiMessages,
    max_tokens: Math.min(
      Math.max(claudeRequest.max_tokens, config.minTokensLimit),
      config.maxTokensLimit,
    ),
    temperature: claudeRequest.temperature,
    stream: claudeRequest.stream ?? false,
  };

  // Optional parameters
  if (claudeRequest.stop_sequences) {
    openaiRequest.stop = claudeRequest.stop_sequences;
  }
  if (claudeRequest.top_p !== undefined && claudeRequest.top_p !== null) {
    openaiRequest.top_p = claudeRequest.top_p;
  }

  // Convert tools
  if (claudeRequest.tools?.length) {
    const openaiTools: OpenAITool[] = [];
    for (const tool of claudeRequest.tools) {
      const openaiTool = normalizeTool(tool);
      if (openaiTool) {
        openaiTools.push(openaiTool);
      }
    }
    if (openaiTools.length > 0) {
      openaiRequest.tools = openaiTools;
    }
  }

  // Convert tool choice (DeepSeek models reject forced tool_choice — softened to system instruction)
  if (claudeRequest.tool_choice) {
    const choiceType = claudeRequest.tool_choice.type as string | undefined;
    if (choiceType === "auto") {
      openaiRequest.tool_choice = "auto";
    } else if (isDeepSeekModel(openaiModel)) {
      // DeepSeek reasoner rejects forced tool_choice — handled via system instruction instead
      openaiRequest.tool_choice = undefined;
    } else if (choiceType === "any") {
      openaiRequest.tool_choice = "required";
    } else if (choiceType === "tool" && claudeRequest.tool_choice.name) {
      openaiRequest.tool_choice = {
        type: "function",
        function: { name: claudeRequest.tool_choice.name as string },
      };
    }
  }

  // DeepSeek-specific: map thinking and reasoning_effort fields
  if (isDeepSeekModel(openaiModel)) {
    const thinking = thinkingToOpenAi(claudeRequest.thinking);
    if (thinking) openaiRequest.thinking = thinking;
    const effort = reasoningEffortFromOutputConfig(claudeRequest.output_config);
    if (effort) openaiRequest.reasoning_effort = effort;
  }

  // Clean up undefined fields
  if (openaiRequest.tool_choice === undefined) delete openaiRequest.tool_choice;

  // Inject tool_choice system instruction for DeepSeek models
  if (toolChoiceInstruction) {
    const sysMsg = openaiMessages.find((m) => m.role === "system");
    if (sysMsg && typeof sysMsg.content === "string") {
      sysMsg.content = sysMsg.content + "\n\n" + toolChoiceInstruction;
    } else {
      openaiMessages.unshift({ role: "system", content: toolChoiceInstruction });
    }
  }

  return openaiRequest;
}

// ---- Helpers ----

function normalizeTool(tool: unknown): OpenAITool | null {
  if (!tool || typeof tool !== "object") return null;
  const record = tool as Record<string, unknown>;

  const openAiFunction =
    record.function && typeof record.function === "object"
      ? (record.function as Record<string, unknown>)
      : null;

  const rawName = openAiFunction?.name ?? record.name;
  if (typeof rawName !== "string") return null;
  const name = rawName.trim();
  if (!name) return null;

  const rawDescription = openAiFunction?.description ?? record.description;
  const rawParameters = openAiFunction?.parameters ?? record.input_schema;

  return {
    type: Constants.TOOL_FUNCTION as "function",
    function: {
      name,
      description: typeof rawDescription === "string" ? rawDescription : "",
      parameters: normalizeToolParameters(rawParameters),
    },
  };
}

function normalizeToolParameters(parameters: unknown): Record<string, unknown> {
  if (parameters && typeof parameters === "object" && !Array.isArray(parameters)) {
    return parameters as Record<string, unknown>;
  }
  return { type: "object", properties: {} };
}

function extractSystemText(
  system: string | Array<{ type: string; text: string }>,
): string {
  if (typeof system === "string") return system;
  return system
    .filter((block) => block.type === Constants.CONTENT_TEXT)
    .map((block) => block.text)
    .join("\n\n");
}

function convertUserMessage(msg: ClaudeMessage): OpenAIMessage {
  if (msg.content === null || msg.content === undefined) {
    return { role: "user", content: "" };
  }

  if (typeof msg.content === "string") {
    return { role: "user", content: msg.content };
  }

  // Multimodal content
  const parts: OpenAIContentPart[] = [];
  for (const block of msg.content) {
    if (block.type === Constants.CONTENT_TEXT) {
      parts.push({
        type: "text",
        text: (block as { type: "text"; text: string }).text,
      });
    } else if (block.type === Constants.CONTENT_IMAGE) {
      const src = (block as { type: "image"; source: Record<string, string> }).source;
      if (src.type === "base64" && src.media_type && src.data) {
        parts.push({
          type: "image_url",
          image_url: {
            url: `data:${src.media_type};base64,${src.data}`,
          },
        });
      }
    }
  }

  if (parts.length === 1 && parts[0].type === "text") {
    return { role: "user", content: parts[0].text! };
  }
  return { role: "user", content: parts };
}

async function convertAssistantMessage(
  msg: ClaudeMessage,
  model: string,
  config: AppConfig,
  hasToolCall = false,
): Promise<OpenAIMessage> {
  if (msg.content === null || msg.content === undefined) {
    return { role: "assistant", content: null };
  }

  if (typeof msg.content === "string") {
    return { role: "assistant", content: msg.content };
  }

  const textParts: string[] = [];
  const toolCalls: OpenAIToolCall[] = [];

  for (const block of msg.content) {
    if (block.type === Constants.CONTENT_TEXT) {
      textParts.push((block as { type: "text"; text: string }).text);
    } else if (block.type === Constants.CONTENT_TOOL_USE) {
      const toolBlock = block as {
        type: "tool_use";
        id: string;
        name: string;
        input: Record<string, unknown>;
      };
      toolCalls.push({
        id: toolBlock.id,
        type: "function",
        function: {
          name: toolBlock.name,
          arguments: JSON.stringify(toolBlock.input),
        },
      });
    }
  }

  const assistantText = textParts.length > 0 ? textParts.join("") : null;
  const result: OpenAIMessage = {
    role: "assistant",
    content: assistantText,
  };

  if (toolCalls.length > 0) {
    result.tool_calls = toolCalls;
  }

  // Inject reasoning_content for DeepSeek models from thinking blocks in history
  if (isDeepSeekModel(model)) {
    const thinking = thinkingFromAnthropicContent(msg.content);
    if (thinking) {
      result.reasoning_content = thinking;
    } else if (toolCalls.length > 0 || hasToolCall) {
      const toolReasoning = toolCalls
        .map((toolCall) => getToolReasoning(config, model, toolCall.id));
      const resolvedToolReasoning = (await Promise.all(toolReasoning))
        .filter(Boolean)
        .join("\n");
      result.reasoning_content =
        resolvedToolReasoning ||
        (await getAssistantReasoning(config, model, assistantText)) ||
        PLACEHOLDER_REASONING;
    }
  }

  return result;
}

function convertToolResultsInternal(
  toolResults: ClaudeContentBlock[],
): OpenAIMessage[] {
  const toolMessages: OpenAIMessage[] = [];

  for (const block of toolResults) {
    const toolResult = block as {
      type: "tool_result";
      tool_use_id: string;
      content: unknown;
    };
    toolMessages.push({
      role: "tool",
      tool_call_id: toolResult.tool_use_id,
      content: parseToolResultContent(toolResult.content),
    });
  }

  return toolMessages;
}

function parseToolResultContent(content: unknown): string {
  if (content === null || content === undefined) return "No content provided";
  if (typeof content === "string") return content;

  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const item of content) {
      if (typeof item === "string") {
        parts.push(item);
      } else if (typeof item === "object" && item !== null) {
        const obj = item as Record<string, unknown>;
        if (obj.type === Constants.CONTENT_TEXT && typeof obj.text === "string") {
          parts.push(obj.text);
        } else if (typeof obj.text === "string") {
          parts.push(obj.text);
        } else {
          try {
            parts.push(JSON.stringify(obj));
          } catch {
            parts.push(String(obj));
          }
        }
      }
    }
    return parts.join("\n").trim();
  }

  if (typeof content === "object") {
    const obj = content as Record<string, unknown>;
    if (obj.type === Constants.CONTENT_TEXT && typeof obj.text === "string") {
      return obj.text;
    }
    try {
      return JSON.stringify(obj);
    } catch {
      return String(content);
    }
  }

  return String(content);
}

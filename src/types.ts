/** Type definitions for the Claude-to-OpenAI proxy */

// ---- Environment / Cloudflare bindings ----

export interface Env {
  OPENAI_API_KEY: string;
  OPENAI_BASE_URL: string;
  ANTHROPIC_API_KEY?: string;
  AZURE_API_VERSION?: string;
  BIG_MODEL: string;
  MIDDLE_MODEL: string;
  SMALL_MODEL: string;
  MAX_TOKENS_LIMIT: string;
  MIN_TOKENS_LIMIT: string;
  REQUEST_TIMEOUT: string;
  LOG_LEVEL: string;
  CUSTOM_HEADERS?: string; // JSON string of custom headers
  PROXY_MODE?: string; // "openai" (default) or "passthrough"
  ENABLE_MODEL_MAPPING?: string; // "true" to enable Claude→provider model mapping
}

// ---- Claude API types ----

export interface ClaudeContentBlockText {
  type: "text";
  text: string;
}

export interface ClaudeContentBlockImage {
  type: "image";
  source: {
    type: string;
    media_type?: string;
    data?: string;
  };
}

export interface ClaudeContentBlockToolUse {
  type: "tool_use";
  id: string;
  name: string;
  input: Record<string, unknown>;
}

export interface ClaudeContentBlockToolResult {
  type: "tool_result";
  tool_use_id: string;
  content: string | Array<Record<string, unknown>> | Record<string, unknown>;
}

export type ClaudeContentBlock =
  | ClaudeContentBlockText
  | ClaudeContentBlockImage
  | ClaudeContentBlockToolUse
  | ClaudeContentBlockToolResult;

export interface ClaudeSystemContent {
  type: "text";
  text: string;
}

export interface ClaudeMessage {
  role: "user" | "assistant";
  content: string | ClaudeContentBlock[] | null;
}

export interface ClaudeTool {
  name: string;
  description?: string;
  input_schema: Record<string, unknown>;
}

export interface ClaudeThinkingConfig {
  enabled: boolean;
}

export interface ClaudeMessagesRequest {
  model: string;
  max_tokens: number;
  messages: ClaudeMessage[];
  system?: string | ClaudeSystemContent[];
  stop_sequences?: string[];
  stream?: boolean;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  metadata?: Record<string, unknown>;
  tools?: ClaudeTool[];
  tool_choice?: Record<string, unknown>;
  thinking?: ClaudeThinkingConfig;
}

export interface ClaudeTokenCountRequest {
  model: string;
  messages: ClaudeMessage[];
  system?: string | ClaudeSystemContent[];
  tools?: ClaudeTool[];
  thinking?: ClaudeThinkingConfig;
  tool_choice?: Record<string, unknown>;
}

// ---- OpenAI API types ----

export interface OpenAIMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | OpenAIContentPart[] | null;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
  reasoning_content?: string | null;
}

export interface OpenAIContentPart {
  type: "text" | "image_url";
  text?: string;
  image_url?: { url: string };
}

export interface OpenAIToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

export interface OpenAITool {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

export interface OpenAIRequest {
  model: string;
  messages: OpenAIMessage[];
  max_tokens: number;
  temperature?: number;
  stream: boolean;
  stop?: string[];
  top_p?: number;
  tools?: OpenAITool[];
  tool_choice?: string | { type: "function"; function: { name: string } };
  stream_options?: { include_usage: boolean };
}

export interface OpenAIChoice {
  index: number;
  message?: {
    role: string;
    content: string | null;
    tool_calls?: OpenAIToolCall[];
    reasoning_content?: string | null;
  };
  delta?: {
    role?: string;
    content?: string | null;
    reasoning_content?: string | null;
    tool_calls?: Array<{
      index: number;
      id?: string;
      type?: string;
      function?: {
        name?: string;
        arguments?: string;
      };
    }>;
  };
  finish_reason: string | null;
}

export interface OpenAIResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: OpenAIChoice[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    prompt_tokens_details?: {
      cached_tokens?: number;
    };
  };
}

export interface OpenAIStreamChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: OpenAIChoice[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    prompt_tokens_details?: {
      cached_tokens?: number;
    };
  };
}

// ---- Config types ----

export interface AppConfig {
  openaiApiKey: string;
  openaiBaseUrl: string;
  anthropicApiKey?: string;
  azureApiVersion?: string;
  bigModel: string;
  middleModel: string;
  smallModel: string;
  maxTokensLimit: number;
  minTokensLimit: number;
  requestTimeout: number;
  logLevel: string;
  customHeaders: Record<string, string>;
  proxyMode: "openai" | "passthrough";
  enableModelMapping: boolean;
}

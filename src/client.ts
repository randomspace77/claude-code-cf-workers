import type {
  AppConfig,
  OpenAIRequest,
  OpenAIResponse,
  OpenAIStreamChunk,
} from "./types";

/**
 * Send a non-streaming chat completion request to the OpenAI-compatible API.
 */
export async function createChatCompletion(
  config: AppConfig,
  request: OpenAIRequest,
  signal?: AbortSignal,
  apiKey?: string,
): Promise<OpenAIResponse> {
  const url = buildUrl(config);
  const headers = buildHeaders(config, apiKey);

  const body = { ...request, stream: false };
  // Remove stream_options for non-streaming requests
  delete (body as Record<string, unknown>)["stream_options"];

  const response = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new OpenAIError(response.status, classifyOpenAIError(errorBody));
  }

  return (await response.json()) as OpenAIResponse;
}

/**
 * Send a streaming chat completion request and return a ReadableStream of
 * raw SSE lines (each prefixed with "data: ").
 */
export async function createChatCompletionStream(
  config: AppConfig,
  request: OpenAIRequest,
  signal?: AbortSignal,
  apiKey?: string,
): Promise<ReadableStream<string>> {
  const url = buildUrl(config);
  const headers = buildHeaders(config, apiKey);

  const body: OpenAIRequest = {
    ...request,
    stream: true,
    stream_options: { include_usage: true },
  };

  const response = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new OpenAIError(response.status, classifyOpenAIError(errorBody));
  }

  if (!response.body) {
    throw new OpenAIError(500, "No response body for streaming request");
  }

  // Transform the raw byte stream into decoded SSE lines
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  return new ReadableStream<string>({
    async pull(controller) {
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          // Flush remaining buffer
          if (buffer.trim()) {
            controller.enqueue(buffer.trim());
          }
          controller.close();
          return;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        // Keep the last (potentially incomplete) line in buffer
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed) {
            controller.enqueue(trimmed);
          }
        }
      }
    },
    cancel() {
      reader.cancel().catch(() => {});
    },
  });
}

// ---- Helpers ----

function buildUrl(config: AppConfig): string {
  // Remove trailing slashes from the base URL
  let base = config.openaiBaseUrl;
  while (base.endsWith("/")) {
    base = base.slice(0, -1);
  }
  if (config.azureApiVersion) {
    return `${base}/chat/completions?api-version=${encodeURIComponent(config.azureApiVersion)}`;
  }
  return `${base}/chat/completions`;
}

function buildHeaders(config: AppConfig, apiKey?: string): Record<string, string> {
  const effectiveKey = apiKey ?? config.openaiApiKey;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "User-Agent": "claude-code-proxy/1.0.0",
  };

  if (config.azureApiVersion) {
    headers["api-key"] = effectiveKey;
  } else {
    headers["Authorization"] = `Bearer ${effectiveKey}`;
  }

  // Merge custom headers
  for (const [key, value] of Object.entries(config.customHeaders)) {
    headers[key] = value;
  }

  return headers;
}

/**
 * Provide specific error guidance for common OpenAI API issues.
 */
export function classifyOpenAIError(errorDetail: string): string {
  const lower = errorDetail.toLowerCase();

  if (
    lower.includes("unsupported_country_region_territory") ||
    lower.includes("country, region, or territory not supported")
  ) {
    return "OpenAI API is not available in your region. Consider using Azure OpenAI service.";
  }
  if (lower.includes("invalid_api_key") || lower.includes("unauthorized")) {
    return "Invalid API key. Please check your OPENAI_API_KEY configuration.";
  }
  if (lower.includes("rate_limit") || lower.includes("quota")) {
    return "Rate limit exceeded. Please wait and try again, or upgrade your API plan.";
  }
  if (
    lower.includes("model") &&
    (lower.includes("not found") || lower.includes("does not exist"))
  ) {
    return "Model not found. Please check your BIG_MODEL and SMALL_MODEL configuration.";
  }
  if (lower.includes("billing") || lower.includes("payment")) {
    return "Billing issue. Please check your OpenAI account billing status.";
  }

  // Default: return a generic message (don't expose raw error details)
  return "An error occurred while communicating with the API provider.";
}

/**
 * Parse a raw SSE line into a typed chunk.
 * Returns null for non-data lines (comments, empty, etc.) and "[DONE]" markers.
 */
export function parseSSEChunk(line: string): OpenAIStreamChunk | null {
  if (!line.startsWith("data: ")) return null;
  const payload = line.slice(6).trim();
  if (payload === "[DONE]") return null;
  try {
    return JSON.parse(payload) as OpenAIStreamChunk;
  } catch {
    return null;
  }
}

// ---- Error class ----

export class OpenAIError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "OpenAIError";
  }
}

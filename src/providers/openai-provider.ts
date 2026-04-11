import type { AppConfig, ResolvedProvider, ClaudeMessagesRequest } from "../types";
import { convertClaudeToOpenAI } from "../conversion/request";
import { convertOpenAIToClaude, convertOpenAIStreamToClaude } from "../conversion/response";
import { mapModelForProvider } from "../router";

/**
 * Send a request to an OpenAI-compatible provider.
 * Handles: Claude→OpenAI conversion, API call, OpenAI→Claude response conversion.
 */
export async function sendOpenAIRequest(
  provider: ResolvedProvider,
  body: ClaudeMessagesRequest,
  apiKey: string,
  config: AppConfig,
): Promise<Response> {
  // Apply per-provider model mapping
  const mappedBody = { ...body, model: mapModelForProvider(provider, body.model) };

  // Convert Claude request → OpenAI request
  const openaiRequest = convertClaudeToOpenAI(mappedBody, config);

  // Build URL
  let base = provider.baseUrl;
  while (base.endsWith("/")) base = base.slice(0, -1);
  const url = provider.azureApiVersion
    ? `${base}/chat/completions?api-version=${encodeURIComponent(provider.azureApiVersion)}`
    : `${base}/chat/completions`;

  // Build headers
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "User-Agent": "claude-code-proxy-cf-workers/1.0.0",
  };
  if (provider.azureApiVersion) {
    headers["api-key"] = apiKey;
  } else {
    headers["Authorization"] = `Bearer ${apiKey}`;
  }
  // Merge provider-specific headers
  for (const [key, value] of Object.entries(provider.headers)) {
    headers[key] = value;
  }

  const signal = AbortSignal.timeout(provider.timeout * 1000);

  try {
    if (body.stream) {
      return await handleStreamingRequest(url, headers, openaiRequest as unknown as Record<string, unknown>, signal, body, config);
    } else {
      return await handleNonStreamingRequest(url, headers, openaiRequest as unknown as Record<string, unknown>, signal, body, config);
    }
  } catch (err) {
    console.error(`OpenAI provider "${provider.name}" error:`, err);
    const message = err instanceof Error ? err.message : String(err);
    const safeMessage = classifyError(message);
    return errorResponse(502, safeMessage);
  }
}

async function handleStreamingRequest(
  url: string,
  headers: Record<string, string>,
  openaiRequest: Record<string, unknown>,
  signal: AbortSignal,
  originalBody: ClaudeMessagesRequest,
  config: AppConfig,
): Promise<Response> {
  const requestBody = {
    ...openaiRequest,
    stream: true,
    stream_options: { include_usage: true },
  };

  const response = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(requestBody),
    signal,
  });

  if (!response.ok) {
    const errorBody = await response.text();
    return errorResponse(response.status, classifyError(errorBody));
  }

  if (!response.body) {
    return errorResponse(500, "No response body for streaming request");
  }

  // Transform raw byte stream into SSE lines
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const sseStream = new ReadableStream<string>({
    async pull(controller) {
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          if (buffer.trim()) controller.enqueue(buffer.trim());
          controller.close();
          return;
        }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed) controller.enqueue(trimmed);
        }
      }
    },
    cancel() {
      reader.cancel().catch(() => {});
    },
  });

  const claudeStream = convertOpenAIStreamToClaude(sseStream, originalBody, config.logLevel);

  const encoder = new TextEncoder();
  const byteStream = claudeStream.pipeThrough(
    new TransformStream<string, Uint8Array>({
      transform(chunk, controller) {
        controller.enqueue(encoder.encode(chunk));
      },
    }),
  );

  return new Response(byteStream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "Access-Control-Allow-Origin": "*",
    },
  });
}

async function handleNonStreamingRequest(
  url: string,
  headers: Record<string, string>,
  openaiRequest: Record<string, unknown>,
  signal: AbortSignal,
  originalBody: ClaudeMessagesRequest,
  config: AppConfig,
): Promise<Response> {
  const requestBody = { ...openaiRequest, stream: false };
  delete (requestBody as Record<string, unknown>)["stream_options"];

  const response = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(requestBody),
    signal,
  });

  if (!response.ok) {
    const errorBody = await response.text();
    return errorResponse(response.status, classifyError(errorBody));
  }

  const openaiResponse = await response.json();
  const claudeResponse = convertOpenAIToClaude(
    openaiResponse as import("../types").OpenAIResponse,
    originalBody,
    config.logLevel,
  );

  return Response.json(claudeResponse);
}

/**
 * Classify error messages for safe client display.
 * Preserves detailed error text so users can diagnose real issues.
 */
function classifyError(errorDetail: string): string {
  const lower = errorDetail.toLowerCase();

  // Only use generic message for obviously sensitive errors
  if (lower.includes("invalid_api_key") || lower.includes("unauthorized") || lower.includes("invalidauthenticationtoken")) {
    return "Invalid API key. Please check your provider API key configuration.";
  }
  if (lower.includes("rate_limit") || lower.includes("quota")) {
    return "Rate limit exceeded. Please wait and try again.";
  }
  if (lower.includes("billing") || lower.includes("payment")) {
    return "Billing issue. Please check your account billing status.";
  }

  // Return the actual error message for all other cases so users can diagnose
  return errorDetail.length > 200 ? errorDetail.slice(0, 200) + "..." : errorDetail;
}

function errorResponse(status: number, message: string): Response {
  return new Response(
    JSON.stringify({
      type: "error",
      error: { type: "api_error", message },
    }),
    { status, headers: { "Content-Type": "application/json" } },
  );
}

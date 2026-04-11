import type { AppConfig, ClaudeMessagesRequest, ClaudeTokenCountRequest, ClaudeSystemContent } from "./types";
import { loadConfig, validateClientApiKey, extractApiKey, mapModel, isPassthroughModel } from "./config";
import { convertClaudeToOpenAI } from "./conversion/request";
import {
  convertOpenAIToClaude,
  convertOpenAIStreamToClaude,
} from "./conversion/response";
import {
  createChatCompletion,
  createChatCompletionStream,
  classifyOpenAIError,
  OpenAIError,
} from "./client";

// ---- Handlers ----

/**
 * POST /v1/messages – Main proxy endpoint.
 * Routes to OpenAI conversion or Anthropic passthrough based on model name.
 * @param apiKey - The resolved effective API key (server key or client key)
 */
export async function handleMessages(
  request: Request,
  config: AppConfig,
  apiKey: string,
): Promise<Response> {
  // Reject oversized request bodies (10 MB limit)
  const contentLength = request.headers.get("content-length");
  if (contentLength && parseInt(contentLength, 10) > 10 * 1024 * 1024) {
    return errorResponse(413, "Request body too large (max 10 MB)");
  }

  // Read body once for routing decision
  const rawBody = await request.text();
  if (rawBody.length > 10 * 1024 * 1024) {
    return errorResponse(413, "Request body too large (max 10 MB)");
  }

  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(rawBody) as Record<string, unknown>;
  } catch {
    return errorResponse(400, "Invalid JSON in request body");
  }

  const model = typeof parsed.model === "string" ? parsed.model : "";

  // Log request details in DEBUG mode
  if (config.logLevel === "DEBUG") {
    // Full structured body for machine consumption
    console.log({ _tag: "request-body", body: parsed });
    // Human-readable prompt summary
    const messages = Array.isArray(parsed.messages) ? parsed.messages as Array<{ role?: string; content?: unknown }> : [];
    const lines = messages.map((m, i) => {
      const role = String(m.role ?? "unknown").toUpperCase();
      let text: string;
      if (typeof m.content === "string") {
        text = m.content;
      } else if (Array.isArray(m.content)) {
        text = (m.content as Array<Record<string, unknown>>)
          .map((b) => (typeof b.text === "string" ? b.text : `[${b.type ?? "block"}]`))
          .join(" ");
      } else {
        text = "[non-text]";
      }
      return `[${i}] ${role}: ${text}`;
    });
    const system = parsed.system
      ? typeof parsed.system === "string"
        ? parsed.system
        : JSON.stringify(parsed.system)
      : "(none)";
    console.log(
      `\n===== REQUEST PROMPT =====\n` +
      `Model: ${model} | Stream: ${!!parsed.stream} | MaxTokens: ${parsed.max_tokens ?? "default"}\n` +
      `System: ${system}\n` +
      `--- Messages (${messages.length}) ---\n` +
      lines.join("\n") +
      `\n=========================\n`
    );
  }

  if (isPassthroughModel(config, model)) {
    return handleMessagesPassthrough(rawBody, parsed, config, apiKey);
  }

  return handleMessagesOpenAI(parsed, config, apiKey);
}

/**
 * Passthrough mode: forward the Anthropic-format request directly to the backend.
 */
async function handleMessagesPassthrough(
  rawBody: string,
  parsed: Record<string, unknown>,
  config: AppConfig,
  apiKey: string,
): Promise<Response> {
  try {
    // Optionally apply model mapping
    let forwardBody = rawBody;
    if (config.enableModelMapping && parsed.model) {
      parsed.model = mapModel(config, parsed.model as string);
      forwardBody = JSON.stringify(parsed);
    }

    // Build target URL
    let base = config.openaiBaseUrl;
    while (base.endsWith("/")) base = base.slice(0, -1);
    const url = `${base}/messages`;

    // Build headers for Anthropic-compatible API
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "User-Agent": "claude-code-proxy-cf-workers/1.0.0",
    };

    // Merge custom headers (can override defaults)
    for (const [key, value] of Object.entries(config.customHeaders)) {
      headers[key] = value;
    }

    const response = await fetch(url, {
      method: "POST",
      headers,
      body: forwardBody,
      signal: AbortSignal.timeout(config.requestTimeout * 1000),
    });

    // Forward response headers
    const responseHeaders: Record<string, string> = {
      "Access-Control-Allow-Origin": "*",
    };
    const contentType = response.headers.get("Content-Type");
    if (contentType) {
      responseHeaders["Content-Type"] = contentType;
    }

    return new Response(response.body, {
      status: response.status,
      headers: responseHeaders,
    });
  } catch (err) {
    console.error("Passthrough error:", err);
    const message = err instanceof Error ? err.message : String(err);
    return errorResponse(502, `Passthrough request failed: ${message}`);
  }
}

/**
 * OpenAI conversion mode: convert Claude request to OpenAI format and back.
 */
async function handleMessagesOpenAI(
  parsed: Record<string, unknown>,
  config: AppConfig,
  apiKey: string,
): Promise<Response> {

  const body = parsed as unknown as ClaudeMessagesRequest;

  // Basic validation
  if (!body.model || !body.messages || !Array.isArray(body.messages)) {
    return errorResponse(400, "Missing required fields: model, messages");
  }

  try {
    const openaiRequest = convertClaudeToOpenAI(body, config);

    if (body.stream) {
      // Streaming
      const signal = AbortSignal.timeout(config.requestTimeout * 1000);
      const openaiStream = await createChatCompletionStream(
        config,
        openaiRequest,
        signal,
        apiKey,
      );
      const claudeStream = convertOpenAIStreamToClaude(openaiStream, body, config.logLevel);

      // Encode string chunks into bytes for the response
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
    } else {
      // Non-streaming
      const signal = AbortSignal.timeout(config.requestTimeout * 1000);
      const openaiResponse = await createChatCompletion(config, openaiRequest, signal, apiKey);
      const claudeResponse = convertOpenAIToClaude(openaiResponse, body, config.logLevel);
      return Response.json(claudeResponse);
    }
  } catch (err) {
    if (err instanceof OpenAIError) {
      return errorResponse(err.status, err.message);
    }
    // Log internal errors but don't expose raw details to client
    console.error("Unexpected error processing request:", err);
    const message =
      err instanceof Error ? err.message : String(err);
    const classified = classifyOpenAIError(message);
    // If classifyOpenAIError returned the raw message, replace with generic
    const safeMessage = classified === message
      ? "An unexpected error occurred while processing the request"
      : classified;
    return errorResponse(500, safeMessage);
  }
}

/**
 * POST /v1/messages/count_tokens – Token counting endpoint.
 */
export async function handleCountTokens(
  request: Request,
): Promise<Response> {
  let body: ClaudeTokenCountRequest;
  try {
    body = (await request.json()) as ClaudeTokenCountRequest;
  } catch {
    return errorResponse(400, "Invalid JSON in request body");
  }

  let totalChars = 0;

  // System message
  if (body.system) {
    if (typeof body.system === "string") {
      totalChars += body.system.length;
    } else if (Array.isArray(body.system)) {
      for (const block of body.system as ClaudeSystemContent[]) {
        if (block.text) totalChars += block.text.length;
      }
    }
  }

  // Messages
  for (const msg of body.messages ?? []) {
    if (msg.content === null || msg.content === undefined) continue;
    if (typeof msg.content === "string") {
      totalChars += msg.content.length;
    } else if (Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if ("text" in block && typeof block.text === "string") {
          totalChars += block.text.length;
        }
      }
    }
  }

  // Rough estimation: ~4 characters per token
  const estimatedTokens = Math.max(1, Math.floor(totalChars / 4));

  return Response.json({ input_tokens: estimatedTokens });
}

/**
 * GET /health – Health check endpoint.
 */
export function handleHealth(config: AppConfig): Response {
  return Response.json({
    status: "healthy",
    timestamp: new Date().toISOString(),
  });
}

/**
 * GET / – Root endpoint.
 * Only exposes non-sensitive service info; internal config is hidden.
 */
export function handleRoot(): Response {
  return Response.json({
    message: "Claude-to-OpenAI API Proxy (CF Workers) v1.0.0",
    status: "running",
    endpoints: {
      messages: "/v1/messages",
      count_tokens: "/v1/messages/count_tokens",
      health: "/health",
    },
  });
}

// ---- Auth middleware ----

/**
 * Validate the client API key and resolve the effective API key for backend calls.
 * Returns the effective API key string on success, or a Response (error) if invalid.
 *
 * Key resolution:
 * - If OPENAI_API_KEY is configured (managed mode): use server key
 * - Otherwise (passthrough mode): use client-provided key
 * - If ANTHROPIC_API_KEY is configured: additionally validate client key against it
 */
export function authenticate(
  request: Request,
  config: AppConfig,
): Response | string {
  const clientKey = extractApiKey(request.headers);

  // If ANTHROPIC_API_KEY is configured, validate client key against it
  if (!validateClientApiKey(config, clientKey)) {
    return new Response(
      JSON.stringify({
        type: "error",
        error: {
          type: "authentication_error",
          message:
            "Invalid API key. Please provide a valid Anthropic API key.",
        },
      }),
      {
        status: 401,
        headers: { "Content-Type": "application/json" },
      },
    );
  }

  // Resolve effective API key: server key takes priority, then client key
  const effectiveKey = config.openaiApiKey || clientKey;
  if (!effectiveKey) {
    return new Response(
      JSON.stringify({
        type: "error",
        error: {
          type: "authentication_error",
          message:
            "API key is required. Provide your API key via x-api-key header or Authorization: Bearer header.",
        },
      }),
      {
        status: 401,
        headers: { "Content-Type": "application/json" },
      },
    );
  }

  return effectiveKey;
}

// ---- Helpers ----

function errorResponse(
  status: number,
  message: string,
): Response {
  return new Response(
    JSON.stringify({
      type: "error",
      error: { type: "api_error", message },
    }),
    {
      status,
      headers: { "Content-Type": "application/json" },
    },
  );
}

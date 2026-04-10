import type { AppConfig, ClaudeMessagesRequest, ClaudeTokenCountRequest, ClaudeSystemContent } from "./types";
import { loadConfig, validateClientApiKey, extractApiKey, mapModel } from "./config";
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
 * Routes to OpenAI conversion or Anthropic passthrough based on PROXY_MODE.
 * @param apiKey - The resolved effective API key (server key or client key)
 */
export async function handleMessages(
  request: Request,
  config: AppConfig,
  apiKey: string,
): Promise<Response> {
  if (config.proxyMode === "passthrough") {
    return handleMessagesPassthrough(request, config, apiKey);
  }

  return handleMessagesOpenAI(request, config, apiKey);
}

/**
 * Passthrough mode: forward the Anthropic-format request directly to the backend.
 */
async function handleMessagesPassthrough(
  request: Request,
  config: AppConfig,
  apiKey: string,
): Promise<Response> {
  try {
    // Read raw body
    const rawBody = await request.text();

    // Optionally apply model mapping
    let forwardBody = rawBody;
    if (config.enableModelMapping) {
      try {
        const parsed = JSON.parse(rawBody);
        if (parsed.model) {
          parsed.model = mapModel(config, parsed.model);
          forwardBody = JSON.stringify(parsed);
        }
      } catch {
        // If parsing fails, forward as-is
      }
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
      "User-Agent": "claude-code-proxy/1.0.0",
    };

    // Merge custom headers (can override defaults)
    for (const [key, value] of Object.entries(config.customHeaders)) {
      headers[key] = value;
    }

    const response = await fetch(url, {
      method: "POST",
      headers,
      body: forwardBody,
    });

    // Forward response headers
    const responseHeaders: Record<string, string> = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "*",
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
  request: Request,
  config: AppConfig,
  apiKey: string,
): Promise<Response> {

  // Parse body
  let body: ClaudeMessagesRequest;
  try {
    body = (await request.json()) as ClaudeMessagesRequest;
  } catch {
    return errorResponse(400, "Invalid JSON in request body");
  }

  // Basic validation
  if (!body.model || !body.messages || !Array.isArray(body.messages)) {
    return errorResponse(400, "Missing required fields: model, messages");
  }

  try {
    const openaiRequest = convertClaudeToOpenAI(body, config);

    if (body.stream) {
      // Streaming
      const openaiStream = await createChatCompletionStream(
        config,
        openaiRequest,
        undefined,
        apiKey,
      );
      const claudeStream = convertOpenAIStreamToClaude(openaiStream, body);

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
          "Access-Control-Allow-Headers": "*",
        },
      });
    } else {
      // Non-streaming
      const openaiResponse = await createChatCompletion(config, openaiRequest, undefined, apiKey);
      const claudeResponse = convertOpenAIToClaude(openaiResponse, body);
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
    key_mode: config.openaiApiKey ? "managed" : "passthrough",
    client_api_key_validation: Boolean(config.anthropicApiKey),
  });
}

/**
 * GET / – Root endpoint.
 */
export function handleRoot(config: AppConfig): Response {
  return Response.json({
    message: "Claude-to-OpenAI API Proxy v1.0.0",
    status: "running",
    config: {
      proxy_mode: config.proxyMode,
      key_mode: config.openaiApiKey ? "managed" : "passthrough",
      openai_base_url: config.openaiBaseUrl,
      max_tokens_limit: config.maxTokensLimit,
      client_api_key_validation: Boolean(config.anthropicApiKey),
      enable_model_mapping: config.enableModelMapping,
      big_model: config.bigModel,
      middle_model: config.middleModel,
      small_model: config.smallModel,
    },
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

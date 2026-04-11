import type { AppConfig, ClaudeMessagesRequest, ClaudeTokenCountRequest, ClaudeSystemContent } from "./types";
import { resolveProvider } from "./router";
import { dispatchToProvider } from "./providers";

// ---- Handlers ----

/**
 * POST /v1/messages – Main proxy endpoint.
 * Routes to the appropriate provider based on model name.
 */
export async function handleMessages(
  request: Request,
  config: AppConfig,
  clientApiKey: string,
): Promise<Response> {
  // Reject oversized request bodies (10 MB limit)
  const contentLength = request.headers.get("content-length");
  if (contentLength && parseInt(contentLength, 10) > 10 * 1024 * 1024) {
    return errorResponse(413, "Request body too large (max 10 MB)");
  }

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
    console.log({ _tag: "request-body", body: parsed });
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
    console.log({
      _tag: "request-prompt",
      model,
      stream: !!parsed.stream,
      maxTokens: parsed.max_tokens ?? "default",
      system: parsed.system
        ? typeof parsed.system === "string"
          ? parsed.system
          : JSON.stringify(parsed.system)
        : "(none)",
      messageCount: messages.length,
      messages: lines,
    });
  }

  const body = parsed as unknown as ClaudeMessagesRequest;

  // Basic validation
  if (!body.model || !body.messages || !Array.isArray(body.messages)) {
    return errorResponse(400, "Missing required fields: model, messages");
  }

  // Resolve provider via router
  const provider = resolveProvider(
    model,
    config.routing,
    config.defaultProvider,
    config.providers,
  );

  if (!provider) {
    return errorResponse(400, `No provider configured for model "${model}"`);
  }

  if (config.logLevel === "DEBUG") {
    console.log({
      _tag: "provider-resolved",
      model,
      provider: provider.name,
      protocol: provider.protocol,
      baseUrl: provider.baseUrl,
    });
  }

  return dispatchToProvider(provider, body, clientApiKey, config);
}

/**
 * POST /v1/messages/count_tokens – Token counting endpoint.
 */
export async function handleCountTokens(
  request: Request,
): Promise<Response> {
  // Reject oversized request bodies (10 MB limit)
  const contentLength = request.headers.get("content-length");
  if (contentLength && parseInt(contentLength, 10) > 10 * 1024 * 1024) {
    return errorResponse(413, "Request body too large (max 10 MB)");
  }

  let body: ClaudeTokenCountRequest;
  try {
    body = (await request.json()) as ClaudeTokenCountRequest;
  } catch {
    return errorResponse(400, "Invalid JSON in request body");
  }

  let totalChars = 0;

  if (body.system) {
    if (typeof body.system === "string") {
      totalChars += body.system.length;
    } else if (Array.isArray(body.system)) {
      for (const block of body.system as ClaudeSystemContent[]) {
        if (block.text) totalChars += block.text.length;
      }
    }
  }

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

// ---- Helpers ----

function errorResponse(status: number, message: string): Response {
  return new Response(
    JSON.stringify({
      type: "error",
      error: { type: "api_error", message },
    }),
    { status, headers: { "Content-Type": "application/json" } },
  );
}

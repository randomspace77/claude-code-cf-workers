import type { AppConfig, ResolvedProvider, ClaudeMessagesRequest } from "../types";
import { mapModelForProvider } from "../router";

/**
 * Send a request to an Anthropic-compatible provider (passthrough mode).
 * Forwards the Claude-format request directly, with optional model mapping.
 */
export async function sendAnthropicRequest(
  provider: ResolvedProvider,
  body: ClaudeMessagesRequest,
  apiKey: string,
  config: AppConfig,
): Promise<Response> {
  try {
    // Apply per-provider model mapping if configured
    const mappedModel = mapModelForProvider(provider, body.model);
    const forwardBody = mappedModel !== body.model
      ? JSON.stringify({ ...body, model: mappedModel })
      : JSON.stringify(body);

    // Build target URL
    let base = provider.baseUrl;
    while (base.endsWith("/")) base = base.slice(0, -1);
    const url = `${base}/messages`;

    // Build headers for Anthropic-compatible API
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "User-Agent": "claude-code-proxy-cf-workers/1.0.0",
    };

    // Merge provider-specific headers
    for (const [key, value] of Object.entries(provider.headers)) {
      headers[key] = value;
    }

    const response = await fetch(url, {
      method: "POST",
      headers,
      body: forwardBody,
      signal: AbortSignal.timeout(provider.timeout * 1000),
    });

    // Forward response
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
    console.error(`Anthropic provider "${provider.name}" error:`, err);
    const raw = err instanceof Error ? err.message : String(err);
    const safeMessage = classifyAnthropicError(raw);
    return new Response(
      JSON.stringify({
        type: "error",
        error: { type: "api_error", message: safeMessage },
      }),
      { status: 502, headers: { "Content-Type": "application/json" } },
    );
  }
}

/**
 * Classify error messages for safe client display (no raw details exposed).
 */
function classifyAnthropicError(errorDetail: string): string {
  const lower = errorDetail.toLowerCase();

  if (lower.includes("unauthorized") || (lower.includes("invalid") && lower.includes("key"))) {
    return "Invalid API key. Please check your provider API key configuration.";
  }
  if (lower.includes("rate_limit") || lower.includes("rate limit") || lower.includes("quota")) {
    return "Rate limit exceeded. Please wait and try again.";
  }
  if (lower.includes("model") && (lower.includes("not found") || lower.includes("does not exist"))) {
    return "Model not found. Please check your model configuration.";
  }
  if (lower.includes("timeout") || lower.includes("timed out")) {
    return "Request timed out. The provider did not respond in time.";
  }
  if (lower.includes("billing") || lower.includes("payment")) {
    return "Billing issue. Please check your account billing status.";
  }

  return "An error occurred while communicating with the API provider.";
}

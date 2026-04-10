import type { Env, AppConfig } from "./types";

/**
 * Parse environment bindings into a typed application config.
 * Returns config even if OPENAI_API_KEY is missing (validated at request time).
 */
export function loadConfig(env: Env): AppConfig {
  const bigModel = env.BIG_MODEL || "gpt-4o";
  const customHeaders = parseCustomHeaders(env.CUSTOM_HEADERS);
  const proxyMode = env.PROXY_MODE === "passthrough" ? "passthrough" : "openai";
  const enableModelMapping = env.ENABLE_MODEL_MAPPING === "true";

  return {
    openaiApiKey: env.OPENAI_API_KEY || "",
    openaiBaseUrl: env.OPENAI_BASE_URL || "https://api.openai.com/v1",
    anthropicApiKey: env.ANTHROPIC_API_KEY,
    azureApiVersion: env.AZURE_API_VERSION,
    bigModel,
    middleModel: env.MIDDLE_MODEL || bigModel,
    smallModel: env.SMALL_MODEL || "gpt-4o-mini",
    maxTokensLimit: parseInt(env.MAX_TOKENS_LIMIT || "16384", 10),
    minTokensLimit: parseInt(env.MIN_TOKENS_LIMIT || "4096", 10),
    requestTimeout: parseInt(env.REQUEST_TIMEOUT || "90", 10),
    logLevel: env.LOG_LEVEL || "WARNING",
    customHeaders,
    proxyMode,
    enableModelMapping,
  };
}

/**
 * Parse custom headers from a JSON string.
 * Format: `{"Header-Name": "value", ...}`
 */
function parseCustomHeaders(
  raw: string | undefined,
): Record<string, string> {
  if (!raw) return {};
  try {
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
      const result: Record<string, string> = {};
      for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
        if (typeof value === "string") {
          result[key] = value;
        }
      }
      return result;
    }
  } catch {
    // ignore parse errors
  }
  return {};
}

/**
 * Validate a client-provided API key against the configured ANTHROPIC_API_KEY.
 * Returns true if validation passes (or is not configured).
 */
export function validateClientApiKey(
  config: AppConfig,
  clientApiKey: string | null,
): boolean {
  // If no ANTHROPIC_API_KEY configured, skip validation
  if (!config.anthropicApiKey) return true;
  if (!clientApiKey) return false;
  // Constant-time comparison to prevent timing attacks
  return timingSafeEqual(clientApiKey, config.anthropicApiKey);
}

/**
 * Constant-time string comparison to prevent timing attacks.
 */
function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return result === 0;
}

/**
 * Extract API key from request headers.
 */
export function extractApiKey(headers: Headers): string | null {
  const xApiKey = headers.get("x-api-key");
  if (xApiKey) return xApiKey;

  const authorization = headers.get("authorization");
  if (authorization?.startsWith("Bearer ")) {
    return authorization.slice(7);
  }

  return null;
}

/**
 * Map a Claude model name to the configured OpenAI model.
 * When enableModelMapping is false (default), returns the model name as-is.
 */
export function mapModel(config: AppConfig, claudeModel: string): string {
  // When model mapping is disabled, forward model-id as-is
  if (!config.enableModelMapping) {
    return claudeModel;
  }

  // If it's already an OpenAI / known provider model, return as-is
  if (
    claudeModel.startsWith("gpt-") ||
    claudeModel.startsWith("o1-") ||
    claudeModel.startsWith("o3-") ||
    claudeModel.startsWith("o4-") ||
    claudeModel.startsWith("ep-") ||
    claudeModel.startsWith("doubao-") ||
    claudeModel.startsWith("deepseek-") ||
    claudeModel.startsWith("glm-") ||
    claudeModel.startsWith("qwen-") ||
    claudeModel.startsWith("gemini-")
  ) {
    return claudeModel;
  }

  const lower = claudeModel.toLowerCase();
  if (lower.includes("haiku")) return config.smallModel;
  if (lower.includes("sonnet")) return config.middleModel;
  if (lower.includes("opus")) return config.bigModel;

  // Default to big model for unknown models
  return config.bigModel;
}

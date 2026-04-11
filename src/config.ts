import type { Env, AppConfig, ResolvedProvider, ProvidersJsonConfig, ProviderUserConfig } from "./types";
import { getKnownProvider } from "./providers/registry";

/**
 * Parse environment bindings into a typed application config.
 *
 * Two modes:
 * 1. Multi-provider mode: `PROVIDERS` JSON is set → parse it and resolve providers
 * 2. Legacy mode: no `PROVIDERS` → auto-generate single-provider config from legacy vars
 */
export function loadConfig(env: Env): AppConfig {
  const globalTimeout = parseInt(env.REQUEST_TIMEOUT || "90", 10);
  const logLevel = env.LOG_LEVEL || "WARNING";
  const maxTokensLimit = parseInt(env.MAX_TOKENS_LIMIT || "16384", 10);
  const minTokensLimit = parseInt(env.MIN_TOKENS_LIMIT || "4096", 10);
  const anthropicApiKey = env.ANTHROPIC_API_KEY;

  // Legacy single-provider fields (always parsed for backward compat)
  const bigModel = env.BIG_MODEL || "gpt-4o";
  const customHeaders = parseCustomHeaders(env.CUSTOM_HEADERS);
  const passthroughModels = parsePassthroughModels(env.PASSTHROUGH_MODELS);
  const enableModelMapping = env.ENABLE_MODEL_MAPPING === "true";

  if (env.PROVIDERS) {
    // --- Multi-provider mode ---
    const parsed = parseProvidersJson(env.PROVIDERS);
    const providers = resolveProviders(parsed, env, globalTimeout);
    return {
      anthropicApiKey,
      logLevel,
      requestTimeout: globalTimeout,
      maxTokensLimit,
      minTokensLimit,
      defaultProvider: parsed.default,
      routing: parsed.routing ?? {},
      providers,
      // Legacy fields (not used in multi-provider mode, but kept for type compat)
      openaiApiKey: env.OPENAI_API_KEY || "",
      openaiBaseUrl: env.OPENAI_BASE_URL || "https://api.openai.com/v1",
      azureApiVersion: env.AZURE_API_VERSION,
      bigModel,
      middleModel: env.MIDDLE_MODEL || bigModel,
      smallModel: env.SMALL_MODEL || "gpt-4o-mini",
      customHeaders,
      passthroughModels,
      enableModelMapping,
    };
  }

  // --- Legacy single-provider mode ---
  const legacyProvider = buildLegacyProvider(env, globalTimeout, customHeaders);
  const providers: Record<string, ResolvedProvider> = { default: legacyProvider };

  // Build routing from PASSTHROUGH_MODELS
  const routing: Record<string, string> = {};
  if (passthroughModels.length > 0) {
    // Create a passthrough provider with anthropic protocol
    const passthroughProvider: ResolvedProvider = {
      ...legacyProvider,
      name: "passthrough",
      protocol: "anthropic",
    };
    providers["passthrough"] = passthroughProvider;
    for (const prefix of passthroughModels) {
      routing[`${prefix}*`] = "passthrough";
    }
  }

  return {
    anthropicApiKey,
    logLevel,
    requestTimeout: globalTimeout,
    maxTokensLimit,
    minTokensLimit,
    defaultProvider: "default",
    routing,
    providers,
    // Legacy fields
    openaiApiKey: env.OPENAI_API_KEY || "",
    openaiBaseUrl: env.OPENAI_BASE_URL || "https://api.openai.com/v1",
    azureApiVersion: env.AZURE_API_VERSION,
    bigModel,
    middleModel: env.MIDDLE_MODEL || bigModel,
    smallModel: env.SMALL_MODEL || "gpt-4o-mini",
    customHeaders,
    passthroughModels,
    enableModelMapping,
  };
}

// ---- Multi-provider helpers ----

function parseProvidersJson(raw: string): ProvidersJsonConfig {
  try {
    const parsed = JSON.parse(raw) as ProvidersJsonConfig;
    if (!parsed.default || typeof parsed.default !== "string") {
      throw new Error("PROVIDERS JSON must have a 'default' field");
    }
    return parsed;
  } catch (err) {
    if (err instanceof SyntaxError) {
      throw new Error(`Invalid PROVIDERS JSON: ${err.message}`);
    }
    throw err;
  }
}

/**
 * Resolve all providers from the PROVIDERS JSON config.
 * Merges known provider defaults with user overrides, and resolves API keys.
 */
function resolveProviders(
  config: ProvidersJsonConfig,
  env: Env,
  globalTimeout: number,
): Record<string, ResolvedProvider> {
  const result: Record<string, ResolvedProvider> = {};
  const providerConfigs = config.providers ?? {};

  // Ensure the default provider exists in the config (even if not explicitly listed)
  if (!providerConfigs[config.default]) {
    providerConfigs[config.default] = {};
  }

  // Also ensure all providers referenced in routing exist
  if (config.routing) {
    for (const providerName of Object.values(config.routing)) {
      if (!providerConfigs[providerName]) {
        providerConfigs[providerName] = {};
      }
    }
  }

  for (const [name, userConfig] of Object.entries(providerConfigs)) {
    const resolved = resolveOneProvider(name, userConfig, env, globalTimeout);
    if (resolved) {
      result[name] = resolved;
    }
  }

  return result;
}

function resolveOneProvider(
  name: string,
  userConfig: ProviderUserConfig,
  env: Env,
  globalTimeout: number,
): ResolvedProvider | null {
  const known = getKnownProvider(name);

  // Determine base URL: user override > known provider > null
  const baseUrl = userConfig.baseUrl || known?.baseUrl;
  if (!baseUrl) {
    console.warn(`Provider "${name}" has no baseUrl and is not a known provider. Skipping.`);
    return null;
  }

  // Determine protocol: user override > known provider > default "openai"
  const protocol = userConfig.protocol || known?.protocol || "openai";

  // Resolve API key: PROVIDER_<NAME>_API_KEY env var
  const envKeyName = `PROVIDER_${name.toUpperCase().replace(/-/g, "_")}_API_KEY`;
  const apiKey = env[envKeyName] || "";

  // Merge headers: known defaults + user overrides
  const headers: Record<string, string> = {
    ...(known?.defaultHeaders ?? {}),
    ...(userConfig.headers ?? {}),
  };

  // Blocklist: prevent overriding sensitive headers
  const blocklist = new Set(["authorization", "api-key", "host", "content-type"]);
  for (const key of Object.keys(headers)) {
    if (blocklist.has(key.toLowerCase())) {
      delete headers[key];
    }
  }

  return {
    name,
    baseUrl,
    protocol,
    apiKey,
    timeout: userConfig.timeout ?? globalTimeout,
    headers,
    azureApiVersion: userConfig.azureApiVersion,
    modelMapping: userConfig.modelMapping,
  };
}

// ---- Legacy helpers ----

function buildLegacyProvider(
  env: Env,
  timeout: number,
  customHeaders: Record<string, string>,
): ResolvedProvider {
  const bigModel = env.BIG_MODEL || "gpt-4o";
  const enableModelMapping = env.ENABLE_MODEL_MAPPING === "true";

  let modelMapping: Record<string, string> | undefined;
  if (enableModelMapping) {
    modelMapping = {
      opus: bigModel,
      sonnet: env.MIDDLE_MODEL || bigModel,
      haiku: env.SMALL_MODEL || "gpt-4o-mini",
    };
  }

  return {
    name: "default",
    baseUrl: env.OPENAI_BASE_URL || "https://api.openai.com/v1",
    protocol: "openai",
    apiKey: env.OPENAI_API_KEY || "",
    timeout,
    headers: { ...customHeaders },
    azureApiVersion: env.AZURE_API_VERSION,
    modelMapping,
  };
}

/**
 * Parse custom headers from a JSON string.
 */
function parseCustomHeaders(raw: string | undefined): Record<string, string> {
  if (!raw) return {};
  const blocklist = new Set(["authorization", "api-key", "host", "content-type"]);
  try {
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
      const result: Record<string, string> = {};
      for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
        if (typeof value === "string" && !blocklist.has(key.toLowerCase())) {
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
 * Parse comma-separated passthrough model prefixes.
 */
function parsePassthroughModels(raw: string | undefined): string[] {
  if (!raw) return [];
  return raw
    .split(",")
    .map((s) => s.trim().toLowerCase())
    .filter((s) => s.length > 0);
}

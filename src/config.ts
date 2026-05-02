import type { Env, AppConfig, ResolvedProvider, ProvidersJsonConfig, ProviderUserConfig, ReasoningCacheNamespace } from "./types";
import { getKnownProvider } from "./providers/registry";

/**
 * Parse environment bindings into a typed application config.
 *
 * Two modes:
 * 1. Multi-provider mode: `PROVIDERS` JSON is set → parse it and resolve providers
 * 2. Legacy mode: no `PROVIDERS` → auto-generate single-provider config from legacy vars
 */
export function loadConfig(env: Env): AppConfig {
  const globalTimeout = parseInt(envString(env.REQUEST_TIMEOUT) || "90", 10);
  const logLevel = envString(env.LOG_LEVEL) || "WARNING";
  const maxTokensLimit = parseInt(envString(env.MAX_TOKENS_LIMIT) || "16384", 10);
  const minTokensLimit = parseInt(envString(env.MIN_TOKENS_LIMIT) || "4096", 10);
  const anthropicApiKey = envString(env.ANTHROPIC_API_KEY);
  const reasoningCache = isKVNamespace(env.REASONING_CACHE) ? env.REASONING_CACHE : undefined;
  const reasoningCacheTtlSeconds = parseInt(
    envString(env.REASONING_CACHE_TTL_SECONDS) || "2592000",
    10,
  );

  // Legacy single-provider fields (always parsed for backward compat)
  const bigModel = envString(env.BIG_MODEL) || "gpt-4o";
  const customHeaders = parseCustomHeaders(envString(env.CUSTOM_HEADERS));
  const passthroughModels = parsePassthroughModels(envString(env.PASSTHROUGH_MODELS));
  const enableModelMapping = envString(env.ENABLE_MODEL_MAPPING) === "true";

  const providersJson = envString(env.PROVIDERS);
  if (providersJson) {
    // --- Multi-provider mode ---
    const parsed = parseProvidersJson(providersJson);
    const providers = resolveProviders(parsed, env, globalTimeout);
    console.log({
      _tag: "config-mode",
      mode: "multi-provider",
      default: parsed.default,
      routing: Object.keys(parsed.routing ?? {}),
      providers: Object.keys(providers),
    });
    return {
      anthropicApiKey,
      logLevel,
      requestTimeout: globalTimeout,
      maxTokensLimit,
      minTokensLimit,
      reasoningCache,
      reasoningCacheTtlSeconds,
      defaultProvider: parsed.default,
      routing: parsed.routing ?? {},
      providers,
      // Legacy fields (not used in multi-provider mode, but kept for type compat)
      openaiApiKey: envString(env.OPENAI_API_KEY) || "",
      openaiBaseUrl: envString(env.OPENAI_BASE_URL) || "https://api.openai.com/v1",
      azureApiVersion: envString(env.AZURE_API_VERSION),
      bigModel,
      middleModel: envString(env.MIDDLE_MODEL) || bigModel,
      smallModel: envString(env.SMALL_MODEL) || "gpt-4o-mini",
      customHeaders,
      passthroughModels,
      enableModelMapping,
    };
  }

  // --- Legacy single-provider mode ---
  console.warn({
    _tag: "config-mode",
    mode: "legacy",
    message: "PROVIDERS env var not set — using legacy single-provider mode",
    baseUrl: envString(env.OPENAI_BASE_URL) || "https://api.openai.com/v1",
  });
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
    reasoningCache,
    reasoningCacheTtlSeconds,
    defaultProvider: "default",
    routing,
    providers,
    // Legacy fields
    openaiApiKey: envString(env.OPENAI_API_KEY) || "",
    openaiBaseUrl: envString(env.OPENAI_BASE_URL) || "https://api.openai.com/v1",
    azureApiVersion: envString(env.AZURE_API_VERSION),
    bigModel,
    middleModel: envString(env.MIDDLE_MODEL) || bigModel,
    smallModel: envString(env.SMALL_MODEL) || "gpt-4o-mini",
    customHeaders,
    passthroughModels,
    enableModelMapping,
  };
}

// ---- Multi-provider helpers ----

function parseProvidersJson(raw: string): ProvidersJsonConfig {
  try {
    // Strip BOM and surrounding whitespace (common when pasting into Dashboard)
    const cleaned = raw.replace(/^\uFEFF/, "").trim();
    const parsed = JSON.parse(cleaned) as ProvidersJsonConfig;
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
  const apiKey = envString(env[envKeyName]) || "";

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
  const bigModel = envString(env.BIG_MODEL) || "gpt-4o";
  const enableModelMapping = envString(env.ENABLE_MODEL_MAPPING) === "true";

  let modelMapping: Record<string, string> | undefined;
  if (enableModelMapping) {
    modelMapping = {
      opus: bigModel,
      sonnet: envString(env.MIDDLE_MODEL) || bigModel,
      haiku: envString(env.SMALL_MODEL) || "gpt-4o-mini",
    };
  }

  return {
    name: "default",
    baseUrl: envString(env.OPENAI_BASE_URL) || "https://api.openai.com/v1",
    protocol: "openai",
    apiKey: envString(env.OPENAI_API_KEY) || "",
    timeout,
    headers: { ...customHeaders },
    azureApiVersion: envString(env.AZURE_API_VERSION),
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

function envString(value: string | ReasoningCacheNamespace | undefined): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function isKVNamespace(value: unknown): value is ReasoningCacheNamespace {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as ReasoningCacheNamespace).get === "function" &&
    typeof (value as ReasoningCacheNamespace).put === "function"
  );
}

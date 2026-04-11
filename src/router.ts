import type { ResolvedProvider } from "./types";

/**
 * Resolve which provider should handle a given model name.
 *
 * Resolution order:
 * 1. Check explicit routing rules (glob patterns, first match wins)
 * 2. Fall back to the default provider
 *
 * @param model - The model name from the request (e.g. "glm-5.1", "gpt-4o")
 * @param routing - Map of glob patterns to provider names
 * @param defaultProviderName - Name of the default provider
 * @param providers - Map of provider names to resolved configs
 * @returns The resolved provider, or undefined if no match (shouldn't happen if default is valid)
 */
export function resolveProvider(
  model: string,
  routing: Record<string, string>,
  defaultProviderName: string,
  providers: Record<string, ResolvedProvider>,
): ResolvedProvider | undefined {
  const lowerModel = model.toLowerCase();

  // Check explicit routing rules
  for (const [pattern, providerName] of Object.entries(routing)) {
    if (globMatch(lowerModel, pattern.toLowerCase())) {
      const provider = providers[providerName];
      if (provider) return provider;
    }
  }

  // Fall back to default provider
  return providers[defaultProviderName];
}

/**
 * Simple glob pattern matching.
 * Supports `*` (match any characters) and `?` (match single character).
 * Case-insensitive (caller should lowercase both inputs).
 *
 * Examples:
 *   globMatch("glm-5.1", "glm-*") → true
 *   globMatch("gpt-4o", "gpt-*") → true
 *   globMatch("claude-3.5-sonnet", "claude-*") → true
 *   globMatch("deepseek-chat", "deep*") → true
 *   globMatch("gpt-4o", "glm-*") → false
 */
export function globMatch(text: string, pattern: string): boolean {
  // Convert glob to regex: escape special regex chars, then replace glob wildcards
  let regexStr = "";
  let prevStar = false;
  for (const ch of pattern) {
    if (ch === "*") {
      // Collapse consecutive * into a single .* to prevent ReDoS
      if (!prevStar) {
        regexStr += ".*";
        prevStar = true;
      }
    } else if (ch === "?") {
      regexStr += ".";
      prevStar = false;
    } else if (".+^${}()|[]\\".includes(ch)) {
      regexStr += "\\" + ch;
      prevStar = false;
    } else {
      regexStr += ch;
      prevStar = false;
    }
  }
  return new RegExp(`^${regexStr}$`).test(text);
}

/**
 * Apply per-provider model mapping.
 * If the provider has a modelMapping config, map Claude model names
 * (containing "opus", "sonnet", "haiku") to provider-specific models.
 * Otherwise return the model name as-is.
 */
export function mapModelForProvider(
  provider: ResolvedProvider,
  claudeModel: string,
): string {
  if (!provider.modelMapping || Object.keys(provider.modelMapping).length === 0) {
    return claudeModel;
  }

  const lower = claudeModel.toLowerCase();
  for (const [keyword, mappedModel] of Object.entries(provider.modelMapping)) {
    if (lower.includes(keyword.toLowerCase())) {
      return mappedModel;
    }
  }

  // If no mapping matched, return as-is
  return claudeModel;
}

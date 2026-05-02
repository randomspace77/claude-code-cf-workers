import type { AppConfig } from "../types";

type CacheKind = "tool" | "assistant";

export async function getToolReasoning(
  config: AppConfig,
  model: string,
  id: string | null | undefined,
): Promise<string | undefined> {
  if (!id) return undefined;
  return getReasoning(config, "tool", model, id);
}

export async function setToolReasoning(
  config: AppConfig,
  model: string,
  id: string | null | undefined,
  reasoning: string | null | undefined,
): Promise<void> {
  if (!id || !reasoning) return;
  await setReasoning(config, "tool", model, id, reasoning);
}

export async function getAssistantReasoning(
  config: AppConfig,
  model: string,
  text: string | null | undefined,
): Promise<string | undefined> {
  if (!text) return undefined;
  return getReasoning(config, "assistant", model, text);
}

export async function setAssistantReasoning(
  config: AppConfig,
  model: string,
  text: string | null | undefined,
  reasoning: string | null | undefined,
): Promise<void> {
  if (!text || !reasoning) return;
  await setReasoning(config, "assistant", model, text, reasoning);
}

async function getReasoning(
  config: AppConfig,
  kind: CacheKind,
  model: string,
  value: string,
): Promise<string | undefined> {
  if (!config.reasoningCache) return undefined;
  try {
    return (await config.reasoningCache.get(await cacheKey(kind, model, value))) ?? undefined;
  } catch (err) {
    console.warn("Failed to read DeepSeek reasoning cache:", err);
    return undefined;
  }
}

async function setReasoning(
  config: AppConfig,
  kind: CacheKind,
  model: string,
  value: string,
  reasoning: string,
): Promise<void> {
  if (!config.reasoningCache) return;
  try {
    await config.reasoningCache.put(await cacheKey(kind, model, value), reasoning, {
      expirationTtl: config.reasoningCacheTtlSeconds ?? 2592000,
    });
  } catch (err) {
    console.warn("Failed to write DeepSeek reasoning cache:", err);
  }
}

async function cacheKey(kind: CacheKind, model: string, value: string): Promise<string> {
  const digest = await sha256(`${model}\0${value}`);
  return `deepseek-reasoning:${kind}:${digest}`;
}

async function sha256(value: string): Promise<string> {
  const data = new TextEncoder().encode(value);
  const hash = await crypto.subtle.digest("SHA-256", data);
  return [...new Uint8Array(hash)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

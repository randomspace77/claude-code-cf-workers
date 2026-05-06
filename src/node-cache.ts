import { createConnection, type Socket } from "node:net";
import { connect as tlsConnect } from "node:tls";
import type { ReasoningCacheNamespace } from "./types";

type RedisValue = string | number | null | RedisValue[];

interface PendingRedisCommand {
  resolve: (value: RedisValue) => void;
  reject: (err: Error) => void;
}

export async function createNodeReasoningCache(
  env: NodeJS.ProcessEnv = process.env,
): Promise<ReasoningCacheNamespace | undefined> {
  const backend = (env.REASONING_CACHE_BACKEND || "memory").trim().toLowerCase();

  if (backend === "memory") {
    console.log("Reasoning cache backend: memory");
    return new MemoryReasoningCache();
  }

  if (backend === "redis") {
    const cache = new RedisReasoningCache(env.REDIS_URL || "redis://redis:6379");
    await cache.connect();
    console.log("Reasoning cache backend: redis");
    return cache;
  }

  if (backend === "none" || backend === "off" || backend === "disabled") {
    console.log("Reasoning cache backend: disabled");
    return undefined;
  }

  throw new Error(
    `Unsupported REASONING_CACHE_BACKEND "${backend}". Use memory, redis, or none.`,
  );
}

class MemoryReasoningCache implements ReasoningCacheNamespace {
  private values = new Map<string, { value: string; expiresAt: number | null }>();

  async get(key: string): Promise<string | null> {
    const entry = this.values.get(key);
    if (!entry) return null;
    if (entry.expiresAt !== null && entry.expiresAt <= Date.now()) {
      this.values.delete(key);
      return null;
    }
    return entry.value;
  }

  async put(
    key: string,
    value: string,
    options?: { expirationTtl?: number },
  ): Promise<void> {
    const expiresAt = ttlToExpiresAt(options?.expirationTtl);
    this.values.set(key, { value, expiresAt });
  }
}

class RedisReasoningCache implements ReasoningCacheNamespace {
  private readonly url: URL;
  private socket: Socket | undefined;
  private connectPromise: Promise<void> | undefined;
  private buffer = Buffer.alloc(0);
  private pending: PendingRedisCommand[] = [];

  constructor(redisUrl: string) {
    this.url = new URL(redisUrl);
    if (this.url.protocol !== "redis:" && this.url.protocol !== "rediss:") {
      throw new Error("REDIS_URL must use redis:// or rediss://");
    }
  }

  async connect(): Promise<void> {
    if (this.socket && !this.socket.destroyed) return;
    if (!this.connectPromise) {
      this.connectPromise = this.openConnection().finally(() => {
        this.connectPromise = undefined;
      });
    }
    await this.connectPromise;
  }

  async get(key: string): Promise<string | null> {
    const value = await this.command(["GET", key]);
    return typeof value === "string" ? value : null;
  }

  async put(
    key: string,
    value: string,
    options?: { expirationTtl?: number },
  ): Promise<void> {
    const ttl = options?.expirationTtl;
    if (ttl && ttl > 0) {
      await this.command(["SET", key, value, "EX", String(Math.floor(ttl))]);
      return;
    }
    await this.command(["SET", key, value]);
  }

  private async openConnection(): Promise<void> {
    const socket = await openRedisSocket(this.url);
    this.socket = socket;
    this.buffer = Buffer.alloc(0);

    socket.on("data", (chunk: Buffer) => this.handleData(chunk));
    socket.on("error", (err: Error) => this.handleSocketFailure(err));
    socket.on("close", () => {
      if (this.socket === socket) {
        this.handleSocketFailure(new Error("Redis connection closed"));
      }
    });

    try {
      const password = decodeURIComponent(this.url.password);
      const username = decodeURIComponent(this.url.username);
      if (password) {
        await this.commandOnOpenSocket(
          username ? ["AUTH", username, password] : ["AUTH", password],
        );
      }

      const db = this.url.pathname.replace(/^\//, "");
      if (db) {
        await this.commandOnOpenSocket(["SELECT", db]);
      }
    } catch (err) {
      socket.destroy();
      throw err;
    }
  }

  private async command(parts: string[]): Promise<RedisValue> {
    await this.connect();
    return this.commandOnOpenSocket(parts);
  }

  private commandOnOpenSocket(parts: string[]): Promise<RedisValue> {
    const socket = this.socket;
    if (!socket || socket.destroyed) {
      throw new Error("Redis connection is not available");
    }

    return new Promise<RedisValue>((resolve, reject) => {
      this.pending.push({ resolve, reject });
      socket.write(encodeRedisCommand(parts), (err) => {
        if (err) {
          const pending = this.pending.pop();
          pending?.reject(err);
        }
      });
    });
  }

  private handleData(chunk: Buffer): void {
    this.buffer = Buffer.concat([this.buffer, chunk]);

    try {
      while (this.pending.length > 0) {
        const parsed = parseRedisValue(this.buffer, 0);
        if (!parsed) return;

        this.buffer = this.buffer.subarray(parsed.offset);
        const pending = this.pending.shift();
        if (!pending) continue;

        if (parsed.value instanceof Error) {
          pending.reject(parsed.value);
        } else {
          pending.resolve(parsed.value);
        }
      }
    } catch (err) {
      this.handleSocketFailure(err instanceof Error ? err : new Error(String(err)));
    }
  }

  private handleSocketFailure(err: Error): void {
    if (this.socket && !this.socket.destroyed) {
      this.socket.destroy();
    }
    this.socket = undefined;
    this.buffer = Buffer.alloc(0);
    const pending = this.pending.splice(0);
    for (const command of pending) {
      command.reject(err);
    }
  }
}

function ttlToExpiresAt(ttlSeconds: number | undefined): number | null {
  if (!ttlSeconds || ttlSeconds <= 0) return null;
  return Date.now() + Math.floor(ttlSeconds) * 1000;
}

function openRedisSocket(url: URL): Promise<Socket> {
  const host = url.hostname || "localhost";
  const port = Number(url.port || (url.protocol === "rediss:" ? 6380 : 6379));

  return new Promise((resolve, reject) => {
    const socket = url.protocol === "rediss:"
      ? tlsConnect({ host, port, servername: host })
      : createConnection({ host, port });
    const event = url.protocol === "rediss:" ? "secureConnect" : "connect";

    const cleanup = () => {
      socket.off(event, onConnect);
      socket.off("error", onError);
    };
    const onConnect = () => {
      cleanup();
      resolve(socket);
    };
    const onError = (err: Error) => {
      cleanup();
      reject(err);
    };

    socket.once(event, onConnect);
    socket.once("error", onError);
  });
}

function encodeRedisCommand(parts: string[]): Buffer {
  const chunks = [`*${parts.length}\r\n`];
  for (const part of parts) {
    chunks.push(`$${Buffer.byteLength(part)}\r\n${part}\r\n`);
  }
  return Buffer.from(chunks.join(""));
}

function parseRedisValue(
  buffer: Buffer,
  offset: number,
): { value: RedisValue | Error; offset: number } | null {
  if (offset >= buffer.length) return null;
  const type = String.fromCharCode(buffer[offset]);
  const lineEnd = buffer.indexOf("\r\n", offset);
  if (lineEnd === -1) return null;
  const line = buffer.subarray(offset + 1, lineEnd).toString();
  const next = lineEnd + 2;

  if (type === "+") return { value: line, offset: next };
  if (type === "-") return { value: new Error(line), offset: next };
  if (type === ":") return { value: Number(line), offset: next };

  if (type === "$") {
    const length = Number(line);
    if (length === -1) return { value: null, offset: next };
    const end = next + length;
    if (buffer.length < end + 2) return null;
    return {
      value: buffer.subarray(next, end).toString(),
      offset: end + 2,
    };
  }

  if (type === "*") {
    const length = Number(line);
    if (length === -1) return { value: null, offset: next };
    const values: RedisValue[] = [];
    let cursor = next;
    for (let i = 0; i < length; i++) {
      const parsed = parseRedisValue(buffer, cursor);
      if (!parsed) return null;
      if (parsed.value instanceof Error) throw parsed.value;
      values.push(parsed.value);
      cursor = parsed.offset;
    }
    return { value: values, offset: cursor };
  }

  throw new Error(`Unsupported Redis response type "${type}"`);
}

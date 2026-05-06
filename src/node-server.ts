import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { Readable } from "node:stream";
import type { Env, ReasoningCacheNamespace } from "./types";
import { handleRequest } from "./app";
import { createNodeReasoningCache } from "./node-cache";

const PORT = parseInt(process.env.PORT || "3000", 10);
const HOST = process.env.HOST || "0.0.0.0";

/**
 * Build the Env object from process.env and Node-only bindings.
 * Uses a Proxy so dynamic keys like PROVIDER_<NAME>_API_KEY resolve automatically.
 */
function buildEnv(reasoningCache: ReasoningCacheNamespace | undefined): Env {
  return new Proxy({} as Env, {
    get(_target, prop: string | symbol) {
      if (prop === "REASONING_CACHE") {
        return reasoningCache;
      }
      if (typeof prop !== "string") return undefined;
      return process.env[prop];
    },
    has(_target, prop: string | symbol) {
      if (prop === "REASONING_CACHE") {
        return reasoningCache !== undefined;
      }
      if (typeof prop !== "string") return false;
      return prop in process.env;
    },
  });
}

/**
 * Convert a Node.js IncomingMessage into a Web API Request.
 */
function toWebRequest(req: IncomingMessage): Request {
  const host = req.headers.host || `localhost:${PORT}`;
  const url = new URL(req.url || "/", `http://${host}`);

  const headers = new Headers();
  for (const [key, value] of Object.entries(req.headers)) {
    if (value === undefined) continue;
    if (Array.isArray(value)) {
      for (const v of value) headers.append(key, v);
    } else {
      headers.set(key, value);
    }
  }

  const method = (req.method || "GET").toUpperCase();
  const init: RequestInit & { duplex?: string } = { method, headers };

  if (method !== "GET" && method !== "HEAD") {
    init.body = Readable.toWeb(req) as ReadableStream;
    init.duplex = "half";
  }

  return new Request(url.toString(), init);
}

/**
 * Pipe a Web API Response back through a Node.js ServerResponse.
 */
async function sendWebResponse(webRes: Response, res: ServerResponse): Promise<void> {
  const headersObj: Record<string, string> = {};
  webRes.headers.forEach((value, key) => {
    headersObj[key] = value;
  });
  res.writeHead(webRes.status, headersObj);

  if (!webRes.body) {
    res.end();
    return;
  }

  const reader = webRes.body.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      res.write(value);
    }
  } catch (err) {
    console.error("Error streaming response:", err);
  } finally {
    res.end();
  }
}

// ---- Main ----

const env = buildEnv(await createNodeReasoningCache());

const server = createServer(async (req, res) => {
  try {
    const webRequest = toWebRequest(req);
    const webResponse = await handleRequest(webRequest, env);
    await sendWebResponse(webResponse, res);
  } catch (err) {
    console.error("Server error:", err);
    if (!res.headersSent) {
      res.writeHead(500, { "Content-Type": "application/json" });
    }
    res.end(JSON.stringify({
      type: "error",
      error: { type: "api_error", message: "Internal server error" },
    }));
  }
});

server.listen(PORT, HOST, () => {
  console.log(`Claude Code Proxy running at http://${HOST}:${PORT}`);
});

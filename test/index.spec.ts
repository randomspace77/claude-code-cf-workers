import { env } from "cloudflare:workers";
import { describe, it, expect } from "vitest";
import worker from "../src/index";
import type { Env } from "../src/types";

// The env from cloudflare:workers gets populated from wrangler.toml [vars]
const testEnv = env as unknown as Env;

describe("Claude Code Proxy Worker", () => {
  describe("GET /", () => {
    it("returns proxy information", async () => {
      const request = new Request("http://localhost/");
      const response = await worker.fetch(request, testEnv);

      expect(response.status).toBe(200);
      const body = (await response.json()) as Record<string, unknown>;
      expect(body.message).toBe("Claude-to-OpenAI API Proxy v1.0.0");
      expect(body.status).toBe("running");
      expect(body).toHaveProperty("endpoints");
    });
  });

  describe("GET /health", () => {
    it("returns health status", async () => {
      const request = new Request("http://localhost/health");
      const response = await worker.fetch(request, testEnv);

      expect(response.status).toBe(200);
      const body = (await response.json()) as Record<string, unknown>;
      expect(body.status).toBe("healthy");
      expect(body).toHaveProperty("timestamp");
    });
  });

  describe("OPTIONS (CORS)", () => {
    it("returns CORS headers for preflight", async () => {
      const request = new Request("http://localhost/v1/messages", {
        method: "OPTIONS",
      });
      const response = await worker.fetch(request, testEnv);

      expect(response.status).toBe(200);
      expect(response.headers.get("Access-Control-Allow-Origin")).toBe("*");
      expect(response.headers.get("Access-Control-Allow-Methods")).toContain("POST");
    });
  });

  describe("POST /v1/messages", () => {
    it("rejects when no API key available (no server key, no client key)", async () => {
      const request = new Request("http://localhost/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-3-5-sonnet-20241022",
          max_tokens: 100,
          messages: [{ role: "user", content: "Hello" }],
        }),
      });
      const response = await worker.fetch(request, testEnv);

      // Without any API key, should return 401
      expect(response.status).toBe(401);
      const body = (await response.json()) as Record<string, unknown>;
      expect(body.type).toBe("error");
    });

    it("rejects invalid JSON with configured API key", async () => {
      const envWithKey = { ...testEnv, OPENAI_API_KEY: "sk-test-key" };
      const request = new Request("http://localhost/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "not json",
      });
      const response = await worker.fetch(request, envWithKey as unknown as Env);

      expect(response.status).toBe(400);
    });

    it("rejects requests missing required fields", async () => {
      const envWithKey = { ...testEnv, OPENAI_API_KEY: "sk-test-key" };
      const request = new Request("http://localhost/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: "claude-3-5-sonnet-20241022" }),
      });
      const response = await worker.fetch(request, envWithKey as unknown as Env);

      expect(response.status).toBe(400);
      const body = (await response.json()) as Record<string, unknown>;
      expect(body.type).toBe("error");
    });
  });

  describe("POST /v1/messages/count_tokens", () => {
    it("estimates token count", async () => {
      const request = new Request("http://localhost/v1/messages/count_tokens", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": "client-key-123",
        },
        body: JSON.stringify({
          model: "claude-3-5-sonnet-20241022",
          messages: [
            { role: "user", content: "Hello, how are you?" },
          ],
        }),
      });
      const response = await worker.fetch(request, testEnv);

      expect(response.status).toBe(200);
      const body = (await response.json()) as Record<string, unknown>;
      expect(body).toHaveProperty("input_tokens");
      expect(typeof body.input_tokens).toBe("number");
      expect(body.input_tokens).toBeGreaterThan(0);
    });
  });

  describe("404 handling", () => {
    it("returns 404 for unknown routes", async () => {
      const request = new Request("http://localhost/unknown-route", {
        headers: { "x-api-key": "client-key-123" },
      });
      const response = await worker.fetch(request, testEnv);

      expect(response.status).toBe(404);
    });
  });
});

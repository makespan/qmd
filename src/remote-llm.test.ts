/**
 * remote-llm.test.ts - Tests for RemoteLLM using mock servers
 *
 * Run with: bun test src/remote-llm.test.ts
 */

import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import { RemoteLLM } from "./remote-llm";
import type { RerankDocument } from "./llm";

// =============================================================================
// Mock servers
// =============================================================================

/** Simple 4-dimensional embeddings for testing */
function mockEmbedding(text: string): number[] {
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
  }
  const dim = 4;
  const emb: number[] = [];
  for (let i = 0; i < dim; i++) {
    emb.push(Math.sin(hash + i * 1000) * 0.5);
  }
  return emb;
}

function createMockEmbedServer() {
  return Bun.serve({
    port: 0,
    async fetch(req) {
      const url = new URL(req.url);

      if (url.pathname === "/health") {
        return Response.json({ status: "ok" });
      }

      if (url.pathname === "/tokenize" && req.method === "POST") {
        const body = (await req.json()) as any;
        const tokens = (body.content as string)
          .split(/\s+/)
          .filter(Boolean)
          .map((_: string, i: number) => i + 1);
        return Response.json({ tokens });
      }

      if (url.pathname === "/detokenize" && req.method === "POST") {
        const body = (await req.json()) as any;
        const content = (body.tokens as number[])
          .map((t: number) => `tok${t}`)
          .join(" ");
        return Response.json({ content });
      }

      if (url.pathname === "/v1/embeddings" && req.method === "POST") {
        const body = (await req.json()) as any;
        const inputs = Array.isArray(body.input) ? body.input : [body.input];
        const data = inputs.map((text: string, index: number) => ({
          embedding: mockEmbedding(text),
          index,
          object: "embedding",
        }));
        return Response.json({
          data,
          model: body.model || "mock-embed",
          object: "list",
          usage: { prompt_tokens: 10, total_tokens: 10 },
        });
      }

      return new Response("Not found", { status: 404 });
    },
  });
}

function createMockOpenAIEmbedServer() {
  // Simulates OpenAI API: requires auth, model field, no /tokenize
  return Bun.serve({
    port: 0,
    async fetch(req) {
      const url = new URL(req.url);

      if (url.pathname === "/v1/embeddings" && req.method === "POST") {
        const auth = req.headers.get("Authorization");
        if (!auth || !auth.startsWith("Bearer sk-")) {
          return new Response("Unauthorized", { status: 401 });
        }

        const body = (await req.json()) as any;
        if (!body.model) {
          return Response.json(
            { error: { message: "model is required" } },
            { status: 400 }
          );
        }

        const inputs = Array.isArray(body.input) ? body.input : [body.input];
        const data = inputs.map((text: string, index: number) => ({
          embedding: mockEmbedding(text),
          index,
          object: "embedding",
        }));
        return Response.json({
          data,
          model: body.model,
          object: "list",
          usage: { prompt_tokens: 10, total_tokens: 10 },
        });
      }

      // No /tokenize endpoint (like real OpenAI)
      return new Response("Not found", { status: 404 });
    },
  });
}

function createMockGenerateServer() {
  return Bun.serve({
    port: 0,
    async fetch(req) {
      const url = new URL(req.url);

      if (url.pathname === "/health") {
        return Response.json({ status: "ok" });
      }

      if (url.pathname === "/v1/chat/completions" && req.method === "POST") {
        const body = (await req.json()) as any;
        const userMessage =
          body.messages?.find((m: any) => m.role === "user")?.content || "";

        let content: string;
        if (body.grammar) {
          const query = userMessage.replace(
            "/no_think Expand this search query: ",
            ""
          );
          content = `vec: ${query} meaning and context\nlex: ${query}\nhyde: A document about ${query}\n`;
        } else {
          content = `Response to: ${userMessage}`;
        }

        return Response.json({
          id: "mock-chat-1",
          object: "chat.completion",
          choices: [
            {
              index: 0,
              message: { role: "assistant", content },
              finish_reason: "stop",
            },
          ],
          model: body.model || "mock-generate",
          usage: {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
          },
        });
      }

      return new Response("Not found", { status: 404 });
    },
  });
}

function createMockRerankServer() {
  return Bun.serve({
    port: 0,
    async fetch(req) {
      const url = new URL(req.url);

      if (url.pathname === "/health") {
        return Response.json({ status: "ok" });
      }

      if (url.pathname === "/v1/rerank" && req.method === "POST") {
        const body = (await req.json()) as any;
        const query = (body.query as string).toLowerCase();
        const documents = body.documents as string[];

        const queryWords = new Set(query.split(/\s+/));
        const results = documents.map((doc: string, index: number) => {
          const docWords = doc.toLowerCase().split(/\s+/);
          const overlap = docWords.filter((w: string) => queryWords.has(w))
            .length;
          const relevance_score = Math.min(
            overlap / Math.max(queryWords.size, 1),
            1.0
          );
          return { index, relevance_score };
        });

        return Response.json({
          results,
          model: body.model || "mock-rerank",
          object: "list",
        });
      }

      return new Response("Not found", { status: 404 });
    },
  });
}

// =============================================================================
// Tests: llama.cpp server backend
// =============================================================================

describe("RemoteLLM (llama.cpp server)", () => {
  let embedServer: any;
  let generateServer: any;
  let rerankServer: any;
  let llm: RemoteLLM;

  beforeAll(() => {
    embedServer = createMockEmbedServer();
    generateServer = createMockGenerateServer();
    rerankServer = createMockRerankServer();

    llm = new RemoteLLM({
      embedUrl: `http://localhost:${embedServer.port}`,
      generateUrl: `http://localhost:${generateServer.port}`,
      rerankUrl: `http://localhost:${rerankServer.port}`,
    });
  });

  afterAll(() => {
    embedServer.stop(true);
    generateServer.stop(true);
    rerankServer.stop(true);
  });

  // ===========================================================================
  // Tokenization
  // ===========================================================================

  describe("tokenize", () => {
    test("returns token array from server", async () => {
      const tokens = await llm.tokenize("Hello world test");
      expect(tokens).toBeInstanceOf(Array);
      expect(tokens.length).toBe(3);
    });

    test("hasNativeTokenizer is true after successful tokenize", async () => {
      await llm.tokenize("test");
      expect(llm.hasNativeTokenizer).toBe(true);
    });

    test("handles empty string", async () => {
      const tokens = await llm.tokenize("");
      expect(tokens).toBeInstanceOf(Array);
      expect(tokens.length).toBe(0);
    });
  });

  describe("detokenize", () => {
    test("returns string from tokens", async () => {
      const text = await llm.detokenize([1, 2, 3]);
      expect(typeof text).toBe("string");
      expect(text.length).toBeGreaterThan(0);
    });
  });

  describe("countTokens", () => {
    test("returns token count", async () => {
      const count = await llm.countTokens("one two three four");
      expect(count).toBe(4);
    });
  });

  // ===========================================================================
  // Embeddings
  // ===========================================================================

  describe("embed", () => {
    test("returns embedding result", async () => {
      const result = await llm.embed("Hello world");
      expect(result).not.toBeNull();
      expect(result!.embedding).toBeInstanceOf(Array);
      expect(result!.embedding.length).toBe(4);
    });

    test("returns consistent embeddings for same input", async () => {
      const r1 = await llm.embed("test text");
      const r2 = await llm.embed("test text");
      expect(r1).not.toBeNull();
      expect(r2).not.toBeNull();
      for (let i = 0; i < r1!.embedding.length; i++) {
        expect(r1!.embedding[i]).toBeCloseTo(r2!.embedding[i]!, 5);
      }
    });

    test("returns different embeddings for different inputs", async () => {
      const r1 = await llm.embed("cats");
      const r2 = await llm.embed("database optimization");
      expect(r1).not.toBeNull();
      expect(r2).not.toBeNull();
      const allSame = r1!.embedding.every(
        (v, i) => Math.abs(v - r2!.embedding[i]!) < 1e-6
      );
      expect(allSame).toBe(false);
    });
  });

  describe("embedBatch", () => {
    test("returns embeddings for multiple texts", async () => {
      const texts = ["Hello", "World", "Test"];
      const results = await llm.embedBatch(texts);
      expect(results).toHaveLength(3);
      for (const r of results) {
        expect(r).not.toBeNull();
        expect(r!.embedding.length).toBe(4);
      }
    });

    test("handles empty array", async () => {
      const results = await llm.embedBatch([]);
      expect(results).toHaveLength(0);
    });

    test("preserves order", async () => {
      const texts = ["aaa", "bbb", "ccc"];
      const batch = await llm.embedBatch(texts);
      const individual = await Promise.all(texts.map((t) => llm.embed(t)));
      for (let i = 0; i < texts.length; i++) {
        for (let j = 0; j < batch[i]!.embedding.length; j++) {
          expect(batch[i]!.embedding[j]).toBeCloseTo(
            individual[i]!.embedding[j]!,
            5
          );
        }
      }
    });
  });

  // ===========================================================================
  // Generation
  // ===========================================================================

  describe("generate", () => {
    test("returns generated text", async () => {
      const result = await llm.generate("Tell me about cats");
      expect(result).not.toBeNull();
      expect(result!.text).toContain("cats");
      expect(result!.done).toBe(true);
    });
  });

  // ===========================================================================
  // Query Expansion
  // ===========================================================================

  describe("expandQuery", () => {
    test("returns query expansions with correct types", async () => {
      const result = await llm.expandQuery("test query");
      expect(result.length).toBeGreaterThanOrEqual(1);
      for (const q of result) {
        expect(["lex", "vec", "hyde"]).toContain(q.type);
        expect(q.text.length).toBeGreaterThan(0);
      }
    });

    test("can exclude lexical queries", async () => {
      const result = await llm.expandQuery("authentication setup", {
        includeLexical: false,
      });
      const lexEntries = result.filter((q) => q.type === "lex");
      expect(lexEntries).toHaveLength(0);
    });

    test("falls back gracefully on error", async () => {
      const brokenLlm = new RemoteLLM({
        generateUrl: "http://localhost:1",
      });
      const result = await brokenLlm.expandQuery("test");
      expect(result.length).toBeGreaterThanOrEqual(1);
      const vecEntries = result.filter((q) => q.type === "vec");
      expect(vecEntries.length).toBeGreaterThanOrEqual(1);
    });
  });

  // ===========================================================================
  // Reranking
  // ===========================================================================

  describe("rerank", () => {
    test("scores documents by relevance", async () => {
      const query = "capital of France";
      const documents: RerankDocument[] = [
        { file: "food.txt", text: "Pizza is delicious food" },
        { file: "france.txt", text: "The capital of France is Paris" },
        { file: "random.txt", text: "Nothing relevant here" },
      ];
      const result = await llm.rerank(query, documents);
      expect(result.results).toHaveLength(3);
      expect(result.results[0]!.file).toBe("france.txt");
    });

    test("handles empty document list", async () => {
      const result = await llm.rerank("test query", []);
      expect(result.results).toHaveLength(0);
    });

    test("handles single document", async () => {
      const result = await llm.rerank("test", [
        { file: "doc.md", text: "content" },
      ]);
      expect(result.results).toHaveLength(1);
    });

    test("preserves original file paths", async () => {
      const documents: RerankDocument[] = [
        { file: "path/to/doc1.md", text: "content one" },
        { file: "another/path/doc2.md", text: "content two" },
      ];
      const result = await llm.rerank("query", documents);
      const files = result.results.map((r) => r.file).sort();
      expect(files).toEqual(["another/path/doc2.md", "path/to/doc1.md"]);
    });

    test("returns scores between 0 and 1", async () => {
      const documents: RerankDocument[] = [
        { file: "a.md", text: "The quick brown fox" },
        { file: "b.md", text: "Machine learning data" },
      ];
      const result = await llm.rerank("fox", documents);
      for (const doc of result.results) {
        expect(doc.score).toBeGreaterThanOrEqual(0);
        expect(doc.score).toBeLessThanOrEqual(1);
      }
    });
  });

  // ===========================================================================
  // Model Info & Lifecycle
  // ===========================================================================

  describe("modelExists", () => {
    test("returns exists:true when server is healthy", async () => {
      const result = await llm.modelExists("some-model");
      expect(result.exists).toBe(true);
    });
  });

  describe("lifecycle", () => {
    test("dispose is safe to call multiple times", async () => {
      const disposable = new RemoteLLM({ embedUrl: "http://localhost:1" });
      await disposable.dispose();
      await disposable.dispose();
    });

    test("unloadIdleResources is no-op", async () => {
      await llm.unloadIdleResources();
    });
  });
});

// =============================================================================
// Tests: OpenAI-compatible backend
// =============================================================================

describe("RemoteLLM (OpenAI-compatible)", () => {
  let openaiServer: any;
  let llm: RemoteLLM;

  beforeAll(() => {
    openaiServer = createMockOpenAIEmbedServer();
    llm = new RemoteLLM({
      embedUrl: `http://localhost:${openaiServer.port}`,
      embedApiKey: "sk-test-key-123",
      embedModel: "text-embedding-3-small",
      embedFormat: "raw",
    });
  });

  afterAll(() => {
    openaiServer.stop(true);
  });

  test("sends Authorization header", async () => {
    const result = await llm.embed("Hello world");
    expect(result).not.toBeNull();
    expect(result!.embedding.length).toBe(4);
  });

  test("sends model in request body", async () => {
    const result = await llm.embed("test");
    expect(result).not.toBeNull();
    // The mock validates model is present and returns 400 without it
    expect(result!.model).toBe("text-embedding-3-small");
  });

  test("fails without API key", async () => {
    const noKeyLlm = new RemoteLLM({
      embedUrl: `http://localhost:${openaiServer.port}`,
      embedModel: "text-embedding-3-small",
    });
    const result = await noKeyLlm.embed("test");
    expect(result).toBeNull(); // 401 from server
  });

  test("batch embedding works with auth", async () => {
    const results = await llm.embedBatch(["hello", "world"]);
    expect(results).toHaveLength(2);
    expect(results[0]).not.toBeNull();
    expect(results[1]).not.toBeNull();
  });

  test("isRawEmbedFormat is true", () => {
    expect(llm.isRawEmbedFormat).toBe(true);
  });

  test("hasNativeTokenizer is false (no /tokenize endpoint)", async () => {
    // Reset tokenize detection by creating a fresh instance
    const freshLlm = new RemoteLLM({
      embedUrl: `http://localhost:${openaiServer.port}`,
      embedApiKey: "sk-test-key-123",
      embedModel: "text-embedding-3-small",
    });
    expect(freshLlm.hasNativeTokenizer).toBe(false);

    // After a tokenize attempt, it should still be false (server returns 404)
    const tokens = await freshLlm.tokenize("hello world");
    expect(freshLlm.hasNativeTokenizer).toBe(false);
    // Falls back to approximate tokenization
    expect(tokens.length).toBeGreaterThan(0);
  });

  test("approximate tokenization works for chunking", async () => {
    const freshLlm = new RemoteLLM({
      embedUrl: `http://localhost:${openaiServer.port}`,
      embedApiKey: "sk-test-key-123",
    });
    // Force approximate mode
    const tokens = await freshLlm.tokenize("hello world");
    expect(tokens.length).toBeGreaterThan(0);

    const count = await freshLlm.countTokens(
      "This is a longer text for counting"
    );
    expect(count).toBeGreaterThan(0);
  });
});

// =============================================================================
// Tests: Embedding format
// =============================================================================

describe("Embedding format", () => {
  test("embeddinggemma format is default", () => {
    const llm = new RemoteLLM({ embedUrl: "http://localhost:1" });
    expect(llm.isRawEmbedFormat).toBe(false);
  });

  test("raw format can be configured", () => {
    const llm = new RemoteLLM({
      embedUrl: "http://localhost:1",
      embedFormat: "raw",
    });
    expect(llm.isRawEmbedFormat).toBe(true);
  });
});

// =============================================================================
// Tests: Error handling
// =============================================================================

describe("Error handling", () => {
  test("embed returns null on connection error", async () => {
    const brokenLlm = new RemoteLLM({ embedUrl: "http://localhost:1" });
    const result = await brokenLlm.embed("test");
    expect(result).toBeNull();
  });

  test("embedBatch returns nulls on connection error", async () => {
    const brokenLlm = new RemoteLLM({ embedUrl: "http://localhost:1" });
    const results = await brokenLlm.embedBatch(["a", "b"]);
    expect(results).toHaveLength(2);
    expect(results[0]).toBeNull();
    expect(results[1]).toBeNull();
  });

  test("generate returns null on connection error", async () => {
    const brokenLlm = new RemoteLLM({ generateUrl: "http://localhost:1" });
    const result = await brokenLlm.generate("test");
    expect(result).toBeNull();
  });

  test("rerank returns zero scores on connection error", async () => {
    const brokenLlm = new RemoteLLM({ rerankUrl: "http://localhost:1" });
    const result = await brokenLlm.rerank("test", [
      { file: "a.md", text: "content" },
    ]);
    expect(result.results).toHaveLength(1);
    expect(result.results[0]!.score).toBe(0);
  });

  test("throws when URL not configured", async () => {
    const noUrlLlm = new RemoteLLM({});
    await expect(noUrlLlm.tokenize("test")).rejects.toThrow(
      "No URL configured"
    );
  });
});

// =============================================================================
// Tests: Factory integration
// =============================================================================

describe("Factory integration", () => {
  test("getRemoteLLMConfig returns null when no env vars set", async () => {
    const { getRemoteLLMConfig } = await import("./llm");
    const saved = {
      embed: process.env.QMD_EMBED_URL,
      generate: process.env.QMD_GENERATE_URL,
      rerank: process.env.QMD_RERANK_URL,
    };
    delete process.env.QMD_EMBED_URL;
    delete process.env.QMD_GENERATE_URL;
    delete process.env.QMD_RERANK_URL;

    try {
      const config = getRemoteLLMConfig();
      expect(config).toBeNull();
    } finally {
      if (saved.embed) process.env.QMD_EMBED_URL = saved.embed;
      if (saved.generate) process.env.QMD_GENERATE_URL = saved.generate;
      if (saved.rerank) process.env.QMD_RERANK_URL = saved.rerank;
    }
  });

  test("getRemoteLLMConfig returns full config from env", async () => {
    const { getRemoteLLMConfig } = await import("./llm");
    const saved = {
      url: process.env.QMD_EMBED_URL,
      key: process.env.QMD_EMBED_API_KEY,
      model: process.env.QMD_EMBED_MODEL,
      format: process.env.QMD_EMBED_FORMAT,
    };
    process.env.QMD_EMBED_URL = "https://api.openai.com";
    process.env.QMD_EMBED_API_KEY = "sk-test";
    process.env.QMD_EMBED_MODEL = "text-embedding-3-small";
    process.env.QMD_EMBED_FORMAT = "raw";

    try {
      const config = getRemoteLLMConfig();
      expect(config).not.toBeNull();
      expect(config!.embedUrl).toBe("https://api.openai.com");
      expect(config!.embedApiKey).toBe("sk-test");
      expect(config!.embedModel).toBe("text-embedding-3-small");
      expect(config!.embedFormat).toBe("raw");
    } finally {
      if (saved.url) process.env.QMD_EMBED_URL = saved.url;
      else delete process.env.QMD_EMBED_URL;
      if (saved.key) process.env.QMD_EMBED_API_KEY = saved.key;
      else delete process.env.QMD_EMBED_API_KEY;
      if (saved.model) process.env.QMD_EMBED_MODEL = saved.model;
      else delete process.env.QMD_EMBED_MODEL;
      if (saved.format) process.env.QMD_EMBED_FORMAT = saved.format;
      else delete process.env.QMD_EMBED_FORMAT;
    }
  });

  test("isRawEmbedFormat checks env var directly", async () => {
    const { isRawEmbedFormat } = await import("./llm");
    const saved = process.env.QMD_EMBED_FORMAT;

    process.env.QMD_EMBED_FORMAT = "raw";
    expect(isRawEmbedFormat()).toBe(true);

    delete process.env.QMD_EMBED_FORMAT;
    expect(isRawEmbedFormat()).toBe(false);

    process.env.QMD_EMBED_FORMAT = "embeddinggemma";
    expect(isRawEmbedFormat()).toBe(false);

    if (saved) process.env.QMD_EMBED_FORMAT = saved;
    else delete process.env.QMD_EMBED_FORMAT;
  });
});

// =============================================================================
// Tests: Format functions with raw mode
// =============================================================================

describe("Format functions", () => {
  test("formatQueryForEmbedding uses embeddinggemma format by default", async () => {
    const saved = process.env.QMD_EMBED_FORMAT;
    delete process.env.QMD_EMBED_FORMAT;

    try {
      const { formatQueryForEmbedding } = await import("./llm");
      const result = formatQueryForEmbedding("test query");
      expect(result).toBe("task: search result | query: test query");
    } finally {
      if (saved) process.env.QMD_EMBED_FORMAT = saved;
    }
  });

  test("formatQueryForEmbedding returns raw text in raw mode", async () => {
    const saved = process.env.QMD_EMBED_FORMAT;
    process.env.QMD_EMBED_FORMAT = "raw";

    try {
      const { formatQueryForEmbedding } = await import("./llm");
      const result = formatQueryForEmbedding("test query");
      expect(result).toBe("test query");
    } finally {
      if (saved) process.env.QMD_EMBED_FORMAT = saved;
      else delete process.env.QMD_EMBED_FORMAT;
    }
  });

  test("formatDocForEmbedding uses embeddinggemma format by default", async () => {
    const saved = process.env.QMD_EMBED_FORMAT;
    delete process.env.QMD_EMBED_FORMAT;

    try {
      const { formatDocForEmbedding } = await import("./llm");
      const result = formatDocForEmbedding("doc text", "My Title");
      expect(result).toBe("title: My Title | text: doc text");
    } finally {
      if (saved) process.env.QMD_EMBED_FORMAT = saved;
    }
  });

  test("formatDocForEmbedding returns plain text with title in raw mode", async () => {
    const saved = process.env.QMD_EMBED_FORMAT;
    process.env.QMD_EMBED_FORMAT = "raw";

    try {
      const { formatDocForEmbedding } = await import("./llm");
      const result = formatDocForEmbedding("doc text", "My Title");
      expect(result).toBe("My Title\n\ndoc text");
    } finally {
      if (saved) process.env.QMD_EMBED_FORMAT = saved;
      else delete process.env.QMD_EMBED_FORMAT;
    }
  });

  test("formatDocForEmbedding returns just text without title in raw mode", async () => {
    const saved = process.env.QMD_EMBED_FORMAT;
    process.env.QMD_EMBED_FORMAT = "raw";

    try {
      const { formatDocForEmbedding } = await import("./llm");
      const result = formatDocForEmbedding("doc text");
      expect(result).toBe("doc text");
    } finally {
      if (saved) process.env.QMD_EMBED_FORMAT = saved;
      else delete process.env.QMD_EMBED_FORMAT;
    }
  });
});

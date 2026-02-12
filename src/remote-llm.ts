/**
 * remote-llm.ts - Remote LLM backend for QMD
 *
 * Calls external HTTP API servers instead of loading models in-process.
 * Works with llama.cpp server, OpenAI API, and any OpenAI-compatible service.
 *
 * Configure via environment variables:
 *   QMD_EMBED_URL       - Embedding server URL
 *   QMD_EMBED_API_KEY   - API key for embedding server
 *   QMD_EMBED_MODEL     - Model name to send in requests
 *   QMD_EMBED_FORMAT    - "embeddinggemma" (default) or "raw"
 *   QMD_GENERATE_URL    - Generation server URL
 *   QMD_GENERATE_API_KEY - API key for generation server
 *   QMD_GENERATE_MODEL  - Model name for generation
 *   QMD_RERANK_URL      - Reranking server URL
 *   QMD_RERANK_API_KEY  - API key for reranking server
 *   QMD_RERANK_MODEL    - Model name for reranking
 *
 * Examples:
 *   # llama.cpp server (no auth needed)
 *   QMD_EMBED_URL=http://localhost:8081
 *
 *   # OpenAI
 *   QMD_EMBED_URL=https://api.openai.com
 *   QMD_EMBED_API_KEY=sk-...
 *   QMD_EMBED_MODEL=text-embedding-3-small
 *   QMD_EMBED_FORMAT=raw
 *
 *   # Any OpenAI-compatible service (Ollama, vLLM, Together, etc.)
 *   QMD_EMBED_URL=http://localhost:11434
 *   QMD_EMBED_MODEL=nomic-embed-text
 *   QMD_EMBED_FORMAT=raw
 */

import type {
  LLMEngine,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  Queryable,
  QueryType,
  RerankDocument,
  RerankOptions,
  RerankResult,
  RerankDocumentResult,
} from "./llm";

export type RemoteLLMConfig = {
  embedUrl?: string;
  embedApiKey?: string;
  embedModel?: string;
  embedFormat?: "embeddinggemma" | "raw";
  generateUrl?: string;
  generateApiKey?: string;
  generateModel?: string;
  rerankUrl?: string;
  rerankApiKey?: string;
  rerankModel?: string;
};

/**
 * Build a RemoteLLMConfig from environment variables.
 */
export function configFromEnv(): RemoteLLMConfig {
  return {
    embedUrl: process.env.QMD_EMBED_URL,
    embedApiKey: process.env.QMD_EMBED_API_KEY,
    embedModel: process.env.QMD_EMBED_MODEL,
    embedFormat: (process.env.QMD_EMBED_FORMAT as "embeddinggemma" | "raw") || undefined,
    generateUrl: process.env.QMD_GENERATE_URL,
    generateApiKey: process.env.QMD_GENERATE_API_KEY,
    generateModel: process.env.QMD_GENERATE_MODEL,
    rerankUrl: process.env.QMD_RERANK_URL,
    rerankApiKey: process.env.QMD_RERANK_API_KEY,
    rerankModel: process.env.QMD_RERANK_MODEL,
  };
}

// =============================================================================
// Approximate tokenizer for APIs without /tokenize endpoint
// =============================================================================

/**
 * Simple token approximation: ~4 characters per token.
 * Used as fallback when the remote server doesn't support /tokenize.
 * Good enough for chunking purposes.
 */
function approxTokenize(text: string): number[] {
  const CHARS_PER_TOKEN = 4;
  const count = Math.max(1, Math.ceil(text.length / CHARS_PER_TOKEN));
  return Array.from({ length: count }, (_, i) => i);
}

function approxDetokenize(tokens: readonly any[], originalLength?: number): string {
  // We can't truly detokenize without the model, but for chunking
  // the caller uses slice on the original text anyway
  return `[${tokens.length} tokens]`;
}

/**
 * Remote LLM implementation that calls HTTP API servers.
 *
 * Compatible with:
 * - llama.cpp server (full support including tokenize/rerank)
 * - OpenAI API (embeddings + chat completions)
 * - Any OpenAI-compatible service (Ollama, vLLM, Together, Groq, etc.)
 */
export class RemoteLLM implements LLMEngine {
  private embedUrl: string;
  private embedApiKey: string;
  private embedModel: string;
  private embedFormat: "embeddinggemma" | "raw";
  private generateUrl: string;
  private generateApiKey: string;
  private generateModel: string;
  private rerankUrl: string;
  private rerankApiKey: string;
  private rerankModel: string;
  private disposed = false;

  // Whether the embed server supports /tokenize (auto-detected on first call)
  private _tokenizeSupported: boolean | null = null;

  constructor(config: RemoteLLMConfig) {
    this.embedUrl = (config.embedUrl || "").replace(/\/+$/, "");
    this.embedApiKey = config.embedApiKey || "";
    this.embedModel = config.embedModel || "";
    this.embedFormat = config.embedFormat || "embeddinggemma";
    this.generateUrl = (config.generateUrl || "").replace(/\/+$/, "");
    this.generateApiKey = config.generateApiKey || "";
    this.generateModel = config.generateModel || "";
    this.rerankUrl = (config.rerankUrl || "").replace(/\/+$/, "");
    this.rerankApiKey = config.rerankApiKey || "";
    this.rerankModel = config.rerankModel || "";
  }

  /**
   * Whether this instance uses raw embedding format (no embeddinggemma prefixes).
   */
  get isRawEmbedFormat(): boolean {
    return this.embedFormat === "raw";
  }

  /**
   * Whether the remote server has a real tokenizer.
   * Auto-detected on first tokenize call. Before detection, returns false
   * so callers use character-based chunking as the safe default.
   */
  get hasNativeTokenizer(): boolean {
    return this._tokenizeSupported === true;
  }

  // ===========================================================================
  // Helper
  // ===========================================================================

  private requireUrl(url: string, capability: string): string {
    if (!url) {
      throw new Error(
        `No URL configured for ${capability}. Set QMD_${capability.toUpperCase()}_URL environment variable.`
      );
    }
    return url;
  }

  private buildHeaders(apiKey: string): Record<string, string> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (apiKey) {
      headers["Authorization"] = `Bearer ${apiKey}`;
    }
    return headers;
  }

  private async fetchJSON(
    url: string,
    body: object,
    apiKey: string = ""
  ): Promise<any> {
    const resp = await fetch(url, {
      method: "POST",
      headers: this.buildHeaders(apiKey),
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`HTTP ${resp.status} from ${url}: ${text}`);
    }
    return resp.json();
  }

  // ===========================================================================
  // Tokenization
  // ===========================================================================

  /**
   * Tokenize text. Tries the server's /tokenize endpoint first (llama.cpp),
   * falls back to character-based approximation (OpenAI, etc.).
   */
  async tokenize(text: string): Promise<readonly number[]> {
    const url = this.requireUrl(this.embedUrl, "embed");

    // If we already know tokenize isn't supported, use approximation
    if (this._tokenizeSupported === false) {
      return approxTokenize(text);
    }

    try {
      const data = await this.fetchJSON(
        `${url}/tokenize`,
        { content: text },
        this.embedApiKey
      );
      this._tokenizeSupported = true;
      return data.tokens as number[];
    } catch (error) {
      // Server doesn't support /tokenize — use approximation
      this._tokenizeSupported = false;
      return approxTokenize(text);
    }
  }

  async detokenize(tokens: readonly any[]): Promise<string> {
    const url = this.requireUrl(this.embedUrl, "embed");

    if (this._tokenizeSupported === false) {
      return approxDetokenize(tokens);
    }

    try {
      const data = await this.fetchJSON(
        `${url}/detokenize`,
        { tokens: Array.from(tokens) },
        this.embedApiKey
      );
      this._tokenizeSupported = true;
      return data.content as string;
    } catch {
      this._tokenizeSupported = false;
      return approxDetokenize(tokens);
    }
  }

  async countTokens(text: string): Promise<number> {
    const tokens = await this.tokenize(text);
    return tokens.length;
  }

  // ===========================================================================
  // Embeddings
  // ===========================================================================

  async embed(
    text: string,
    options: EmbedOptions = {}
  ): Promise<EmbeddingResult | null> {
    try {
      const url = this.requireUrl(this.embedUrl, "embed");
      const body: any = { input: [text] };
      if (this.embedModel) body.model = this.embedModel;

      const data = await this.fetchJSON(
        `${url}/v1/embeddings`,
        body,
        this.embedApiKey
      );
      return {
        embedding: data.data[0].embedding,
        model: this.embedModel || "remote:" + this.embedUrl,
      };
    } catch (error) {
      console.error("Remote embedding error:", error);
      return null;
    }
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];
    try {
      const url = this.requireUrl(this.embedUrl, "embed");
      const body: any = { input: texts };
      if (this.embedModel) body.model = this.embedModel;

      const data = await this.fetchJSON(
        `${url}/v1/embeddings`,
        body,
        this.embedApiKey
      );
      // Sort by index to ensure correct ordering
      const sorted = (data.data as any[]).sort(
        (a: any, b: any) => a.index - b.index
      );
      return sorted.map((d: any) => ({
        embedding: d.embedding,
        model: this.embedModel || "remote:" + this.embedUrl,
      }));
    } catch (error) {
      console.error("Remote batch embedding error:", error);
      return texts.map(() => null);
    }
  }

  // ===========================================================================
  // Text Generation
  // ===========================================================================

  async generate(
    prompt: string,
    options: GenerateOptions = {}
  ): Promise<GenerateResult | null> {
    try {
      const url = this.requireUrl(this.generateUrl, "generate");
      const body: any = {
        messages: [{ role: "user", content: prompt }],
        max_tokens: options.maxTokens ?? 150,
        temperature: options.temperature ?? 0.7,
        top_p: 0.8,
      };
      if (this.generateModel) body.model = this.generateModel;

      const data = await this.fetchJSON(
        `${url}/v1/chat/completions`,
        body,
        this.generateApiKey
      );
      return {
        text: data.choices[0].message.content,
        model: this.generateModel || "remote:" + this.generateUrl,
        done: true,
      };
    } catch (error) {
      console.error("Remote generate error:", error);
      return null;
    }
  }

  // ===========================================================================
  // Query Expansion (grammar-constrained generation)
  // ===========================================================================

  private static EXPANSION_GRAMMAR = `
root ::= line+
line ::= type ": " content "\\n"
type ::= "lex" | "vec" | "hyde"
content ::= [^\\n]+
`.trim();

  async expandQuery(
    query: string,
    options: { context?: string; includeLexical?: boolean } = {}
  ): Promise<Queryable[]> {
    const includeLexical = options.includeLexical ?? true;
    const prompt = `/no_think Expand this search query: ${query}`;

    try {
      const url = this.requireUrl(this.generateUrl, "generate");
      const body: any = {
        messages: [{ role: "user", content: prompt }],
        max_tokens: 600,
        temperature: 0.7,
        top_p: 0.8,
      };
      if (this.generateModel) body.model = this.generateModel;

      // Only include grammar for llama.cpp compatible servers
      // OpenAI doesn't support GBNF grammar, so we'll parse best-effort
      body.grammar = RemoteLLM.EXPANSION_GRAMMAR;
      body.repeat_penalty = 1.0;
      body.presence_penalty = 0.5;
      body.repeat_last_n = 64;

      const data = await this.fetchJSON(
        `${url}/v1/chat/completions`,
        body,
        this.generateApiKey
      );

      const result = data.choices[0].message.content as string;
      return this.parseExpandedQuery(result, query, includeLexical);
    } catch (error) {
      console.error("Remote query expansion failed:", error);
      const fallback: Queryable[] = [{ type: "vec", text: query }];
      if (includeLexical) fallback.unshift({ type: "lex", text: query });
      return fallback;
    }
  }

  private parseExpandedQuery(
    result: string,
    query: string,
    includeLexical: boolean
  ): Queryable[] {
    const lines = result.trim().split("\n");
    const queryLower = query.toLowerCase();
    const queryTerms = queryLower
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter(Boolean);

    const hasQueryTerm = (text: string): boolean => {
      const lower = text.toLowerCase();
      if (queryTerms.length === 0) return true;
      return queryTerms.some((term) => lower.includes(term));
    };

    const queryables: Queryable[] = lines
      .map((line) => {
        const colonIdx = line.indexOf(":");
        if (colonIdx === -1) return null;
        const type = line.slice(0, colonIdx).trim();
        if (type !== "lex" && type !== "vec" && type !== "hyde") return null;
        const text = line.slice(colonIdx + 1).trim();
        if (!hasQueryTerm(text)) return null;
        return { type: type as QueryType, text };
      })
      .filter((q): q is Queryable => q !== null);

    const filtered = includeLexical
      ? queryables
      : queryables.filter((q) => q.type !== "lex");
    if (filtered.length > 0) return filtered;

    // Fallback
    const fallback: Queryable[] = [
      { type: "hyde", text: `Information about ${query}` },
      { type: "lex", text: query },
      { type: "vec", text: query },
    ];
    return includeLexical
      ? fallback
      : fallback.filter((q) => q.type !== "lex");
  }

  // ===========================================================================
  // Reranking
  // ===========================================================================

  async rerank(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions = {}
  ): Promise<RerankResult> {
    if (documents.length === 0) {
      return { results: [], model: this.rerankModel || "remote:" + this.rerankUrl };
    }

    try {
      const url = this.requireUrl(this.rerankUrl, "rerank");
      const body: any = {
        query,
        documents: documents.map((d) => d.text),
      };
      if (this.rerankModel) body.model = this.rerankModel;

      const data = await this.fetchJSON(
        `${url}/v1/rerank`,
        body,
        this.rerankApiKey
      );

      const results: RerankDocumentResult[] = (data.results as any[])
        .map((r: any) => ({
          file: documents[r.index]!.file,
          score: r.relevance_score,
          index: r.index,
        }))
        .sort((a, b) => b.score - a.score);

      return {
        results,
        model: this.rerankModel || "remote:" + this.rerankUrl,
      };
    } catch (error) {
      console.error("Remote rerank error:", error);
      return {
        results: documents.map((d, i) => ({
          file: d.file,
          score: 0,
          index: i,
        })),
        model: this.rerankModel || "remote:" + this.rerankUrl,
      };
    }
  }

  // ===========================================================================
  // Model Info
  // ===========================================================================

  async modelExists(modelUri: string): Promise<ModelInfo> {
    const urls = [this.embedUrl, this.generateUrl, this.rerankUrl].filter(
      Boolean
    );
    for (const url of urls) {
      try {
        const resp = await fetch(`${url}/health`, { method: "GET" });
        if (resp.ok) {
          return { name: modelUri, exists: true };
        }
      } catch {
        // Server not reachable
      }
    }
    return { name: modelUri, exists: urls.length > 0 };
  }

  // ===========================================================================
  // Lifecycle
  // ===========================================================================

  async unloadIdleResources(): Promise<void> {
    // No-op for remote backend
  }

  async dispose(): Promise<void> {
    this.disposed = true;
  }
}

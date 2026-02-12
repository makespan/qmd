/**
 * remote-llm.ts - Remote LLM backend for QMD using llama.cpp server HTTP API
 *
 * Enables sharing model inference across multiple QMD instances by calling
 * external llama.cpp server(s) instead of loading models in-process.
 *
 * Configure via environment variables:
 *   QMD_EMBED_URL    - Embedding server (e.g., http://localhost:8081)
 *   QMD_GENERATE_URL - Generation server (e.g., http://localhost:8082)
 *   QMD_RERANK_URL   - Reranking server (e.g., http://localhost:8083)
 *
 * Each URL points to a llama.cpp server instance loaded with the appropriate model:
 *   - Embedding:  llama-server -m embeddinggemma-300M-Q8_0.gguf --embedding --port 8081
 *   - Generation: llama-server -m qmd-query-expansion-1.7B-q4_k_m.gguf --port 8082
 *   - Reranking:  llama-server -m qwen3-reranker-0.6b-q8_0.gguf --reranking --port 8083
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
  generateUrl?: string;
  rerankUrl?: string;
};

/**
 * Remote LLM implementation that calls llama.cpp server(s) via HTTP.
 *
 * Implements the same LLMEngine interface as LlamaCpp, enabling transparent
 * substitution when QMD_*_URL environment variables are set.
 */
export class RemoteLLM implements LLMEngine {
  private embedUrl: string;
  private generateUrl: string;
  private rerankUrl: string;
  private disposed = false;

  constructor(config: RemoteLLMConfig) {
    this.embedUrl = (config.embedUrl || "").replace(/\/+$/, "");
    this.generateUrl = (config.generateUrl || "").replace(/\/+$/, "");
    this.rerankUrl = (config.rerankUrl || "").replace(/\/+$/, "");
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

  private async fetchJSON(url: string, body: object): Promise<any> {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`HTTP ${resp.status} from ${url}: ${text}`);
    }
    return resp.json();
  }

  // ===========================================================================
  // Tokenization (via embedding server)
  // ===========================================================================

  async tokenize(text: string): Promise<readonly number[]> {
    const url = this.requireUrl(this.embedUrl, "embed");
    const data = await this.fetchJSON(`${url}/tokenize`, { content: text });
    return data.tokens as number[];
  }

  async detokenize(tokens: readonly any[]): Promise<string> {
    const url = this.requireUrl(this.embedUrl, "embed");
    const data = await this.fetchJSON(`${url}/detokenize`, {
      tokens: Array.from(tokens),
    });
    return data.content as string;
  }

  async countTokens(text: string): Promise<number> {
    const tokens = await this.tokenize(text);
    return tokens.length;
  }

  // ===========================================================================
  // Embeddings
  // ===========================================================================

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    try {
      const url = this.requireUrl(this.embedUrl, "embed");
      const data = await this.fetchJSON(`${url}/v1/embeddings`, {
        input: [text],
      });
      return {
        embedding: data.data[0].embedding,
        model: "remote:" + this.embedUrl,
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
      // llama.cpp server supports batched input natively
      const data = await this.fetchJSON(`${url}/v1/embeddings`, {
        input: texts,
      });
      // Sort by index to ensure correct ordering
      const sorted = (data.data as any[]).sort(
        (a: any, b: any) => a.index - b.index
      );
      return sorted.map((d: any) => ({
        embedding: d.embedding,
        model: "remote:" + this.embedUrl,
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
      const data = await this.fetchJSON(`${url}/v1/chat/completions`, {
        messages: [{ role: "user", content: prompt }],
        max_tokens: options.maxTokens ?? 150,
        temperature: options.temperature ?? 0.7,
        top_k: 20,
        top_p: 0.8,
      });
      return {
        text: data.choices[0].message.content,
        model: "remote:" + this.generateUrl,
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
      const data = await this.fetchJSON(`${url}/v1/chat/completions`, {
        messages: [{ role: "user", content: prompt }],
        grammar: RemoteLLM.EXPANSION_GRAMMAR,
        max_tokens: 600,
        temperature: 0.7,
        top_k: 20,
        top_p: 0.8,
        repeat_penalty: 1.0,
        presence_penalty: 0.5,
        repeat_last_n: 64,
      });

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
      return { results: [], model: "remote:" + this.rerankUrl };
    }

    try {
      const url = this.requireUrl(this.rerankUrl, "rerank");
      const data = await this.fetchJSON(`${url}/v1/rerank`, {
        query,
        documents: documents.map((d) => d.text),
      });

      const results: RerankDocumentResult[] = (data.results as any[])
        .map((r: any) => ({
          file: documents[r.index]!.file,
          score: r.relevance_score,
          index: r.index,
        }))
        .sort((a, b) => b.score - a.score);

      return {
        results,
        model: "remote:" + this.rerankUrl,
      };
    } catch (error) {
      console.error("Remote rerank error:", error);
      // Return documents with zero scores on error
      return {
        results: documents.map((d, i) => ({ file: d.file, score: 0, index: i })),
        model: "remote:" + this.rerankUrl,
      };
    }
  }

  // ===========================================================================
  // Model Info
  // ===========================================================================

  async modelExists(modelUri: string): Promise<ModelInfo> {
    // For remote backends, check if the server is reachable
    const urls = [this.embedUrl, this.generateUrl, this.rerankUrl].filter(Boolean);
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
    // No-op for remote backend - server manages its own resources
  }

  async dispose(): Promise<void> {
    this.disposed = true;
    // No-op for remote backend - we don't own the server lifecycle
  }
}

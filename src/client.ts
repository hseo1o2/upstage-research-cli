import { promises as fs } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';

export type JsonSchema = Record<string, unknown>;

export interface ParseAnchorCoordinate {
  x: number;
  y: number;
}

export interface ParseAnchor {
  markdown: string;
  elementId?: string;
  page?: number;
  category?: string;
  section?: string;
  coordinates?: ParseAnchorCoordinate[];
}

export interface UpstageMessage {
  role: 'system' | 'user' | 'assistant';
  content: string | Array<Record<string, unknown>>;
}

export interface DocumentParseResult {
  markdown: string;
  mode: 'sync' | 'async';
  raw: unknown;
  anchors: ParseAnchor[];
}

export interface InformationExtractionOptions {
  schemaName: string;
  schema: JsonSchema;
  filePath: string;
  instruction: string;
  contextText?: string;
  extraBody?: Record<string, unknown>;
}

export interface ChatCompletionOptions {
  model: string;
  messages: UpstageMessage[];
  temperature?: number;
  maxTokens?: number;
}

export interface StructuredChatOptions extends ChatCompletionOptions {
  schemaName: string;
  schema: JsonSchema;
  extraBody?: Record<string, unknown>;
}

export interface UpstageClientOptions {
  apiKey: string;
  baseUrl?: string;
  pollIntervalMs?: number;
  progress?: (message: string) => void;
  cacheDir?: string;
  cacheEnabled?: boolean;
  cacheOnly?: boolean;
  requestTimeoutMs?: number;
  maxRetries?: number;
}

export function resolveDefaultCacheDir(cwd: string = process.cwd()): string {
  return process.env.UPSTAGE_RESEARCH_CACHE_DIR?.trim() || path.resolve(cwd, '.upstage-research-cache');
}

const DEFAULT_BASE_URL = 'https://api.upstage.ai/v1';
const DEFAULT_V2_BASE_URL = 'https://api.upstage.ai/v2';
const DEFAULT_POLL_INTERVAL_MS = 5_000;
const DEFAULT_CACHE_DIR = resolveDefaultCacheDir();
const MAX_ASYNC_POLLS = 120;
const ASYNC_SIZE_THRESHOLD_BYTES = 15 * 1024 * 1024;
const CACHE_VERSION = 1;
const DEFAULT_INFORMATION_EXTRACT_MODEL =
  process.env.UPSTAGE_INFORMATION_EXTRACT_MODEL?.trim() || 'information-extract';
const DEFAULT_REQUEST_TIMEOUT_MS = Math.max(
  5_000,
  Number.parseInt(process.env.UPSTAGE_RESEARCH_REQUEST_TIMEOUT_MS ?? '', 10) || 90_000,
);
const DEFAULT_MAX_RETRIES = Math.max(
  0,
  Number.parseInt(process.env.UPSTAGE_RESEARCH_MAX_RETRIES ?? '', 10) || 2,
);

class UpstageApiError extends Error {
  public readonly status: number;
  public readonly payload?: unknown;

  public constructor(message: string, status: number, payload?: unknown) {
    super(message);
    this.name = 'UpstageApiError';
    this.status = status;
    this.payload = payload;
  }
}

export class UpstageClient {
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly v2BaseUrl: string;
  private readonly pollIntervalMs: number;
  private readonly progress?: (message: string) => void;
  private readonly cacheDir: string;
  private readonly cacheEnabled: boolean;
  private readonly cacheOnly: boolean;
  private readonly requestTimeoutMs: number;
  private readonly maxRetries: number;
  private readonly inflight = new Map<string, Promise<unknown>>();

  public constructor(options: UpstageClientOptions) {
    this.apiKey = options.apiKey;
    this.baseUrl = options.baseUrl ?? DEFAULT_BASE_URL;
    this.v2BaseUrl = DEFAULT_V2_BASE_URL;
    this.pollIntervalMs = options.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS;
    this.progress = options.progress;
    this.cacheDir = options.cacheDir ?? DEFAULT_CACHE_DIR;
    this.cacheEnabled =
      options.cacheEnabled
      ?? !['1', 'true', 'yes'].includes((process.env.UPSTAGE_RESEARCH_DISABLE_CACHE ?? '').trim().toLowerCase());
    this.cacheOnly =
      options.cacheOnly
      ?? ['1', 'true', 'yes'].includes((process.env.UPSTAGE_RESEARCH_CACHE_ONLY ?? '').trim().toLowerCase());
    this.requestTimeoutMs = options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;
    this.maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES;
  }

  public async parseDocument(filePath: string, preferAsync?: boolean): Promise<DocumentParseResult> {
    const resolvedPath = path.resolve(filePath);
    const stats = await fs.stat(resolvedPath);
    const shouldPreferAsync = preferAsync ?? stats.size >= ASYNC_SIZE_THRESHOLD_BYTES;
    const cacheKey = await this.buildCacheKey({
      version: CACHE_VERSION,
      operation: 'document-parse',
      filePath: resolvedPath,
      fileHash: await hashFile(resolvedPath),
      shouldPreferAsync,
    });
    const cached = await this.readCache<unknown>('document-parse', cacheKey);
    if (cached) {
      this.progress?.(`Cache hit: document parse for ${path.basename(resolvedPath)}`);
      return this.hydrateDocumentParseResult(cached);
    }
    if (this.cacheOnly) {
      throw new Error(`Cache miss: document parse for ${path.basename(resolvedPath)} (cache-only mode).`);
    }

    return await this.memoizeInFlight(`document-parse:${cacheKey}`, async () => {
      let result: DocumentParseResult;

      if (shouldPreferAsync) {
        try {
          result = await this.parseDocumentAsync(resolvedPath);
          await this.writeCache('document-parse', cacheKey, result);
          return result;
        } catch (error) {
          this.progress?.(
            `Async document parse unavailable for ${path.basename(resolvedPath)}. Falling back to sync parsing.`,
          );
        }
      }

      try {
        result = await this.parseDocumentSync(resolvedPath);
        await this.writeCache('document-parse', cacheKey, result);
        return result;
      } catch (error) {
        if (this.shouldFallbackToAsync(error)) {
          this.progress?.(
            `Sync document parse timed out or was rejected for ${path.basename(resolvedPath)}. Retrying with async parsing.`,
          );
          result = await this.parseDocumentAsync(resolvedPath);
          await this.writeCache('document-parse', cacheKey, result);
          return result;
        }

        throw error;
      }
    });
  }

  public async informationExtract<T>(options: InformationExtractionOptions): Promise<T> {
    const resolvedPath = path.resolve(options.filePath);
    const cacheKey = await this.buildCacheKey({
      version: CACHE_VERSION,
      operation: 'information-extract',
      model: DEFAULT_INFORMATION_EXTRACT_MODEL,
      filePath: resolvedPath,
      fileHash: await hashFile(resolvedPath),
      schemaName: options.schemaName,
      schema: options.schema,
      instruction: options.instruction,
      contextText: options.contextText ?? '',
      extraBody: options.extraBody ?? {},
    });
    const cached = await this.readCache<T>('information-extract', cacheKey);
    if (cached) {
      this.progress?.(`Cache hit: information extraction for ${path.basename(resolvedPath)}`);
      return cached;
    }
    if (this.cacheOnly) {
      throw new Error(`Cache miss: information extraction for ${path.basename(resolvedPath)} (cache-only mode).`);
    }

    return await this.memoizeInFlight(`information-extract:${cacheKey}`, async () => {
      const fileId = await this.uploadUserDataFile(resolvedPath);
      const promptParts = [options.instruction.trim()];
      if (options.contextText?.trim()) {
        promptParts.push(`Context:\n${options.contextText.trim()}`);
      }

      const body = {
        model: DEFAULT_INFORMATION_EXTRACT_MODEL,
        input: [
          {
            role: 'user',
            content: [
              {
                type: 'input_text',
                text: promptParts.join('\n\n'),
              },
              {
                type: 'input_file',
                file_id: fileId,
              },
            ],
          },
        ],
        text: {
          format: {
            type: 'json_schema',
            name: options.schemaName,
            schema: options.schema,
            strict: true,
          },
        },
        ...(options.extraBody ?? {}),
      };

      const response = await this.postJsonV2('/responses', body);
      const parsed = this.parseResponsesJson<T>(response);
      await this.writeCache('information-extract', cacheKey, parsed);
      return parsed;
    });
  }

  public async chatCompletion(options: ChatCompletionOptions): Promise<string> {
    const cacheKey = await this.buildCacheKey({
      version: CACHE_VERSION,
      operation: 'chat-completion',
      model: options.model,
      messages: options.messages,
      temperature: options.temperature ?? 0.2,
      maxTokens: options.maxTokens ?? null,
    });
    const cached = await this.readCache<string>('chat-completion', cacheKey);
    if (cached) {
      this.progress?.(`Cache hit: chat completion (${options.model})`);
      return cached;
    }
    if (this.cacheOnly) {
      throw new Error(`Cache miss: chat completion (${options.model}) (cache-only mode).`);
    }

    return await this.memoizeInFlight(`chat-completion:${cacheKey}`, async () => {
      const response = await this.postJson('/chat/completions', {
        model: options.model,
        messages: options.messages,
        temperature: options.temperature ?? 0.2,
        max_tokens: options.maxTokens,
      });

      const content = this.extractCompletionText(response);
      if (!content) {
        throw new Error('Chat completion response did not include message content.');
      }

      const parsed = content.trim();
      await this.writeCache('chat-completion', cacheKey, parsed);
      return parsed;
    });
  }

  public async chatStructured<T>(options: StructuredChatOptions): Promise<T> {
    const cacheKey = await this.buildCacheKey({
      version: CACHE_VERSION,
      operation: 'chat-structured',
      model: options.model,
      schemaName: options.schemaName,
      schema: options.schema,
      messages: options.messages,
      temperature: options.temperature ?? 0.2,
      maxTokens: options.maxTokens ?? null,
      extraBody: options.extraBody ?? {},
    });
    const cached = await this.readCache<T>('chat-structured', cacheKey);
    if (cached) {
      this.progress?.(`Cache hit: structured chat (${options.model}/${options.schemaName})`);
      return cached;
    }
    if (this.cacheOnly) {
      throw new Error(
        `Cache miss: structured chat (${options.model}/${options.schemaName}) (cache-only mode).`,
      );
    }

    return await this.memoizeInFlight(`chat-structured:${cacheKey}`, async () => {
      const response = await this.postJson('/chat/completions', {
        model: options.model,
        messages: options.messages,
        temperature: options.temperature ?? 0.2,
        max_tokens: options.maxTokens,
        response_format: {
          type: 'json_schema',
          json_schema: {
            name: options.schemaName,
            strict: true,
            schema: options.schema,
          },
        },
        ...(options.extraBody ?? {}),
      });

      const parsed = this.parseJsonCompletion<T>(response);
      await this.writeCache('chat-structured', cacheKey, parsed);
      return parsed;
    });
  }

  private async parseDocumentSync(filePath: string): Promise<DocumentParseResult> {
    const form = await this.buildDocumentForm(filePath);
    const response = await this.fetchJson('/document-digitization', {
      method: 'POST',
      body: form,
    });
    return this.normalizeParsePayload(response, 'sync');
  }

  private async parseDocumentAsync(filePath: string): Promise<DocumentParseResult> {
    const form = await this.buildDocumentForm(filePath);
    const submission = await this.fetchJson('/document-digitization/async', {
      method: 'POST',
      body: form,
    });

    const requestId = this.pickFirstString(submission, ['request_id', 'id']);
    if (!requestId) {
      return this.normalizeParsePayload(submission, 'async');
    }

    return await this.pollAsyncParse(requestId);
  }

  private async pollAsyncParse(requestId: string): Promise<DocumentParseResult> {
    for (let attempt = 0; attempt < MAX_ASYNC_POLLS; attempt += 1) {
      if (attempt > 0) {
        await delay(this.pollIntervalMs);
      }

      const statusPayload = await this.fetchJson(
        `/document-digitization/requests/${encodeURIComponent(requestId)}`,
        { method: 'GET' },
      );
      const status = String((statusPayload as Record<string, unknown>).status ?? '').toLowerCase();
      const completedPages = (statusPayload as Record<string, unknown>).completed_pages;
      const totalPages = (statusPayload as Record<string, unknown>).total_pages;

      this.progress?.(
        `Document parse ${requestId}: ${status || 'processing'} (${completedPages ?? 0}/${totalPages ?? '?'})`,
      );

      if (['completed', 'succeeded', 'done'].includes(status)) {
        return await this.materializeAsyncParse(statusPayload);
      }

      if (['failed', 'error', 'cancelled'].includes(status)) {
        const failureMessage = this.pickFirstString(statusPayload, ['failure_message', 'message']);
        throw new Error(`Async document parse failed: ${failureMessage ?? 'unknown failure'}`);
      }
    }

    throw new Error(`Timed out while waiting for async document parse request ${requestId}.`);
  }

  private async materializeAsyncParse(statusPayload: unknown): Promise<DocumentParseResult> {
    const inlineMarkdown = this.extractMarkdown(statusPayload);
    if (inlineMarkdown) {
      return this.hydrateDocumentParseResult({
        markdown: inlineMarkdown,
        mode: 'async',
        raw: statusPayload,
      });
    }

    const directDownloadUrl = this.pickFirstString(statusPayload, ['download_url']);
    const batchDownloadUrls = Array.isArray((statusPayload as Record<string, unknown>).batches)
      ? ((statusPayload as Record<string, unknown>).batches as Array<Record<string, unknown>>)
          .slice()
          .sort(
            (left, right) =>
              Number(left.start_page ?? 0) - Number(right.start_page ?? 0),
          )
          .map((batch) => this.pickFirstString(batch, ['download_url']))
          .filter((value): value is string => Boolean(value))
      : [];

    const downloadUrls = directDownloadUrl ? [directDownloadUrl, ...batchDownloadUrls] : batchDownloadUrls;

    if (downloadUrls.length === 0) {
      throw new Error('Async document parse completed without a downloadable result.');
    }

    const chunks: string[] = [];
    const payloads: unknown[] = [];

    for (const downloadUrl of downloadUrls) {
      const payload = await this.fetchDownloadPayload(downloadUrl);
      payloads.push(payload);

      const markdown = this.extractMarkdown(payload) || (typeof payload === 'string' ? payload : '');
      if (markdown.trim()) {
        chunks.push(markdown.trim());
      }
    }

    if (chunks.length === 0) {
      throw new Error('Async document parse download did not include markdown content.');
    }

    return this.hydrateDocumentParseResult({
      markdown: chunks.join('\n\n'),
      mode: 'async',
      raw: {
        status: statusPayload,
        downloads: payloads,
      },
    });
  }

  private async fetchDownloadPayload(downloadUrl: string): Promise<unknown> {
    const tryFetch = async (authorized: boolean): Promise<Response> =>
      fetch(downloadUrl, {
        method: 'GET',
        headers: authorized ? this.buildJsonHeaders() : undefined,
      });

    let response = await tryFetch(false);
    if (!response.ok && [401, 403].includes(response.status)) {
      response = await tryFetch(true);
    }

    const text = await response.text();
    if (!response.ok) {
      throw new UpstageApiError(
        `Upstage download request failed with status ${response.status}`,
        response.status,
        text,
      );
    }

    try {
      return JSON.parse(text);
    } catch {
      return text;
    }
  }

  private async buildDocumentForm(filePath: string): Promise<FormData> {
    const form = new FormData();
    const fileName = path.basename(filePath);
    const fileBuffer = await fs.readFile(filePath);

    form.append(
      'document',
      new Blob([fileBuffer], {
        type: inferMimeType(fileName),
      }),
      fileName,
    );
    form.append('model', 'document-parse');
    form.append('ocr', 'auto');
    form.append('output_formats', "['markdown']");

    return form;
  }

  private normalizeParsePayload(payload: unknown, mode: 'sync' | 'async'): DocumentParseResult {
    const markdown = this.extractMarkdown(payload);
    if (!markdown) {
      throw new Error('Document parse response did not include markdown content.');
    }

    return this.hydrateDocumentParseResult({
      markdown,
      mode,
      raw: payload,
    });
  }

  private hydrateDocumentParseResult(payload: unknown): DocumentParseResult {
    const candidate = payload && typeof payload === 'object' ? payload as Record<string, unknown> : {};
    const raw = Object.prototype.hasOwnProperty.call(candidate, 'raw') ? candidate.raw : payload;
    const markdown =
      (typeof candidate.markdown === 'string' && candidate.markdown.trim()
        ? candidate.markdown.trim()
        : this.extractMarkdown(raw))
      ?? '';
    if (!markdown) {
      throw new Error('Document parse result did not include markdown content.');
    }

    const mode = candidate.mode === 'async' ? 'async' : 'sync';
    const existingAnchors = Array.isArray(candidate.anchors) ? candidate.anchors : [];

    return {
      markdown,
      mode,
      raw,
      anchors: existingAnchors.length > 0 ? normalizeParseAnchors(existingAnchors) : this.extractParseAnchors(raw),
    };
  }

  private extractParseAnchors(raw: unknown): ParseAnchor[] {
    const anchors = collectParseElementPayloads(raw)
      .map((element, index) => normalizeParseAnchor(element, index))
      .filter((anchor): anchor is ParseAnchor => Boolean(anchor))
      .sort(compareParseAnchors);

    let currentSection: string | undefined;
    return anchors.map((anchor) => {
      if (isSectionAnchor(anchor)) {
        currentSection = stripHeadingMarkdown(anchor.markdown) || currentSection;
        return {
          ...anchor,
          section: currentSection,
        };
      }

      return currentSection ? { ...anchor, section: anchor.section ?? currentSection } : anchor;
    });
  }

  private extractMarkdown(payload: unknown): string | undefined {
    if (!payload || typeof payload !== 'object') {
      return undefined;
    }

    const content = (payload as Record<string, unknown>).content;
    if (content && typeof content === 'object') {
      const markdown = (content as Record<string, unknown>).markdown;
      if (typeof markdown === 'string' && markdown.trim()) {
        return markdown.trim();
      }
    }

    return undefined;
  }

  private extractCompletionText(payload: unknown): string | undefined {
    if (!payload || typeof payload !== 'object') {
      return undefined;
    }

    const choices = (payload as Record<string, unknown>).choices;
    if (!Array.isArray(choices) || choices.length === 0) {
      return undefined;
    }

    const firstChoice = choices[0] as Record<string, unknown>;
    const message = firstChoice.message as Record<string, unknown> | undefined;
    const content = message?.content;

    if (typeof content === 'string') {
      return content;
    }

    if (Array.isArray(content)) {
      return content
        .map((part) => {
          if (typeof part === 'string') {
            return part;
          }

          if (part && typeof part === 'object') {
            const typedPart = part as Record<string, unknown>;
            if (typeof typedPart.text === 'string') {
              return typedPart.text;
            }

            if (typeof typedPart.content === 'string') {
              return typedPart.content;
            }
          }

          return '';
        })
        .join('');
    }

    return undefined;
  }

  private extractResponsesOutputText(payload: unknown): string | undefined {
    if (!payload || typeof payload !== 'object') {
      return undefined;
    }

    const output = (payload as Record<string, unknown>).output;
    if (!Array.isArray(output)) {
      return undefined;
    }

    for (const item of output) {
      if (!item || typeof item !== 'object') {
        continue;
      }

      const content = (item as Record<string, unknown>).content;
      if (!Array.isArray(content)) {
        continue;
      }

      for (const contentItem of content) {
        if (!contentItem || typeof contentItem !== 'object') {
          continue;
        }

        const text = (contentItem as Record<string, unknown>).text;
        if (typeof text === 'string' && text.trim()) {
          return text;
        }
      }
    }

    return undefined;
  }

  private parseJsonCompletion<T>(payload: unknown): T {
    const content = this.extractCompletionText(payload);
    if (!content) {
      throw new Error('Information extraction response did not include JSON content.');
    }

    const parsed = parseJsonLoose(content);
    return parsed as T;
  }

  private parseResponsesJson<T>(payload: unknown): T {
    const content = this.extractResponsesOutputText(payload);
    if (!content) {
      throw new Error('Information extraction response did not include structured text output.');
    }

    const parsed = parseJsonLoose(content);
    return parsed as T;
  }

  private async postJson(pathname: string, body: Record<string, unknown>): Promise<unknown> {
    return await this.fetchJson(pathname, {
      method: 'POST',
      headers: this.buildJsonHeaders(),
      body: JSON.stringify(body),
    });
  }

  private async postJsonV2(pathname: string, body: Record<string, unknown>): Promise<unknown> {
    return await this.fetchJsonAbsolute(`${this.v2BaseUrl}${pathname}`, {
      method: 'POST',
      headers: this.buildJsonHeaders(),
      body: JSON.stringify(body),
    });
  }

  private async fetchJson(pathname: string, init: RequestInit): Promise<unknown> {
    return await this.fetchJsonAbsolute(`${this.baseUrl}${pathname}`, init);
  }

  private async fetchJsonAbsolute(url: string, init: RequestInit): Promise<unknown> {
    const headers =
      init.body instanceof FormData
        ? this.buildAuthHeaders()
        : {
            ...this.buildJsonHeaders(),
            ...(init.headers ?? {}),
          };
    const pathname = safePathname(url);

    for (let attempt = 0; attempt <= this.maxRetries; attempt += 1) {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), this.requestTimeoutMs);

      try {
        const response = await fetch(url, {
          ...init,
          headers,
          signal: controller.signal,
        });

        const text = await response.text();
        let payload: unknown = text;

        try {
          payload = text ? JSON.parse(text) : {};
        } catch {
          payload = text;
        }

        if (!response.ok) {
          const message =
            this.pickFirstString(payload, ['error.message', 'message']) ??
            `Upstage request failed with status ${response.status}`;
          const error = new UpstageApiError(message, response.status, payload);
          if (attempt < this.maxRetries && shouldRetryStatus(response.status)) {
            this.progress?.(
              `Retrying Upstage request ${pathname} after status ${response.status} (${attempt + 1}/${this.maxRetries + 1})`,
            );
            await delay(backoffDelayMs(attempt));
            continue;
          }

          throw error;
        }

        return payload;
      } catch (error) {
        if (attempt < this.maxRetries && isRetryableFetchError(error)) {
          this.progress?.(
            `Retrying Upstage request ${pathname} after transient failure (${attempt + 1}/${this.maxRetries + 1})`,
          );
          await delay(backoffDelayMs(attempt));
          continue;
        }

        throw error;
      } finally {
        clearTimeout(timeout);
      }
    }

    throw new Error(`Upstage request failed after ${this.maxRetries + 1} attempts: ${pathname}`);
  }

  private async uploadUserDataFile(filePath: string): Promise<string> {
    this.progress?.(`Uploading ${path.basename(filePath)} for information extraction`);
    const form = new FormData();
    const fileName = path.basename(filePath);
    const fileBuffer = await fs.readFile(filePath);

    form.append(
      'file',
      new Blob([fileBuffer], {
        type: inferMimeType(fileName),
      }),
      fileName,
    );
    form.append('purpose', 'user_data');

    const response = await this.fetchJsonAbsolute(`${this.v2BaseUrl}/files`, {
      method: 'POST',
      body: form,
    });
    const fileId = this.pickFirstString(response, ['id']);
    if (!fileId) {
      throw new Error('Upstage file upload did not return a file id.');
    }

    return fileId;
  }

  private buildAuthHeaders(): HeadersInit {
    return {
      Authorization: `Bearer ${this.apiKey}`,
    };
  }

  private buildJsonHeaders(): HeadersInit {
    return {
      Authorization: `Bearer ${this.apiKey}`,
      'Content-Type': 'application/json',
    };
  }

  private pickFirstString(payload: unknown, keys: string[]): string | undefined {
    for (const key of keys) {
      const value = readDotPath(payload, key);
      if (typeof value === 'string' && value.trim()) {
        return value.trim();
      }
    }

    return undefined;
  }

  private shouldFallbackToAsync(error: unknown): boolean {
    if (!(error instanceof UpstageApiError)) {
      return false;
    }

    return [408, 413, 422, 429, 500, 502, 503, 504].includes(error.status);
  }

  private async buildCacheKey(payload: Record<string, unknown>): Promise<string> {
    return hashString(stableStringify(payload));
  }

  private async readCache<T>(namespace: string, key: string): Promise<T | undefined> {
    if (!this.cacheEnabled) {
      return undefined;
    }

    const cachePath = path.join(this.cacheDir, namespace, `${key}.json`);
    try {
      const raw = await fs.readFile(cachePath, 'utf8');
      return JSON.parse(raw) as T;
    } catch {
      return undefined;
    }
  }

  private async writeCache(namespace: string, key: string, value: unknown): Promise<void> {
    if (!this.cacheEnabled) {
      return;
    }

    const cachePath = path.join(this.cacheDir, namespace, `${key}.json`);
    await fs.mkdir(path.dirname(cachePath), { recursive: true });
    await fs.writeFile(cachePath, JSON.stringify(value), 'utf8');
  }

  private async memoizeInFlight<T>(key: string, factory: () => Promise<T>): Promise<T> {
    const existing = this.inflight.get(key);
    if (existing) {
      return await existing as T;
    }

    const promise = factory()
      .finally(() => {
        this.inflight.delete(key);
      });

    this.inflight.set(key, promise);
    return await promise;
  }
}

function parseJsonLoose(input: string): unknown {
  const trimmed = input.trim();
  const directCandidates = [
    trimmed,
    trimmed.replace(/^```json\s*/i, '').replace(/^```\s*/i, '').replace(/\s*```$/, ''),
  ];

  for (const candidate of directCandidates) {
    try {
      return JSON.parse(candidate);
    } catch {
      continue;
    }
  }

  const firstBrace = Math.min(
    ...['{', '[']
      .map((token) => trimmed.indexOf(token))
      .filter((index) => index >= 0),
  );

  if (Number.isFinite(firstBrace)) {
    const lastCurly = trimmed.lastIndexOf('}');
    const lastSquare = trimmed.lastIndexOf(']');
    const lastIndex = Math.max(lastCurly, lastSquare);

    if (lastIndex > firstBrace) {
      return JSON.parse(trimmed.slice(firstBrace, lastIndex + 1));
    }
  }

  throw new Error('Failed to parse JSON response from Upstage.');
}

function readDotPath(payload: unknown, pathExpression: string): unknown {
  return pathExpression.split('.').reduce<unknown>((current, key) => {
    if (!current || typeof current !== 'object') {
      return undefined;
    }

    return (current as Record<string, unknown>)[key];
  }, payload);
}

function inferMimeType(fileName: string): string {
  const extension = path.extname(fileName).toLowerCase();

  switch (extension) {
    case '.pdf':
      return 'application/pdf';
    case '.md':
      return 'text/markdown';
    case '.txt':
      return 'text/plain';
    case '.png':
      return 'image/png';
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg';
    case '.docx':
      return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
    case '.pptx':
      return 'application/vnd.openxmlformats-officedocument.presentationml.presentation';
    case '.xlsx':
      return 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
    case '.hwp':
      return 'application/x-hwp';
    case '.hwpx':
      return 'application/x-hwp+zip';
    default:
      return 'application/octet-stream';
  }
}

async function hashFile(filePath: string): Promise<string> {
  const content = await fs.readFile(filePath);
  return hashString(content);
}

function hashString(input: string | Uint8Array): string {
  return createHash('sha256').update(input).digest('hex');
}

function stableStringify(value: unknown): string {
  if (value === null || typeof value !== 'object') {
    return JSON.stringify(value);
  }

  if (Array.isArray(value)) {
    return `[${value.map((item) => stableStringify(item)).join(',')}]`;
  }

  const entries = Object.entries(value as Record<string, unknown>)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, entryValue]) => `${JSON.stringify(key)}:${stableStringify(entryValue)}`);

  return `{${entries.join(',')}}`;
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function collectParseElementPayloads(raw: unknown): unknown[] {
  const payloads: unknown[] = [];
  const visit = (candidate: unknown) => {
    if (!candidate || typeof candidate !== 'object') {
      return;
    }

    const elements = (candidate as Record<string, unknown>).elements;
    if (Array.isArray(elements)) {
      payloads.push(...elements);
    }

    const downloads = (candidate as Record<string, unknown>).downloads;
    if (Array.isArray(downloads)) {
      downloads.forEach(visit);
    }
  };

  visit(raw);
  return payloads;
}

function normalizeParseAnchor(candidate: unknown, index: number): ParseAnchor | undefined {
  if (!candidate || typeof candidate !== 'object') {
    return undefined;
  }

  const element = candidate as Record<string, unknown>;
  const markdown = extractAnchorMarkdown(element);
  if (!markdown) {
    return undefined;
  }

  return {
    markdown,
    elementId: stringifyElementId(element.id, index),
    page: normalizeNumber(element.page),
    category: typeof element.category === 'string' && element.category.trim() ? element.category.trim() : undefined,
    coordinates: normalizeCoordinates(element.coordinates),
  };
}

function normalizeParseAnchors(candidates: unknown[]): ParseAnchor[] {
  return candidates
    .map((candidate, index) => normalizeStoredParseAnchor(candidate) ?? normalizeParseAnchor(candidate, index))
    .filter((anchor): anchor is ParseAnchor => Boolean(anchor))
    .sort(compareParseAnchors);
}

function normalizeStoredParseAnchor(candidate: unknown): ParseAnchor | undefined {
  if (!candidate || typeof candidate !== 'object') {
    return undefined;
  }

  const value = candidate as Record<string, unknown>;
  const markdown = typeof value.markdown === 'string' && value.markdown.trim() ? value.markdown.trim() : '';
  if (!markdown) {
    return undefined;
  }

  return {
    markdown,
    elementId: typeof value.elementId === 'string' && value.elementId.trim() ? value.elementId.trim() : undefined,
    page: normalizeNumber(value.page),
    category: typeof value.category === 'string' && value.category.trim() ? value.category.trim() : undefined,
    section: typeof value.section === 'string' && value.section.trim() ? value.section.trim() : undefined,
    coordinates: normalizeCoordinates(value.coordinates),
  };
}

function extractAnchorMarkdown(element: Record<string, unknown>): string {
  const content = element.content;
  if (content && typeof content === 'object') {
    const markdown = (content as Record<string, unknown>).markdown;
    if (typeof markdown === 'string' && markdown.trim()) {
      return markdown.trim();
    }

    const text = (content as Record<string, unknown>).text;
    if (typeof text === 'string' && text.trim()) {
      return text.trim();
    }
  }

  return '';
}

function stringifyElementId(value: unknown, fallbackIndex: number): string {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return String(value);
  }
  if (typeof value === 'string' && value.trim()) {
    return value.trim();
  }

  return `anchor-${fallbackIndex + 1}`;
}

function normalizeNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function normalizeCoordinates(value: unknown): ParseAnchorCoordinate[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }

  const normalized = value
    .map((item) => {
      if (!item || typeof item !== 'object') {
        return undefined;
      }

      const coordinate = item as Record<string, unknown>;
      const x = normalizeNumber(coordinate.x);
      const y = normalizeNumber(coordinate.y);
      if (x === undefined || y === undefined) {
        return undefined;
      }

      return { x, y };
    })
    .filter((item): item is ParseAnchorCoordinate => Boolean(item));

  return normalized.length > 0 ? normalized : undefined;
}

function compareParseAnchors(left: ParseAnchor, right: ParseAnchor): number {
  const pageDiff = (left.page ?? 0) - (right.page ?? 0);
  if (pageDiff !== 0) {
    return pageDiff;
  }

  const leftId = Number.parseInt(left.elementId ?? '', 10);
  const rightId = Number.parseInt(right.elementId ?? '', 10);
  if (Number.isFinite(leftId) && Number.isFinite(rightId) && leftId !== rightId) {
    return leftId - rightId;
  }

  return (left.elementId ?? '').localeCompare(right.elementId ?? '');
}

function isSectionAnchor(anchor: ParseAnchor): boolean {
  const category = (anchor.category ?? '').toLowerCase();
  const markdown = anchor.markdown.trim();
  if (!(category.startsWith('heading') || /^#+\s+/.test(markdown))) {
    return false;
  }

  const heading = stripHeadingMarkdown(markdown);
  if (!heading || heading.length > 120) {
    return false;
  }

  const tokenCount = heading.split(/\s+/).filter(Boolean).length;
  if (/^[a-z]/.test(heading) && tokenCount > 6) {
    return false;
  }
  if (/[.!?]$/.test(heading) && tokenCount > 6) {
    return false;
  }

  return true;
}

function stripHeadingMarkdown(value: string): string {
  return value
    .replace(/^#+\s*/, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function shouldRetryStatus(status: number): boolean {
  return [408, 409, 425, 429, 500, 502, 503, 504].includes(status);
}

function isRetryableFetchError(error: unknown): boolean {
  if (error instanceof UpstageApiError) {
    return shouldRetryStatus(error.status);
  }
  if (error instanceof Error) {
    return error.name === 'AbortError' || /fetch failed|network|timeout|timed out|econnreset|enotfound/i.test(error.message);
  }

  return false;
}

function backoffDelayMs(attempt: number): number {
  return Math.min(10_000, 750 * 2 ** attempt);
}

function safePathname(url: string): string {
  try {
    return new URL(url).pathname;
  } catch {
    return url;
  }
}

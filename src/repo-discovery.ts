import { createHash } from 'node:crypto';
import { promises as fs } from 'node:fs';
import path from 'node:path';

import { resolveDefaultCacheDir } from './client';

export interface RepositoryCandidate {
  url: string;
  fullName: string;
  description?: string;
  stars?: number;
  source: 'document' | 'github_search';
  confidence: 'high' | 'medium' | 'low';
  reason: string;
}

export interface RepositoryDiscoveryResult {
  titleGuess: string;
  arxivId?: string;
  candidates: RepositoryCandidate[];
  searchQueries: string[];
  searchAttempted: boolean;
}

interface RepositorySearchItem {
  full_name?: string;
  html_url?: string;
  description?: string;
  stargazers_count?: number;
  homepage?: string;
}

export interface DiscoverRepositoriesOptions {
  filePath: string;
  markdown: string;
  cacheOnly?: boolean;
  progress?: (message: string) => void;
  cacheDir?: string;
}

const CACHE_VERSION = 1;
const SEARCH_RESULT_LIMIT = 5;
const REQUEST_TIMEOUT_MS = 15_000;

export async function discoverPaperRepositories(
  options: DiscoverRepositoriesOptions,
): Promise<RepositoryDiscoveryResult> {
  const titleGuess = inferPaperTitle(options.filePath, options.markdown);
  const arxivId = extractArxivId(options.filePath, options.markdown);
  const inlineCandidates = extractInlineGitHubCandidates(options.markdown);
  const cacheDir = options.cacheDir ?? resolveDefaultCacheDir(path.dirname(path.resolve(options.filePath)));
  const searchQueries = buildSearchQueries(titleGuess, arxivId);

  if (inlineCandidates.length > 0) {
    return {
      titleGuess,
      arxivId,
      candidates: inlineCandidates,
      searchQueries,
      searchAttempted: false,
    };
  }

  const cacheKey = buildCacheKey(titleGuess, arxivId, options.filePath);
  const cachePath = path.join(cacheDir, 'repo-discovery', `${cacheKey}.json`);
  const cached = await readJsonCache<RepositoryDiscoveryResult>(cachePath);
  if (cached) {
    options.progress?.(`Cache hit: repository discovery for ${path.basename(options.filePath)}`);
    return cached;
  }

  if (options.cacheOnly) {
    return {
      titleGuess,
      arxivId,
      candidates: [],
      searchQueries,
      searchAttempted: false,
    };
  }

  const candidates = await searchGitHubRepositories(titleGuess, arxivId, searchQueries, options.progress);
  const result: RepositoryDiscoveryResult = {
    titleGuess,
    arxivId,
    candidates,
    searchQueries,
    searchAttempted: searchQueries.length > 0,
  };
  await writeJsonCache(cachePath, result);
  return result;
}

function inferPaperTitle(filePath: string, markdown: string): string {
  const headingMatch = markdown.match(/^#\s+(.+)$/m);
  if (headingMatch?.[1]?.trim() && !isGenericSectionHeading(headingMatch[1])) {
    return headingMatch[1].trim();
  }

  const stemTitle = path.basename(filePath, path.extname(filePath)).replace(/[-_]\d{4}\.\d{4,5}(v\d+)?$/i, '').trim();
  if (stemTitle && /[A-Za-z]/.test(stemTitle)) {
    return stemTitle;
  }

  const lines = markdown
    .replace(/\r\n/g, '\n')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .filter((line) => !/^\[.*\]$/.test(line))
    .filter((line) => !/^(arxiv:|published|abstract|introduction|#)/i.test(line))
    .filter((line) => !isGenericSectionHeading(line))
    .slice(0, 12);

  for (const line of lines) {
    if (line.length >= 12 && /[A-Za-z]/.test(line) && !/@/.test(line)) {
      return line.replace(/\s+/g, ' ').trim();
    }
  }

  return path.basename(filePath, path.extname(filePath)).replace(/[-_]\d{4}\.\d{4,5}(v\d+)?$/i, '').trim();
}

function isGenericSectionHeading(value: string): boolean {
  return /^\d+(\.\d+)*\s+(abstract|introduction|related work|background|experiments?|evaluation|results?|method|methods|conclusion|discussion)\b/i.test(value)
    || /^(abstract|introduction|related work|background|experiments?|evaluation|results?|method|methods|conclusion|discussion)\b/i.test(value);
}

function extractArxivId(filePath: string, markdown: string): string | undefined {
  const fileMatch = path.basename(filePath).match(/(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+\-\d{4}-\d{5,})/i);
  if (fileMatch?.[1]) {
    return fileMatch[1].replace(/v\d+$/i, '');
  }

  const markdownMatch = markdown.match(/arXiv:(\d{4}\.\d{4,5})(?:v\d+)?/i);
  return markdownMatch?.[1];
}

function extractInlineGitHubCandidates(markdown: string): RepositoryCandidate[] {
  const matches = Array.from(
    new Set(
      Array.from(markdown.matchAll(/https?:\/\/github\.com\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+/g)).map((match) => match[0]),
    ),
  );

  return matches.slice(0, 5).map((url) => ({
    url,
    fullName: url.replace(/^https?:\/\/github\.com\//, ''),
    source: 'document',
    confidence: 'high',
    reason: 'Repository URL was found directly in the parsed paper.',
  }));
}

function buildSearchQueries(titleGuess: string, arxivId?: string): string[] {
  const queries = new Set<string>();
  const compactTitle = titleGuess.replace(/\s+/g, ' ').trim();
  if (compactTitle) {
    queries.add(`"${compactTitle}" in:name,description,readme`);
    const titleTokens = compactTitle
      .toLowerCase()
      .replace(/[^a-z0-9\s-]+/g, ' ')
      .split(/\s+/)
      .filter((token) => token.length >= 3)
      .slice(0, 6);
    if (titleTokens.length > 0) {
      queries.add(`${titleTokens.join(' ')} in:name,description,readme`);
    }
  }
  if (arxivId) {
    queries.add(`${arxivId} in:name,description,readme`);
  }

  return Array.from(queries).slice(0, 3);
}

async function searchGitHubRepositories(
  titleGuess: string,
  arxivId: string | undefined,
  queries: string[],
  progress?: (message: string) => void,
): Promise<RepositoryCandidate[]> {
  const allCandidates: RepositoryCandidate[] = [];
  const seen = new Set<string>();

  for (const query of queries) {
    progress?.(`Searching GitHub repositories for "${titleGuess}"`);
    const url = new URL('https://api.github.com/search/repositories');
    url.searchParams.set('q', query);
    url.searchParams.set('sort', 'stars');
    url.searchParams.set('order', 'desc');
    url.searchParams.set('per_page', String(SEARCH_RESULT_LIMIT));

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
      const response = await fetch(url, {
        headers: {
          Accept: 'application/vnd.github+json',
          'User-Agent': 'upstage-research-cli',
          ...(process.env.GITHUB_TOKEN?.trim()
            ? { Authorization: `Bearer ${process.env.GITHUB_TOKEN.trim()}` }
            : {}),
        },
        signal: controller.signal,
      });

      if (!response.ok) {
        continue;
      }

      const payload = await response.json() as { items?: RepositorySearchItem[] };
      for (const item of payload.items ?? []) {
        const normalizedUrl = item.html_url?.trim();
        const fullName = item.full_name?.trim();
        if (!normalizedUrl || !fullName || seen.has(normalizedUrl)) {
          continue;
        }

        seen.add(normalizedUrl);
        allCandidates.push(scoreRepositoryCandidate(item, titleGuess, arxivId));
      }
    } catch {
      continue;
    } finally {
      clearTimeout(timeout);
    }
  }

  return allCandidates
    .sort((left, right) => {
      const confidenceRank = confidenceWeight(right.confidence) - confidenceWeight(left.confidence);
      if (confidenceRank !== 0) {
        return confidenceRank;
      }
      return (right.stars ?? 0) - (left.stars ?? 0);
    })
    .filter((candidate, _index, array) =>
      array.some((item) => item.confidence !== 'low')
        ? candidate.confidence !== 'low'
        : true,
    )
    .slice(0, 3);
}

function scoreRepositoryCandidate(
  item: RepositorySearchItem,
  titleGuess: string,
  arxivId?: string,
): RepositoryCandidate {
  const haystack = `${item.full_name ?? ''} ${item.description ?? ''} ${item.homepage ?? ''}`.toLowerCase();
  const titleTokens = titleGuess
    .toLowerCase()
    .replace(/[^a-z0-9\s-]+/g, ' ')
    .split(/\s+/)
    .filter((token) => token.length >= 3);
  const matchedTokens = titleTokens.filter((token) => haystack.includes(token));
  const matchRatio = titleTokens.length > 0 ? matchedTokens.length / titleTokens.length : 0;
  const includesArxiv = Boolean(arxivId && haystack.includes(arxivId.toLowerCase()));

  let confidence: RepositoryCandidate['confidence'] = 'low';
  if (includesArxiv || matchRatio >= 0.7) {
    confidence = 'high';
  } else if (matchRatio >= 0.4) {
    confidence = 'medium';
  }

  const reasons: string[] = [];
  if (includesArxiv) {
    reasons.push(`Matched arXiv identifier ${arxivId}`);
  }
  if (matchedTokens.length > 0) {
    reasons.push(`Matched title tokens: ${matchedTokens.slice(0, 4).join(', ')}`);
  }
  if (!reasons.length) {
    reasons.push('Returned from GitHub repository search.');
  }

  return {
    url: item.html_url ?? '',
    fullName: item.full_name ?? '',
    description: item.description ?? undefined,
    stars: item.stargazers_count ?? undefined,
    source: 'github_search',
    confidence,
    reason: reasons.join('; '),
  };
}

function confidenceWeight(value: RepositoryCandidate['confidence']): number {
  switch (value) {
    case 'high':
      return 3;
    case 'medium':
      return 2;
    default:
      return 1;
  }
}

function buildCacheKey(titleGuess: string, arxivId: string | undefined, filePath: string): string {
  return createHash('sha256')
    .update(
      JSON.stringify({
        version: CACHE_VERSION,
        titleGuess,
        arxivId: arxivId ?? '',
        fileName: path.basename(filePath),
      }),
    )
    .digest('hex');
}

async function readJsonCache<T>(cachePath: string): Promise<T | null> {
  try {
    const raw = await fs.readFile(cachePath, 'utf8');
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

async function writeJsonCache(cachePath: string, value: unknown): Promise<void> {
  await fs.mkdir(path.dirname(cachePath), { recursive: true });
  await fs.writeFile(cachePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

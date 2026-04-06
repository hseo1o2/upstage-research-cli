import type { ParseAnchor, ParseAnchorCoordinate } from './client';
import { getEvidenceAliasRegistry } from './registry-config';

export interface EvidenceOptions {
  maxSnippets?: number;
  maxCharsPerSnippet?: number;
  minScore?: number;
  anchors?: ParseAnchor[];
}

export interface EvidenceRecord {
  id: string;
  text: string;
  score: number;
  sourceKind?: 'block' | 'anchor' | 'table';
  matchedQuery?: string;
  page?: number;
  elementId?: string;
  category?: string;
  section?: string;
  coordinates?: ParseAnchorCoordinate[];
}

export interface EvidenceRecordMergeOptions {
  maxSnippets?: number;
  anchors?: ParseAnchor[];
}

export interface FocusedEvidenceOptions extends EvidenceOptions {
  maxPerQuery?: number;
}

const DEFAULT_MAX_SNIPPETS = 3;
const DEFAULT_MAX_CHARS = 220;
const DEFAULT_MIN_SCORE = 2;
const STOPWORDS = new Set([
  'the', 'and', 'for', 'with', 'that', 'from', 'this', 'into', 'using', 'use',
  'our', 'their', 'they', 'are', 'was', 'were', 'have', 'has', 'had', 'can',
  'could', 'should', 'would', 'your', 'about', 'than', 'then', 'when', 'where',
  'which', 'what', 'why', 'how', 'not', 'but', 'paper', 'method', 'methods',
  'results', 'experiment', 'evaluation', 'model', 'models', 'dataset', 'datasets',
  'metric', 'metrics', 'based', 'approach', 'proposed', 'study', 'task', 'tasks',
  'data', 'performance', 'system', 'systems', 'analysis', 'section', 'table',
]);

export function buildEvidenceSnippets(
  markdown: string,
  queryTexts: string[],
  options: EvidenceOptions = {},
): string[] {
  return buildEvidenceRecords(markdown, queryTexts, options).map((record) => record.text);
}

export function buildEvidenceRecords(
  markdown: string,
  queryTexts: string[],
  options: EvidenceOptions = {},
): EvidenceRecord[] {
  const maxSnippets = options.maxSnippets ?? DEFAULT_MAX_SNIPPETS;
  const maxCharsPerSnippet = options.maxCharsPerSnippet ?? DEFAULT_MAX_CHARS;
  const minScore = options.minScore ?? DEFAULT_MIN_SCORE;
  const blocks = splitMarkdownIntoBlocks(markdown, maxCharsPerSnippet);
  const expandedQueries = expandEvidenceQueries(queryTexts);
  const tokens = Array.from(
    new Set(
      expandedQueries
        .flatMap((queryText) => extractQueryTokens(queryText))
        .filter(Boolean),
    ),
  );
  const normalizedQueries = Array.from(
    new Set(
      expandedQueries
        .map((queryText) => normalizeQueryPhrase(queryText))
        .filter(Boolean),
    ),
  );

  if (tokens.length === 0 || blocks.length === 0) {
    return [];
  }

  const scored = blocks
    .map((block) => scoreBlock(block, tokens, normalizedQueries))
    .filter((candidate) => candidate.score >= minScore)
    .sort((left, right) => right.score - left.score || left.block.length - right.block.length);

  const snippets: EvidenceRecord[] = [];
  const seen = new Set<string>();
  const coveredTokens = new Set<string>();

  while (snippets.length < maxSnippets && scored.length > 0) {
    let bestIndex = -1;
    let bestGain = -1;

    for (let index = 0; index < scored.length; index += 1) {
      const candidate = scored[index];
      const normalized = candidate.block.toLowerCase();
      if (seen.has(normalized)) {
        continue;
      }

      const newTokenGain = candidate.matchedTokens.filter((token) => !coveredTokens.has(token)).length;
      const combinedGain = newTokenGain * 100 + candidate.score;
      if (combinedGain > bestGain) {
        bestGain = combinedGain;
        bestIndex = index;
      }
    }

    if (bestIndex === -1) {
      break;
    }

    const candidate = scored.splice(bestIndex, 1)[0];
    const normalized = candidate.block.toLowerCase();
    seen.add(normalized);
    candidate.matchedTokens.forEach((token) => coveredTokens.add(token));
    snippets.push({
      id: `E${snippets.length + 1}`,
      text: candidate.block,
      score: candidate.score,
      sourceKind: 'block',
    });
  }

  return applyAnchorProvenance(snippets, options.anchors);
}

export function buildAnchoredEvidenceRecords(
  markdown: string,
  queryTexts: string[],
  options: EvidenceOptions = {},
): EvidenceRecord[] {
  const maxSnippets = options.maxSnippets ?? DEFAULT_MAX_SNIPPETS;
  const maxCharsPerSnippet = options.maxCharsPerSnippet ?? DEFAULT_MAX_CHARS;
  const minScore = options.minScore ?? DEFAULT_MIN_SCORE;
  const expandedQueries = expandEvidenceQueries(queryTexts);
  const tokens = Array.from(
    new Set(
      expandedQueries
        .flatMap((queryText) => extractQueryTokens(queryText))
        .filter(Boolean),
    ),
  );
  const normalizedQueries = Array.from(
    new Set(
      expandedQueries
        .map((queryText) => normalizeQueryPhrase(queryText))
        .filter(Boolean),
    ),
  );
  const lines = markdown.replace(/\r\n/g, '\n').split('\n');

  if (tokens.length === 0 || lines.length === 0) {
    return [];
  }

  const scored = lines
    .map((line, index) => {
      const window = compactSnippet(
        [lines[index - 1] ?? '', line, lines[index + 1] ?? ''].join(' '),
        maxCharsPerSnippet,
      );
      return scoreBlock(window, tokens, normalizedQueries);
    })
    .filter((candidate) => candidate.block && candidate.score >= minScore)
    .sort((left, right) => right.score - left.score || left.block.length - right.block.length);

  return mergeEvidenceRecordGroups(
    [
      scored.map((candidate) => ({
        id: '',
        text: candidate.block,
        score: candidate.score,
        sourceKind: 'anchor' as const,
      })),
    ],
    {
      maxSnippets,
      anchors: options.anchors,
    },
  );
}

export function buildTableEvidenceRecords(
  markdown: string,
  queryTexts: string[],
  options: EvidenceOptions = {},
): EvidenceRecord[] {
  const maxSnippets = options.maxSnippets ?? DEFAULT_MAX_SNIPPETS;
  const maxCharsPerSnippet = options.maxCharsPerSnippet ?? DEFAULT_MAX_CHARS;
  const minScore = options.minScore ?? DEFAULT_MIN_SCORE;
  const expandedQueries = expandEvidenceQueries(queryTexts);
  const tokens = Array.from(
    new Set(
      expandedQueries
        .flatMap((queryText) => extractQueryTokens(queryText))
        .filter(Boolean),
    ),
  );
  const normalizedQueries = Array.from(
    new Set(
      expandedQueries
        .map((queryText) => normalizeQueryPhrase(queryText))
        .filter(Boolean),
    ),
  );

  if (tokens.length === 0) {
    return [];
  }

  const candidates = extractMarkdownTableCandidates(markdown, maxCharsPerSnippet);
  if (candidates.length === 0) {
    return [];
  }

  return mergeEvidenceRecordGroups(
    [
      candidates
        .map((candidate) => {
          const scored = scoreBlock(candidate, tokens, normalizedQueries);
          return {
            id: '',
            text: scored.block,
            score: scored.score + 2,
            sourceKind: 'table' as const,
          };
        })
        .filter((candidate) => candidate.score >= minScore)
        .sort((left, right) => right.score - left.score || left.text.length - right.text.length),
    ],
    { maxSnippets, anchors: options.anchors },
  );
}

export function buildFocusedEvidenceRecords(
  markdown: string,
  queryTexts: string[],
  options: FocusedEvidenceOptions = {},
): EvidenceRecord[] {
  const maxSnippets = options.maxSnippets ?? DEFAULT_MAX_SNIPPETS;
  const maxPerQuery = options.maxPerQuery ?? 1;
  const focusedGroups = queryTexts
    .map((queryText) => queryText.trim())
    .filter(Boolean)
    .map((queryText) =>
      mergeEvidenceRecordGroups(
        [
          buildTableEvidenceRecords(markdown, [queryText], {
            maxSnippets: maxPerQuery,
            maxCharsPerSnippet: options.maxCharsPerSnippet,
            minScore: options.minScore,
          }).map((record) => ({ ...record, matchedQuery: queryText })),
          buildAnchoredEvidenceRecords(markdown, [queryText], {
            maxSnippets: maxPerQuery,
            maxCharsPerSnippet: options.maxCharsPerSnippet,
            minScore: options.minScore,
          }).map((record) => ({ ...record, matchedQuery: queryText })),
          buildEvidenceRecords(markdown, [queryText], {
            maxSnippets: maxPerQuery,
            maxCharsPerSnippet: options.maxCharsPerSnippet,
            minScore: options.minScore,
          }).map((record) => ({ ...record, matchedQuery: queryText })),
        ],
        { maxSnippets: maxPerQuery, anchors: options.anchors },
      ),
    );

  return mergeEvidenceRecordGroups(focusedGroups, { maxSnippets, anchors: options.anchors });
}

export function mergeEvidenceRecordGroups(
  groups: EvidenceRecord[][],
  options: EvidenceRecordMergeOptions = {},
): EvidenceRecord[] {
  const maxSnippets = options.maxSnippets ?? DEFAULT_MAX_SNIPPETS;
  const merged: EvidenceRecord[] = [];
  const seen = new Set<string>();

  for (const group of groups) {
    for (const record of group) {
      const key = normalizeQueryPhrase(record.text);
      if (!key || seen.has(key)) {
        continue;
      }

      seen.add(key);
      merged.push({ ...record, id: '' });
      if (merged.length >= maxSnippets) {
        return applyAnchorProvenance(
          merged.slice(0, maxSnippets).map((item, index) => ({
            ...item,
            id: `E${index + 1}`,
          })),
          options.anchors,
        );
      }
    }
  }

  return applyAnchorProvenance(
    merged.map((item, index) => ({
      ...item,
      id: `E${index + 1}`,
    })),
    options.anchors,
  );
}

export function expandEvidenceQueries(queryTexts: string[]): string[] {
  const expanded = new Set<string>();

  for (const queryText of queryTexts) {
    const trimmed = queryText.trim();
    if (!trimmed) {
      continue;
    }

    for (const variant of expandEvidenceQuery(trimmed)) {
      if (variant) {
        expanded.add(variant);
      }
    }
  }

  return Array.from(expanded);
}

function splitMarkdownIntoBlocks(markdown: string, maxCharsPerSnippet: number): string[] {
  const normalized = markdown
    .replace(/\r\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n');
  const paragraphBlocks = normalized
    .split(/\n\s*\n/)
    .map((block) => compactSnippet(block, maxCharsPerSnippet))
    .filter(Boolean);

  if (paragraphBlocks.length > 0) {
    return paragraphBlocks;
  }

  return normalized
    .split('\n')
    .map((line) => compactSnippet(line, maxCharsPerSnippet))
    .filter(Boolean);
}

function compactSnippet(value: string, maxCharsPerSnippet: number): string {
  const compact = value.replace(/\s+/g, ' ').trim();
  if (!compact) {
    return '';
  }

  if (compact.length <= maxCharsPerSnippet) {
    return compact;
  }

  return `${compact.slice(0, Math.max(0, maxCharsPerSnippet - 3)).trimEnd()}...`;
}

function extractMarkdownTableCandidates(markdown: string, maxCharsPerSnippet: number): string[] {
  const lines = markdown.replace(/\r\n/g, '\n').split('\n');
  const candidates: string[] = [];

  for (let index = 0; index < lines.length; index += 1) {
    if (!isMarkdownTableLine(lines[index])) {
      continue;
    }

    const start = index;
    let end = index;
    while (end + 1 < lines.length && isMarkdownTableLine(lines[end + 1])) {
      end += 1;
    }

    const tableLines = lines.slice(start, end + 1);
    const rowLines = tableLines.filter((line) => !isMarkdownTableDivider(line));
    const headerLine = rowLines[0] ?? '';
    const preceding = collectNearbyContextLines(lines, start - 1, -1, 2)
      .filter((line) => /^table\b|^figure\b|^fig\b|^chart\b|^#/i.test(line));
    const following = collectNearbyContextLines(lines, end + 1, 1, 1)
      .filter((line) => /^table\b|^figure\b|^fig\b|^chart\b/i.test(line));
    const tableContext = [...preceding, ...following]
      .map((line) => line.trim())
      .filter(Boolean);
    const tablePrefix = tableContext.join(' ');

    for (const rowLine of rowLines) {
      for (const variant of [
        rowLine,
        [headerLine, rowLine].filter(Boolean).join(' '),
        [tablePrefix, headerLine, rowLine].filter(Boolean).join(' '),
      ]) {
        const combined = compactSnippet(variant, maxCharsPerSnippet);
        if (combined) {
          candidates.push(combined);
        }
      }
    }

    const wholeTableSnippet = compactSnippet(
      [tablePrefix, ...rowLines.slice(0, 4)].filter(Boolean).join(' '),
      maxCharsPerSnippet,
    );
    if (wholeTableSnippet) {
      candidates.push(wholeTableSnippet);
    }

    index = end;
  }

  return candidates;
}

function collectNearbyContextLines(
  lines: string[],
  startIndex: number,
  step: -1 | 1,
  limit: number,
): string[] {
  const collected: string[] = [];
  let index = startIndex;

  while (index >= 0 && index < lines.length && collected.length < limit) {
    const compact = lines[index].replace(/\s+/g, ' ').trim();
    if (compact) {
      collected.push(compact);
    }
    index += step;
  }

  return step === -1 ? collected.reverse() : collected;
}

function isMarkdownTableLine(line: string): boolean {
  const trimmed = line.trim();
  return trimmed.startsWith('|') && trimmed.endsWith('|');
}

function isMarkdownTableDivider(line: string): boolean {
  const compact = line.replace(/\s+/g, '');
  return /^\|(?:-+\|)+$/.test(compact);
}

function extractQueryTokens(input: string): string[] {
  return input
    .toLowerCase()
    .replace(/[^a-z0-9가-힣\s-]/g, ' ')
    .split(/\s+/)
    .map((token) => token.trim())
    .filter((token) => token.length >= 3 || /[가-힣]{2,}/.test(token))
    .filter((token) => !STOPWORDS.has(token));
}

function scoreBlock(
  block: string,
  queryTokens: string[],
  normalizedQueries: string[],
): { block: string; score: number; matchedTokens: string[] } {
  const normalized = ` ${block.toLowerCase()} `;
  let score = 0;
  const matchedTokens: string[] = [];
  for (const token of queryTokens) {
    if (normalized.includes(` ${token} `) || normalized.includes(token)) {
      score += token.length >= 8 ? 2 : 1;
      matchedTokens.push(token);
    }
  }

  const compactBlock = normalizeQueryPhrase(block);
  for (const query of normalizedQueries) {
    if (!query) {
      continue;
    }

    if (compactBlock.includes(query)) {
      score += query.split(' ').length >= 3 ? 4 : 2;
    }
  }

  return { block, score, matchedTokens };
}

export function normalizeQueryPhrase(input: string): string {
  return input
    .toLowerCase()
    .replace(/[^a-z0-9가-힣\s-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function expandEvidenceQuery(input: string): string[] {
  const variants = new Set<string>();
  const add = (value: string) => {
    const compact = value.replace(/\s+/g, ' ').trim();
    if (compact) {
      variants.add(compact);
    }
  };

  add(input);
  add(input.replace(/[-_/]+/g, ' '));
  add(input.replace(/\s*\([^)]*\)/g, ' '));
  add(input.replace(/\(([^)]+)\)/g, ' $1 '));

  const coarseSegments = input
    .split(/[,:;]+/)
    .map((segment) => segment.trim())
    .filter((segment) => segment.length >= 3);
  for (const segment of coarseSegments) {
    add(segment);
  }

  if (coarseSegments.length > 1) {
    for (const segment of coarseSegments.slice(1)) {
      add(segment.replace(/^(and|or)\s+/i, ''));
    }
  }

  const normalized = normalizeQueryPhrase(input);
  for (const alias of aliasesForNormalizedQuery(normalized)) {
    add(alias);
  }

  return Array.from(variants);
}

function aliasesForNormalizedQuery(normalized: string): string[] {
  const aliases = new Set<string>();
  const add = (...values: string[]) => {
    for (const value of values) {
      const compact = value.trim();
      if (compact) {
        aliases.add(compact);
      }
    }
  };

  if (!normalized) {
    return [];
  }

  for (const rule of getEvidenceAliasRegistry().aliases) {
    if (new RegExp(rule.pattern).test(normalized)) {
      add(...rule.values);
    }
  }
  if (normalized.startsWith('amazon ')) {
    add(normalized.replace(/^amazon\s+/, ''));
  }
  if (/^pbmc \d+k$/.test(normalized)) {
    add(normalized.replace(/\s+\d+k$/, ''), normalized.replace(/\s+/, ' (') + ')');
  }
  if (/^pancreas \d+k$/.test(normalized)) {
    add('pancreas', normalized.replace(/\s+/, ' (') + ')');
  }

  return Array.from(aliases);
}

function applyAnchorProvenance(records: EvidenceRecord[], anchors: ParseAnchor[] | undefined): EvidenceRecord[] {
  if (!anchors || anchors.length === 0) {
    return records;
  }

  return records.map((record) => {
    if (record.page !== undefined || record.elementId || record.category || record.section) {
      return record;
    }

    const anchor = findBestMatchingAnchor(record, anchors);
    if (!anchor) {
      return record;
    }

    return {
      ...record,
      page: anchor.page,
      elementId: anchor.elementId,
      category: anchor.category,
      section: anchor.section,
      coordinates: anchor.coordinates,
    };
  });
}

function findBestMatchingAnchor(record: EvidenceRecord, anchors: ParseAnchor[]): ParseAnchor | undefined {
  const snippet = normalizeQueryPhrase(record.text);
  if (!snippet) {
    return undefined;
  }

  const snippetTokens = extractQueryTokens(record.text);
  const normalizedQuery = normalizeQueryPhrase(record.matchedQuery ?? '');
  let bestScore = -1;
  let bestAnchor: ParseAnchor | undefined;

  for (const anchor of anchors) {
    const normalizedAnchor = normalizeQueryPhrase(anchor.markdown);
    if (!normalizedAnchor) {
      continue;
    }

    let score = 0;
    if (normalizedAnchor === snippet) {
      score += 16;
    }
    if (normalizedAnchor.includes(snippet) || snippet.includes(normalizedAnchor)) {
      score += 10;
    }

    const tokenMatches = snippetTokens.filter((token) =>
      normalizedAnchor.includes(` ${token} `) || normalizedAnchor.includes(token),
    ).length;
    score += tokenMatches * 2;

    if (normalizedQuery && containsVariant(normalizedAnchor, normalizedQuery)) {
      score += 4;
    }
    if (record.sourceKind === 'table' && /table|chart|caption/i.test(anchor.category ?? '')) {
      score += 3;
    }
    if (record.sourceKind === 'anchor' && /heading/i.test(anchor.category ?? '')) {
      score += 1;
    }
    if (record.sourceKind === 'block' && /paragraph|list|caption|table/i.test(anchor.category ?? '')) {
      score += 1;
    }

    if (score > bestScore) {
      bestScore = score;
      bestAnchor = anchor;
    }
  }

  return bestScore >= 4 ? bestAnchor : undefined;
}

function containsVariant(normalizedAnchor: string, normalizedQuery: string): boolean {
  if (!normalizedQuery) {
    return false;
  }

  if (normalizedAnchor.includes(normalizedQuery)) {
    return true;
  }

  return expandEvidenceQuery(normalizedQuery)
    .map((variant) => normalizeQueryPhrase(variant))
    .filter(Boolean)
    .some((variant) => normalizedAnchor.includes(variant));
}

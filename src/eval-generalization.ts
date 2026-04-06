import { expandEvidenceQueries, normalizeQueryPhrase } from './evidence';
import type {
  Baseline,
  EvaluationMetric,
  MetricPlan,
  PaperEvaluationSummary,
} from './eval-types';
import type { RepositoryCandidate, RepositoryDiscoveryResult } from './repo-discovery';

export type MetricShape =
  | 'judge'
  | 'preference_model'
  | 'geometry'
  | 'ranking_retrieval'
  | 'generation_similarity'
  | 'structured_prediction'
  | 'generic_protocol';

export type UnresolvedPolicy = 'repo_first' | 'methodological_scaffold' | 'explicit_unresolved';

export function reconcileSummaryWithEvidence(
  summary: PaperEvaluationSummary,
  markdown: string,
): PaperEvaluationSummary {
  const evaluationMetrics = reconcileMetrics(summary.evaluation_metrics ?? [], markdown, summary.task_type ?? '');
  const datasets = reconcileDatasets(summary.datasets ?? [], markdown);
  const baselines = reconcileBaselines(summary.baselines ?? [], markdown, evaluationMetrics);
  return {
    ...summary,
    evaluation_metrics: evaluationMetrics,
    datasets,
    baselines,
  };
}

export function reconcileMetrics(
  metrics: EvaluationMetric[],
  markdown: string,
  taskType: string,
): EvaluationMetric[] {
  const normalizedMetrics = normalizeMetrics(metrics);
  if (normalizedMetrics.length === 0) {
    return normalizedMetrics;
  }

  const normalizedMarkdown = normalizeQueryPhrase(markdown);
  const normalizedTaskType = normalizeQueryPhrase(taskType);
  if (!normalizedMarkdown) {
    return normalizedMetrics;
  }

  const scored = normalizedMetrics
    .map((metric) => ({
      metric,
      score: scoreMetricEvidence(metric, normalizedMarkdown, normalizedTaskType),
    }))
    .sort((left, right) => right.score - left.score || (left.metric.name?.length ?? 0) - (right.metric.name?.length ?? 0));

  const grounded = scored.filter((item) => item.score >= 1.5).map((item) => item.metric);
  if (grounded.length > 0) {
    return grounded.slice(0, 10);
  }

  const partiallyGrounded = scored.filter((item) => item.score >= 1.0).map((item) => item.metric);
  if (partiallyGrounded.length > 0) {
    return partiallyGrounded.slice(0, 10);
  }

  return normalizedMetrics;
}

export function reconcileDatasets(datasets: string[], markdown: string): string[] {
  const normalizedDatasets = normalizeStringList(datasets, 12, 96);
  if (normalizedDatasets.length === 0) {
    return normalizedDatasets;
  }

  const datasetContext = buildDatasetEvidenceCorpus(markdown);
  const normalizedContext = normalizeQueryPhrase(datasetContext);
  const normalizedMarkdown = normalizeQueryPhrase(markdown);
  if (!normalizedMarkdown && !normalizedContext) {
    return normalizedDatasets;
  }

  const scored = normalizedDatasets
    .map((dataset) => ({
      dataset,
      contextScore: scoreDatasetGrounding(dataset, normalizedContext, normalizedDatasets),
      globalScore: scoreDatasetGrounding(dataset, normalizedMarkdown, normalizedDatasets),
    }))
    .sort((left, right) =>
      right.contextScore - left.contextScore
      || right.globalScore - left.globalScore
      || left.dataset.length - right.dataset.length,
    );

  const grounded = scored
    .filter((item) => item.contextScore >= 1.5 || (item.contextScore >= 1 && item.globalScore >= 2))
    .map((item) => item.dataset);

  const recovered = backfillGroundedDatasets(normalizedDatasets, grounded, markdown, normalizedContext || normalizedMarkdown);
  const replaced = replaceUngroundedDatasetsWithEvidence(normalizedDatasets, grounded, recovered);
  if (replaced.length > 0) {
    return replaced.slice(0, 12);
  }

  return normalizedDatasets;
}

export function reconcileBaselines(
  baselines: Baseline[],
  markdown: string,
  evaluationMetrics: EvaluationMetric[],
): Baseline[] {
  const normalizedBaselines = normalizeBaselines(baselines);
  if (normalizedBaselines.length === 0) {
    return normalizedBaselines;
  }

  const normalizedMarkdown = normalizeQueryPhrase(markdown);
  if (!normalizedMarkdown) {
    return normalizedBaselines;
  }

  const metricNames = (evaluationMetrics ?? [])
    .map((metric) => normalizeLooseText(metric.name, 96))
    .filter(Boolean);

  const scored = normalizedBaselines
    .map((baseline) => {
      const groundedNameScore = scoreGroundedMention(baseline.model_name ?? '', normalizedMarkdown);
      const supportedScores = Object.entries(baseline.scores ?? {})
        .filter(([scoreName]) => isSupportedBaselineScore(scoreName, metricNames, normalizedMarkdown))
        .slice(0, 8);

      return {
        baseline,
        groundedNameScore,
        supportedScores,
        score: groundedNameScore + supportedScores.length * 0.5,
      };
    })
    .sort((left, right) => right.score - left.score || (left.baseline.model_name?.length ?? 0) - (right.baseline.model_name?.length ?? 0));

  const grounded = scored
    .filter((item) => item.groundedNameScore >= 1 || item.supportedScores.length > 0)
    .map(({ baseline, supportedScores, groundedNameScore }) => ({
      ...baseline,
      scores:
        supportedScores.length > 0
          ? Object.fromEntries(supportedScores)
          : groundedNameScore >= 1
            ? Object.fromEntries(Object.entries(baseline.scores ?? {}).slice(0, 4))
            : baseline.scores,
    }));

  if (grounded.length > 0) {
    return grounded.slice(0, 10);
  }

  return normalizedBaselines;
}

export function applyGeneralizedFallbackPolicy(
  metricPlans: MetricPlan[],
  discovery: RepositoryDiscoveryResult,
  language: 'ko' | 'en',
): MetricPlan[] {
  return metricPlans.map((plan) => {
    if (!requiresFallbackLead(plan)) {
      return plan;
    }

    const shape = inferMetricShape(plan);
    const policy = selectUnresolvedPolicy(shape, discovery.candidates);
    const fallbackNote = buildPolicyNote(policy, shape, discovery.candidates, language);
    return {
      ...plan,
      note: [plan.note?.trim(), fallbackNote].filter(Boolean).join(' '),
    };
  });
}

export function requiresFallbackLead(plan: MetricPlan): boolean {
  return (
    plan.kind === 'custom_placeholder'
    || plan.kind === 'feature_coefficient_placeholder'
    || plan.kind === 'judge_score_aggregate'
  );
}

export function inferMetricShape(plan: MetricPlan): MetricShape {
  const corpus = `${plan.metricName} ${plan.formula ?? ''} ${plan.note ?? ''}`.toLowerCase();
  if (plan.kind === 'judge_score_aggregate' || /(judge|faithfulness|factuality|groundedness|helpfulness|coherence|fluency|relevance)/.test(corpus)) {
    return 'judge';
  }
  if (plan.kind === 'feature_coefficient_placeholder' || /(bradley|bias|preference|coefficient|pairwise)/.test(corpus)) {
    return 'preference_model';
  }
  if (/(rmsd|steric|intersection|geometry|distance|docking|structure|torsion|pose)/.test(corpus)) {
    return 'geometry';
  }
  if (/(retrieval|rank|ranking|top-k|top k|@k|ndcg|mrr|hit rate|hit ratio|precision@|recall@|tr@|ir@)/.test(corpus)) {
    return 'ranking_retrieval';
  }
  if (/(bleu|rouge|bertscore|cider|spice|caption|similarity|wer|cer|edit)/.test(corpus)) {
    return 'generation_similarity';
  }
  if (/(iou|map|f1|accuracy|auc|precision|recall|calibration|perplexity|entropy|pass@k|success rate)/.test(corpus)) {
    return 'structured_prediction';
  }
  return 'generic_protocol';
}

function selectUnresolvedPolicy(
  shape: MetricShape,
  candidates: RepositoryCandidate[],
): UnresolvedPolicy {
  const hasHighConfidenceRepo = candidates.some((candidate) => candidate.confidence === 'high');
  if (hasHighConfidenceRepo) {
    return 'repo_first';
  }

  if (shape === 'judge' || shape === 'ranking_retrieval' || shape === 'generation_similarity' || shape === 'structured_prediction') {
    return 'methodological_scaffold';
  }

  if (shape === 'geometry' || shape === 'preference_model') {
    return 'explicit_unresolved';
  }

  return 'methodological_scaffold';
}

function buildPolicyNote(
  policy: UnresolvedPolicy,
  shape: MetricShape,
  candidates: RepositoryCandidate[],
  language: 'ko' | 'en',
): string {
  const top = candidates[0];
  switch (policy) {
    case 'repo_first':
      return language === 'ko'
        ? `우선 author code 후보를 확인하세요: ${top.url} (${top.reason}) 저자 코드가 없으면 논문 근거와 baseline row를 기준으로 ${describeMethodologicalFallback(shape, language)}`
        : `Inspect the author-code candidate first: ${top.url} (${top.reason}) If author code is unavailable, ${describeMethodologicalFallback(shape, language)}`;
    case 'methodological_scaffold':
      return language === 'ko'
        ? describeMethodologicalFallback(shape, language)
        : describeMethodologicalFallback(shape, language);
    case 'explicit_unresolved':
      return language === 'ko'
        ? `현재 논문 설명만으로는 ${describeShapeLabel(shape, language)} 프로토콜을 안전하게 완전 자동화하기 어렵습니다. 저자 코드 또는 부록 기준으로 수동 확인을 유지하세요.`
        : `The paper alone does not define the ${describeShapeLabel(shape, language)} protocol clearly enough for safe full automation. Keep this as an explicit manual-review item unless author code or appendix details are available.`;
  }
}

function describeMethodologicalFallback(shape: MetricShape, language: 'ko' | 'en'): string {
  switch (shape) {
    case 'judge':
      return language === 'ko'
        ? 'evidence snippet, reported rubric, baseline comparison을 바탕으로 judge 입력/출력 스키마와 점수 rubric을 재구성하세요.'
        : 'reconstruct the judge input/output schema and scoring rubric from evidence snippets, reported rubrics, and baseline comparisons.';
    case 'ranking_retrieval':
      return language === 'ko'
        ? 'reported top-k / retrieval definition, query-gallery split, baseline row를 기준으로 evaluator interface를 재구성하세요.'
        : 'reconstruct the evaluator interface from the reported top-k or retrieval definition, query-gallery split, and baseline rows.';
    case 'generation_similarity':
      return language === 'ko'
        ? 'reported metric formula, tokenizer assumptions, and baseline rows를 기준으로 text-generation evaluator를 재구성하세요.'
        : 'reconstruct the text-generation evaluator from the reported metric formula, tokenizer assumptions, and baseline rows.';
    case 'structured_prediction':
      return language === 'ko'
        ? 'reported metric formula, threshold/top-k rule, and baseline rows를 기준으로 evaluation schema를 재구성하세요.'
        : 'reconstruct the evaluation schema from the reported metric formula, threshold or top-k rule, and baseline rows.';
    default:
      return language === 'ko'
        ? 'evidence snippet, reported metric formula/value, baseline row를 기준으로 평가 입력/출력 스키마를 재구성하세요.'
        : 'reconstruct the evaluator input/output schema from evidence snippets, reported metric formulas/values, and baseline rows.';
  }
}

function describeShapeLabel(shape: MetricShape, language: 'ko' | 'en'): string {
  switch (shape) {
    case 'judge':
      return language === 'ko' ? 'judge-based metric' : 'judge-based metric';
    case 'preference_model':
      return language === 'ko' ? 'preference-model' : 'preference-model';
    case 'geometry':
      return language === 'ko' ? 'geometry/structure' : 'geometry/structure';
    case 'ranking_retrieval':
      return language === 'ko' ? 'ranking/retrieval' : 'ranking/retrieval';
    case 'generation_similarity':
      return language === 'ko' ? 'generation-similarity' : 'generation-similarity';
    case 'structured_prediction':
      return language === 'ko' ? 'structured-prediction' : 'structured-prediction';
    default:
      return language === 'ko' ? 'paper-specific' : 'paper-specific';
  }
}

function scoreGroundedMention(rawValue: string, normalizedMarkdown: string): number {
  let bestScore = 0;
  for (const variant of expandEvidenceQueries([rawValue])) {
    const normalizedVariant = normalizeQueryPhrase(variant);
    if (!normalizedVariant) {
      continue;
    }

    const paddedMarkdown = ` ${normalizedMarkdown} `;
    const paddedVariant = ` ${normalizedVariant} `;
    if (paddedMarkdown.includes(paddedVariant) || normalizedMarkdown.includes(normalizedVariant)) {
      bestScore = Math.max(bestScore, 3);
      continue;
    }

    const parts = normalizedVariant
      .split(/\s+/)
      .filter((part) => part.length >= 2 || /\d/.test(part));
    if (parts.length === 0) {
      continue;
    }

    const matched = parts.filter((part) => paddedMarkdown.includes(` ${part} `) || normalizedMarkdown.includes(part)).length;
    const ratio = matched / parts.length;
    if (ratio >= 0.8) {
      bestScore = Math.max(bestScore, 2);
    } else if (ratio >= 0.5) {
      bestScore = Math.max(bestScore, 1);
    }
  }

  return bestScore;
}

export function expandDatasetSurfaceForms(dataset: string, peers: string[] = []): string[] {
  const variants = new Set<string>();
  for (const variant of expandEvidenceQueries([dataset])) {
    const compact = variant.replace(/\s+/g, ' ').trim();
    if (compact) {
      variants.add(compact);
    }
  }

  const sharedPrefix = detectSharedDatasetPrefix(dataset, peers);
  if (sharedPrefix) {
    const suffix = dataset.replace(new RegExp(`^${escapeRegExp(sharedPrefix)}\\s+`, 'i'), '').trim();
    if (suffix) {
      variants.add(suffix);
    }
  }

  return Array.from(variants);
}

export function datasetsEquivalent(left: string, right: string, peers: string[] = []): boolean {
  const leftVariants = expandDatasetSurfaceForms(left, peers)
    .map((variant) => normalizeQueryPhrase(variant))
    .filter(Boolean);
  const rightVariants = expandDatasetSurfaceForms(right, peers)
    .map((variant) => normalizeQueryPhrase(variant))
    .filter(Boolean);

  for (const leftVariant of leftVariants) {
    for (const rightVariant of rightVariants) {
      if (leftVariant === rightVariant) {
        return true;
      }
      if (leftVariant.includes(rightVariant) || rightVariant.includes(leftVariant)) {
        return true;
      }
      if (hasMeaningfulTokenOverlap(leftVariant, rightVariant)) {
        return true;
      }
    }
  }

  return false;
}

function scoreDatasetGrounding(
  dataset: string,
  normalizedMarkdown: string,
  peers: string[],
): number {
  let bestScore = 0;
  for (const variant of expandDatasetSurfaceForms(dataset, peers)) {
    bestScore = Math.max(bestScore, scoreGroundedMention(variant, normalizedMarkdown));
  }
  return bestScore;
}

function backfillGroundedDatasets(
  originalDatasets: string[],
  groundedDatasets: string[],
  markdown: string,
  normalizedMarkdown: string,
): string[] {
  const recovered = [...groundedDatasets];
  const seen = new Set(recovered.map((dataset) => dataset.toLowerCase()));
  const desiredCount = Math.min(12, Math.max(originalDatasets.length, groundedDatasets.length));
  const candidates = extractDatasetCandidates(markdown, originalDatasets)
    .map((dataset) => ({
      dataset,
      score: scoreDatasetGrounding(dataset, normalizedMarkdown, originalDatasets),
    }))
    .filter((item) => item.score >= 1.0)
    .sort((left, right) => right.score - left.score || left.dataset.length - right.dataset.length);

  for (const candidate of candidates) {
    const key = candidate.dataset.toLowerCase();
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    recovered.push(candidate.dataset);
    if (recovered.length >= desiredCount) {
      break;
    }
  }

  return recovered;
}

function replaceUngroundedDatasetsWithEvidence(
  originalDatasets: string[],
  groundedDatasets: string[],
  evidenceDatasets: string[],
): string[] {
  if (groundedDatasets.length === 0) {
    return evidenceDatasets.slice(0, 12);
  }

  const result: string[] = [];
  const seen = new Set<string>();
  for (const dataset of groundedDatasets) {
    const key = dataset.toLowerCase();
    if (!seen.has(key)) {
      seen.add(key);
      result.push(dataset);
    }
  }

  const missingSlots = Math.max(0, Math.min(12, originalDatasets.length) - result.length);
  if (missingSlots === 0) {
    return result.slice(0, 12);
  }

  for (const dataset of evidenceDatasets) {
    if (result.some((item) => datasetsEquivalent(item, dataset, [...originalDatasets, ...evidenceDatasets]))) {
      continue;
    }

    result.push(dataset);
    if (result.length >= Math.min(12, originalDatasets.length)) {
      break;
    }
  }

  return result.slice(0, 12);
}

function scoreMetricEvidence(
  metric: EvaluationMetric,
  normalizedMarkdown: string,
  normalizedTaskType: string,
): number {
  const name = normalizeLooseText(metric.name, 96);
  if (!name) {
    return 0;
  }

  let score = scoreGroundedMention(name, normalizedMarkdown);
  const formula = normalizeLooseText(metric.formula, 220);
  if (formula) {
    score += scoreGroundedMention(formula, normalizedMarkdown) * 0.35;
  }

  const valueText = normalizeScalarValue(metric.value);
  if (valueText && normalizedMarkdown.includes(valueText)) {
    score += 0.4;
  }

  if (normalizedTaskType && hasMeaningfulTokenOverlap(name, normalizedTaskType)) {
    score += 0.2;
  }

  return score;
}

function isSupportedBaselineScore(
  scoreName: string,
  metricNames: string[],
  normalizedMarkdown: string,
): boolean {
  if (scoreGroundedMention(scoreName, normalizedMarkdown) >= 1) {
    return true;
  }

  const normalizedScoreName = normalizeQueryPhrase(scoreName);
  if (!normalizedScoreName) {
    return false;
  }

  return metricNames.some((metricName) => isMetricAliasMatch(scoreName, metricName));
}

function isMetricAliasMatch(left: string, right: string): boolean {
  const leftVariants = expandEvidenceQueries([left])
    .map((variant) => normalizeQueryPhrase(variant))
    .filter(Boolean);
  const rightVariants = expandEvidenceQueries([right])
    .map((variant) => normalizeQueryPhrase(variant))
    .filter(Boolean);

  for (const leftVariant of leftVariants) {
    for (const rightVariant of rightVariants) {
      if (leftVariant === rightVariant) {
        return true;
      }
      if (leftVariant.includes(rightVariant) || rightVariant.includes(leftVariant)) {
        return true;
      }
      if (hasMeaningfulTokenOverlap(leftVariant, rightVariant)) {
        return true;
      }
    }
  }

  return false;
}

function hasMeaningfulTokenOverlap(left: string, right: string): boolean {
  const leftTokens = tokenizeLoosePhrase(left);
  const rightTokens = tokenizeLoosePhrase(right);
  if (leftTokens.length === 0 || rightTokens.length === 0) {
    return false;
  }

  const leftSet = new Set(leftTokens);
  const overlap = rightTokens.filter((token) => leftSet.has(token)).length;
  return overlap >= Math.max(1, Math.min(leftTokens.length, rightTokens.length) / 2);
}

function tokenizeLoosePhrase(value: string): string[] {
  return normalizeQueryPhrase(value)
    .split(/\s+/)
    .filter((token) => token.length >= 2 || /\d/.test(token));
}

function normalizeMetrics(metrics: EvaluationMetric[]): EvaluationMetric[] {
  const seen = new Set<string>();
  const normalized: EvaluationMetric[] = [];

  for (const metric of metrics) {
    const name = normalizeLooseText(metric.name, 96);
    if (!name) {
      continue;
    }

    const key = name.toLowerCase();
    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    normalized.push({
      name,
      formula: normalizeLooseText(metric.formula, 220),
      value: metric.value ?? null,
    });

    if (normalized.length >= 10) {
      break;
    }
  }

  return normalized;
}

function normalizeBaselines(baselines: Baseline[]): Baseline[] {
  const seen = new Set<string>();
  const normalized: Baseline[] = [];

  for (const baseline of baselines) {
    const modelName = normalizeLooseText(baseline.model_name, 96);
    if (!modelName) {
      continue;
    }

    const key = modelName.toLowerCase();
    if (seen.has(key)) {
      continue;
    }

    const scores = Object.fromEntries(
      Object.entries(baseline.scores ?? {})
        .map(([scoreName, rawValue]) => [normalizeLooseText(scoreName, 72), rawValue] as const)
        .filter(([scoreName, rawValue]) => Boolean(scoreName) && rawValue !== null && rawValue !== undefined),
    );

    seen.add(key);
    normalized.push({
      model_name: modelName,
      scores,
    });

    if (normalized.length >= 10) {
      break;
    }
  }

  return normalized;
}

function extractDatasetCandidates(markdown: string, originalDatasets: string[]): string[] {
  const candidates: string[] = [];
  const lines = markdown.replace(/\r\n/g, '\n').split('\n');

  for (const line of lines) {
    const compact = line.replace(/\s+/g, ' ').trim();
    if (!compact) {
      continue;
    }

    if (/\|/.test(compact) && /\bdataset\b/i.test(compact)) {
      const cells = compact
        .split('|')
        .map((cell) => sanitizeDatasetCandidate(cell))
        .filter(Boolean);
      if (cells.length > 1) {
        candidates.push(...cells.slice(1).map((cell) => canonicalizeDatasetCandidate(cell, originalDatasets)));
      }
    }
  }

  return normalizeStringList(candidates, 12, 48);
}

function canonicalizeDatasetCandidate(candidate: string, originalDatasets: string[]): string {
  const normalizedCandidate = sanitizeDatasetCandidate(candidate);
  if (!normalizedCandidate) {
    return '';
  }

  const exact = originalDatasets.find((dataset) => datasetsEquivalent(dataset, normalizedCandidate, originalDatasets));
  if (exact) {
    return exact;
  }

  return normalizedCandidate;
}

function buildDatasetEvidenceCorpus(markdown: string): string {
  const lines = markdown.replace(/\r\n/g, '\n').split('\n');
  const blocks = markdown
    .replace(/\r\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .split(/\n\s*\n/)
    .map((block) => block.replace(/\s+/g, ' ').trim())
    .filter(Boolean);

  const datasetLines = lines
    .map((line) => line.replace(/\s+/g, ' ').trim())
    .filter(Boolean)
    .filter((line) =>
      /\bdataset\b|\bdatasets\b|\bbenchmark\b|\bbenchmarks\b|\bsubcategor(?:y|ies)\b|\bevaluat(?:e|ed|ion)\b|\btrain\b|\btest\b|\bvalid\b|\bvalidation\b/.test(line.toLowerCase()),
    );
  const datasetBlocks = blocks.filter((block) =>
    /\bdataset\b|\bdatasets\b|\bbenchmark\b|\bbenchmarks\b|\bsubcategor(?:y|ies)\b|\bevaluat(?:e|ed|ion)\b/.test(block.toLowerCase()),
  );
  const tableBlocks = extractDatasetContextTableBlocks(markdown)
    .filter((block: string) => /\bdataset\b|\btrain\b|\btest\b|\bvalid\b|\bvalidation\b/.test(block.toLowerCase()));

  return [...datasetLines, ...datasetBlocks, ...tableBlocks]
    .join('\n')
    .trim();
}

function extractDatasetContextTableBlocks(markdown: string): string[] {
  const lines = markdown.replace(/\r\n/g, '\n').split('\n');
  const blocks: string[] = [];

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index].replace(/\s+/g, ' ').trim();
    if (!line.startsWith('|') || !line.endsWith('|')) {
      continue;
    }

    const start = index;
    let end = index;
    while (end + 1 < lines.length) {
      const next = lines[end + 1].replace(/\s+/g, ' ').trim();
      if (!next.startsWith('|') || !next.endsWith('|')) {
        break;
      }
      end += 1;
    }

    const tableLines = lines
      .slice(start, end + 1)
      .map((item) => item.replace(/\s+/g, ' ').trim())
      .filter(Boolean)
      .slice(0, 4);
    const block = tableLines.join(' ');
    if (block) {
      blocks.push(block);
    }
    index = end;
  }

  return blocks;
}

function detectSharedDatasetPrefix(dataset: string, peers: string[]): string | null {
  const compact = dataset.replace(/\s+/g, ' ').trim();
  const tokens = compact.split(' ');
  if (tokens.length < 2) {
    return null;
  }

  const prefix = tokens[0];
  const count = peers.filter((peer) =>
    peer !== dataset
    && new RegExp(`^${escapeRegExp(prefix)}\\s+`, 'i').test(peer),
  ).length;

  return count >= 1 ? prefix : null;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function sanitizeDatasetCandidate(value: string): string {
  const compact = value
    .replace(/^[#>*\s-]+/, '')
    .replace(/\s+/g, ' ')
    .trim();
  if (!compact) {
    return '';
  }
  if (/^(dataset|datasets|metric|metrics|model|models|method|methods)$/i.test(compact)) {
    return '';
  }
  if (/^[-:]+$/.test(compact)) {
    return '';
  }
  if (!/[A-Za-z]/.test(compact)) {
    return '';
  }
  if (compact.length > 48) {
    return '';
  }
  return compact;
}

function normalizeScalarValue(value: number | string | null | undefined): string {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? String(value) : '';
  }

  if (typeof value !== 'string') {
    return '';
  }

  const trimmed = value.trim();
  if (!trimmed || /^(n\/a|none|null|not specified|unknown|-|needs review|확인 필요|추출 실패)$/i.test(trimmed)) {
    return '';
  }

  return trimmed.toLowerCase();
}

function normalizeStringList(values: string[], maxItems: number, maxChars: number): string[] {
  const seen = new Set<string>();
  const normalized: string[] = [];

  for (const rawValue of values) {
    const value = normalizeLooseText(rawValue, maxChars);
    if (!value) {
      continue;
    }

    const key = value.toLowerCase();
    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    normalized.push(value);
    if (normalized.length >= maxItems) {
      break;
    }
  }

  return normalized;
}

function normalizeLooseText(value: string | undefined, maxChars: number): string {
  if (!value) {
    return '';
  }

  const compact = value
    .replace(/\s+/g, ' ')
    .replace(/^[•*-]\s*/, '')
    .trim();
  if (!compact || /^(n\/a|none|null|not specified|unknown|-|needs review|확인 필요|추출 실패)$/i.test(compact)) {
    return '';
  }

  return compact.length > maxChars ? `${compact.slice(0, Math.max(0, maxChars - 3)).trimEnd()}...` : compact;
}

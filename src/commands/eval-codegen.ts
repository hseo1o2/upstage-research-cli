import { promises as fs } from 'node:fs';
import path from 'node:path';

import { UpstageClient, type JsonSchema } from '../client';
import { resolveApiKey } from '../config';
import {
  buildAnchoredEvidenceRecords,
  buildEvidenceRecords,
  buildFocusedEvidenceRecords,
  buildTableEvidenceRecords,
  expandEvidenceQueries,
  mergeEvidenceRecordGroups,
  normalizeQueryPhrase,
  type EvidenceRecord,
} from '../evidence';
import {
  buildMetricPlans as buildMetricPlansFromRegistry,
  inferDomain as inferDomainFromRegistry,
  inferTaskMode as inferTaskModeFromRegistry,
  isDirectPredictionMetricKind as isDirectPredictionMetricKindFromRegistry,
  isRankingMetricKind as isRankingMetricKindFromRegistry,
  libraryHintsByDomain as libraryHintsByDomainFromRegistry,
} from '../eval-metric-registry';
import { buildEvaluationVerification, renderVerificationSection } from '../eval-verifier';
import {
  applyGeneralizedFallbackPolicy,
  reconcileSummaryWithEvidence,
  requiresFallbackLead,
} from '../eval-generalization';
import { normalizeOutputFormat, resolvePrimaryOutputPath, writeOutputFile } from '../output';
import {
  discoverPaperRepositories,
  type RepositoryCandidate,
  type RepositoryDiscoveryResult,
} from '../repo-discovery';

export interface EvalCodegenOptions {
  lang?: string;
  framework?: string;
  includePrompt?: boolean;
  format?: string;
  out?: string;
  saveReport?: string;
  saveCode?: string;
  verifyOnly?: boolean;
  cacheOnly?: boolean;
}

interface EvaluationMetric {
  name?: string;
  formula?: string;
  value?: number | string | null;
}

interface Baseline {
  model_name?: string;
  scores?: Record<string, number | string | null>;
}

interface PaperEvaluationSummary {
  task_type?: string;
  evaluation_metrics?: EvaluationMetric[];
  datasets?: string[];
  baselines?: Baseline[];
  implementation_details?: string;
}

type TaskMode =
  | 'pairwise_preference'
  | 'prediction_reference'
  | 'ranking'
  | 'forecasting'
  | 'classification'
  | 'vision';

interface EvaluationGuidanceCriterion {
  name: string;
  description: string;
}

interface EvaluationGuidance {
  reproduction_focus: string;
  implementation_cautions: string[];
  judge_criteria: EvaluationGuidanceCriterion[];
  judge_failure_conditions: string[];
}

type MetricPlanKind =
  | 'pairwise_win_rate'
  | 'feature_coefficient_placeholder'
  | 'accuracy'
  | 'top_k_accuracy'
  | 'exact_match'
  | 'precision'
  | 'precision_at_k'
  | 'recall'
  | 'recall_at_k'
  | 'f1'
  | 'silhouette_score'
  | 'normalized_mutual_info'
  | 'adjusted_rand_index'
  | 'average_bio_score'
  | 'batch_integration_score'
  | 'principal_component_regression_score'
  | 'average_batch_score'
  | 'matthews_corrcoef'
  | 'mse'
  | 'pearson'
  | 'spearman'
  | 'kendall_tau'
  | 'auc'
  | 'average_precision'
  | 'brier_score'
  | 'expected_calibration_error'
  | 'negative_log_likelihood'
  | 'perplexity'
  | 'iou'
  | 'map'
  | 'mask_map'
  | 'fid'
  | 'psnr'
  | 'ssim'
  | 'mae'
  | 'rmse'
  | 'mape'
  | 'smape'
  | 'dtw'
  | 'edit_similarity'
  | 'token_overlap_recall'
  | 'sigmoid_score_average'
  | 'greedy_token_similarity'
  | 'bertscore_precision'
  | 'bertscore_recall'
  | 'bertscore_f1'
  | 'wer'
  | 'cer'
  | 'average_return'
  | 'average_max_state_value'
  | 'success_rate'
  | 'pass_at_k'
  | 'normalized_score'
  | 'expected_entropy'
  | 'amino_acid_recovery'
  | 'runtime_seconds'
  | 'speedup_ratio'
  | 'rank_percent'
  | 'winning_count'
  | 'path_length'
  | 'judge_score_aggregate'
  | 'relevant_noise_sensitivity'
  | 'irrelevant_noise_sensitivity'
  | 'rmsd'
  | 'ndcg'
  | 'mrr'
  | 'hit_rate'
  | 'library_metric'
  | 'custom_placeholder';

interface MetricPlan {
  metricName: string;
  outputKey: string;
  functionName: string;
  kind: MetricPlanKind;
  formula?: string;
  note?: string;
  resultKey?: string;
}

interface MetricResolutionDecision {
  metric_name: string;
  recommended_kind: MetricPlanKind;
  note: string;
  required_metadata_keys: string[];
}

interface MetricResolutionResult {
  resolutions: MetricResolutionDecision[];
}

const METRIC_RESOLUTION_KIND_VALUES: MetricPlanKind[] = [
  'pairwise_win_rate',
  'feature_coefficient_placeholder',
  'accuracy',
  'top_k_accuracy',
  'exact_match',
  'precision',
  'precision_at_k',
  'recall',
  'recall_at_k',
  'f1',
  'silhouette_score',
  'normalized_mutual_info',
  'adjusted_rand_index',
  'average_bio_score',
  'batch_integration_score',
  'principal_component_regression_score',
  'average_batch_score',
  'matthews_corrcoef',
  'mse',
  'pearson',
  'spearman',
  'kendall_tau',
  'auc',
  'average_precision',
  'brier_score',
  'expected_calibration_error',
  'negative_log_likelihood',
  'perplexity',
  'iou',
  'map',
  'mask_map',
  'fid',
  'psnr',
  'ssim',
  'mae',
  'rmse',
  'mape',
  'smape',
  'dtw',
  'edit_similarity',
  'token_overlap_recall',
  'sigmoid_score_average',
  'greedy_token_similarity',
  'bertscore_precision',
  'bertscore_recall',
  'bertscore_f1',
  'wer',
  'cer',
  'average_return',
  'average_max_state_value',
  'success_rate',
  'pass_at_k',
  'normalized_score',
  'expected_entropy',
  'amino_acid_recovery',
  'runtime_seconds',
  'speedup_ratio',
  'rank_percent',
  'winning_count',
  'path_length',
  'judge_score_aggregate',
  'relevant_noise_sensitivity',
  'irrelevant_noise_sensitivity',
  'rmsd',
  'ndcg',
  'mrr',
  'hit_rate',
  'library_metric',
  'custom_placeholder',
];

const DEFAULT_SOLAR_MODEL = process.env.UPSTAGE_SOLAR_MODEL?.trim() || 'solar-pro3';
const EVALUATION_GUIDANCE_SCHEMA: JsonSchema = {
  type: 'object',
  additionalProperties: false,
  required: [
    'reproduction_focus',
    'implementation_cautions',
    'judge_criteria',
    'judge_failure_conditions',
  ],
  properties: {
    reproduction_focus: { type: 'string' },
    implementation_cautions: {
      type: 'array',
      items: { type: 'string' },
    },
    judge_criteria: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['name', 'description'],
        properties: {
          name: { type: 'string' },
          description: { type: 'string' },
        },
      },
    },
    judge_failure_conditions: {
      type: 'array',
      items: { type: 'string' },
    },
  },
};

const METRIC_RESOLUTION_SCHEMA: JsonSchema = {
  type: 'object',
  additionalProperties: false,
  required: ['resolutions'],
  properties: {
    resolutions: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['metric_name', 'recommended_kind', 'note', 'required_metadata_keys'],
        properties: {
          metric_name: { type: 'string' },
          recommended_kind: {
            type: 'string',
            enum: [...METRIC_RESOLUTION_KIND_VALUES],
          },
          note: { type: 'string' },
          required_metadata_keys: {
            type: 'array',
            items: { type: 'string' },
          },
        },
      },
    },
  },
};

export async function evalCodegenCommand(
  paper: string,
  options: EvalCodegenOptions,
): Promise<void> {
  const apiKey = await resolveApiKey();
  if (!apiKey) {
    throw new Error(
      'UPSTAGE_API_KEY is not set. Export it or run `upstage-research install --skills` to store it in config.',
    );
  }

  const inputPath = path.resolve(paper);
  const language = pickLanguage([paper]);
  const requestedLanguage = options.lang?.trim() || 'python';
  const requestedFramework = options.framework?.trim() || 'auto';
  const client = new UpstageClient({
    apiKey,
    progress: (message) => console.error(message),
    cacheOnly: options.cacheOnly,
  });

  const schema = await loadSchema(
    path.resolve(
      __dirname,
      '..',
      '..',
      'skills',
      'paper-eval-codegen',
      'references',
      'eval-schema.json',
    ),
  );
  const metricsByDomain = await fs.readFile(
    path.resolve(
      __dirname,
      '..',
      '..',
      'skills',
      'paper-eval-codegen',
      'references',
      'metrics-by-domain.md',
    ),
    'utf8',
  );
  const promptTemplates = await fs.readFile(
    path.resolve(
      __dirname,
      '..',
      '..',
      'skills',
      'paper-eval-codegen',
      'references',
      'prompt-templates.md',
    ),
    'utf8',
  );

  console.error(`Parsing ${path.basename(inputPath)}`);
  const parsed = await client.parseDocument(inputPath);
  const experimentMarkdown = selectExperimentExcerpt(parsed.markdown);

  console.error('Extracting experiment structure');
  const extractedSummary = await client.informationExtract<PaperEvaluationSummary>({
    schemaName: 'paper_evaluation_summary',
    schema,
    filePath: inputPath,
    instruction:
      'Extract evaluation setup details from this academic paper. Focus on the experiment section, result tables, baselines, datasets, implementation details, and directly measurable evaluation metrics with values. Prefer executable metrics such as accuracy, F1, BLEU, ROUGE, MAE, MSE, RMSE, mAP, IoU, NDCG, hit rate, reward, return, success rate, RMSD, and similar reported scores. Do not treat qualitative claims, training fit statements, theoretical properties, or complexity claims as evaluation metrics unless the paper explicitly reports them as reproducible measured metrics with a clear formula. Use concise factual text. If a field is missing, return an empty string or an empty array.',
    contextText:
      `File name: ${path.basename(inputPath)}\n\n` +
      `Experiment markdown excerpt from document parse:\n${clipText(experimentMarkdown, 28_000)}`,
  });
  const repairedSummary = await maybeRepairEvaluationSummary(
    client,
    extractedSummary,
    schema,
    experimentMarkdown,
    path.basename(inputPath),
    language,
  );
  const summary = reconcileSummaryWithEvidence(
    enrichEvaluationSummary(repairedSummary, experimentMarkdown),
    parsed.markdown,
  );

  const domain = inferDomainFromRegistry(summary);
  const taskMode = inferTaskModeFromRegistry(summary);
  const recommendedLibraries = libraryHintsByDomainFromRegistry(domain);
  const promptEvidenceRecords = buildEvaluationEvidenceRecords(summary, experimentMarkdown, parsed.anchors);
  const promptEvidenceSnippets = promptEvidenceRecords.map((record) => record.text);
  const discoveredRepositories = await discoverPaperRepositories({
    filePath: inputPath,
    markdown: parsed.markdown,
    cacheOnly: options.cacheOnly,
    progress: (message) => console.error(message),
  });
  const metricPlans = applyGeneralizedFallbackPolicy(await resolveMetricPlans(
    client,
    summary,
    experimentMarkdown,
    promptEvidenceSnippets,
    taskMode,
    language,
  ), discoveredRepositories, language);

  console.error('Generating structured evaluation guidance');
  const guidance = await generateEvaluationGuidance(
    client,
    summary,
    experimentMarkdown,
    promptEvidenceSnippets,
    promptTemplates,
    domain,
    taskMode,
    language,
  );

  console.error('Building evaluation artifact');
  const evaluationCode =
    requestedLanguage.toLowerCase() === 'python'
      ? buildPythonEvaluationCode(summary, guidance, domain, taskMode, requestedFramework, metricPlans)
      : await generateEvaluationCodeWithModel(
          client,
          summary,
          experimentMarkdown,
          requestedLanguage,
          requestedFramework,
          domain,
          recommendedLibraries,
          metricsByDomain,
        );
  const evidenceRecords = buildEvaluationEvidenceRecords(summary, parsed.markdown, parsed.anchors);
  const verification = buildEvaluationVerification(summary, metricPlans, evidenceRecords, language);
  const judgePrompt = options.includePrompt ? buildJudgePrompt(summary, guidance, language) : null;

  const judgePromptSection = options.includePrompt
    ? buildJudgePromptSection(judgePrompt ?? '', language)
    : '';

  const markdown = [
    language === 'ko' ? '## 평가 코드' : '## Evaluation Code',
    '',
    ensureCodeFence(evaluationCode, requestedLanguage),
    '',
    buildChecklistSection(summary, domain, requestedFramework, parsed.mode, language, guidance, metricPlans),
    '',
    renderVerificationSection(verification, language),
    '',
    buildRepositorySection(discoveredRepositories, metricPlans, language),
    '',
    buildMetricsSection(summary, language),
    '',
    buildEvidenceSection(evidenceRecords, language),
    judgePromptSection ? `\n${judgePromptSection}` : '',
  ]
    .join('\n')
    .trim();

  const outputFormat = normalizeOutputFormat(options.format);
  const jsonPayload = {
    paper: path.basename(inputPath),
    parse_mode: parsed.mode,
    requested_language: requestedLanguage,
    requested_framework: requestedFramework,
    summary,
    domain,
    task_mode: taskMode,
    metric_plans: metricPlans,
    guidance,
    verification,
    evidence_records: evidenceRecords,
    repository_discovery: discoveredRepositories,
    evaluation_code: evaluationCode,
    judge_prompt: judgePrompt,
    markdown,
  };
  const primaryOutput = options.verifyOnly
    ? (
      outputFormat === 'json'
        ? `${JSON.stringify({
          paper: jsonPayload.paper,
          parse_mode: jsonPayload.parse_mode,
          summary: jsonPayload.summary,
          domain: jsonPayload.domain,
          task_mode: jsonPayload.task_mode,
          metric_plans: jsonPayload.metric_plans,
          verification: jsonPayload.verification,
          evidence_records: jsonPayload.evidence_records,
        }, null, 2)}\n`
        : `${renderVerificationSection(verification, language)}\n\n${buildEvidenceSection(evidenceRecords, language)}\n`
    )
    : (
      outputFormat === 'json'
        ? `${JSON.stringify(jsonPayload, null, 2)}\n`
        : `${markdown}\n`
    );

  const reportPath = resolvePrimaryOutputPath(options.out, options.saveReport);
  if (reportPath) {
    await writeOutputFile(reportPath, primaryOutput);
    console.error(`Wrote report to ${reportPath}`);
  }

  if (options.saveCode?.trim()) {
    const codePath = path.resolve(options.saveCode.trim());
    await writeOutputFile(codePath, `${evaluationCode.trimEnd()}\n`);
    console.error(`Wrote evaluation code to ${codePath}`);
  }

  process.stdout.write(primaryOutput);
}

async function loadSchema(schemaPath: string): Promise<JsonSchema> {
  const raw = await fs.readFile(schemaPath, 'utf8');
  return JSON.parse(raw) as JsonSchema;
}

async function generateEvaluationGuidance(
  client: UpstageClient,
  summary: PaperEvaluationSummary,
  experimentMarkdown: string,
  evidenceSnippets: string[],
  promptTemplates: string,
  domain: string,
  taskMode: TaskMode,
  language: 'ko' | 'en',
): Promise<EvaluationGuidance> {
  try {
    const guidance = await client.chatStructured<EvaluationGuidance>({
      model: DEFAULT_SOLAR_MODEL,
      schemaName: 'evaluation_guidance',
      schema: EVALUATION_GUIDANCE_SCHEMA,
      messages: [
        {
          role: 'system',
          content:
            language === 'ko'
              ? '당신은 논문 재현용 평가 파이프라인을 설계하는 실무형 ML 엔지니어다. 길게 설명하지 말고 실행에 필요한 주의점과 평가 기준만 구조화해서 반환하라.'
              : 'You are a pragmatic ML engineer designing paper reproduction evaluations. Return only concise implementation cautions and judging criteria in structured form.',
        },
        {
          role: 'user',
          content:
            `Research area hint: ${domain}\n` +
            `Task mode: ${taskMode}\n\n` +
            `Structured extraction:\n${JSON.stringify(summary, null, 2)}\n\n` +
            `Evidence snippets:\n${evidenceSnippets.map((snippet) => `- ${snippet}`).join('\n') || '- none'}\n\n` +
            `Prompt templates:\n${clipText(promptTemplates, 6_000)}\n\n` +
            `Experiment markdown excerpt:\n${clipText(experimentMarkdown, 12_000)}\n\n` +
            'Return only high-signal guidance that will help generate reliable evaluation artifacts.',
        },
      ],
    });

    return normalizeEvaluationGuidance(guidance, language);
  } catch {
    return buildFallbackEvaluationGuidance(summary, taskMode, evidenceSnippets, language);
  }
}

async function generateEvaluationCodeWithModel(
  client: UpstageClient,
  summary: PaperEvaluationSummary,
  experimentMarkdown: string,
  requestedLanguage: string,
  requestedFramework: string,
  domain: string,
  recommendedLibraries: string,
  metricsByDomain: string,
): Promise<string> {
  return await client.chatCompletion({
    model: DEFAULT_SOLAR_MODEL,
    temperature: 0.1,
    messages: [
      {
        role: 'system',
        content:
          'You write practical evaluation code for research reproduction. Return only one fenced code block and no commentary. Use only the paper metrics when they are available. Do not add generic benchmark metrics unless the paper is missing them.',
      },
      {
        role: 'user',
        content:
          `Generate runnable ${requestedLanguage} evaluation code for this paper.\n\n` +
          `Requested framework: ${requestedFramework}\n` +
          `Research area hint: ${domain}\n` +
          `Recommended libraries: ${recommendedLibraries}\n\n` +
          `Structured extraction:\n${JSON.stringify(summary, null, 2)}\n\n` +
          `Reference notes:\n${metricsByDomain}\n\n` +
          `Experiment markdown excerpt:\n${clipText(experimentMarkdown, 18_000)}\n\n` +
          'Requirements:\n' +
          '- Include imports.\n' +
          '- Define a main evaluation function.\n' +
          '- Use the paper metrics and datasets when available.\n' +
          '- Do not invent generic benchmark metrics when the paper already specifies them.\n' +
          '- Every referenced variable, helper, and import must be defined inside the code block or clearly passed as a function argument.\n' +
          '- Return only one fenced code block.',
      },
    ],
  });
}

async function resolveMetricPlans(
  client: UpstageClient,
  summary: PaperEvaluationSummary,
  experimentMarkdown: string,
  evidenceSnippets: string[],
  taskMode: TaskMode,
  language: 'ko' | 'en',
): Promise<MetricPlan[]> {
  const initialPlans = buildMetricPlansFromRegistry(summary, taskMode);
  const unresolvedPlans = initialPlans.filter((plan) =>
    plan.kind === 'custom_placeholder' || plan.kind === 'feature_coefficient_placeholder',
  );

  if (unresolvedPlans.length === 0) {
    return initialPlans;
  }

  try {
    const resolved = await client.chatStructured<MetricResolutionResult>({
      model: DEFAULT_SOLAR_MODEL,
      schemaName: 'metric_resolution',
      schema: METRIC_RESOLUTION_SCHEMA,
      temperature: 0.1,
      messages: [
        {
          role: 'system',
          content:
            language === 'ko'
              ? '당신은 논문 평가지표를 안전한 구현 family로 분류하는 ML 평가 엔지니어다. 강한 근거가 있을 때만 supported family로 매핑하고, 애매하면 custom_placeholder를 유지하라. 새로운 metric을 발명하지 마라.'
              : 'You map paper metrics onto safe implementation families for evaluation code. Only map when the evidence is strong. Keep custom_placeholder when ambiguous. Do not invent new metrics.',
        },
        {
          role: 'user',
          content:
            `Task mode: ${taskMode}\n` +
            `Structured extraction:\n${JSON.stringify(summary, null, 2)}\n\n` +
            `Evidence snippets:\n${evidenceSnippets.map((snippet) => `- ${snippet}`).join('\n') || '- none'}\n\n` +
            `Unresolved metrics:\n${JSON.stringify(
              unresolvedPlans.map((plan) => ({
                metric_name: plan.metricName,
                formula: plan.formula ?? '',
                note: plan.note ?? '',
              })),
              null,
              2,
            )}\n\n` +
            `Supported implementation families:\n${describeSupportedMetricKinds()}\n\n` +
            `Experiment markdown excerpt:\n${clipText(experimentMarkdown, 12_000)}\n\n` +
            'Return one resolution object per unresolved metric. Use custom_placeholder if the paper-specific procedure is not clear enough.',
        },
      ],
    });

    return applyMetricResolutionDecisions(initialPlans, normalizeMetricResolutionResult(resolved));
  } catch {
    return initialPlans;
  }
}

function normalizeMetricResolutionResult(result: MetricResolutionResult): MetricResolutionResult {
  return {
    resolutions: (result.resolutions ?? [])
      .filter((resolution) => resolution.metric_name?.trim())
      .map((resolution) => ({
        metric_name: resolution.metric_name.trim(),
        recommended_kind: METRIC_RESOLUTION_KIND_VALUES.includes(resolution.recommended_kind)
          ? resolution.recommended_kind
          : 'custom_placeholder',
        note: resolution.note?.trim() || '',
        required_metadata_keys: (resolution.required_metadata_keys ?? [])
          .map((key) => key.trim())
          .filter(Boolean)
          .slice(0, 8),
      })),
  };
}

function applyMetricResolutionDecisions(
  metricPlans: MetricPlan[],
  result: MetricResolutionResult,
): MetricPlan[] {
  const decisionByMetric = new Map(
    result.resolutions.map((resolution) => [resolution.metric_name.toLowerCase(), resolution]),
  );

  return metricPlans.map((plan) => {
    if (plan.kind !== 'custom_placeholder' && plan.kind !== 'feature_coefficient_placeholder') {
      return plan;
    }

    const resolution = decisionByMetric.get(plan.metricName.toLowerCase());
    if (!resolution || resolution.recommended_kind === 'custom_placeholder') {
      return resolution
        ? {
            ...plan,
            note: mergeMetricNotes(plan.note, resolution.note, resolution.required_metadata_keys),
          }
        : plan;
    }

    return {
      ...plan,
      kind: resolution.recommended_kind,
      note: mergeMetricNotes(plan.note, resolution.note, resolution.required_metadata_keys),
    };
  });
}

function mergeMetricNotes(
  primary: string | undefined,
  secondary: string | undefined,
  requiredMetadataKeys: string[],
): string {
  const unique = new Set<string>();
  const basePieces = [primary?.trim(), secondary?.trim()].filter((piece): piece is string => Boolean(piece));
  const pieces = basePieces.filter((piece) => {
    const normalized = piece.replace(/\s+/g, ' ').trim();
    if (!normalized || unique.has(normalized)) {
      return false;
    }

    unique.add(normalized);
    return true;
  }) as string[];
  if (requiredMetadataKeys.length > 0) {
    pieces.push(`Metadata keys: ${requiredMetadataKeys.join(', ')}`);
  }

  return pieces.join(' ');
}

function describeSupportedMetricKinds(): string {
  return [
    '- accuracy / top_k_accuracy: exact correctness on labels, optionally top-k candidates',
    '- silhouette_score / normalized_mutual_info / adjusted_rand_index / average_bio_score / batch_integration_score / principal_component_regression_score / average_batch_score: clustering and batch integration metrics for embedding-based evaluations',
    '- precision / recall / f1 / matthews_corrcoef / auc / average_precision / brier_score / expected_calibration_error / negative_log_likelihood / perplexity',
    '- ndcg / mrr / hit_rate / precision_at_k / recall_at_k: ranking or retrieval metrics',
    '- iou / map / mask_map / fid / psnr / ssim: vision generation, detection, segmentation, restoration',
    '- mae / rmse / mape / smape / dtw / pearson / spearman / kendall_tau / rmsd: numeric, sequence, structure, or correlation metrics',
    '- wer / cer / edit_similarity / token_overlap_recall / sigmoid_score_average / greedy_token_similarity: string, token, or lightweight lexical/embedding similarity metrics',
    '- average_return / average_max_state_value / normalized_score / success_rate / pass_at_k / expected_entropy / runtime_seconds / speedup_ratio / rank_percent / winning_count / path_length / amino_acid_recovery / judge_score_aggregate',
    '- library_metric: standard text metrics like BLEU, ROUGE, BERTScore',
    '- custom_placeholder: keep this when the metric requires a paper-specific protocol or insufficient evidence is available',
  ].join('\n');
}

function selectExperimentExcerpt(markdown: string): string {
  const section = extractSectionByHeadings(markdown, {
    startPatterns: [
      /^#+\s*(experiments?|experimental setup|evaluation|results?|benchmarks?|implementation details?|ablations?|analysis|case stud(?:y|ies)|main results?)\b/i,
      /^#+\s*(실험|평가|결과|벤치마크|구현|분석|어블레이션|사례 연구)\b/i,
    ],
    stopPatterns: [
      /^#+\s*(conclusion|discussion|limitations?|future work|references?|appendix)\b/i,
      /^#+\s*(결론|토론|한계|향후 연구|참고문헌|부록)\b/i,
    ],
    limit: 32_000,
  });

  if (section.trim()) {
    return section;
  }

  return markdown.slice(0, 32_000);
}

function extractSectionByHeadings(
  markdown: string,
  options: {
    startPatterns: RegExp[];
    stopPatterns: RegExp[];
    limit: number;
  },
): string {
  const lines = markdown.split('\n');
  const selected: string[] = [];
  let collecting = false;

  for (const line of lines) {
    if (options.startPatterns.some((pattern) => pattern.test(line))) {
      collecting = true;
    }

    if (collecting) {
      if (
        selected.length > 0 &&
        options.stopPatterns.some((pattern) => pattern.test(line))
      ) {
        break;
      }

      selected.push(line);
      if (selected.join('\n').length >= options.limit) {
        break;
      }
    }
  }

  return selected.join('\n').trim();
}

function buildPythonEvaluationCode(
  summary: PaperEvaluationSummary,
  guidance: EvaluationGuidance,
  domain: string,
  taskMode: TaskMode,
  framework: string,
  metricPlans: MetricPlan[],
): string {
  const lines: string[] = [];
  const usesTextAlignmentHelpers = metricPlans.some((plan) =>
    ['token_overlap_recall', 'sigmoid_score_average', 'greedy_token_similarity'].includes(plan.kind),
  );
  const usesClusterMetrics = metricPlans.some((plan) =>
    [
      'silhouette_score',
      'normalized_mutual_info',
      'adjusted_rand_index',
      'average_bio_score',
      'batch_integration_score',
      'principal_component_regression_score',
      'average_batch_score',
    ].includes(plan.kind),
  );

  lines.push('from __future__ import annotations');
  lines.push('');
  lines.push('from typing import Any, Dict, List, Mapping, Optional, Sequence');
  lines.push('import math');
  if (usesTextAlignmentHelpers) {
    lines.push('import re');
    lines.push('from difflib import SequenceMatcher');
  }
  lines.push('');
  lines.push(`# Task type: ${fallbackText(summary.task_type, 'unknown')}`);
  lines.push(`# Research area hint: ${domain}`);
  lines.push(`# Framework hint: ${framework}`);
  lines.push(`# Datasets: ${formatList(summary.datasets, 'not specified')}`);
  lines.push(`# Metrics: ${metricPlans.map((plan) => plan.metricName).join(', ') || 'not specified'}`);
  lines.push('');
  lines.push(`REPRODUCTION_FOCUS = ${toPythonString(fallbackText(guidance.reproduction_focus, 'Reproduce the reported evaluation setup as closely as possible.'))}`);
  lines.push(`DATASETS = ${toPythonList(summary.datasets ?? [])}`);
  lines.push(`IMPLEMENTATION_CAUTIONS = ${toPythonList(guidance.implementation_cautions)}`);
  lines.push(`EXPECTED_BASELINES = ${toPythonBaselines(summary.baselines ?? [])}`);
  lines.push('');
  lines.push('def _safe_divide(numerator: float, denominator: float) -> float:');
  lines.push('    return numerator / denominator if denominator else 0.0');
  lines.push('');
  lines.push('def _flatten_numeric(value: Any) -> List[float]:');
  lines.push('    if isinstance(value, Mapping):');
  lines.push('        flattened: List[float] = []');
  lines.push('        for nested in value.values():');
  lines.push('            flattened.extend(_flatten_numeric(nested))');
  lines.push('        return flattened');
  lines.push('    if isinstance(value, (list, tuple, set)):');
  lines.push('        flattened: List[float] = []');
  lines.push('        for nested in value:');
  lines.push('            flattened.extend(_flatten_numeric(nested))');
  lines.push('        return flattened');
  lines.push('    if isinstance(value, bool):');
  lines.push('        return [1.0 if value else 0.0]');
  lines.push('    if isinstance(value, (int, float)):');
  lines.push('        return [float(value)]');
  lines.push('    if isinstance(value, str):');
  lines.push('        try:');
  lines.push('            return [float(value.strip())]');
  lines.push('        except ValueError:');
  lines.push('            return []');
  lines.push('    return []');
  lines.push('');
  lines.push('def _to_scalar(value: Any, default: float = 0.0) -> float:');
  lines.push('    flattened = _flatten_numeric(value)');
  lines.push('    return _safe_divide(sum(flattened), float(len(flattened))) if flattened else default');
  lines.push('');
  if (usesClusterMetrics) {
    lines.push('def _coerce_embedding_rows(values: Sequence[Any]) -> List[List[float]]:');
    lines.push('    rows: List[List[float]] = []');
    lines.push('    for item in values:');
    lines.push('        if isinstance(item, Mapping):');
    lines.push('            extracted = None');
    lines.push('            for key in ("embedding", "embeddings", "vector", "vectors", "features", "representation"):');
    lines.push('                if key in item:');
    lines.push('                    extracted = item[key]');
    lines.push('                    break');
    lines.push('            numeric = _flatten_numeric(extracted) if extracted is not None else _flatten_numeric(item)');
    lines.push('        else:');
    lines.push('            numeric = _flatten_numeric(item)');
    lines.push('        if numeric:');
    lines.push('            rows.append(numeric)');
    lines.push('    return rows');
    lines.push('');
    lines.push('def _extract_label_sequence(values: Sequence[Any], keys: Sequence[str]) -> List[str]:');
    lines.push('    labels: List[str] = []');
    lines.push('    for item in values:');
    lines.push('        if isinstance(item, Mapping):');
    lines.push('            value = None');
    lines.push('            for key in keys:');
    lines.push('                if key in item and item[key] is not None:');
    lines.push('                    value = item[key]');
    lines.push('                    break');
    lines.push('            if value is None and isinstance(item.get("metadata"), Mapping):');
    lines.push('                metadata = item.get("metadata", {})');
    lines.push('                for key in keys:');
    lines.push('                    if key in metadata and metadata[key] is not None:');
    lines.push('                        value = metadata[key]');
    lines.push('                        break');
    lines.push('            labels.append("" if value is None else str(value))');
    lines.push('        else:');
    lines.push('            labels.append(str(item))');
    lines.push('    return labels');
    lines.push('');
    lines.push('def _label_indices(labels: Sequence[str]) -> List[int]:');
    lines.push('    index_by_label: Dict[str, int] = {}');
    lines.push('    encoded: List[int] = []');
    lines.push('    for label in labels:');
    lines.push('        if label not in index_by_label:');
    lines.push('            index_by_label[label] = len(index_by_label)');
    lines.push('        encoded.append(index_by_label[label])');
    lines.push('    return encoded');
    lines.push('');
    lines.push('def _euclidean_distance(left: Sequence[float], right: Sequence[float]) -> float:');
    lines.push('    size = max(len(left), len(right))');
    lines.push('    total = 0.0');
    lines.push('    for index in range(size):');
    lines.push('        left_value = float(left[index]) if index < len(left) else 0.0');
    lines.push('        right_value = float(right[index]) if index < len(right) else 0.0');
    lines.push('        total += (left_value - right_value) ** 2');
    lines.push('    return math.sqrt(total)');
    lines.push('');
    lines.push('def _silhouette_from_embeddings(rows: Sequence[Sequence[float]], labels: Sequence[str]) -> float:');
    lines.push('    if len(rows) <= 1 or len(rows) != len(labels):');
    lines.push('        return 0.0');
    lines.push('    unique_labels = sorted(set(labels))');
    lines.push('    if len(unique_labels) <= 1:');
    lines.push('        return 0.0');
    lines.push('    distances = [[0.0] * len(rows) for _ in rows]');
    lines.push('    for row_index in range(len(rows)):');
    lines.push('        for col_index in range(row_index + 1, len(rows)):');
    lines.push('            distance = _euclidean_distance(rows[row_index], rows[col_index])');
    lines.push('            distances[row_index][col_index] = distance');
    lines.push('            distances[col_index][row_index] = distance');
    lines.push('    scores: List[float] = []');
    lines.push('    for row_index, label in enumerate(labels):');
    lines.push('        same_cluster = [distances[row_index][col_index] for col_index in range(len(rows)) if col_index != row_index and labels[col_index] == label]');
    lines.push('        intra = _safe_divide(sum(same_cluster), float(len(same_cluster))) if same_cluster else 0.0');
    lines.push('        nearest = None');
    lines.push('        for other_label in unique_labels:');
    lines.push('            if other_label == label:');
    lines.push('                continue');
    lines.push('            candidate = [distances[row_index][col_index] for col_index in range(len(rows)) if labels[col_index] == other_label]');
    lines.push('            if not candidate:');
    lines.push('                continue');
    lines.push('            mean_distance = _safe_divide(sum(candidate), float(len(candidate)))');
    lines.push('            nearest = mean_distance if nearest is None else min(nearest, mean_distance)');
    lines.push('        if nearest is None:');
    lines.push('            scores.append(0.0)');
    lines.push('            continue');
    lines.push('        denominator = max(intra, nearest)');
    lines.push('        scores.append(_safe_divide(nearest - intra, denominator) if denominator else 0.0)');
    lines.push('    return _safe_divide(sum(scores), float(len(scores)))');
    lines.push('');
    lines.push('def _contingency_table(left_labels: Sequence[str], right_labels: Sequence[str]) -> Dict[str, Any]:');
    lines.push('    rows = _label_indices(left_labels)');
    lines.push('    cols = _label_indices(right_labels)');
    lines.push('    counts: Dict[int, Dict[int, int]] = {}');
    lines.push('    row_totals: Dict[int, int] = {}');
    lines.push('    col_totals: Dict[int, int] = {}');
    lines.push('    for row_label, col_label in zip(rows, cols):');
    lines.push('        bucket = counts.setdefault(row_label, {})');
    lines.push('        bucket[col_label] = bucket.get(col_label, 0) + 1');
    lines.push('        row_totals[row_label] = row_totals.get(row_label, 0) + 1');
    lines.push('        col_totals[col_label] = col_totals.get(col_label, 0) + 1');
    lines.push('    return {"counts": counts, "row_totals": row_totals, "col_totals": col_totals, "total": len(rows)}');
    lines.push('');
    lines.push('def _comb2(value: int) -> float:');
    lines.push('    return float(value * (value - 1) / 2) if value >= 2 else 0.0');
    lines.push('');
    lines.push('def _normalized_mutual_info_score(left_labels: Sequence[str], right_labels: Sequence[str]) -> float:');
    lines.push('    table = _contingency_table(left_labels, right_labels)');
    lines.push('    total = float(table["total"])');
    lines.push('    if total <= 0.0:');
    lines.push('        return 0.0');
    lines.push('    mutual_information = 0.0');
    lines.push('    for row_label, bucket in table["counts"].items():');
    lines.push('        for col_label, count in bucket.items():');
    lines.push('            probability = count / total');
    lines.push('            row_probability = table["row_totals"][row_label] / total');
    lines.push('            col_probability = table["col_totals"][col_label] / total');
    lines.push('            if probability > 0.0 and row_probability > 0.0 and col_probability > 0.0:');
    lines.push('                mutual_information += probability * math.log(probability / (row_probability * col_probability))');
    lines.push('    row_entropy = -sum((count / total) * math.log(count / total) for count in table["row_totals"].values() if count)');
    lines.push('    col_entropy = -sum((count / total) * math.log(count / total) for count in table["col_totals"].values() if count)');
    lines.push('    denominator = math.sqrt(row_entropy * col_entropy) if row_entropy > 0.0 and col_entropy > 0.0 else 0.0');
    lines.push('    return _safe_divide(mutual_information, denominator) if denominator else 0.0');
    lines.push('');
    lines.push('def _adjusted_rand_index_score(left_labels: Sequence[str], right_labels: Sequence[str]) -> float:');
    lines.push('    table = _contingency_table(left_labels, right_labels)');
    lines.push('    total = table["total"]');
    lines.push('    if total <= 1:');
    lines.push('        return 0.0');
    lines.push('    index = sum(_comb2(count) for bucket in table["counts"].values() for count in bucket.values())');
    lines.push('    row_pairs = sum(_comb2(count) for count in table["row_totals"].values())');
    lines.push('    col_pairs = sum(_comb2(count) for count in table["col_totals"].values())');
    lines.push('    all_pairs = _comb2(total)');
    lines.push('    expected = _safe_divide(row_pairs * col_pairs, all_pairs)');
    lines.push('    maximum = 0.5 * (row_pairs + col_pairs)');
    lines.push('    denominator = maximum - expected');
    lines.push('    return _safe_divide(index - expected, denominator) if denominator else 0.0');
    lines.push('');
    lines.push('def _batch_effect_ratio(rows: Sequence[Sequence[float]], labels: Sequence[str]) -> float:');
    lines.push('    if not rows or len(rows) != len(labels):');
    lines.push('        return 0.0');
    lines.push('    dimension = max((len(row) for row in rows), default=0)');
    lines.push('    if dimension == 0:');
    lines.push('        return 0.0');
    lines.push('    global_mean = [0.0] * dimension');
    lines.push('    for row in rows:');
    lines.push('        for index in range(dimension):');
    lines.push('            global_mean[index] += float(row[index]) if index < len(row) else 0.0');
    lines.push('    global_mean = [value / float(len(rows)) for value in global_mean]');
    lines.push('    total_ss = 0.0');
    lines.push('    group_sums: Dict[str, List[float]] = {}');
    lines.push('    group_counts: Dict[str, int] = {}');
    lines.push('    for row, label in zip(rows, labels):');
    lines.push('        group_sums.setdefault(label, [0.0] * dimension)');
    lines.push('        group_counts[label] = group_counts.get(label, 0) + 1');
    lines.push('        for index in range(dimension):');
    lines.push('            value = float(row[index]) if index < len(row) else 0.0');
    lines.push('            group_sums[label][index] += value');
    lines.push('            total_ss += (value - global_mean[index]) ** 2');
    lines.push('    if total_ss == 0.0:');
    lines.push('        return 0.0');
    lines.push('    between_ss = 0.0');
    lines.push('    for label, sums in group_sums.items():');
    lines.push('        count = float(group_counts[label])');
    lines.push('        mean = [value / count for value in sums]');
    lines.push('        between_ss += count * sum((mean[index] - global_mean[index]) ** 2 for index in range(dimension))');
    lines.push('    return _safe_divide(between_ss, total_ss)');
    lines.push('');
  }
  if (usesTextAlignmentHelpers) {
    lines.push('def _text_tokens(value: Any) -> List[str]:');
    lines.push('    if isinstance(value, Mapping):');
    lines.push('        for key in ("text", "prediction", "reference", "candidate", "answer", "content"):');
    lines.push('            if key in value:');
    lines.push('                return _text_tokens(value[key])');
    lines.push('        return [token for token in re.split(r"[^0-9A-Za-z_]+", " ".join(str(key) for key in value.keys()).lower()) if token]');
    lines.push('    if isinstance(value, str):');
    lines.push('        return [token for token in re.split(r"[^0-9A-Za-z_]+", value.lower()) if token]');
    lines.push('    if isinstance(value, (list, tuple, set)):');
    lines.push('        if all(isinstance(item, (int, float, bool)) for item in value):');
    lines.push('            return [str(item) for item in value]');
    lines.push('        flattened: List[str] = []');
    lines.push('        for item in value:');
    lines.push('            flattened.extend(_text_tokens(item))');
    lines.push('        return flattened');
    lines.push('    return [token for token in re.split(r"[^0-9A-Za-z_]+", str(value).lower()) if token]');
    lines.push('');
    lines.push('def _token_similarity(left: str, right: str) -> float:');
    lines.push('    return SequenceMatcher(None, left, right).ratio()');
    lines.push('');
    lines.push('def _greedy_token_similarity(left_tokens: Sequence[str], right_tokens: Sequence[str]) -> float:');
    lines.push('    if not left_tokens or not right_tokens:');
    lines.push('        return 0.0');
    lines.push('    scores = [max(_token_similarity(left, right) for right in right_tokens) for left in left_tokens]');
    lines.push('    return _safe_divide(sum(scores), float(len(scores)))');
    lines.push('');
    lines.push('def _pair_similarity_score(prediction: Any, reference: Any) -> float:');
    lines.push('    left_numeric = _flatten_numeric(prediction)');
    lines.push('    right_numeric = _flatten_numeric(reference)');
    lines.push('    if left_numeric and right_numeric and len(left_numeric) == len(right_numeric):');
    lines.push('        numerator = sum(left * right for left, right in zip(left_numeric, right_numeric))');
    lines.push('        left_norm = math.sqrt(sum(left * left for left in left_numeric))');
    lines.push('        right_norm = math.sqrt(sum(right * right for right in right_numeric))');
    lines.push('        cosine = _safe_divide(numerator, left_norm * right_norm)');
    lines.push('        if cosine != 0.0:');
    lines.push('            return cosine');
    lines.push('    left_tokens = _text_tokens(prediction)');
    lines.push('    right_tokens = _text_tokens(reference)');
    lines.push('    if left_tokens and right_tokens:');
    lines.push('        joined_left = " ".join(left_tokens)');
    lines.push('        joined_right = " ".join(right_tokens)');
    lines.push('        lexical = SequenceMatcher(None, joined_left, joined_right).ratio()');
    lines.push('        if lexical != 0.0:');
    lines.push('            return lexical');
    lines.push('        return _greedy_token_similarity(left_tokens, right_tokens)');
    lines.push('    return _to_scalar(prediction)');
    lines.push('');
  }
  lines.push('def _extract_vote_metric_inputs(pairwise_votes: Sequence[Mapping[str, Any]], metric_key: str) -> Dict[str, List[float]]:');
  lines.push('    prediction_keys = [f"{metric_key}_predictions", f"{metric_key}_prediction", "predictions", "prediction_scores", "automatic_scores", "judge_scores", "scores", "score"]');
  lines.push('    reference_keys = [f"{metric_key}_references", f"{metric_key}_reference", "references", "reference_scores", "human_scores", "labels", "targets", "ground_truth"]');
  lines.push('    predictions: List[float] = []');
  lines.push('    references: List[float] = []');
  lines.push('    for vote in pairwise_votes:');
  lines.push('        metadata = vote.get("metadata", {}) if isinstance(vote, Mapping) else {}');
  lines.push('        prediction_value = next((metadata[key] for key in prediction_keys if key in metadata), None)');
  lines.push('        reference_value = next((metadata[key] for key in reference_keys if key in metadata), None)');
  lines.push('        if prediction_value is None or reference_value is None:');
  lines.push('            continue');
  lines.push('        if isinstance(prediction_value, (list, tuple)) and isinstance(reference_value, (list, tuple)):');
  lines.push('            for left, right in zip(prediction_value, reference_value):');
  lines.push('                predictions.append(_to_scalar(left))');
  lines.push('                references.append(_to_scalar(right))');
  lines.push('            continue');
  lines.push('        predictions.append(_to_scalar(prediction_value))');
  lines.push('        references.append(_to_scalar(reference_value))');
  lines.push('    return {"predictions": predictions, "references": references}');
  lines.push('');

  if (metricPlans.some((plan) =>
    plan.kind === 'library_metric' || ['bertscore_precision', 'bertscore_recall', 'bertscore_f1'].includes(plan.kind),
  )) {
    lines.push('def _require_optional_dependency(module_name: str):');
    lines.push('    try:');
    lines.push('        module = __import__(module_name)');
    lines.push('    except ImportError:');
    lines.push('        return None');
    lines.push('    return module');
    lines.push('');
  }

  if (metricPlans.some((plan) =>
    ['bertscore_precision', 'bertscore_recall', 'bertscore_f1'].includes(plan.kind),
  )) {
    lines.push('def _compute_bertscore_payload(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, Any]:');
    lines.push('    evaluate = _require_optional_dependency("evaluate")');
    lines.push('    if evaluate is None:');
    lines.push('        return {"value": None, "note": "Install `evaluate` to compute BERTScore metrics."}');
    lines.push('    metric = evaluate.load("bertscore")');
    lines.push('    raw = metric.compute(predictions=list(predictions), references=list(references), lang="en")');
    lines.push('    precision = raw.get("precision", [])');
    lines.push('    recall = raw.get("recall", [])');
    lines.push('    f1 = raw.get("f1", [])');
    lines.push('    return {');
    lines.push('        "precision": _safe_divide(sum(float(value) for value in precision), float(len(precision))) if precision else None,');
    lines.push('        "recall": _safe_divide(sum(float(value) for value in recall), float(len(recall))) if recall else None,');
    lines.push('        "f1": _safe_divide(sum(float(value) for value in f1), float(len(f1))) if f1 else None,');
    lines.push('    }');
    lines.push('');
  }

  lines.push(...buildMetricFunctionBlocks(metricPlans, taskMode));
  lines.push(...buildEntryPointBlock(summary, metricPlans, taskMode));
  lines.push(...buildExampleUsageBlock(taskMode, metricPlans));

  return lines.join('\n').trim();
}

function buildMetricFunctionBlocks(metricPlans: MetricPlan[], taskMode: TaskMode): string[] {
  const blocks: string[] = [];
  const addedFunctions = new Set<string>();

  for (const plan of metricPlans) {
    if (addedFunctions.has(plan.functionName)) {
      continue;
    }

    addedFunctions.add(plan.functionName);
    switch (plan.kind) {
      case 'pairwise_win_rate':
        blocks.push(
          `def ${plan.functionName}(model_name: str, pairwise_votes: Sequence[Mapping[str, Any]]) -> float:`,
          '    relevant_votes = [vote for vote in pairwise_votes if vote.get("winner") == model_name or vote.get("loser") == model_name]',
          '    wins = sum(1 for vote in relevant_votes if vote.get("winner") == model_name)',
          '    return _safe_divide(float(wins), float(len(relevant_votes)))',
          '',
        );
        break;
      case 'feature_coefficient_placeholder':
        blocks.push(
          `def ${plan.functionName}(pairwise_votes: Sequence[Mapping[str, Any]], *, feature_name: str) -> Dict[str, Any]:`,
          `    """Placeholder for ${plan.metricName}. Fit the paper's preferred coefficient model here."""`,
          '    return {',
          '        "value": None,',
          '        "feature_name": feature_name,',
          `        "note": ${toPythonString(plan.note || 'TODO: implement the paper-specific coefficient fitting routine.')},`,
          plan.formula
            ? `        "formula": ${toPythonString(plan.formula)},`
            : '        "formula": None,',
          '        "num_votes": len(pairwise_votes),',
          '    }',
          '',
        );
        break;
      case 'accuracy':
      case 'exact_match':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> float:`,
          '    aligned = list(zip(predictions, references))',
          '    correct = sum(1 for prediction, reference in aligned if prediction == reference)',
          '    return _safe_divide(float(correct), float(len(aligned)))',
          '',
        );
        break;
      case 'top_k_accuracy':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any], *, k: int = 5) -> float:`,
          '    hits = 0',
          '    total = 0',
          '    for prediction, reference in zip(predictions, references):',
          '        total += 1',
          '        if isinstance(prediction, Mapping):',
          '            candidates = prediction.get("candidates") or prediction.get("labels") or prediction.get("top_k") or []',
          '        elif isinstance(prediction, (list, tuple, set)):',
          '            candidates = list(prediction)',
          '        else:',
          '            candidates = [prediction]',
          '        if reference in list(candidates)[:k]:',
          '            hits += 1',
          '    return _safe_divide(float(hits), float(total))',
          '',
        );
        break;
      case 'precision':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> float:`,
          '    aligned = list(zip(predictions, references))',
          '    true_positives = sum(1 for prediction, reference in aligned if bool(prediction) and bool(reference))',
          '    predicted_positives = sum(1 for prediction, _reference in aligned if bool(prediction))',
          '    return _safe_divide(float(true_positives), float(predicted_positives))',
          '',
        );
        break;
      case 'precision_at_k':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Sequence[Any]], references: Sequence[Sequence[Any]], *, k: int = 10) -> float:`,
          '    scores: List[float] = []',
          '    for ranked_items, relevant_items in zip(predictions, references):',
          '        top_items = list(ranked_items)[:k]',
          '        if not top_items:',
          '            scores.append(0.0)',
          '            continue',
          '        relevant_set = set(relevant_items)',
          '        hits = sum(1 for item in top_items if item in relevant_set)',
          '        scores.append(_safe_divide(float(hits), float(len(top_items))))',
          '    return _safe_divide(sum(scores), float(len(scores)))',
          '',
        );
        break;
      case 'recall':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> float:`,
          '    aligned = list(zip(predictions, references))',
          '    true_positives = sum(1 for prediction, reference in aligned if bool(prediction) and bool(reference))',
          '    actual_positives = sum(1 for _prediction, reference in aligned if bool(reference))',
          '    return _safe_divide(float(true_positives), float(actual_positives))',
          '',
        );
        break;
      case 'recall_at_k':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Sequence[Any]], references: Sequence[Sequence[Any]], *, k: int = 10) -> float:`,
          '    scores: List[float] = []',
          '    for ranked_items, relevant_items in zip(predictions, references):',
          '        relevant_list = list(relevant_items)',
          '        relevant_set = set(relevant_list)',
          '        hits = sum(1 for item in list(ranked_items)[:k] if item in relevant_set)',
          '        scores.append(_safe_divide(float(hits), float(len(relevant_list))))',
          '    return _safe_divide(sum(scores), float(len(scores)))',
          '',
        );
        break;
      case 'f1':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> float:`,
          '    precision = _safe_divide(',
          '        float(sum(1 for prediction, reference in zip(predictions, references) if bool(prediction) and bool(reference))),',
          '        float(sum(1 for prediction in predictions if bool(prediction))),',
          '    )',
          '    recall = _safe_divide(',
          '        float(sum(1 for prediction, reference in zip(predictions, references) if bool(prediction) and bool(reference))),',
          '        float(sum(1 for reference in references if bool(reference))),',
          '    )',
          '    return _safe_divide(2.0 * precision * recall, precision + recall)',
          '',
        );
        break;
      case 'silhouette_score':
        blocks.push(
          `def ${plan.functionName}(*, predictions: Optional[Sequence[Any]] = None, references: Optional[Sequence[Any]] = None, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:`,
          '    metadata = metadata or {}',
          '    rows = _coerce_embedding_rows(metadata.get("embeddings") or predictions or [])',
          '    labels = metadata.get("cluster_labels") or metadata.get("labels") or _extract_label_sequence(references or [], ("cluster_label", "label", "cell_type", "target"))',
          '    if not rows or not labels or len(rows) != len(labels):',
          '        return {"value": 0.0, "note": "Provide embeddings and cluster labels for silhouette-style evaluation."}',
          '    label_values = [str(label) for label in labels]',
          '    return {"value": _silhouette_from_embeddings(rows, label_values), "label_count": len(set(label_values))}',
          '',
        );
        break;
      case 'normalized_mutual_info':
        blocks.push(
          `def ${plan.functionName}(*, predictions: Optional[Sequence[Any]] = None, references: Optional[Sequence[Any]] = None, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:`,
          '    metadata = metadata or {}',
          '    cluster_labels = metadata.get("cluster_labels") or _extract_label_sequence(predictions or [], ("cluster_label", "cluster", "prediction", "label"))',
          '    true_labels = metadata.get("true_labels") or metadata.get("labels") or _extract_label_sequence(references or [], ("true_label", "label", "cell_type", "target"))',
          '    if not cluster_labels and true_labels:',
          '        cluster_labels = list(true_labels)',
          '    if not cluster_labels or not true_labels or len(cluster_labels) != len(true_labels):',
          '        return {"value": 0.0, "note": "Provide cluster labels and true labels for NMI evaluation."}',
          '    return {"value": _normalized_mutual_info_score([str(label) for label in cluster_labels], [str(label) for label in true_labels])}',
          '',
        );
        break;
      case 'adjusted_rand_index':
        blocks.push(
          `def ${plan.functionName}(*, predictions: Optional[Sequence[Any]] = None, references: Optional[Sequence[Any]] = None, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:`,
          '    metadata = metadata or {}',
          '    cluster_labels = metadata.get("cluster_labels") or _extract_label_sequence(predictions or [], ("cluster_label", "cluster", "prediction", "label"))',
          '    true_labels = metadata.get("true_labels") or metadata.get("labels") or _extract_label_sequence(references or [], ("true_label", "label", "cell_type", "target"))',
          '    if not cluster_labels and true_labels:',
          '        cluster_labels = list(true_labels)',
          '    if not cluster_labels or not true_labels or len(cluster_labels) != len(true_labels):',
          '        return {"value": 0.0, "note": "Provide cluster labels and true labels for ARI evaluation."}',
          '    return {"value": _adjusted_rand_index_score([str(label) for label in cluster_labels], [str(label) for label in true_labels])}',
          '',
        );
        break;
      case 'average_bio_score':
        blocks.push(
          `def ${plan.functionName}(*, predictions: Optional[Sequence[Any]] = None, references: Optional[Sequence[Any]] = None, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:`,
          '    metadata = metadata or {}',
          '    rows = _coerce_embedding_rows(metadata.get("embeddings") or predictions or [])',
          '    cluster_labels = metadata.get("cluster_labels") or _extract_label_sequence(predictions or [], ("cluster_label", "cluster", "prediction", "label"))',
          '    true_labels = metadata.get("true_labels") or metadata.get("labels") or _extract_label_sequence(references or [], ("true_label", "label", "cell_type", "target"))',
          '    if not cluster_labels and true_labels:',
          '        cluster_labels = list(true_labels)',
          '    cluster_text = [str(label) for label in cluster_labels] if cluster_labels else []',
          '    true_text = [str(label) for label in true_labels] if true_labels else []',
          '    asw = _silhouette_from_embeddings(rows, cluster_text) if rows and cluster_text and len(rows) == len(cluster_text) else 0.0',
          '    nmi = _normalized_mutual_info_score(cluster_text, true_text) if cluster_text and true_text and len(cluster_text) == len(true_text) else 0.0',
          '    ari = _adjusted_rand_index_score(cluster_text, true_text) if cluster_text and true_text and len(cluster_text) == len(true_text) else 0.0',
          '    return {"value": (asw + nmi + ari) / 3.0, "components": {"asw": asw, "nmi": nmi, "ari": ari}}',
          '',
        );
        break;
      case 'batch_integration_score':
        blocks.push(
          `def ${plan.functionName}(*, predictions: Optional[Sequence[Any]] = None, references: Optional[Sequence[Any]] = None, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:`,
          '    metadata = metadata or {}',
          '    rows = _coerce_embedding_rows(metadata.get("embeddings") or predictions or [])',
          '    batch_labels = metadata.get("batch_labels") or _extract_label_sequence(references or [], ("batch_label", "batch", "batch_id", "source_batch"))',
          '    if not rows or not batch_labels or len(rows) != len(batch_labels):',
          '        return {"value": 0.0, "note": "Provide embeddings and batch labels for batch integration evaluation."}',
          '    batch_text = [str(label) for label in batch_labels]',
          '    silhouette = _silhouette_from_embeddings(rows, batch_text)',
          '    return {"value": max(0.0, 1.0 - abs(silhouette)), "batch_silhouette": silhouette}',
          '',
        );
        break;
      case 'principal_component_regression_score':
        blocks.push(
          `def ${plan.functionName}(*, predictions: Optional[Sequence[Any]] = None, references: Optional[Sequence[Any]] = None, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:`,
          '    metadata = metadata or {}',
          '    embedding_rows = _coerce_embedding_rows(metadata.get("embeddings") or predictions or [])',
          '    original_rows = _coerce_embedding_rows(metadata.get("original_dataset") or metadata.get("original_features") or metadata.get("original_embeddings") or references or [])',
          '    batch_labels = metadata.get("batch_labels") or _extract_label_sequence(references or [], ("batch_label", "batch", "batch_id", "source_batch"))',
          '    if not embedding_rows or not batch_labels or len(embedding_rows) != len(batch_labels):',
          '        return {"value": 0.0, "note": "Provide embeddings and batch labels for PCR evaluation."}',
          '    batch_text = [str(label) for label in batch_labels]',
          '    if not original_rows or len(original_rows) != len(batch_text):',
          '        original_rows = embedding_rows',
          '    original_effect = _batch_effect_ratio(original_rows, batch_text)',
          '    embedding_effect = _batch_effect_ratio(embedding_rows, batch_text)',
          '    if original_effect <= 0.0:',
          '        value = max(0.0, 1.0 - min(max(embedding_effect, 0.0), 1.0))',
          '    else:',
          '        value = max(0.0, 1.0 - min(max(embedding_effect / original_effect, 0.0), 1.0))',
          '    return {"value": value, "original_batch_effect": original_effect, "embedding_batch_effect": embedding_effect}',
          '',
        );
        break;
      case 'average_batch_score':
        blocks.push(
          `def ${plan.functionName}(*, predictions: Optional[Sequence[Any]] = None, references: Optional[Sequence[Any]] = None, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:`,
          '    metadata = metadata or {}',
          '    rows = _coerce_embedding_rows(metadata.get("embeddings") or predictions or [])',
          '    batch_labels = metadata.get("batch_labels") or _extract_label_sequence(references or [], ("batch_label", "batch", "batch_id", "source_batch"))',
          '    batch_text = [str(label) for label in batch_labels] if batch_labels else []',
          '    silhouette = _silhouette_from_embeddings(rows, batch_text) if rows and batch_text and len(rows) == len(batch_text) else 0.0',
          '    batch_score = max(0.0, 1.0 - abs(silhouette)) if batch_text else 0.0',
          '    original_rows = _coerce_embedding_rows(metadata.get("original_dataset") or metadata.get("original_features") or metadata.get("original_embeddings") or references or [])',
          '    if not original_rows or len(original_rows) != len(batch_text):',
          '        original_rows = rows',
          '    original_effect = _batch_effect_ratio(original_rows, batch_text) if batch_text else 0.0',
          '    embedding_effect = _batch_effect_ratio(rows, batch_text) if batch_text else 0.0',
          '    if original_effect <= 0.0:',
          '        pcr_score = max(0.0, 1.0 - min(max(embedding_effect, 0.0), 1.0))',
          '    else:',
          '        pcr_score = max(0.0, 1.0 - min(max(embedding_effect / original_effect, 0.0), 1.0))',
          '    return {"value": (pcr_score + batch_score) / 2.0, "components": {"pcr_score": pcr_score, "batch_integration_score": batch_score}}',
          '',
        );
        break;
      case 'matthews_corrcoef':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> float:`,
          '    aligned = [(bool(prediction), bool(reference)) for prediction, reference in zip(predictions, references)]',
          '    true_positive = sum(1 for prediction, reference in aligned if prediction and reference)',
          '    true_negative = sum(1 for prediction, reference in aligned if not prediction and not reference)',
          '    false_positive = sum(1 for prediction, reference in aligned if prediction and not reference)',
          '    false_negative = sum(1 for prediction, reference in aligned if not prediction and reference)',
          '    numerator = (true_positive * true_negative) - (false_positive * false_negative)',
          '    denominator = math.sqrt(',
          '        float(true_positive + false_positive)',
          '        * float(true_positive + false_negative)',
          '        * float(true_negative + false_positive)',
          '        * float(true_negative + false_negative)',
          '    )',
          '    return _safe_divide(float(numerator), denominator)',
          '',
        );
        break;
      case 'mse':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[float], references: Sequence[float]) -> float:`,
          '    aligned = list(zip(predictions, references))',
          '    return _safe_divide(sum((_to_scalar(prediction) - _to_scalar(reference)) ** 2 for prediction, reference in aligned), float(len(aligned)))',
          '',
        );
        break;
      case 'pearson':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[float], references: Sequence[float]) -> float:`,
          '    aligned = [(_to_scalar(prediction), _to_scalar(reference)) for prediction, reference in zip(predictions, references)]',
          '    if not aligned:',
          '        return 0.0',
          '    prediction_values = [prediction for prediction, _reference in aligned]',
          '    reference_values = [reference for _prediction, reference in aligned]',
          '    prediction_mean = sum(prediction_values) / float(len(prediction_values))',
          '    reference_mean = sum(reference_values) / float(len(reference_values))',
          '    numerator = sum((prediction - prediction_mean) * (reference - reference_mean) for prediction, reference in aligned)',
          '    prediction_denom = math.sqrt(sum((prediction - prediction_mean) ** 2 for prediction in prediction_values))',
          '    reference_denom = math.sqrt(sum((reference - reference_mean) ** 2 for reference in reference_values))',
          '    return _safe_divide(numerator, prediction_denom * reference_denom)',
          '',
        );
        break;
      case 'spearman':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[float], references: Sequence[float]) -> float:`,
          '    def _rank(values: Sequence[float]) -> List[float]:',
          '        ordered = sorted(enumerate(values), key=lambda item: item[1])',
          '        ranks = [0.0] * len(values)',
          '        index = 0',
          '        while index < len(ordered):',
          '            tie_end = index',
          '            while tie_end + 1 < len(ordered) and ordered[tie_end + 1][1] == ordered[index][1]:',
          '                tie_end += 1',
          '            average_rank = (index + tie_end + 2) / 2.0',
          '            for tie_index in range(index, tie_end + 1):',
          '                ranks[ordered[tie_index][0]] = average_rank',
          '            index = tie_end + 1',
          '        return ranks',
          '    prediction_values = [_to_scalar(value) for value in predictions]',
          '    reference_values = [_to_scalar(value) for value in references]',
          '    if not prediction_values or not reference_values:',
          '        return 0.0',
          '    prediction_ranks = _rank(prediction_values)',
          '    reference_ranks = _rank(reference_values)',
          '    aligned = list(zip(prediction_ranks, reference_ranks))',
          '    prediction_mean = sum(prediction_ranks) / float(len(prediction_ranks))',
          '    reference_mean = sum(reference_ranks) / float(len(reference_ranks))',
          '    numerator = sum((prediction - prediction_mean) * (reference - reference_mean) for prediction, reference in aligned)',
          '    prediction_denom = math.sqrt(sum((prediction - prediction_mean) ** 2 for prediction in prediction_ranks))',
          '    reference_denom = math.sqrt(sum((reference - reference_mean) ** 2 for reference in reference_ranks))',
          '    return _safe_divide(numerator, prediction_denom * reference_denom)',
          '',
        );
        break;
      case 'kendall_tau':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[float], references: Sequence[float]) -> float:`,
          '    aligned = [(_to_scalar(prediction), _to_scalar(reference)) for prediction, reference in zip(predictions, references)]',
          '    concordant = 0',
          '    discordant = 0',
          '    for left_index in range(len(aligned)):',
          '        for right_index in range(left_index + 1, len(aligned)):',
          '            prediction_delta = aligned[left_index][0] - aligned[right_index][0]',
          '            reference_delta = aligned[left_index][1] - aligned[right_index][1]',
          '            product = prediction_delta * reference_delta',
          '            if product > 0:',
          '                concordant += 1',
          '            elif product < 0:',
          '                discordant += 1',
          '    return _safe_divide(float(concordant - discordant), float(concordant + discordant))',
          '',
        );
        break;
      case 'auc':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> Dict[str, Any]:`,
          '    try:',
          '        from sklearn.metrics import roc_auc_score',
          '    except ImportError:',
          '        return {"value": None, "note": "Install `scikit-learn` to compute ROC-AUC."}',
          '    return {"value": float(roc_auc_score([_to_scalar(value) for value in references], [_to_scalar(value) for value in predictions]))}',
          '',
        );
        break;
      case 'average_precision':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> Dict[str, Any]:`,
          '    try:',
          '        from sklearn.metrics import average_precision_score',
          '    except ImportError:',
          '        return {"value": None, "note": "Install `scikit-learn` to compute average precision."}',
          '    return {"value": float(average_precision_score([_to_scalar(value) for value in references], [_to_scalar(value) for value in predictions]))}',
          '',
        );
        break;
      case 'brier_score':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> float:`,
          '    def _score(value: Any) -> float:',
          '        if isinstance(value, Mapping):',
          '            for key in ("prob", "probability", "score", "confidence"):',
          '                if key in value:',
          '                    return float(value[key])',
          '        return float(value)',
          '    aligned = list(zip(predictions, references))',
          '    total = sum((_score(prediction) - float(reference)) ** 2 for prediction, reference in aligned)',
          '    return _safe_divide(total, float(len(aligned)))',
          '',
        );
        break;
      case 'expected_calibration_error':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any], *, num_bins: int = 10) -> float:`,
          '    def _score(value: Any) -> float:',
          '        if isinstance(value, Mapping):',
          '            for key in ("prob", "probability", "score", "confidence"):',
          '                if key in value:',
          '                    return float(value[key])',
          '        return float(value)',
          '    bins = [{"count": 0, "correct": 0.0, "confidence": 0.0} for _ in range(num_bins)]',
          '    for prediction, reference in zip(predictions, references):',
          '        confidence = min(max(_score(prediction), 0.0), 1.0)',
          '        bucket = min(int(confidence * num_bins), num_bins - 1)',
          '        bins[bucket]["count"] += 1',
          '        bins[bucket]["confidence"] += confidence',
          '        predicted_label = 1 if confidence >= 0.5 else 0',
          '        bins[bucket]["correct"] += 1.0 if predicted_label == int(reference) else 0.0',
          '    total = float(sum(bucket["count"] for bucket in bins))',
          '    if total == 0.0:',
          '        return 0.0',
          '    error = 0.0',
          '    for bucket in bins:',
          '        if bucket["count"] == 0:',
          '            continue',
          '        accuracy = bucket["correct"] / float(bucket["count"])',
          '        mean_confidence = bucket["confidence"] / float(bucket["count"])',
          '        error += abs(accuracy - mean_confidence) * (bucket["count"] / total)',
          '    return error',
          '',
        );
        break;
      case 'negative_log_likelihood':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> float:`,
          '    def _probability(prediction: Any, reference: Any) -> float:',
          '        if isinstance(prediction, Mapping):',
          '            if "probabilities" in prediction and isinstance(prediction["probabilities"], Mapping):',
          '                return float(prediction["probabilities"].get(reference, 1e-12))',
          '            for key in ("prob", "probability", "score", "confidence"):',
          '                if key in prediction:',
          '                    return float(prediction[key])',
          '        return float(prediction)',
          '    losses = []',
          '    for prediction, reference in zip(predictions, references):',
          '        probability = min(max(_probability(prediction, reference), 1e-12), 1.0)',
          '        if isinstance(reference, (int, float)) and float(reference) in (0.0, 1.0):',
          '            probability = probability if int(reference) == 1 else (1.0 - probability)',
          '        losses.append(-math.log(probability))',
          '    return _safe_divide(sum(losses), float(len(losses)))',
          '',
        );
        break;
      case 'perplexity':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> float:`,
          '    def _probability(prediction: Any, reference: Any) -> float:',
          '        if isinstance(prediction, Mapping):',
          '            if "probabilities" in prediction and isinstance(prediction["probabilities"], Mapping):',
          '                return float(prediction["probabilities"].get(reference, 1e-12))',
          '            for key in ("prob", "probability", "score", "confidence"):',
          '                if key in prediction:',
          '                    return float(prediction[key])',
          '        return float(prediction)',
          '    losses = []',
          '    for prediction, reference in zip(predictions, references):',
          '        probability = min(max(_probability(prediction, reference), 1e-12), 1.0)',
          '        if isinstance(reference, (int, float)) and float(reference) in (0.0, 1.0):',
          '            probability = probability if int(reference) == 1 else (1.0 - probability)',
          '        losses.append(-math.log(probability))',
          '    return math.exp(_safe_divide(sum(losses), float(len(losses))))',
          '',
        );
        break;
      case 'iou':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> float:`,
          '    def _normalize_box(item: Any) -> Optional[Sequence[float]]:',
          '        if isinstance(item, Mapping):',
          '            candidate = item.get("box") or (item.get("boxes")[0] if item.get("boxes") else None)',
          '            if candidate is not None:',
          '                return [float(value) for value in candidate]',
          '        if isinstance(item, (list, tuple)) and len(item) == 4 and all(isinstance(value, (int, float)) for value in item):',
          '            return [float(value) for value in item]',
          '        return None',
          '    def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:',
          '        x1 = max(left[0], right[0])',
          '        y1 = max(left[1], right[1])',
          '        x2 = min(left[2], right[2])',
          '        y2 = min(left[3], right[3])',
          '        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)',
          '        left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])',
          '        right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])',
          '        return _safe_divide(intersection, left_area + right_area - intersection)',
          '    scores: List[float] = []',
          '    for prediction, reference in zip(predictions, references):',
          '        left_box = _normalize_box(prediction)',
          '        right_box = _normalize_box(reference)',
          '        if left_box and right_box:',
          '            scores.append(_box_iou(left_box, right_box))',
          '            continue',
          '        left_set = set(prediction) if not isinstance(prediction, (str, bytes, Mapping)) else set()',
          '        right_set = set(reference) if not isinstance(reference, (str, bytes, Mapping)) else set()',
          '        union = left_set | right_set',
          '        scores.append(_safe_divide(float(len(left_set & right_set)), float(len(union))))',
          '    return _safe_divide(sum(scores), float(len(scores)))',
          '',
        );
        break;
      case 'map':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> Dict[str, Any]:`,
          '    try:',
          '        import torch',
          '        from torchmetrics.detection.mean_ap import MeanAveragePrecision',
          '    except ImportError:',
          '        return {"value": None, "note": "Install `torch` and `torchmetrics` to compute mAP."}',
          '    def _to_detection_batch(batch: Sequence[Any], *, with_scores: bool) -> List[Dict[str, Any]]:',
          '        converted: List[Dict[str, Any]] = []',
          '        for item in batch:',
          '            boxes = item.get("boxes", []) if isinstance(item, Mapping) else []',
          '            labels = item.get("labels", []) if isinstance(item, Mapping) else []',
          '            record: Dict[str, Any] = {',
          '                "boxes": torch.tensor(boxes, dtype=torch.float32),',
          '                "labels": torch.tensor(labels, dtype=torch.int64),',
          '            }',
          '            if with_scores:',
          '                record["scores"] = torch.tensor(item.get("scores", []), dtype=torch.float32)',
          '            converted.append(record)',
          '        return converted',
          '    metric = MeanAveragePrecision()',
          '    metric.update(_to_detection_batch(predictions, with_scores=True), _to_detection_batch(references, with_scores=False))',
          '    result = metric.compute()',
          '    serialized = {key: (float(value) if hasattr(value, "item") else value) for key, value in result.items()}',
          plan.resultKey
            ? `    return {"value": serialized.get(${toPythonString(plan.resultKey)}), "result_key": ${toPythonString(plan.resultKey)}, "all_metrics": serialized}`
            : '    return {"value": serialized.get("map"), "result_key": "map", "all_metrics": serialized}',
          '',
        );
        break;
      case 'mask_map':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> Dict[str, Any]:`,
          '    try:',
          '        import torch',
          '        from torchmetrics.detection.mean_ap import MeanAveragePrecision',
          '    except ImportError:',
          '        return {"value": None, "note": "Install `torch` and `torchmetrics` to compute mask AP."}',
          '    def _to_mask_batch(batch: Sequence[Any], *, with_scores: bool) -> List[Dict[str, Any]]:',
          '        converted: List[Dict[str, Any]] = []',
          '        for item in batch:',
          '            masks = item.get("masks", []) if isinstance(item, Mapping) else []',
          '            labels = item.get("labels", []) if isinstance(item, Mapping) else []',
          '            record: Dict[str, Any] = {',
          '                "masks": torch.tensor(masks, dtype=torch.uint8),',
          '                "labels": torch.tensor(labels, dtype=torch.int64),',
          '            }',
          '            if with_scores:',
          '                record["scores"] = torch.tensor(item.get("scores", []), dtype=torch.float32)',
          '            converted.append(record)',
          '        return converted',
          '    metric = MeanAveragePrecision(iou_type="segm")',
          '    metric.update(_to_mask_batch(predictions, with_scores=True), _to_mask_batch(references, with_scores=False))',
          '    result = metric.compute()',
          '    serialized = {key: (float(value) if hasattr(value, "item") else value) for key, value in result.items()}',
          plan.resultKey
            ? `    return {"value": serialized.get(${toPythonString(plan.resultKey)}), "result_key": ${toPythonString(plan.resultKey)}, "all_metrics": serialized}`
            : '    return {"value": serialized.get("map"), "result_key": "map", "all_metrics": serialized}',
          '',
        );
        break;
      case 'fid':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> Dict[str, Any]:`,
          '    try:',
          '        import torch',
          '        from torchmetrics.image.fid import FrechetInceptionDistance',
          '    except ImportError:',
          '        return {"value": None, "note": "Install `torch` and `torchmetrics` to compute FID."}',
          '    def _to_batch(item: Any):',
          '        tensor = torch.tensor(item, dtype=torch.uint8)',
          '        if tensor.ndim == 3:',
          '            tensor = tensor.unsqueeze(0)',
          '        return tensor',
          '    metric = FrechetInceptionDistance(normalize=False)',
          '    for batch in references:',
          '        metric.update(_to_batch(batch), real=True)',
          '    for batch in predictions:',
          '        metric.update(_to_batch(batch), real=False)',
          '    return {"value": float(metric.compute())}',
          '',
        );
        break;
      case 'psnr':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any], *, max_value: float = 255.0) -> float:`,
          '    def _flatten(value: Any) -> List[float]:',
          '        if isinstance(value, (list, tuple)):',
          '            flattened: List[float] = []',
          '            for item in value:',
          '                flattened.extend(_flatten(item))',
          '            return flattened',
          '        return [float(value)]',
          '    scores: List[float] = []',
          '    for prediction, reference in zip(predictions, references):',
          '        left = _flatten(prediction)',
          '        right = _flatten(reference)',
          '        mse = _safe_divide(sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right)), float(len(left)))',
          '        if mse == 0.0:',
          '            scores.append(float("inf"))',
          '            continue',
          '        scores.append(20.0 * math.log10(max_value) - 10.0 * math.log10(mse))',
          '    finite_scores = [score for score in scores if math.isfinite(score)]',
          '    return _safe_divide(sum(finite_scores), float(len(finite_scores))) if finite_scores else float("inf")',
          '',
        );
        break;
      case 'ssim':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> float:`,
          '    def _flatten(value: Any) -> List[float]:',
          '        if isinstance(value, (list, tuple)):',
          '            flattened: List[float] = []',
          '            for item in value:',
          '                flattened.extend(_flatten(item))',
          '            return flattened',
          '        return [float(value)]',
          '    scores: List[float] = []',
          '    c1 = 0.01 ** 2',
          '    c2 = 0.03 ** 2',
          '    for prediction, reference in zip(predictions, references):',
          '        left = _flatten(prediction)',
          '        right = _flatten(reference)',
          '        if not left or not right:',
          '            continue',
          '        mean_left = _safe_divide(sum(left), float(len(left)))',
          '        mean_right = _safe_divide(sum(right), float(len(right)))',
          '        var_left = _safe_divide(sum((value - mean_left) ** 2 for value in left), float(len(left)))',
          '        var_right = _safe_divide(sum((value - mean_right) ** 2 for value in right), float(len(right)))',
          '        covariance = _safe_divide(sum((left_value - mean_left) * (right_value - mean_right) for left_value, right_value in zip(left, right)), float(len(left)))',
          '        numerator = (2.0 * mean_left * mean_right + c1) * (2.0 * covariance + c2)',
          '        denominator = (mean_left ** 2 + mean_right ** 2 + c1) * (var_left + var_right + c2)',
          '        scores.append(_safe_divide(numerator, denominator))',
          '    return _safe_divide(sum(scores), float(len(scores)))',
          '',
        );
        break;
      case 'mae':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[float], references: Sequence[float]) -> float:`,
          '    aligned = list(zip(predictions, references))',
          '    return _safe_divide(sum(abs(_to_scalar(prediction) - _to_scalar(reference)) for prediction, reference in aligned), float(len(aligned)))',
          '',
        );
        break;
      case 'rmse':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[float], references: Sequence[float]) -> float:`,
          '    aligned = list(zip(predictions, references))',
          '    mse = _safe_divide(sum((_to_scalar(prediction) - _to_scalar(reference)) ** 2 for prediction, reference in aligned), float(len(aligned)))',
          '    return math.sqrt(mse)',
          '',
        );
        break;
      case 'mape':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[float], references: Sequence[float]) -> float:`,
          '    aligned = [(_to_scalar(prediction), _to_scalar(reference)) for prediction, reference in zip(predictions, references) if _to_scalar(reference) != 0.0]',
          '    total = sum(abs((prediction - reference) / reference) for prediction, reference in aligned)',
          '    return _safe_divide(total, float(len(aligned)))',
          '',
        );
        break;
      case 'smape':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[float], references: Sequence[float]) -> float:`,
          '    aligned = list(zip(predictions, references))',
          '    total = 0.0',
          '    for prediction, reference in aligned:',
          '        prediction_value = _to_scalar(prediction)',
          '        reference_value = _to_scalar(reference)',
          '        denominator = abs(prediction_value) + abs(reference_value)',
          '        if denominator == 0.0:',
          '            continue',
          '        total += (2.0 * abs(prediction_value - reference_value)) / denominator',
          '    return _safe_divide(total, float(len(aligned)))',
          '',
        );
        break;
      case 'dtw':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Sequence[float]], references: Sequence[Sequence[float]]) -> float:`,
          '    def _dtw(left: Sequence[float], right: Sequence[float]) -> float:',
          '        rows = len(left)',
          '        cols = len(right)',
          '        table = [[float("inf")] * (cols + 1) for _ in range(rows + 1)]',
          '        table[0][0] = 0.0',
          '        for i in range(1, rows + 1):',
          '            for j in range(1, cols + 1):',
          '                cost = abs(float(left[i - 1]) - float(right[j - 1]))',
          '                table[i][j] = cost + min(table[i - 1][j], table[i][j - 1], table[i - 1][j - 1])',
          '        return table[rows][cols]',
          '    aligned = list(zip(predictions, references))',
          '    return _safe_divide(sum(_dtw(prediction, reference) for prediction, reference in aligned), float(len(aligned)))',
          '',
        );
        break;
      case 'edit_similarity':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> float:`,
          '    def _to_sequence(value: Any) -> List[str]:',
          '        if isinstance(value, str):',
          '            return list(value)',
          '        if isinstance(value, (list, tuple)):',
          '            return [str(item) for item in value]',
          '        return [str(value)]',
          '    def _levenshtein(left: List[str], right: List[str]) -> int:',
          '        rows = len(left)',
          '        cols = len(right)',
          '        table = [[0] * (cols + 1) for _ in range(rows + 1)]',
          '        for row in range(rows + 1):',
          '            table[row][0] = row',
          '        for col in range(cols + 1):',
          '            table[0][col] = col',
          '        for row in range(1, rows + 1):',
          '            for col in range(1, cols + 1):',
          '                cost = 0 if left[row - 1] == right[col - 1] else 1',
          '                table[row][col] = min(table[row - 1][col] + 1, table[row][col - 1] + 1, table[row - 1][col - 1] + cost)',
          '        return table[rows][cols]',
          '    similarities: List[float] = []',
          '    for prediction, reference in zip(predictions, references):',
          '        left = _to_sequence(prediction)',
          '        right = _to_sequence(reference)',
          '        scale = max(len(left), len(right), 1)',
          '        similarities.append(1.0 - (_levenshtein(left, right) / float(scale)))',
          '    return _safe_divide(sum(similarities), float(len(similarities)))',
          '',
        );
        break;
      case 'token_overlap_recall':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> Dict[str, Any]:`,
          '    scores: List[float] = []',
          '    for prediction, reference in zip(predictions, references):',
          '        prediction_tokens = set(_text_tokens(prediction))',
          '        reference_tokens = set(_text_tokens(reference))',
          '        if not reference_tokens:',
          '            scores.append(0.0)',
          '            continue',
          '        overlap = sum(1 for token in reference_tokens if token in prediction_tokens)',
          '        scores.append(_safe_divide(float(overlap), float(len(reference_tokens))))',
          '    return {"value": _safe_divide(sum(scores), float(len(scores))), "approximation": "token overlap recall from extracted metric formula"}',
          '',
        );
        break;
      case 'sigmoid_score_average':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> Dict[str, Any]:`,
          '    transformed: List[float] = []',
          '    for prediction, reference in zip(predictions, references):',
          '        similarity = _pair_similarity_score(prediction, reference)',
          '        transformed.append(1.0 / (1.0 + math.exp(-similarity)))',
          '    return {"value": _safe_divide(sum(transformed), float(len(transformed))), "approximation": "sigmoid over similarity scores inferred from predictions/references"}',
          '',
        );
        break;
      case 'greedy_token_similarity':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> Dict[str, Any]:`,
          '    scores: List[float] = []',
          '    for prediction, reference in zip(predictions, references):',
          '        prediction_tokens = _text_tokens(prediction)',
          '        reference_tokens = _text_tokens(reference)',
          '        scores.append(_greedy_token_similarity(reference_tokens, prediction_tokens))',
          '    return {"value": _safe_divide(sum(scores), float(len(scores))), "approximation": "greedy lexical token similarity from extracted metric formula"}',
          '',
        );
        break;
      case 'bertscore_precision':
      case 'bertscore_recall':
      case 'bertscore_f1':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, Any]:`,
          '    payload = _compute_bertscore_payload(predictions, references)',
          '    if payload.get("precision") is None and payload.get("recall") is None and payload.get("f1") is None:',
          '        return payload',
          plan.kind === 'bertscore_precision'
            ? '    return {"value": payload.get("precision"), "component": "precision"}'
            : plan.kind === 'bertscore_recall'
              ? '    return {"value": payload.get("recall"), "component": "recall"}'
              : '    return {"value": payload.get("f1"), "component": "f1"}',
          '',
        );
        break;
      case 'wer':
      case 'cer':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> float:`,
          '    def _to_units(value: Any) -> List[str]:',
          '        if isinstance(value, str):',
          plan.kind === 'wer'
            ? '            return value.split()'
            : '            return list(value)',
          '        if isinstance(value, (list, tuple)):',
          '            return [str(item) for item in value]',
          '        return [str(value)]',
          '    def _levenshtein(left: List[str], right: List[str]) -> int:',
          '        rows = len(left)',
          '        cols = len(right)',
          '        table = [[0] * (cols + 1) for _ in range(rows + 1)]',
          '        for row in range(rows + 1):',
          '            table[row][0] = row',
          '        for col in range(cols + 1):',
          '            table[0][col] = col',
          '        for row in range(1, rows + 1):',
          '            for col in range(1, cols + 1):',
          '                cost = 0 if left[row - 1] == right[col - 1] else 1',
          '                table[row][col] = min(table[row - 1][col] + 1, table[row][col - 1] + 1, table[row - 1][col - 1] + cost)',
          '        return table[rows][cols]',
          '    errors = []',
          '    for prediction, reference in zip(predictions, references):',
          '        left = _to_units(prediction)',
          '        right = _to_units(reference)',
          '        errors.append(_safe_divide(float(_levenshtein(left, right)), float(max(len(right), 1))))',
          '    return _safe_divide(sum(errors), float(len(errors)))',
          '',
        );
        break;
      case 'average_return':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> Dict[str, Any]:`,
          '    def _trajectory_return(item: Any) -> float:',
          '        if isinstance(item, Mapping):',
          '            for key in ("episode_rewards", "rewards", "trajectory_rewards", "returns", "reward_curve"):',
          '                if key in item:',
          '                    return _trajectory_return(item[key])',
          '            for key in ("return", "episode_return", "score", "reward"):',
          '                if key in item:',
          '                    return float(item[key])',
          '        if isinstance(item, (list, tuple)):',
          '            return float(sum(_to_scalar(value) for value in item))',
          '        return _to_scalar(item)',
          '    returns = [_trajectory_return(item) for item in predictions]',
          '    reference_returns = [_trajectory_return(item) for item in references] if references else []',
          '    payload: Dict[str, Any] = {"value": _safe_divide(sum(returns), float(len(returns))), "num_episodes": len(returns)}',
          '    if reference_returns:',
          '        payload["reference_value"] = _safe_divide(sum(reference_returns), float(len(reference_returns)))',
          '    return payload',
          '',
        );
        break;
      case 'average_max_state_value':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> Dict[str, Any]:`,
          '    def _state_values(item: Any) -> List[float]:',
          '        if isinstance(item, Mapping):',
          '            for key in ("q_values", "action_values", "predicted_action_values", "state_action_values", "scores", "logits"):',
          '                if key in item:',
          '                    return _state_values(item[key])',
          '            scalar = _to_scalar(item, default=float("nan"))',
          '            return [] if math.isnan(scalar) else [scalar]',
          '        flattened = _flatten_numeric(item)',
          '        return flattened',
          '    maxima = [max(values) for values in (_state_values(item) for item in predictions) if values]',
          '    payload: Dict[str, Any] = {"value": _safe_divide(sum(maxima), float(len(maxima))), "num_states": len(maxima)}',
          '    reference_maxima = [max(values) for values in (_state_values(item) for item in references) if values] if references else []',
          '    if reference_maxima:',
          '        payload["reference_value"] = _safe_divide(sum(reference_maxima), float(len(reference_maxima)))',
          '    return payload',
          '',
        );
        break;
      case 'success_rate':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> Dict[str, Any]:`,
          '    def _is_success(value: Any) -> bool:',
          '        if isinstance(value, Mapping):',
          '            for key in ("success", "passed", "correct", "solved"):',
          '                if key in value:',
          '                    return bool(value[key])',
          '        if isinstance(value, str):',
          '            return value.strip().lower() in {"true", "success", "passed", "correct", "solved", "yes"}',
          '        return bool(value)',
          '    successes = 0',
          '    total = 0',
          '    if references:',
          '        for prediction, reference in zip(predictions, references):',
          '            total += 1',
          '            if _is_success(prediction) == _is_success(reference):',
          '                successes += 1',
          '    else:',
          '        for prediction in predictions:',
          '            total += 1',
          '            if _is_success(prediction):',
          '                successes += 1',
          '    return {"value": _safe_divide(float(successes), float(total)), "num_items": total}',
          '',
        );
        break;
      case 'pass_at_k':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Sequence[Any]], references: Sequence[Any], *, k: int = 1) -> Dict[str, Any]:`,
          '    def _accepted(reference: Any) -> List[Any]:',
          '        if isinstance(reference, Mapping):',
          '            candidate = reference.get("accepted") or reference.get("references") or reference.get("solutions") or []',
          '            return list(candidate) if isinstance(candidate, (list, tuple, set)) else [candidate]',
          '        if isinstance(reference, (list, tuple, set)):',
          '            return list(reference)',
          '        return [reference]',
          '    passed = 0',
          '    total = 0',
          '    for candidates, reference in zip(predictions, references):',
          '        total += 1',
          '        accepted = _accepted(reference)',
          '        if any(candidate in accepted for candidate in list(candidates)[:k]):',
          '            passed += 1',
          '    return {"value": _safe_divide(float(passed), float(total)), "num_items": total, "k": k}',
          '',
        );
        break;
      case 'normalized_score':
        blocks.push(
          `def ${plan.functionName}(*, predictions: Optional[Sequence[Any]] = None, references: Optional[Sequence[Any]] = None, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:`,
          '    metadata = metadata or {}',
          '    raw_scores = metadata.get("raw_scores", predictions or [])',
          '    numeric_scores = [_to_scalar(value) for value in raw_scores] if raw_scores else []',
          '    random_score = metadata.get("random_score")',
          '    human_score = metadata.get("human_score")',
          '    if not numeric_scores or random_score is None or human_score is None:',
          '        return {"value": None, "note": "Provide `raw_scores`, `random_score`, and `human_score` in metadata."}',
          '    mean_score = _safe_divide(sum(numeric_scores), float(len(numeric_scores)))',
          '    normalized = 100.0 * _safe_divide(mean_score - float(random_score), float(human_score) - float(random_score))',
          '    return {"value": normalized, "raw_mean": mean_score}',
          '',
        );
        break;
      case 'expected_entropy':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> Dict[str, Any]:`,
          '    def _extract_probs(item: Any) -> List[float]:',
          '        if isinstance(item, Mapping):',
          '            candidate = item.get("probs") or item.get("probabilities") or []',
          '            return [float(value) for value in candidate]',
          '        if isinstance(item, (list, tuple)):',
          '            return [float(value) for value in item]',
          '        return []',
          '    entropies: List[float] = []',
          '    for item in predictions:',
          '        probabilities = [value for value in _extract_probs(item) if value > 0.0]',
          '        if probabilities:',
          '            entropies.append(-sum(value * math.log(value) for value in probabilities))',
          '    return {"value": _safe_divide(sum(entropies), float(len(entropies))), "num_items": len(entropies)}',
          '',
        );
        break;
      case 'runtime_seconds':
        blocks.push(
          `def ${plan.functionName}(*, predictions: Optional[Sequence[Any]] = None, references: Optional[Sequence[Any]] = None, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:`,
          '    metadata = metadata or {}',
          '    durations = metadata.get("durations") or metadata.get("wall_clock_seconds") or predictions or []',
          '    numeric_durations = [_to_scalar(value) for value in durations] if durations else []',
          '    payload: Dict[str, Any] = {"value": _safe_divide(sum(numeric_durations), float(len(numeric_durations))), "num_runs": len(numeric_durations)}',
          '    reference_durations = [_to_scalar(value) for value in references] if references else []',
          '    if reference_durations:',
          '        payload["reference_value"] = _safe_divide(sum(reference_durations), float(len(reference_durations)))',
          '    return payload',
          '',
        );
        break;
      case 'speedup_ratio':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> Dict[str, Any]:`,
          '    candidate_times = [_to_scalar(value) for value in predictions] if predictions else []',
          '    baseline_times = [_to_scalar(value) for value in references] if references else []',
          '    if not candidate_times or not baseline_times:',
          '        return {"value": None, "note": "Provide candidate runtimes in predictions and baseline runtimes in references."}',
          '    candidate_mean = _safe_divide(sum(candidate_times), float(len(candidate_times)))',
          '    baseline_mean = _safe_divide(sum(baseline_times), float(len(baseline_times)))',
          '    return {"value": _safe_divide(baseline_mean, candidate_mean), "candidate_runtime": candidate_mean, "baseline_runtime": baseline_mean}',
          '',
        );
        break;
      case 'amino_acid_recovery':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> float:`,
          '    recovered = 0',
          '    total = 0',
          '    for prediction, reference in zip(predictions, references):',
          '        prediction_tokens = list(str(prediction))',
          '        reference_tokens = list(str(reference))',
          '        for prediction_token, reference_token in zip(prediction_tokens, reference_tokens):',
          '            total += 1',
          '            if prediction_token == reference_token:',
          '                recovered += 1',
          '    return _safe_divide(float(recovered), float(total))',
          '',
        );
        break;
      case 'rank_percent':
        blocks.push(
          `def ${plan.functionName}(*, predictions: Optional[Sequence[Any]] = None, references: Optional[Sequence[Any]] = None, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:`,
          '    metadata = metadata or {}',
          '    candidate_score = metadata.get("candidate_score")',
          '    cohort_scores = metadata.get("cohort_scores") or []',
          '    if candidate_score is None and predictions:',
          '        candidate_score = predictions[0]',
          '    numeric_cohort = [float(value) for value in cohort_scores] if cohort_scores else []',
          '    if candidate_score is None or not numeric_cohort:',
          '        return {"value": None, "note": "Provide `candidate_score` and `cohort_scores` in metadata."}',
          '    less_equal = sum(1 for value in numeric_cohort if value <= float(candidate_score))',
          '    percentile = 100.0 * _safe_divide(float(less_equal), float(len(numeric_cohort)))',
          '    return {"value": percentile, "candidate_score": float(candidate_score)}',
          '',
        );
        break;
      case 'winning_count':
        blocks.push(
          `def ${plan.functionName}(*, predictions: Optional[Sequence[Any]] = None, references: Optional[Sequence[Any]] = None, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:`,
          '    metadata = metadata or {}',
          '    score_rows = metadata.get("score_rows") or []',
          '    winning_count = 0',
          '    for row in score_rows:',
          '        if not isinstance(row, Mapping):',
          '            continue',
          '        candidate = row.get("candidate")',
          '        baselines = row.get("baselines") or []',
          '        numeric_baselines = [float(value) for value in baselines] if baselines else []',
          '        if candidate is not None and (not numeric_baselines or float(candidate) >= max(numeric_baselines)):',
          '            winning_count += 1',
          '    return {"value": float(winning_count), "num_rows": len(score_rows)}',
          '',
        );
        break;
      case 'path_length':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Any], references: Sequence[Any]) -> Dict[str, Any]:`,
          '    distances = [float(value) for value in predictions if isinstance(value, (int, float))]',
          '    return {"value": _safe_divide(sum(distances), float(len(distances))), "num_steps": len(distances)}',
          '',
        );
        break;
      case 'judge_score_aggregate':
        blocks.push(
          `def ${plan.functionName}(*, predictions: Optional[Sequence[Any]] = None, references: Optional[Sequence[Any]] = None, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:`,
          '    metadata = metadata or {}',
          `    metric_key = ${toPythonString(plan.outputKey)}`,
          '    score_candidates = metadata.get(f"{metric_key}_scores") or metadata.get("judge_scores") or []',
          '    numeric_scores = [float(value) for value in score_candidates if isinstance(value, (int, float))]',
          '    if not numeric_scores:',
          '        return {"value": None, "note": "Provide judge scores in metadata to aggregate this metric."}',
          '    return {"value": _safe_divide(sum(numeric_scores), float(len(numeric_scores))), "num_scores": len(numeric_scores)}',
          '',
        );
        break;
      case 'relevant_noise_sensitivity':
      case 'irrelevant_noise_sensitivity':
        blocks.push(
          `def ${plan.functionName}(*, predictions: Optional[Sequence[Any]] = None, references: Optional[Sequence[Any]] = None, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:`,
          '    metadata = metadata or {}',
          '    claim_records = metadata.get("claim_records") or metadata.get("response_claims") or []',
          '    bucket_name = "relevant" if "relevant" in ' + toPythonString(plan.kind) + ' else "irrelevant"',
          '    numerator = 0',
          '    denominator = 0',
          '    for claim in claim_records:',
          '        if not isinstance(claim, Mapping):',
          '            continue',
          '        is_correct = bool(claim.get("is_correct") or claim.get("supported"))',
          '        if is_correct:',
          '            continue',
          '        denominator += 1',
          '        if bucket_name == "relevant":',
          '            entailed = bool(claim.get("entailed_in_relevant") or claim.get("entailed_in_relevant_chunks"))',
          '        else:',
          '            entailed = bool(claim.get("entailed_in_irrelevant") or claim.get("entailed_in_irrelevant_chunks"))',
          '        if entailed:',
          '            numerator += 1',
          '    if denominator == 0:',
          '        return {"value": None, "note": "Provide claim-level metadata such as `claim_records` with correctness and relevant/irrelevant entailment flags."}',
          '    return {"value": _safe_divide(float(numerator), float(denominator)), "num_incorrect_claims": denominator, "bucket": bucket_name}',
          '',
        );
        break;
      case 'rmsd':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Sequence[Sequence[float]]], references: Sequence[Sequence[Sequence[float]]]) -> float:`,
          '    def _flatten(points: Sequence[Sequence[float]]) -> List[float]:',
          '        flattened: List[float] = []',
          '        for point in points:',
          '            flattened.extend(float(value) for value in point)',
          '        return flattened',
          '    aligned = list(zip(predictions, references))',
          '    squared = []',
          '    for prediction, reference in aligned:',
          '        left = _flatten(prediction)',
          '        right = _flatten(reference)',
          '        squared.extend((left_value - right_value) ** 2 for left_value, right_value in zip(left, right))',
          '    return math.sqrt(_safe_divide(sum(squared), float(len(squared))))',
          '',
        );
        break;
      case 'ndcg':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Sequence[Any]], references: Sequence[Sequence[Any]], *, k: int = 10) -> float:`,
          '    def _dcg(items: Sequence[Any], relevant_items: Sequence[Any]) -> float:',
          '        total = 0.0',
          '        for index, item in enumerate(items[:k], start=1):',
          '            if item in relevant_items:',
          '                total += 1.0 / math.log2(index + 1)',
          '        return total',
          '    scores = []',
          '    for items, relevant_items in zip(predictions, references):',
          '        ideal = _dcg(list(relevant_items), relevant_items)',
          '        scores.append(_safe_divide(_dcg(items, relevant_items), ideal))',
          '    return _safe_divide(sum(scores), float(len(scores)))',
          '',
        );
        break;
      case 'mrr':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Sequence[Any]], references: Sequence[Sequence[Any]]) -> float:`,
          '    reciprocal_ranks: List[float] = []',
          '    for items, relevant_items in zip(predictions, references):',
          '        score = 0.0',
          '        for index, item in enumerate(items, start=1):',
          '            if item in relevant_items:',
          '                score = 1.0 / float(index)',
          '                break',
          '        reciprocal_ranks.append(score)',
          '    return _safe_divide(sum(reciprocal_ranks), float(len(reciprocal_ranks)))',
          '',
        );
        break;
      case 'hit_rate':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[Sequence[Any]], references: Sequence[Sequence[Any]], *, k: int = 10) -> float:`,
          '    hits = 0',
          '    total = 0',
          '    for items, relevant_items in zip(predictions, references):',
          '        total += 1',
          '        if any(item in relevant_items for item in items[:k]):',
          '            hits += 1',
          '    return _safe_divide(float(hits), float(total))',
          '',
        );
        break;
      case 'library_metric':
        blocks.push(
          `def ${plan.functionName}(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, Any]:`,
          `    """Thin wrapper for ${plan.metricName}. Install optional dependencies before use."""`,
          '    evaluate = _require_optional_dependency("evaluate")',
          '    if evaluate is None:',
          '        return {',
          '            "value": None,',
          `            "note": ${toPythonString('Install `evaluate` to compute this metric.')},`,
          `            "metric": ${toPythonString(plan.metricName)},`,
          '        }',
          `    metric_name = ${toPythonString(plan.metricName.toLowerCase())}`,
          '    try:',
          '        metric = evaluate.load(metric_name)',
          '    except Exception as exc:',
          '        return {',
          '            "value": None,',
          `            "note": ${toPythonString('Optional metric backend is unavailable for this metric name. Install the matching evaluator or replace with a paper-specific implementation.')},`,
          '            "metric": metric_name,',
          '            "error": str(exc),',
          '        }',
          '    return metric.compute(predictions=list(predictions), references=list(references))',
          '',
        );
        break;
      case 'custom_placeholder':
        blocks.push(
          `def ${plan.functionName}(*, predictions: Optional[Sequence[Any]] = None, references: Optional[Sequence[Any]] = None, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:`,
          `    """Placeholder for ${plan.metricName}. Replace this with the paper-specific implementation."""`,
          '    return {',
          '        "value": None,',
          `        "note": ${toPythonString(plan.note || 'TODO: implement the paper-specific metric.')},`,
          plan.formula
            ? `        "formula": ${toPythonString(plan.formula)},`
            : '        "formula": None,',
          '        "metadata_keys": sorted((metadata or {}).keys()),',
          '    }',
          '',
        );
        break;
      default:
        blocks.push('');
        break;
    }
  }

  if (blocks.length === 0) {
    blocks.push(
      'def evaluate_placeholder(*, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:',
      '    return {"note": "TODO: define the evaluation routine from the paper."}',
      '',
    );
  }

  return blocks;
}

function buildEntryPointBlock(
  summary: PaperEvaluationSummary,
  metricPlans: MetricPlan[],
  taskMode: TaskMode,
): string[] {
  const lines: string[] = [];
  const entrypointName = entrypointNameForTaskMode(taskMode);
  const includesRankingMetrics = metricPlans.some((plan) => isRankingMetricKindFromRegistry(plan.kind));
  const includesMetadataMetrics = metricPlans.some(
    (plan) => !isDirectPredictionMetricKindFromRegistry(plan.kind) && !isRankingMetricKindFromRegistry(plan.kind),
  );

  switch (taskMode) {
    case 'pairwise_preference':
      lines.push(
        `def ${entrypointName}(model_name: str, pairwise_votes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:`,
        '    results: Dict[str, Any] = {',
        '        "task_type": "pairwise_preference",',
        '        "model_name": model_name,',
        '        "datasets": DATASETS,',
        '        "expected_baselines": EXPECTED_BASELINES,',
        '        "metrics": {},',
        '        "cautions": IMPLEMENTATION_CAUTIONS,',
        '    }',
      );
      metricPlans.forEach((plan) => {
        if (plan.kind === 'pairwise_win_rate') {
          lines.push(`    results["metrics"][${toPythonString(plan.outputKey)}] = ${plan.functionName}(model_name, pairwise_votes)`);
          return;
        }

        if (plan.kind === 'feature_coefficient_placeholder') {
          lines.push(
            `    results["metrics"][${toPythonString(plan.outputKey)}] = ${plan.functionName}(pairwise_votes, feature_name=${toPythonString(plan.metricName)})`,
          );
          return;
        }

        if (isDirectPredictionMetricKindFromRegistry(plan.kind)) {
          lines.push(`    _${plan.outputKey}_inputs = _extract_vote_metric_inputs(pairwise_votes, ${toPythonString(plan.outputKey)})`);
          lines.push(
            `    results["metrics"][${toPythonString(plan.outputKey)}] = ${plan.functionName}(_${plan.outputKey}_inputs["predictions"], _${plan.outputKey}_inputs["references"])`,
          );
          return;
        }

        lines.push(
          `    results["metrics"][${toPythonString(plan.outputKey)}] = ${plan.functionName}(metadata={"pairwise_vote_count": len(pairwise_votes)})`,
        );
      });
      lines.push('    return results', '');
      break;
    case 'ranking':
      lines.push(
        `def ${entrypointName}(`,
        '    ranked_items: Sequence[Sequence[Any]],',
        '    ground_truth: Sequence[Sequence[Any]],',
        '    *,',
        '    predictions: Optional[Sequence[Any]] = None,',
        '    references: Optional[Sequence[Any]] = None,',
        '    metadata: Optional[Mapping[str, Any]] = None,',
        '    k: int = 10,',
        ') -> Dict[str, Any]:',
        '    scalar_predictions = list(predictions or [])',
        '    scalar_references = list(references or [])',
        '    results: Dict[str, Any] = {',
        '        "task_type": "ranking",',
        '        "datasets": DATASETS,',
        '        "expected_baselines": EXPECTED_BASELINES,',
        '        "metrics": {},',
        '        "cautions": IMPLEMENTATION_CAUTIONS,',
        '    }',
      );
      metricPlans.forEach((plan) => {
        if (isDirectPredictionMetricKindFromRegistry(plan.kind)) {
          lines.push(`    results["metrics"][${toPythonString(plan.outputKey)}] = ${plan.functionName}(scalar_predictions, scalar_references)`);
        } else if (plan.kind === 'ndcg' || plan.kind === 'hit_rate' || plan.kind === 'precision_at_k' || plan.kind === 'recall_at_k' || plan.kind === 'pass_at_k') {
          lines.push(`    results["metrics"][${toPythonString(plan.outputKey)}] = ${plan.functionName}(ranked_items, ground_truth, k=k)`);
        } else if (plan.kind === 'mrr') {
          lines.push(`    results["metrics"][${toPythonString(plan.outputKey)}] = ${plan.functionName}(ranked_items, ground_truth)`);
        } else {
          lines.push(`    results["metrics"][${toPythonString(plan.outputKey)}] = ${plan.functionName}(predictions=scalar_predictions, references=scalar_references, metadata=metadata or {"k": k})`);
        }
      });
      lines.push('    return results', '');
      break;
    case 'forecasting':
      lines.push(
        `def ${entrypointName}(predictions: Sequence[float], references: Sequence[float]) -> Dict[str, Any]:`,
        '    results: Dict[str, Any] = {',
        '        "task_type": "forecasting",',
        '        "datasets": DATASETS,',
        '        "expected_baselines": EXPECTED_BASELINES,',
        '        "metrics": {},',
        '        "cautions": IMPLEMENTATION_CAUTIONS,',
        '    }',
      );
      metricPlans.forEach((plan) => {
        if (isDirectPredictionMetricKindFromRegistry(plan.kind)) {
          lines.push(`    results["metrics"][${toPythonString(plan.outputKey)}] = ${plan.functionName}(predictions, references)`);
        } else {
          lines.push(`    results["metrics"][${toPythonString(plan.outputKey)}] = ${plan.functionName}(predictions=predictions, references=references)`);
        }
      });
      lines.push('    return results', '');
      break;
    default:
      if (includesRankingMetrics) {
        lines.push(
          `def ${entrypointName}(`,
          '    predictions: Sequence[Any],',
          '    references: Sequence[Any],',
          '    *,',
          '    ranked_items: Optional[Sequence[Sequence[Any]]] = None,',
          '    relevant_items: Optional[Sequence[Sequence[Any]]] = None,',
          '    metadata: Optional[Mapping[str, Any]] = None,',
          '    k: int = 10,',
          ') -> Dict[str, Any]:',
          '    ranking_predictions = list(ranked_items or [])',
          '    ranking_references = list(relevant_items or [])',
          '    results: Dict[str, Any] = {',
          `        "task_type": ${toPythonString(fallbackText(summary.task_type, taskMode))},`,
          '        "datasets": DATASETS,',
          '        "expected_baselines": EXPECTED_BASELINES,',
          '        "metrics": {},',
          '        "cautions": IMPLEMENTATION_CAUTIONS,',
          '    }',
        );
      } else if (includesMetadataMetrics) {
        lines.push(
          `def ${entrypointName}(`,
          '    predictions: Sequence[Any],',
          '    references: Sequence[Any],',
          '    *,',
          '    metadata: Optional[Mapping[str, Any]] = None,',
          ') -> Dict[str, Any]:',
          '    results: Dict[str, Any] = {',
          `        "task_type": ${toPythonString(fallbackText(summary.task_type, taskMode))},`,
          '        "datasets": DATASETS,',
          '        "expected_baselines": EXPECTED_BASELINES,',
          '        "metrics": {},',
          '        "cautions": IMPLEMENTATION_CAUTIONS,',
          '    }',
        );
      } else {
        lines.push(
          `def ${entrypointName}(predictions: Sequence[Any], references: Sequence[Any]) -> Dict[str, Any]:`,
          '    results: Dict[str, Any] = {',
          `        "task_type": ${toPythonString(fallbackText(summary.task_type, taskMode))},`,
          '        "datasets": DATASETS,',
          '        "expected_baselines": EXPECTED_BASELINES,',
          '        "metrics": {},',
          '        "cautions": IMPLEMENTATION_CAUTIONS,',
          '    }',
        );
      }
      metricPlans.forEach((plan) => {
        if (isDirectPredictionMetricKindFromRegistry(plan.kind)) {
          lines.push(`    results["metrics"][${toPythonString(plan.outputKey)}] = ${plan.functionName}(predictions, references)`);
        } else if (plan.kind === 'ndcg' || plan.kind === 'hit_rate' || plan.kind === 'precision_at_k' || plan.kind === 'recall_at_k' || plan.kind === 'pass_at_k') {
          lines.push(`    results["metrics"][${toPythonString(plan.outputKey)}] = ${plan.functionName}(ranking_predictions, ranking_references, k=k)`);
        } else if (plan.kind === 'mrr') {
          lines.push(`    results["metrics"][${toPythonString(plan.outputKey)}] = ${plan.functionName}(ranking_predictions, ranking_references)`);
        } else {
          lines.push(`    results["metrics"][${toPythonString(plan.outputKey)}] = ${plan.functionName}(predictions=predictions, references=references, metadata=metadata)`);
        }
      });
      lines.push('    return results', '');
      break;
  }

  return lines;
}

function buildExampleUsageBlock(taskMode: TaskMode, metricPlans: MetricPlan[]): string[] {
  const entrypointName = entrypointNameForTaskMode(taskMode);
  const includesRankingMetrics = metricPlans.some((plan) => isRankingMetricKindFromRegistry(plan.kind));
  const usesTopKCandidateStyle = metricPlans.some((plan) => plan.kind === 'top_k_accuracy');
  const usesBinaryLabels = metricPlans.some((plan) => ['precision', 'recall', 'f1', 'success_rate'].includes(plan.kind));
  const usesScoreLabels = metricPlans.some((plan) =>
    ['auc', 'average_precision', 'brier_score', 'expected_calibration_error', 'negative_log_likelihood', 'perplexity'].includes(plan.kind),
  );
  const usesDetectionStyle = metricPlans.some((plan) => ['map', 'mask_map', 'iou'].includes(plan.kind));
  const usesImageGenerationStyle = metricPlans.some((plan) => ['fid', 'psnr', 'ssim'].includes(plan.kind));
  const usesClusterStyle = metricPlans.some((plan) =>
    [
      'silhouette_score',
      'normalized_mutual_info',
      'adjusted_rand_index',
      'average_bio_score',
      'batch_integration_score',
      'principal_component_regression_score',
      'average_batch_score',
    ].includes(plan.kind),
  );
  const usesTrajectoryStyle = metricPlans.some((plan) => plan.kind === 'average_return');
  const usesQValueStyle = metricPlans.some((plan) => plan.kind === 'average_max_state_value');
  const usesSequenceDistanceStyle = metricPlans.some((plan) => plan.kind === 'dtw');
  const usesTextDistanceStyle = metricPlans.some((plan) => ['edit_similarity', 'wer', 'cer'].includes(plan.kind));
  const usesStructureStyle = metricPlans.some((plan) => plan.kind === 'rmsd');
  const usesNumericRegression = metricPlans.some((plan) =>
    ['mse', 'mae', 'rmse', 'mape', 'smape', 'pearson', 'spearman', 'kendall_tau'].includes(plan.kind),
  );
  const includesMetadataMetrics = metricPlans.some(
    (plan) => !isDirectPredictionMetricKindFromRegistry(plan.kind) && !isRankingMetricKindFromRegistry(plan.kind),
  );

  switch (taskMode) {
    case 'pairwise_preference':
      return [
        'if __name__ == "__main__":',
        '    example_votes = [',
        '        {"winner": "candidate-model", "loser": "baseline-model", "metadata": {"citation_count": 3}},',
        '        {"winner": "baseline-model", "loser": "candidate-model", "metadata": {"citation_count": 1}},',
        '    ]',
        `    print(${entrypointName}("candidate-model", example_votes))`,
      ];
    case 'ranking':
      return [
        'if __name__ == "__main__":',
        '    ranked_items = [["item_a", "item_b", "item_c"]]',
        '    ground_truth = [["item_b", "item_d"]]',
        '    predictions = ["answer_a"]',
        '    references = ["answer_a"]',
        `    print(${entrypointName}(ranked_items, ground_truth, predictions=predictions, references=references, metadata={"note": "attach paper-specific metadata here"}, k=10))`,
      ];
    case 'forecasting':
      return [
        'if __name__ == "__main__":',
        '    predictions = [0.9, 1.2, 0.7]',
        '    references = [1.0, 1.0, 0.5]',
        `    print(${entrypointName}(predictions, references))`,
      ];
    default:
      if (includesRankingMetrics) {
        return [
          'if __name__ == "__main__":',
          '    predictions = ["example prediction"]',
          '    references = ["example reference"]',
          '    ranked_items = [["item_a", "item_b", "item_c"]]',
          '    relevant_items = [["item_b", "item_d"]]',
          '    metadata = {"note": "attach paper-specific intermediate signals here"}',
          `    print(${entrypointName}(predictions, references, ranked_items=ranked_items, relevant_items=relevant_items, metadata=metadata, k=10))`,
        ];
      }

      if (usesImageGenerationStyle) {
        return [
          'if __name__ == "__main__":',
          '    predictions = [',
          '        [',
          '            [[0, 0], [0, 0]],',
          '            [[0, 0], [0, 0]],',
          '            [[0, 0], [0, 0]],',
          '        ]',
          '    ]',
          '    references = [',
          '        [',
          '            [[255, 255], [255, 255]],',
          '            [[255, 255], [255, 255]],',
          '            [[255, 255], [255, 255]],',
          '        ]',
          '    ]',
          includesMetadataMetrics
            ? `    print(${entrypointName}(predictions, references, metadata={"note": "attach paper-specific metadata here"}))`
            : `    print(${entrypointName}(predictions, references))`,
        ];
      }

      if (usesClusterStyle) {
        return [
          'if __name__ == "__main__":',
          '    predictions = [',
          '        {"embedding": [0.0, 0.1], "cluster_label": "alpha"},',
          '        {"embedding": [0.1, 0.0], "cluster_label": "alpha"},',
          '        {"embedding": [1.0, 1.1], "cluster_label": "beta"},',
          '        {"embedding": [1.1, 1.0], "cluster_label": "beta"},',
          '    ]',
          '    references = [',
          '        {"true_label": "alpha", "batch_label": "batch_a"},',
          '        {"true_label": "alpha", "batch_label": "batch_b"},',
          '        {"true_label": "beta", "batch_label": "batch_a"},',
          '        {"true_label": "beta", "batch_label": "batch_b"},',
          '    ]',
          '    metadata = {',
          '        "embeddings": [item["embedding"] for item in predictions],',
          '        "cluster_labels": [item["cluster_label"] for item in predictions],',
          '        "true_labels": [item["true_label"] for item in references],',
          '        "batch_labels": [item["batch_label"] for item in references],',
          '        "original_dataset": [[0.0, 1.0], [0.2, 0.8], [1.0, 0.0], [0.9, 0.1]],',
          '    }',
          `    print(${entrypointName}(predictions, references, metadata=metadata))`,
        ];
      }

      if (usesTextDistanceStyle) {
        return [
          'if __name__ == "__main__":',
          '    predictions = ["the model generated a short answer"]',
          '    references = ["the model generated short answers"]',
          includesMetadataMetrics
            ? `    print(${entrypointName}(predictions, references, metadata={"note": "attach paper-specific metadata here"}))`
            : `    print(${entrypointName}(predictions, references))`,
        ];
      }

      if (usesDetectionStyle) {
        return [
          'if __name__ == "__main__":',
          '    predictions = [',
          '        {"boxes": [[0.0, 0.0, 1.0, 1.0]], "scores": [0.95], "labels": [1]}',
          '    ]',
          '    references = [',
          '        {"boxes": [[0.0, 0.0, 1.0, 1.0]], "labels": [1]}',
          '    ]',
          includesMetadataMetrics
            ? `    print(${entrypointName}(predictions, references, metadata={"note": "attach paper-specific metadata here"}))`
            : `    print(${entrypointName}(predictions, references))`,
        ];
      }

      if (usesTopKCandidateStyle) {
        return [
          'if __name__ == "__main__":',
          '    predictions = [["cat", "dog", "bird"], ["red", "blue", "green"]]',
          '    references = ["dog", "green"]',
          includesMetadataMetrics
            ? `    print(${entrypointName}(predictions, references, metadata={"note": "attach paper-specific metadata here"}))`
            : `    print(${entrypointName}(predictions, references))`,
        ];
      }

      if (usesQValueStyle) {
        return [
          'if __name__ == "__main__":',
          '    predictions = [',
          '        {"q_values": [0.2, 0.4, 1.1], "episode_rewards": [1.0, 0.5, 2.0]},',
          '        {"q_values": [0.1, 0.7, 0.3], "episode_rewards": [0.3, 0.7, 0.4]},',
          '    ]',
          '    references = [',
          '        {"q_values": [0.1, 0.5, 1.0], "episode_rewards": [0.8, 0.6, 1.8]},',
          '        {"q_values": [0.2, 0.6, 0.4], "episode_rewards": [0.4, 0.5, 0.5]},',
          '    ]',
          includesMetadataMetrics
            ? `    print(${entrypointName}(predictions, references, metadata={"note": "attach paper-specific metadata here"}))`
            : `    print(${entrypointName}(predictions, references))`,
        ];
      }

      if (usesTrajectoryStyle) {
        return [
          'if __name__ == "__main__":',
          '    predictions = [[1.0, 0.5, 2.0], [0.3, 0.7, 0.4]]',
          '    references = [[0.8, 0.6, 1.8], [0.4, 0.5, 0.5]]',
          includesMetadataMetrics
            ? `    print(${entrypointName}(predictions, references, metadata={"note": "attach paper-specific metadata here"}))`
            : `    print(${entrypointName}(predictions, references))`,
        ];
      }

      if (usesStructureStyle) {
        return [
          'if __name__ == "__main__":',
          '    predictions = [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]',
          '    references = [[[0.0, 0.1, 0.0], [1.0, 0.9, 1.0]]]',
          includesMetadataMetrics
            ? `    print(${entrypointName}(predictions, references, metadata={"note": "attach paper-specific metadata here"}))`
            : `    print(${entrypointName}(predictions, references))`,
        ];
      }

      if (usesSequenceDistanceStyle) {
        return [
          'if __name__ == "__main__":',
          '    predictions = [[1.0, 2.0, 3.0], [0.5, 0.7, 0.9]]',
          '    references = [[1.0, 2.5, 2.5], [0.4, 0.8, 1.0]]',
          includesMetadataMetrics
            ? `    print(${entrypointName}(predictions, references, metadata={"note": "attach paper-specific metadata here"}))`
            : `    print(${entrypointName}(predictions, references))`,
        ];
      }

      if (usesNumericRegression) {
        return [
          'if __name__ == "__main__":',
          '    predictions = [0.9, 1.2, 0.7]',
          '    references = [1.0, 1.0, 0.5]',
          includesMetadataMetrics
            ? `    print(${entrypointName}(predictions, references, metadata={"note": "attach paper-specific metadata here"}))`
            : `    print(${entrypointName}(predictions, references))`,
        ];
      }

      if (usesScoreLabels) {
        return [
          'if __name__ == "__main__":',
          '    predictions = [0.9, 0.2, 0.8, 0.6]',
          '    references = [1, 0, 1, 1]',
          includesMetadataMetrics
            ? `    print(${entrypointName}(predictions, references, metadata={"note": "attach paper-specific metadata here"}))`
            : `    print(${entrypointName}(predictions, references))`,
        ];
      }

      if (usesBinaryLabels) {
        return [
          'if __name__ == "__main__":',
          '    predictions = [1, 0, 1, 1]',
          '    references = [1, 0, 1, 1]',
          includesMetadataMetrics
            ? `    print(${entrypointName}(predictions, references, metadata={"note": "attach paper-specific metadata here"}))`
            : `    print(${entrypointName}(predictions, references))`,
        ];
      }

      return [
        'if __name__ == "__main__":',
        '    predictions = ["example prediction"]',
        '    references = ["example reference"]',
        includesMetadataMetrics
          ? `    print(${entrypointName}(predictions, references, metadata={"note": "attach paper-specific metadata here"}))`
          : `    print(${entrypointName}(predictions, references))`,
      ];
  }
}

function buildChecklistSection(
  summary: PaperEvaluationSummary,
  domain: string,
  framework: string,
  parseMode: 'sync' | 'async',
  language: 'ko' | 'en',
  guidance: EvaluationGuidance,
  metricPlans: MetricPlan[],
): string {
  const lines: string[] = [];
  const unresolved = metricPlans.filter((plan) => requiresFallbackLead(plan));
  lines.push(language === 'ko' ? '## 실험 재현 체크리스트' : '## Reproduction Checklist');
  lines.push('');
  lines.push(
    language === 'ko'
      ? `- [ ] 태스크 유형: ${fallbackText(summary.task_type, '확인 필요')}`
      : `- [ ] Task type: ${fallbackText(summary.task_type, 'Needs review')}`,
  );
  lines.push(
    language === 'ko'
      ? `- [ ] 연구 분야 힌트: ${domain}`
      : `- [ ] Research area hint: ${domain}`,
  );
  lines.push(
    language === 'ko'
      ? `- [ ] 프레임워크: ${framework}`
      : `- [ ] Framework: ${framework}`,
  );
  lines.push(
    language === 'ko'
      ? `- [ ] 데이터셋: ${formatList(summary.datasets, '확인 필요')}`
      : `- [ ] Datasets: ${formatList(summary.datasets, 'Needs review')}`,
  );
  lines.push(
    language === 'ko'
      ? `- [ ] 평가지표: ${formatMetricChecklist(summary.evaluation_metrics, '확인 필요')}`
      : `- [ ] Metrics: ${formatMetricChecklist(summary.evaluation_metrics, 'Needs review')}`,
  );
  lines.push(
    language === 'ko'
      ? `- [ ] 베이스라인: ${formatBaselines(summary.baselines, '확인 필요')}`
      : `- [ ] Baselines: ${formatBaselines(summary.baselines, 'Needs review')}`,
  );
  lines.push(
    language === 'ko'
      ? `- [ ] 구현 세부사항: ${fallbackText(summary.implementation_details, '저자 구현 또는 부록 확인 필요')}`
      : `- [ ] Implementation details: ${fallbackText(summary.implementation_details, 'Review appendix or author code')}`,
  );
  lines.push(
    language === 'ko'
      ? `- [ ] 재현 초점: ${fallbackText(guidance.reproduction_focus, '핵심 재현 초점을 다시 정의하세요.')}`
      : `- [ ] Reproduction focus: ${fallbackText(guidance.reproduction_focus, 'Clarify the main reproduction target.')}`,
  );
  for (const caution of guidance.implementation_cautions.slice(0, 4)) {
    lines.push(language === 'ko' ? `- [ ] 주의: ${caution}` : `- [ ] Caution: ${caution}`);
  }
  lines.push(
    language === 'ko'
      ? `- [ ] 문서 파싱 모드: \`${parseMode}\``
      : `- [ ] Document parse mode: \`${parseMode}\``,
  );
  for (const plan of unresolved.slice(0, 4)) {
    lines.push(
      language === 'ko'
        ? `- [ ] 수동 검토 필요 metric: ${plan.metricName} (${fallbackText(plan.note, '논문 프로토콜 확인 필요')})`
        : `- [ ] Manual review metric: ${plan.metricName} (${fallbackText(plan.note, 'Inspect the paper-specific protocol.')})`,
    );
  }

  return lines.join('\n');
}

function buildRepositorySection(
  discovery: RepositoryDiscoveryResult,
  metricPlans: MetricPlan[],
  language: 'ko' | 'en',
): string {
  const needsLead = metricPlans.some((plan) => requiresFallbackLead(plan));
  if (!needsLead && discovery.candidates.length === 0) {
    return '';
  }

  const lines: string[] = [];
  lines.push(language === 'ko' ? '## Author Code / 구현 단서' : '## Author Code / Implementation Leads');
  lines.push('');

  if (discovery.candidates.length === 0) {
    lines.push(
      language === 'ko'
        ? '- 문서 내부 GitHub 링크나 신뢰도 높은 검색 결과를 찾지 못했습니다.'
        : '- No document-linked or high-confidence GitHub repository was found.',
    );
  } else {
    for (const candidate of discovery.candidates.slice(0, 3)) {
      const stars = candidate.stars !== undefined ? `, stars=${candidate.stars}` : '';
      lines.push(`- ${candidate.url} (${candidate.source}, confidence=${candidate.confidence}${stars})`);
      lines.push(language === 'ko' ? `  근거: ${candidate.reason}` : `  Reason: ${candidate.reason}`);
    }
  }

  if (needsLead) {
    lines.push(
      language === 'ko'
        ? '- 방법론 fallback: 저자 코드를 찾지 못하면 evidence snippet, reported metric formula/value, baseline row를 기준으로 입력/출력 스키마와 judge rubric을 재구성하세요.'
        : '- Methodological fallback: when author code is unavailable, reconstruct the evaluator input/output schema and judge rubric from evidence snippets, reported metric formulas/values, and baseline rows.',
    );
  }

  return lines.join('\n');
}

function buildMetricsSection(summary: PaperEvaluationSummary, language: 'ko' | 'en'): string {
  const sections: string[] = [];
  const metrics = summary.evaluation_metrics ?? [];
  const baselines = summary.baselines ?? [];

  sections.push(language === 'ko' ? '## 보고된 지표' : '## Reported Metrics');
  sections.push('');

  if (metrics.length === 0) {
    sections.push(language === 'ko' ? '- 추출된 지표가 없습니다.' : '- No metrics were extracted.');
  } else {
    sections.push(
      '| Metric | Formula | Value |\n| --- | --- | --- |',
    );
    for (const metric of metrics) {
      sections.push(
        `| ${escapeTableCell(fallbackText(metric.name, '-'))} | ${escapeTableCell(
          fallbackText(metric.formula, '-'),
        )} | ${escapeTableCell(formatMetricValue(metric.value))} |`,
      );
    }
  }

  sections.push('');
  sections.push(language === 'ko' ? '## 베이스라인 스냅샷' : '## Baseline Snapshot');
  sections.push('');

  if (baselines.length === 0) {
    sections.push(language === 'ko' ? '- 추출된 베이스라인이 없습니다.' : '- No baselines were extracted.');
  } else {
    sections.push('| Baseline | Scores |\n| --- | --- |');
    for (const baseline of baselines) {
      const scores = baseline.scores
        ? Object.entries(baseline.scores)
            .map(([name, value]) => `${name}=${formatMetricValue(value)}`)
            .join(', ')
        : '-';
      sections.push(
        `| ${escapeTableCell(fallbackText(baseline.model_name, '-'))} | ${escapeTableCell(scores || '-')} |`,
      );
    }
  }

  return sections.join('\n');
}

function buildEvidenceSection(
  evidenceRecords: EvidenceRecord[],
  language: 'ko' | 'en',
): string {
  const sections: string[] = [];
  sections.push(language === 'ko' ? '## 근거 발췌' : '## Evidence Snippets');
  sections.push('');
  if (evidenceRecords.length === 0) {
    sections.push(language === 'ko' ? '- 근거 발췌를 찾지 못했습니다.' : '- No evidence snippets were found.');
    return sections.join('\n');
  }

  for (const record of evidenceRecords) {
    const locator = formatEvidenceLocator(record, language);
    sections.push(locator ? `- [${record.id} ${locator}] ${record.text}` : `- [${record.id}] ${record.text}`);
  }

  return sections.join('\n');
}

function formatEvidenceLocator(record: EvidenceRecord, language: 'ko' | 'en'): string {
  const parts: string[] = [];
  if (record.page !== undefined) {
    parts.push(`p.${record.page}`);
  }
  if (record.section?.trim()) {
    parts.push(language === 'ko' ? `section ${record.section.trim()}` : `section ${record.section.trim()}`);
  }
  if (record.category?.trim()) {
    const category = record.elementId ? `${record.category.trim()}#${record.elementId}` : record.category.trim();
    parts.push(category);
  } else if (record.elementId) {
    parts.push(`#${record.elementId}`);
  }

  return parts.join(' · ');
}

function buildJudgePromptSection(judgePrompt: string, language: 'ko' | 'en'): string {
  return [
    language === 'ko' ? '## LLM-as-judge 프롬프트' : '## LLM-as-Judge Prompt',
    '',
    judgePrompt.trim(),
  ].join('\n');
}

function buildJudgePrompt(
  summary: PaperEvaluationSummary,
  guidance: EvaluationGuidance,
  language: 'ko' | 'en',
): string {
  const datasets = formatList(summary.datasets, language === 'ko' ? '명시되지 않음' : 'Not specified');
  const metrics = formatMetricChecklist(summary.evaluation_metrics, language === 'ko' ? '명시되지 않음' : 'Not specified');
  const baselines = formatBaselines(summary.baselines, language === 'ko' ? '명시되지 않음' : 'Not specified');

  const lines: string[] = [];
  if (language === 'ko') {
    lines.push(`당신은 "${fallbackText(summary.task_type, '논문 평가')}" 재현 실험을 검토하는 평가자입니다.`);
    lines.push('후보 결과물이나 실험 보고서를 아래 기준으로 평가하세요.');
    lines.push('');
    lines.push('평가 컨텍스트');
    lines.push(`- 데이터셋: ${datasets}`);
    lines.push(`- 핵심 지표: ${metrics}`);
    lines.push(`- 비교 기준 베이스라인: ${baselines}`);
    lines.push(`- 재현 초점: ${fallbackText(guidance.reproduction_focus, '논문과 동일한 평가 절차를 최대한 보존하세요.')}`);
    lines.push('');
    lines.push('평가 기준');
    guidance.judge_criteria.forEach((criterion, index) => {
      lines.push(`${index + 1}. ${criterion.name}: ${criterion.description}`);
    });
    lines.push('');
    lines.push('실패 조건');
    guidance.judge_failure_conditions.forEach((condition) => {
      lines.push(`- ${condition}`);
    });
    lines.push('');
    lines.push('출력 형식');
    lines.push('- 간단한 총평');
    lines.push('- 기준별 1-5점');
    lines.push('- 최종 판정');
  } else {
    lines.push(`You are evaluating a reproduction artifact for "${fallbackText(summary.task_type, 'paper evaluation')}".`);
    lines.push('Score the candidate output or experiment report against the paper-specific setup below.');
    lines.push('');
    lines.push('Evaluation context');
    lines.push(`- Datasets: ${datasets}`);
    lines.push(`- Core metrics: ${metrics}`);
    lines.push(`- Baselines: ${baselines}`);
    lines.push(`- Reproduction focus: ${fallbackText(guidance.reproduction_focus, 'Preserve the paper-specific evaluation procedure as closely as possible.')}`);
    lines.push('');
    lines.push('Criteria');
    guidance.judge_criteria.forEach((criterion, index) => {
      lines.push(`${index + 1}. ${criterion.name}: ${criterion.description}`);
    });
    lines.push('');
    lines.push('Failure conditions');
    guidance.judge_failure_conditions.forEach((condition) => {
      lines.push(`- ${condition}`);
    });
    lines.push('');
    lines.push('Output format');
    lines.push('- Short rationale');
    lines.push('- 1-5 score for each criterion');
    lines.push('- Final verdict');
  }

  return lines.join('\n').trim();
}

function normalizeEvaluationGuidance(
  guidance: EvaluationGuidance,
  language: 'ko' | 'en',
): EvaluationGuidance {
  const criteria = (guidance.judge_criteria ?? [])
    .filter((criterion) => criterion.name?.trim() && criterion.description?.trim())
    .map((criterion) => ({
      name: criterion.name.trim(),
      description: criterion.description.trim(),
    }))
    .slice(0, 5);
  const cautions = (guidance.implementation_cautions ?? [])
    .map((caution) => caution.trim())
    .filter(Boolean)
    .slice(0, 6);
  const failureConditions = (guidance.judge_failure_conditions ?? [])
    .map((condition) => condition.trim())
    .filter(Boolean)
    .slice(0, 6);

  return {
    reproduction_focus:
      guidance.reproduction_focus?.trim()
      || (language === 'ko'
        ? '논문이 보고한 평가 지표와 입력 형식을 먼저 그대로 맞추세요.'
        : 'Match the paper-reported evaluation metrics and input format before extending the setup.'),
    implementation_cautions:
      cautions.length > 0
        ? cautions
        : [language === 'ko' ? '생성 코드의 가정과 누락된 세부사항을 사람이 다시 검토하세요.' : 'Review the generated code for assumptions and missing implementation details.'],
    judge_criteria:
      criteria.length > 0
        ? criteria
        : [
            {
              name: language === 'ko' ? '지표 충실도' : 'Metric fidelity',
              description: language === 'ko'
                ? '논문이 사용한 지표와 계산 절차를 그대로 반영했는지 확인하세요.'
                : 'Check whether the artifact preserves the paper-reported metrics and calculation procedure.',
            },
            {
              name: language === 'ko' ? '재현 가능성' : 'Reproducibility',
              description: language === 'ko'
                ? '입력 형식, 데이터셋, 베이스라인 비교가 실제로 다시 실행 가능한지 확인하세요.'
                : 'Check whether inputs, datasets, and baseline comparisons are concrete enough to rerun.',
            },
          ],
    judge_failure_conditions:
      failureConditions.length > 0
        ? failureConditions
        : [
            language === 'ko'
              ? '논문에 없는 지표를 임의로 추가했다면 실패로 간주하세요.'
              : 'Fail the artifact if it introduces metrics that were not part of the paper setup.',
            language === 'ko'
              ? '핵심 입력 형식이나 데이터셋 전제가 빠져 있으면 실패로 간주하세요.'
              : 'Fail the artifact if key input-format or dataset assumptions are missing.',
          ],
  };
}

function buildFallbackEvaluationGuidance(
  summary: PaperEvaluationSummary,
  taskMode: TaskMode,
  evidenceSnippets: string[],
  language: 'ko' | 'en',
): EvaluationGuidance {
  const metrics = (summary.evaluation_metrics ?? []).map((metric) => metric.name?.trim()).filter(Boolean) as string[];
  const datasets = (summary.datasets ?? []).filter(Boolean);
  const baseCautions = [
    summary.implementation_details?.trim(),
    evidenceSnippets[0],
    language === 'ko'
      ? '추출된 수치와 표 셀을 논문 원문에서 다시 대조하세요.'
      : 'Cross-check extracted scores and table cells against the original paper.',
    taskMode === 'ranking'
      ? (language === 'ko' ? 'top-k 값과 relevance 정의를 명시적으로 고정하세요.' : 'Fix the top-k cutoff and relevance definition explicitly.')
      : undefined,
  ]
    .filter(Boolean)
    .map((value) => String(value).trim())
    .slice(0, 4);

  return normalizeEvaluationGuidance(
    {
      reproduction_focus:
        language === 'ko'
          ? `${fallbackText(summary.task_type, '실험 태스크')}의 평가 입력 형식, 데이터셋(${formatList(datasets, '확인 필요')}), 지표(${formatList(metrics, '확인 필요')})를 논문과 최대한 동일하게 맞추세요.`
          : `Match the paper's evaluation inputs, datasets (${formatList(datasets, 'needs review')}), and metrics (${formatList(metrics, 'needs review')}) as closely as possible.`,
      implementation_cautions: baseCautions,
      judge_criteria: [
        {
          name: language === 'ko' ? '지표 일치성' : 'Metric fidelity',
          description: language === 'ko'
            ? '논문이 보고한 지표 이름, 계산 단위, 집계 방식이 그대로 보존됐는지 확인하세요.'
            : 'Check whether metric names, aggregation, and scoring units match the paper.',
        },
        {
          name: language === 'ko' ? '입력 형식 일치성' : 'Input fidelity',
          description: language === 'ko'
            ? '예측값과 정답의 입력 구조가 논문 프로토콜과 맞는지 확인하세요.'
            : 'Check whether prediction/reference inputs match the paper protocol.',
        },
      ],
      judge_failure_conditions: [
        language === 'ko'
          ? '논문에 없는 metric이나 baseline을 임의로 추가하면 실패입니다.'
          : 'Fail the artifact if it adds metrics or baselines that were not in the paper.',
        language === 'ko'
          ? '데이터셋 또는 top-k / threshold / judge rubric 같은 핵심 설정이 비어 있으면 실패입니다.'
          : 'Fail the artifact if core settings such as datasets, top-k, thresholds, or judge rubrics are missing.',
      ],
    },
    language,
  );
}

function buildEvaluationEvidenceRecords(
  summary: PaperEvaluationSummary,
  experimentMarkdown: string,
  anchors: import('../client').ParseAnchor[] = [],
): EvidenceRecord[] {
  const implementationHint = compactEvidenceHint(summary.implementation_details ?? '', 260);
  const metricQueries = (summary.evaluation_metrics ?? [])
    .map((metric) => metric.name ?? '')
    .filter((metricName) => metricName.trim());
  const datasetQueries = [...(summary.datasets ?? [])];
  const baselineQueries = buildBaselineEvidenceQueries(summary.baselines ?? []);
  const initialRecords = mergeEvidenceRecordGroups(
    [
      buildFocusedEvidenceRecords(
        experimentMarkdown,
        metricQueries,
        {
          maxSnippets: 2,
          maxPerQuery: 1,
          maxCharsPerSnippet: 220,
          minScore: 1,
          anchors,
        },
      ),
      mergeEvidenceRecordGroups(
        [
          buildAnchoredEvidenceRecords(
            experimentMarkdown,
            metricQueries,
            {
              maxSnippets: 2,
              maxCharsPerSnippet: 220,
              minScore: 1,
              anchors,
            },
          ),
          buildEvidenceRecords(
            experimentMarkdown,
            metricQueries,
            {
              maxSnippets: 2,
              maxCharsPerSnippet: 240,
              minScore: 1,
              anchors,
            },
          ),
        ],
        {
          maxSnippets: 2,
          anchors,
        },
      ),
      mergeEvidenceRecordGroups(
        [
          buildFocusedEvidenceRecords(
            experimentMarkdown,
            baselineQueries,
            {
              maxSnippets: 3,
              maxPerQuery: 1,
              maxCharsPerSnippet: 260,
              minScore: 1,
              anchors,
            },
          ),
          buildTableEvidenceRecords(
            experimentMarkdown,
            baselineQueries,
            {
              maxSnippets: 3,
              maxCharsPerSnippet: 260,
              minScore: 1,
              anchors,
            },
          ),
          buildAnchoredEvidenceRecords(
            experimentMarkdown,
            baselineQueries,
            {
              maxSnippets: 2,
              maxCharsPerSnippet: 240,
              minScore: 1,
              anchors,
            },
          ),
        ],
        {
          maxSnippets: 3,
          anchors,
        },
      ),
      mergeEvidenceRecordGroups(
        [
          buildAnchoredEvidenceRecords(
            experimentMarkdown,
            datasetQueries,
            {
              maxSnippets: 3,
              maxCharsPerSnippet: 230,
              minScore: 1,
              anchors,
            },
          ),
          buildEvidenceRecords(
            experimentMarkdown,
            datasetQueries,
            {
              maxSnippets: 3,
              maxCharsPerSnippet: 260,
              minScore: 1,
              anchors,
            },
          ),
        ],
        {
          maxSnippets: 3,
          anchors,
        },
      ),
      buildFocusedEvidenceRecords(
        experimentMarkdown,
        datasetQueries,
        {
          maxSnippets: 4,
          maxPerQuery: 1,
          maxCharsPerSnippet: 260,
          minScore: 1,
          anchors,
        },
      ),
      mergeEvidenceRecordGroups(
        [
          buildAnchoredEvidenceRecords(
            experimentMarkdown,
            [summary.task_type ?? '', implementationHint],
            {
              maxSnippets: 1,
              maxCharsPerSnippet: 220,
              minScore: 2,
              anchors,
            },
          ),
          buildEvidenceRecords(
            experimentMarkdown,
            [summary.task_type ?? '', implementationHint],
            {
              maxSnippets: 1,
              maxCharsPerSnippet: 220,
              minScore: 2,
              anchors,
            },
          ),
        ],
        { maxSnippets: 1, anchors },
      ),
    ],
    {
      maxSnippets: 12,
      anchors,
    },
  );

  const uncoveredDatasetQueries = (summary.datasets ?? [])
    .filter((dataset) => dataset.trim())
    .filter((dataset) => !containsEvidenceValue(initialRecords, dataset));
  const uncoveredBaselineQueries = baselineQueries
    .filter((query) => query.trim())
    .filter((query) => !containsEvidenceValue(initialRecords, query));

  if (uncoveredDatasetQueries.length === 0 && uncoveredBaselineQueries.length === 0) {
    return initialRecords;
  }

  const uncoveredRecords = mergeEvidenceRecordGroups(
    [
      buildFocusedEvidenceRecords(
        experimentMarkdown,
        uncoveredDatasetQueries,
        {
          maxSnippets: Math.min(6, uncoveredDatasetQueries.length),
          maxPerQuery: 1,
          maxCharsPerSnippet: 260,
          minScore: 1,
          anchors,
        },
      ),
      buildAnchoredEvidenceRecords(
        experimentMarkdown,
        uncoveredDatasetQueries,
        {
          maxSnippets: Math.min(6, uncoveredDatasetQueries.length),
          maxCharsPerSnippet: 240,
          minScore: 1,
          anchors,
        },
      ),
      buildEvidenceRecords(
        experimentMarkdown,
        uncoveredDatasetQueries,
        {
          maxSnippets: Math.min(6, uncoveredDatasetQueries.length),
          maxCharsPerSnippet: 280,
          minScore: 1,
          anchors,
        },
      ),
      buildFocusedEvidenceRecords(
        experimentMarkdown,
        uncoveredBaselineQueries,
        {
          maxSnippets: Math.min(4, uncoveredBaselineQueries.length),
          maxPerQuery: 1,
          maxCharsPerSnippet: 260,
          minScore: 1,
          anchors,
        },
      ),
      buildTableEvidenceRecords(
        experimentMarkdown,
        uncoveredBaselineQueries,
        {
          maxSnippets: Math.min(4, uncoveredBaselineQueries.length),
          maxCharsPerSnippet: 280,
          minScore: 1,
          anchors,
        },
      ),
      buildAnchoredEvidenceRecords(
        experimentMarkdown,
        uncoveredBaselineQueries,
        {
          maxSnippets: Math.min(3, uncoveredBaselineQueries.length),
          maxCharsPerSnippet: 240,
          minScore: 1,
          anchors,
        },
      ),
    ],
    {
      maxSnippets: Math.min(8, uncoveredDatasetQueries.length + uncoveredBaselineQueries.length),
      anchors,
    },
  );

  return mergeEvidenceRecordGroups(
    [initialRecords, uncoveredRecords],
    {
      maxSnippets: 14,
      anchors,
    },
  );
}

function buildBaselineEvidenceQueries(baselines: Baseline[]): string[] {
  const seen = new Set<string>();
  const queries: string[] = [];

  for (const baseline of baselines) {
    const modelName = baseline.model_name?.trim();
    if (!modelName) {
      continue;
    }

    const scoreNames = Object.keys(baseline.scores ?? {})
      .map((scoreName) => scoreName.trim())
      .filter(Boolean)
      .slice(0, 3);
    const candidates = [
      modelName,
      scoreNames.length > 0 ? `${modelName} ${scoreNames.join(' ')}` : '',
    ];

    for (const candidate of candidates) {
      const compact = candidate.replace(/\s+/g, ' ').trim();
      if (!compact) {
        continue;
      }
      const key = compact.toLowerCase();
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);
      queries.push(compact);
    }
  }

  return queries;
}

function containsEvidenceValue(records: EvidenceRecord[], rawValue: string): boolean {
  const normalizedEvidence = normalizeQueryPhrase(records.map((record) => record.text).join('\n'));
  if (!normalizedEvidence) {
    return false;
  }

  for (const variant of expandEvidenceQueries([rawValue])) {
    const normalizedVariant = normalizeQueryPhrase(variant);
    if (!normalizedVariant) {
      continue;
    }

    const paddedEvidence = ` ${normalizedEvidence} `;
    const paddedVariant = ` ${normalizedVariant} `;
    if (paddedEvidence.includes(paddedVariant) || normalizedEvidence.includes(normalizedVariant)) {
      return true;
    }
  }

  return false;
}

function compactEvidenceHint(value: string, maxChars: number): string {
  const compact = value.replace(/\s+/g, ' ').trim();
  if (compact.length <= maxChars) {
    return compact;
  }

  const clipped = compact.slice(0, maxChars);
  const lastBoundary = Math.max(clipped.lastIndexOf('. '), clipped.lastIndexOf('; '), clipped.lastIndexOf(', '));
  return `${(lastBoundary > 80 ? clipped.slice(0, lastBoundary) : clipped).trimEnd()}...`;
}

async function maybeRepairEvaluationSummary(
  client: UpstageClient,
  summary: PaperEvaluationSummary,
  schema: JsonSchema,
  experimentMarkdown: string,
  fileName: string,
  language: 'ko' | 'en',
): Promise<PaperEvaluationSummary> {
  if (!shouldRepairEvaluationSummary(summary)) {
    return normalizeEvaluationSummary(summary);
  }

  const repaired = await client.chatStructured<PaperEvaluationSummary>({
    model: DEFAULT_SOLAR_MODEL,
    schemaName: 'paper_evaluation_summary_repair',
    schema,
    temperature: 0.1,
    messages: [
      {
        role: 'system',
        content:
          language === 'ko'
            ? '당신은 논문 실험 메타데이터를 복원하는 연구 조교다. 제공된 실험 발췌에서 확인 가능한 정보만 추출하고, 없으면 빈 문자열이나 빈 배열을 유지하라.'
            : 'You repair experiment metadata extraction from a parsed paper excerpt. Only use information supported by the excerpt. Keep directly measurable evaluation metrics, and drop qualitative claims or complexity statements unless they are clearly reported as reproducible measured metrics. Leave fields empty when unsupported.',
      },
      {
        role: 'user',
        content:
          `File name: ${fileName}\n\n` +
          `Current extracted summary:\n${JSON.stringify(summary, null, 2)}\n\n` +
          `Experiment excerpt:\n${clipText(experimentMarkdown, 28_000)}`,
      },
    ],
  });

  return normalizeEvaluationSummary(mergeEvaluationSummary(summary, repaired));
}

function inferTaskMode(summary: PaperEvaluationSummary): TaskMode {
  const corpus = [
    summary.task_type ?? '',
    ...(summary.evaluation_metrics ?? []).map((metric) => metric.name ?? ''),
    summary.implementation_details ?? '',
  ]
    .join(' ')
    .toLowerCase();

  if (/(preference|pairwise|bradley|win rate|arena)/.test(corpus)) {
    return 'pairwise_preference';
  }

  if (/(ndcg|mrr|hit@|hr@|precision@|recall@|pass@|top[- ]?k|ranking|retrieval)/.test(corpus)) {
    return 'ranking';
  }

  if (/(forecast|forecasting|time series|temporal|rmse|mse|mae|mape|smape|mase|dtw)/.test(corpus)) {
    return 'forecasting';
  }

  if (/(classification|sentiment|intent|nli|label)/.test(corpus)) {
    return 'classification';
  }

  return 'prediction_reference';
}

function inferDomain(summary: PaperEvaluationSummary): string {
  const corpus = [
    summary.task_type ?? '',
    ...(summary.datasets ?? []),
    ...(summary.evaluation_metrics ?? []).map((metric) => metric.name ?? ''),
  ]
    .join(' ')
    .toLowerCase();

  if (/\b(graph|subgraph|node|edge)\b|knowledge graph|textual graph/.test(corpus)) {
    return 'graph';
  }

  if (/(protein|antibody|amino acid|residue|paratope|epitope|binding affinity|structure prediction|structure evaluation|folding|pdb|alphafold|rosetta|tm-score|gdt|lddt|rmsd|bioinformatics|genomic|proteomic)/.test(corpus)) {
    return 'bioinformatics';
  }

  if (/(translation|summarization|classification|retrieval|search|citation|multilingual|qa|question answering|text|language model|llm|rouge|bleu|bertscore|hallucination|factual|claim|reference-based|nli|entailment|contradiction|neutral)/.test(corpus)) {
    return 'nlp';
  }

  if (/(image|vision|object detection|semantic segmentation|instance segmentation|coco|imagenet|miou|fid|psnr|ssim)/.test(corpus)) {
    return 'cv';
  }

  if (/(reinforcement|policy|reward|gym|episode|return)/.test(corpus)) {
    return 'rl';
  }

  if (/(time series|forecast|forecasting|anomaly|temporal|multivariate|univariate|long-term|ett[h|m]?1|ett[h|m]?2|ecl|weather|traffic|exchange|ili|mae|mase|smape|rmse|mse|dtw)/.test(corpus)) {
    return 'time-series';
  }

  if (
    /(recommendation|recommender|sequential recommendation|collaborative filtering|user-item|click-through|ctr\b)/.test(corpus)
    || (/(ndcg|hit rate|hit@|hr@|mrr\b)/.test(corpus) && /(recommend|user|item|session|click|movie|retail|catalog)/.test(corpus))
  ) {
    return 'recommendation';
  }

  return 'general';
}

function libraryHintsByDomain(domain: string): string {
  switch (domain) {
    case 'bioinformatics':
      return 'biopython, biotite, mdtraj';
    case 'graph':
      return 'networkx, torch_geometric';
    case 'nlp':
      return 'evaluate, nltk, sacrebleu';
    case 'cv':
      return 'torchmetrics, pycocotools';
    case 'rl':
      return 'gymnasium';
    case 'recommendation':
      return 'recbole';
    case 'time-series':
      return 'tsmetric';
    default:
      return 'numpy, pandas';
  }
}

function ensureCodeFence(content: string, language: string): string {
  if (content.trim().startsWith('```')) {
    return content.trim();
  }

  return `\`\`\`${language}\n${content.trim()}\n\`\`\``;
}

function pickLanguage(candidates: string[]): 'ko' | 'en' {
  return candidates.some((candidate) => /[가-힣]/.test(candidate)) ? 'ko' : 'en';
}

function clipText(input: string, maxChars: number): string {
  if (input.length <= maxChars) {
    return input;
  }

  return `${input.slice(0, maxChars)}\n\n[truncated]`;
}

function formatList(values: string[] | undefined, fallback: string): string {
  if (!values || values.length === 0) {
    return fallback;
  }

  return values.map((value) => value.trim()).filter(Boolean).join(', ') || fallback;
}

function formatMetricChecklist(metrics: EvaluationMetric[] | undefined, fallback: string): string {
  if (!metrics || metrics.length === 0) {
    return fallback;
  }

  return metrics
    .map((metric) => {
      const name = metric.name?.trim() || 'unnamed metric';
      const value = formatMetricValue(metric.value);
      return value === '-' ? name : `${name}=${value}`;
    })
    .join(', ');
}

function formatBaselines(baselines: Baseline[] | undefined, fallback: string): string {
  if (!baselines || baselines.length === 0) {
    return fallback;
  }

  return baselines
    .map((baseline) => {
      const modelName = baseline.model_name?.trim() || 'unnamed baseline';
      const scoreText = baseline.scores
        ? Object.entries(baseline.scores)
            .map(([metric, value]) => `${metric}=${formatMetricValue(value)}`)
            .join(', ')
        : '';
      return scoreText ? `${modelName} (${scoreText})` : modelName;
    })
    .join('; ');
}

function formatMetricValue(value: number | string | null | undefined): string {
  if (typeof value === 'number') {
    return Number.isInteger(value) ? `${value}` : `${value}`;
  }

  if (typeof value === 'string' && value.trim()) {
    return value.trim();
  }

  return '-';
}

function fallbackText(value: string | undefined, fallback: string): string {
  return value?.trim() || fallback;
}

function escapeTableCell(value: string): string {
  return value.replace(/\|/g, '\\|').replace(/\n+/g, ' ');
}

function buildMetricPlans(summary: PaperEvaluationSummary, taskMode: TaskMode): MetricPlan[] {
  const metrics = (summary.evaluation_metrics ?? []).slice(0, 8);
  const taskCorpus = [summary.task_type ?? '', ...(summary.datasets ?? [])].join(' ').toLowerCase();
  if (metrics.length === 0) {
    return [
      {
        metricName: fallbackMetricNameForTaskMode(taskMode),
        outputKey: slugifyMetricName(fallbackMetricNameForTaskMode(taskMode)),
        functionName: `compute_${slugifyMetricName(fallbackMetricNameForTaskMode(taskMode))}`,
        kind: taskMode === 'pairwise_preference' ? 'pairwise_win_rate' : 'custom_placeholder',
        note: 'No metric was extracted from the paper. Replace this placeholder with the paper-specific metric.',
      },
    ];
  }

  return metrics.map((metric) => {
    const metricName = fallbackText(metric.name, 'custom metric');
    const slug = slugifyMetricName(metricName);
    const lower = metricName.toLowerCase();
    const formulaLower = fallbackText(metric.formula, '').toLowerCase();

    if (/(win rate|preference win)/.test(lower)) {
      return {
        metricName,
        outputKey: slug,
        functionName: `compute_${slug}`,
        kind: 'pairwise_win_rate',
        formula: metric.formula ?? undefined,
      };
    }

    if (/matthews correlation coefficient|\bmcc\b/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'matthews_corrcoef' };
    }

    if (/(bradley|bradley-terry|bias|response length)/.test(lower)) {
      return {
        metricName,
        outputKey: slug,
        functionName: `estimate_${slug}`,
        kind: 'feature_coefficient_placeholder',
        formula: metric.formula ?? undefined,
        note: `TODO: fit the paper-specific coefficient or preference model for ${metricName}.`,
      };
    }

    if (/accuracy/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'accuracy' };
    }

    if (/(top[- ]?\d+\s*accuracy|top[- ]?k accuracy|top[- ]?\d+\s*acc)/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'top_k_accuracy' };
    }

    if (/exact match/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'exact_match' };
    }

    if (/\bf1\b|f1 score|macro f1|micro f1/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'f1' };
    }

    if (/^pbert$|bertscore precision/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'bertscore_precision' };
    }

    if (/^rbert$|bertscore recall/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'bertscore_recall' };
    }

    if (/^fbert$|bertscore f1/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'bertscore_f1' };
    }

    if (/\bmse\b|mean squared error/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'mse' };
    }

    if (/pearson( correlation)?/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'pearson' };
    }

    if (/spearman('|’)?s?( rank)? correlation/.test(lower) || /spearman/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'spearman' };
    }

    if (/kendall('|’)?s?( rank)? tau|kendall tau/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'kendall_tau' };
    }

    if (/roc-auc|auroc|\bauc\b/.test(lower) && !/pr|average precision/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'auc' };
    }

    if (/average precision|auprc|pr auc|auc-pr/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'average_precision' };
    }

    if (/brier/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'brier_score' };
    }

    if (/\bece\b|expected calibration error|calibration error/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'expected_calibration_error' };
    }

    if (/negative log likelihood|\bnll\b|cross entropy|cross-entropy|log loss/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'negative_log_likelihood' };
    }

    if (/perplexity|\bppl\b/.test(lower) && !/path length|perceptual path length/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'perplexity' };
    }

    if (/precision/.test(lower) && !/average precision|map/.test(lower)) {
      if (/(precision@|precision at k|precision at \d+)/.test(lower)) {
        return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'precision_at_k' };
      }
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'precision' };
    }

    if (/recall/.test(lower)) {
      if (/(recall@|recall at k|recall at \d+)/.test(lower)) {
        return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'recall_at_k' };
      }
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'recall' };
    }

    if (/\biou\b|intersection over union/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'iou' };
    }

    if (/(mask ap|mask average precision|segm ap|segmentation ap)/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'mask_map', resultKey: 'map' };
    }

    if (
      (/^ap(?:50|75|s|m|l)?$/.test(lower)
        || /\bap(?:50|75)\b/.test(lower)
        || /\bap(?:s|m|l)\b/.test(lower)
        || /\bap\s*(?:50|75|s|m|l)\b/.test(lower)
        || /\bmap\b|mean average precision/.test(lower))
      && !/\bmape\b/.test(lower)
    ) {
      const useMaskMetric =
        /(mask|segmentation|instance segmentation|segm)/.test(lower)
        || (
          /(mask|instance segmentation|segm|semantic segmentation)/.test(taskCorpus)
          && !/(object detection|detection|bbox|bounding box)/.test(taskCorpus)
        );
      let resultKey = 'map';
      if (/^ap50$|\bap50\b|\bap\s*50\b/.test(lower)) {
        resultKey = 'map_50';
      } else if (/^ap75$|\bap75\b|\bap\s*75\b/.test(lower)) {
        resultKey = 'map_75';
      } else if (/^aps$|\baps\b|\bap\s*s\b/.test(lower)) {
        resultKey = 'map_small';
      } else if (/^apm$|\bapm\b|\bap\s*m\b/.test(lower)) {
        resultKey = 'map_medium';
      } else if (/^apl$|\bapl\b|\bap\s*l\b/.test(lower)) {
        resultKey = 'map_large';
      }

      return {
        metricName,
        outputKey: slug,
        functionName: `compute_${slug}`,
        kind: useMaskMetric ? 'mask_map' : 'map',
        resultKey,
      };
    }

    if (/\bfid\b|frechet inception distance/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'fid' };
    }

    if (/\bpsnr\b|peak signal-to-noise ratio/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'psnr' };
    }

    if (/\bssim\b|structural similarity/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'ssim' };
    }

    if (/\bmae\b/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'mae' };
    }

    if (/\brmse\b/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'rmse' };
    }

    if (/\bmape\b/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'mape' };
    }

    if (/(smape|s-mape)/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'smape' };
    }

    if (/\bdtw\b|dynamic time warping/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'dtw' };
    }

    if (/edit sim|edit similarity/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'edit_similarity' };
    }

    if (
      /^iter$/.test(lower)
      || /inverse translation edit rate/.test(lower)
      || (/\\mathbb\{h\}/.test(formulaLower) && /s_[xn]/.test(formulaLower))
    ) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'token_overlap_recall' };
    }

    if (
      /^ruse$/.test(lower)
      || /ruse\b/.test(lower)
      || (/\\exp\(-s_i\)/.test(formulaLower) || /1\s*\+\s*\\exp/.test(formulaLower))
    ) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'sigmoid_score_average' };
    }

    if (
      /^yisi(?:-|\s)?1$/.test(lower)
      || /^yisi\b/.test(lower)
      || (/\\max/.test(formulaLower) && /\\mathbf/.test(formulaLower) && /(x_i|hat\{x\}_j)/.test(formulaLower))
    ) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'greedy_token_similarity' };
    }

    if (/relevant noise sensitivity|ns\(i\)|noise sensitivity i/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'relevant_noise_sensitivity' };
    }

    if (/irrelevant noise sensitivity|ns\(ii\)|noise sensitivity ii/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'irrelevant_noise_sensitivity' };
    }

    if (/word error rate|\bwer\b/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'wer' };
    }

    if (/character error rate|\bcer\b/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'cer' };
    }

    if (/reward curve|episode reward|average reward|average return|cumulative reward|return\b/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'average_return' };
    }

    if (/average maximum predicted action[- ]?value|max(?:imum)? predicted q[- ]?value|max q[- ]?value/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'average_max_state_value' };
    }

    if (/success rate|success@|solve rate|solve@/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'success_rate' };
    }

    if (/pass@|pass at k|pass at \d+/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'pass_at_k' };
    }

    if (/average normalized score|human normalized score|normalized score/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'normalized_score' };
    }

    if (/expected entropy/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'expected_entropy' };
    }

    if (/amino acid recovery|\baar\b/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'amino_acid_recovery' };
    }

    if (/rank %|rank percent|rank percentile|percentile rank/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'rank_percent' };
    }

    if (/wini?ng[- ]?counts?/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'winning_count' };
    }

    if (/path length|perceptual path length|\bppl\b/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'path_length' };
    }

    if (/(faithfulness|factuality|utility|correctness|helpfulness|groundedness|relevance|coherence|fluency)/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'judge_score_aggregate' };
    }

    if (/rmsd|root mean square deviation|root-mean-square deviation/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'rmsd' };
    }

    if (/ndcg/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'ndcg' };
    }

    if (/\bmrr\b|mean reciprocal rank/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'mrr' };
    }

    if (/(hit rate|hit ratio|hit@|hr@)/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'hit_rate' };
    }

    if (/(bleu|rouge|bertscore)/.test(lower)) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'library_metric' };
    }

    return {
      metricName,
      outputKey: slug,
      functionName: `compute_${slug}`,
      kind: 'custom_placeholder',
      formula: metric.formula ?? undefined,
      note: `TODO: implement ${metricName} exactly as described in the paper.`,
    };
  });
}

function isRankingMetricKind(kind: MetricPlanKind): boolean {
  return (
    kind === 'ndcg'
    || kind === 'mrr'
    || kind === 'hit_rate'
    || kind === 'precision_at_k'
    || kind === 'recall_at_k'
    || kind === 'pass_at_k'
  );
}

function isDirectPredictionMetricKind(kind: MetricPlanKind): boolean {
  return (
    kind === 'accuracy'
    || kind === 'top_k_accuracy'
    || kind === 'exact_match'
    || kind === 'precision'
    || kind === 'recall'
    || kind === 'f1'
    || kind === 'matthews_corrcoef'
    || kind === 'mse'
    || kind === 'pearson'
    || kind === 'spearman'
    || kind === 'kendall_tau'
    || kind === 'auc'
    || kind === 'average_precision'
    || kind === 'brier_score'
    || kind === 'expected_calibration_error'
    || kind === 'negative_log_likelihood'
    || kind === 'perplexity'
    || kind === 'iou'
    || kind === 'map'
    || kind === 'mask_map'
    || kind === 'fid'
    || kind === 'psnr'
    || kind === 'ssim'
    || kind === 'mae'
    || kind === 'rmse'
    || kind === 'mape'
    || kind === 'smape'
    || kind === 'dtw'
    || kind === 'edit_similarity'
    || kind === 'token_overlap_recall'
    || kind === 'sigmoid_score_average'
    || kind === 'greedy_token_similarity'
    || kind === 'bertscore_precision'
    || kind === 'bertscore_recall'
    || kind === 'bertscore_f1'
    || kind === 'wer'
    || kind === 'cer'
    || kind === 'average_return'
    || kind === 'average_max_state_value'
    || kind === 'success_rate'
    || kind === 'expected_entropy'
    || kind === 'amino_acid_recovery'
    || kind === 'path_length'
    || kind === 'rmsd'
    || kind === 'library_metric'
  );
}

function fallbackMetricNameForTaskMode(taskMode: TaskMode): string {
  switch (taskMode) {
    case 'pairwise_preference':
      return 'win rate';
    case 'ranking':
      return 'ndcg';
    case 'forecasting':
      return 'rmse';
    default:
      return 'custom metric';
  }
}

function entrypointNameForTaskMode(taskMode: TaskMode): string {
  switch (taskMode) {
    case 'pairwise_preference':
      return 'evaluate_pairwise_preferences';
    case 'ranking':
      return 'evaluate_rankings';
    case 'forecasting':
      return 'evaluate_forecasts';
    default:
      return 'evaluate_predictions';
  }
}

function slugifyMetricName(metricName: string): string {
  return metricName
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    || 'metric';
}

function toPythonString(value: string): string {
  return JSON.stringify(value);
}

function toPythonList(values: string[]): string {
  return `[${values.map((value) => JSON.stringify(value)).join(', ')}]`;
}

function toPythonBaselines(baselines: Baseline[]): string {
  const serialized = baselines.map((baseline) => ({
    model_name: baseline.model_name?.trim() || '',
    scores: baseline.scores ?? {},
  }));

  return JSON.stringify(serialized, null, 2)
    .replace(/true/g, 'True')
    .replace(/false/g, 'False')
    .replace(/null/g, 'None');
}

function shouldRepairEvaluationSummary(summary: PaperEvaluationSummary): boolean {
  const hasMetricEvidence = (summary.evaluation_metrics ?? []).some(
    (metric) => metric.name?.trim() && (metric.formula?.trim() || metric.value !== null && metric.value !== undefined),
  );
  const hasBaselineEvidence = (summary.baselines ?? []).some(
    (baseline) => baseline.model_name?.trim()
      && Object.values(baseline.scores ?? {}).some((value) => normalizeMetricValue(value) !== null),
  );
  const filledCount = [
    summary.task_type,
    summary.implementation_details,
  ].filter((value) => value?.trim()).length
    + ((summary.datasets?.length ?? 0) > 0 ? 1 : 0)
    + ((summary.baselines?.length ?? 0) > 0 ? 1 : 0)
    + ((summary.evaluation_metrics?.length ?? 0) > 0 ? 1 : 0);

  return filledCount < 3 || !hasMetricEvidence || ((summary.baselines?.length ?? 0) > 0 && !hasBaselineEvidence);
}

function mergeEvaluationSummary(
  primary: PaperEvaluationSummary,
  fallback: PaperEvaluationSummary,
): PaperEvaluationSummary {
  return {
    task_type: primary.task_type?.trim() || fallback.task_type?.trim() || '',
    evaluation_metrics: [...(primary.evaluation_metrics ?? []), ...(fallback.evaluation_metrics ?? [])],
    datasets: [...(primary.datasets ?? []), ...(fallback.datasets ?? [])],
    baselines: [...(primary.baselines ?? []), ...(fallback.baselines ?? [])],
    implementation_details:
      primary.implementation_details?.trim() || fallback.implementation_details?.trim() || '',
  };
}

function normalizeEvaluationSummary(summary: PaperEvaluationSummary): PaperEvaluationSummary {
  return {
    task_type: normalizeLooseText(summary.task_type, 160),
    evaluation_metrics: normalizeEvaluationMetrics(summary.evaluation_metrics ?? []),
    datasets: normalizeStringList(summary.datasets ?? [], 12, 96),
    baselines: normalizeBaselines(summary.baselines ?? []),
    implementation_details: normalizeLooseText(summary.implementation_details, 420),
  };
}

function enrichEvaluationSummary(
  summary: PaperEvaluationSummary,
  experimentMarkdown: string,
): PaperEvaluationSummary {
  const normalized = normalizeEvaluationSummary(summary);
  const refinedMetrics = refineEvaluationMetrics(normalized, experimentMarkdown);
  return {
    ...normalized,
    evaluation_metrics: refinedMetrics,
    baselines: backfillBaselineEvidence(
      normalized.baselines ?? [],
      refinedMetrics,
      experimentMarkdown,
    ),
  };
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

function normalizeMetricValue(value: number | string | null | undefined): number | string | null {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null;
  }

  if (typeof value !== 'string') {
    return null;
  }

  const trimmed = value.trim();
  if (!trimmed || /^(n\/a|none|null|not specified|unknown|-|needs review)$/i.test(trimmed)) {
    return null;
  }

  if (/^-?\d+(\.\d+)?$/.test(trimmed)) {
    return Number(trimmed);
  }

  return trimmed.length > 80 ? `${trimmed.slice(0, 77).trimEnd()}...` : trimmed;
}

function normalizeEvaluationMetrics(metrics: EvaluationMetric[]): EvaluationMetric[] {
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
      value: normalizeMetricValue(metric.value),
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

    const scores: Record<string, number | string | null> = {};
    for (const [scoreName, rawValue] of Object.entries(baseline.scores ?? {})) {
      const normalizedName = normalizeLooseText(scoreName, 72);
      const normalizedValue = normalizeMetricValue(rawValue);
      if (!normalizedName || normalizedValue === null) {
        continue;
      }

      scores[normalizedName] = normalizedValue;
    }

    seen.add(key);
    normalized.push({
      model_name: modelName,
      scores,
    });

    if (normalized.length >= 10) {
      break;
    }
  }

  const baselinesWithScores = normalized.filter((baseline) => Object.keys(baseline.scores ?? {}).length > 0);
  return baselinesWithScores.length > 0 ? baselinesWithScores : [];
}

function refineEvaluationMetrics(
  summary: PaperEvaluationSummary,
  experimentMarkdown: string,
): EvaluationMetric[] {
  const normalizedMetrics = (summary.evaluation_metrics ?? [])
    .map((metric) => normalizeMetricForContext(metric, summary, experimentMarkdown))
    .filter((metric): metric is EvaluationMetric => Boolean(metric));
  const kept = normalizedMetrics.filter((metric) =>
    !isNonExecutableClaimMetric(metric.name ?? '')
    && !isDescriptiveDatasetStatisticMetric(metric, summary, experimentMarkdown),
  );
  if (kept.length > 0) {
    return kept;
  }

  return inferFallbackMetrics(summary, experimentMarkdown);
}

function normalizeMetricForContext(
  metric: EvaluationMetric,
  summary: PaperEvaluationSummary,
  experimentMarkdown: string,
): EvaluationMetric | null {
  const name = normalizeLooseText(metric.name, 96);
  if (!name) {
    return null;
  }

  const formula = normalizeLooseText(metric.formula, 220);
  const corpus = `${summary.task_type ?? ''}\n${experimentMarkdown}`.toLowerCase();
  const lower = name.toLowerCase();
  const inferredDomain = inferDomainFromRegistry(summary);
  const looksForecasting =
    inferredDomain === 'time-series'
    || /(forecast|forecasting|time series|ett|electricity|traffic|weather|exchange|ili|long-term series)/.test(corpus);

  if ((/^m\s*ap$/i.test(name) || lower === 'map') && (looksForecasting || /\bmape\b|mean absolute percentage error/.test(corpus))) {
    return {
      ...metric,
      name: 'MAPE',
      formula: formula || 'Mean absolute percentage error',
    };
  }

  if (/^test set accuracy$/.test(lower)) {
    return {
      ...metric,
      name: 'accuracy',
      formula: formula || 'Test set accuracy',
    };
  }

  if (/^training set accuracy$/.test(lower)) {
    return {
      ...metric,
      name: 'training set accuracy',
      formula: formula || 'Training set accuracy',
    };
  }

  return {
    ...metric,
    name,
    formula,
  };
}

function isNonExecutableClaimMetric(metricName: string): boolean {
  const lower = metricName.trim().toLowerCase();
  if (!lower) {
    return false;
  }

  return /(training fit|computational complexity|time complexity|complexity|prediction error reduction|theoretical convergence|expressive power|source size|magnification|relative time delay|stellar mass density|maximum magnification)/.test(lower);
}

function isDescriptiveDatasetStatisticMetric(
  metric: EvaluationMetric,
  summary: PaperEvaluationSummary,
  experimentMarkdown: string,
): boolean {
  const name = (metric.name ?? '').trim().toLowerCase();
  const formula = (metric.formula ?? '').trim().toLowerCase();
  const combined = `${name} ${formula}`;
  if (!combined) {
    return false;
  }

  if (
    /(average precision|mean average precision|\bmap\b|\bmape\b|average return|average reward|average bio|average batch|expected entropy|field-level f1|tree edit distance|ted based accuracy)/.test(combined)
  ) {
    return false;
  }

  const looksDescriptiveStatistic =
    /(average .*length|average .*tokens?|average .*words?|average .*characters?|average .*images?|average .*questions?|average .*answers?|unique .*percentage|unique .*ratio|unique .*count|.*per image|.*per page|.*per document|.*per sample|number of .*|count of .*)/.test(combined);
  if (!looksDescriptiveStatistic) {
    return false;
  }

  const referencesDatasetArtifacts = /(question|answer|ocr|token|word|character|image|page|document|dataset|sample|annotation|worker)/.test(combined);
  const referencesPredictionProtocol = /(prediction|predicted|reference|ground truth|target answer|correct answers?|precision|recall|f1|levenshtein|edit distance|accuracy|anls|iou|bleu|rouge|bertscore|score on the test set|evaluation metric)/.test(combined);
  if (!referencesDatasetArtifacts || referencesPredictionProtocol) {
    return false;
  }

  const context = `${summary.task_type ?? ''}\n${experimentMarkdown}`.toLowerCase();
  return /(statistics|dataset statistics|data statistics|question type|analysis|annotation process|dataset creation|corpus)/.test(context);
}

function inferFallbackMetrics(
  summary: PaperEvaluationSummary,
  experimentMarkdown: string,
): EvaluationMetric[] {
  const corpus = `${summary.task_type ?? ''}\n${experimentMarkdown}`.toLowerCase();
  const inferredDomain = inferDomainFromRegistry(summary);
  const looksForecasting =
    inferredDomain === 'time-series'
    || /(forecast|forecasting|time series|ett|electricity|traffic|weather|exchange|ili|long-term series)/.test(corpus);
  const inferred: EvaluationMetric[] = [];

  const maybeAdd = (name: string, formula = '') => {
    if (inferred.some((metric) => metric.name?.toLowerCase() === name.toLowerCase())) {
      return;
    }
    inferred.push({ name, formula, value: null });
  };

  if (/\bmae\b|mean absolute error/.test(corpus)) {
    maybeAdd('MAE', 'Mean absolute error');
  }
  if (/\bmse\b|mean squared error/.test(corpus)) {
    maybeAdd('MSE', 'Mean squared error');
  }
  if (/\brmse\b|root mean squared error/.test(corpus)) {
    maybeAdd('RMSE', 'Root mean squared error');
  }
  if (/\bsmape\b/.test(corpus)) {
    maybeAdd('SMAPE', 'Symmetric mean absolute percentage error');
  }
  if (/\bmape\b/.test(corpus) || (looksForecasting && /\bmap\b/.test(corpus))) {
    maybeAdd('MAPE', 'Mean absolute percentage error');
  }
  if (/\baccuracy\b/.test(corpus)) {
    maybeAdd('accuracy', 'Accuracy');
  }
  if (/\bmicro[- ]?f1\b/.test(corpus)) {
    maybeAdd('micro-F1', 'Micro-averaged F1');
  }
  if (/\bmacro[- ]?f1\b/.test(corpus)) {
    maybeAdd('macro-F1', 'Macro-averaged F1');
  }
  if (/\bf1\b/.test(corpus) && inferred.every((metric) => !/f1/i.test(metric.name ?? ''))) {
    maybeAdd('F1', 'F1 score');
  }
  if (/\bbleu\b/.test(corpus)) {
    maybeAdd('BLEU', 'BLEU score');
  }
  if (/\brouge\b/.test(corpus)) {
    maybeAdd('ROUGE', 'ROUGE score');
  }
  if (/\bbertscore\b/.test(corpus)) {
    maybeAdd('BERTScore', 'BERTScore');
  }
  if (/\bndcg\b/.test(corpus)) {
    maybeAdd('NDCG', 'Normalized discounted cumulative gain');
  }
  if (/\bmrr\b|mean reciprocal rank/.test(corpus)) {
    maybeAdd('MRR', 'Mean reciprocal rank');
  }
  if (/\bhit@|\bhit rate\b/.test(corpus)) {
    maybeAdd('Hit Rate', 'Hit rate');
  }
  if (!looksForecasting && (/\bmap\b|mean average precision/.test(corpus))) {
    maybeAdd('mAP', 'Mean average precision');
  }
  if (/\biou\b|intersection over union/.test(corpus)) {
    maybeAdd('IoU', 'Intersection over union');
  }
  if (/\brmsd\b/.test(corpus)) {
    maybeAdd('RMSD', 'Root mean square deviation');
  }
  if (/\b(return|episode return|average return|reward curve|cumulative reward)\b/.test(corpus)) {
    maybeAdd('average return', 'Average episodic return');
  }
  if (/\bsuccess rate\b/.test(corpus)) {
    maybeAdd('success rate', 'Success rate');
  }

  return inferred.slice(0, 8);
}

function backfillBaselineEvidence(
  baselines: Baseline[],
  metrics: EvaluationMetric[],
  experimentMarkdown: string,
): Baseline[] {
  const blocks = splitSupportBlocks(experimentMarkdown);
  const metricNames = metrics
    .map((metric) => normalizeLooseText(metric.name, 96))
    .filter(Boolean);

  return baselines.map((baseline) => {
    if (Object.keys(baseline.scores ?? {}).length > 0) {
      return baseline;
    }

    const supportBlock = findBaselineSupportBlock(
      blocks,
      baseline.model_name?.trim() || '',
      metricNames,
    );
    if (!supportBlock) {
      return baseline;
    }

    const inferredScores = inferScoresFromSupportBlock(supportBlock, metricNames);
    if (Object.keys(inferredScores).length > 0) {
      return {
        ...baseline,
        scores: inferredScores,
      };
    }

    return baseline;
  });
}

function splitSupportBlocks(markdown: string): string[] {
  const normalized = markdown
    .replace(/\r\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n');

  const paragraphBlocks = normalized
    .split(/\n\s*\n/)
    .map((block) => block.replace(/\s+/g, ' ').trim())
    .filter(Boolean);

  if (paragraphBlocks.length > 0) {
    return paragraphBlocks;
  }

  return normalized
    .split('\n')
    .map((line) => line.replace(/\s+/g, ' ').trim())
    .filter(Boolean);
}

function findBaselineSupportBlock(
  blocks: string[],
  baselineName: string,
  metricNames: string[],
): string {
  const normalizedBaseline = normalizeSupportText(baselineName);
  if (!normalizedBaseline) {
    return '';
  }

  const baselineTokens = normalizedBaseline.split(' ').filter((token) => token.length >= 3);
  let bestBlock = '';
  let bestScore = -1;

  for (const block of blocks) {
    if (/!\[image\]/i.test(block)) {
      continue;
    }
    if (/^figure\s+\d+/i.test(block) || /^fig\./i.test(block)) {
      continue;
    }

    const normalizedBlock = normalizeSupportText(block);
    if (!normalizedBlock) {
      continue;
    }

    const exactBaselineHit = normalizedBlock.includes(normalizedBaseline);
    const matchedTokens = baselineTokens.filter((token) => normalizedBlock.includes(token)).length;
    if (!exactBaselineHit && matchedTokens === 0) {
      continue;
    }

    const numericMatches = block.match(/-?\d+(?:\.\d+)?/g)?.length ?? 0;
    const metricHits = metricNames.filter((metricName) => normalizedBlock.includes(normalizeSupportText(metricName))).length;
    const tableLike = block.includes('|') || /table\s+\d+/i.test(block);
    if (metricHits === 0 && numericMatches < 2 && !tableLike) {
      continue;
    }
    const score = (exactBaselineHit ? 6 : 0) + matchedTokens * 2 + metricHits * 2 + Math.min(numericMatches, 6);
    if (score > bestScore) {
      bestScore = score;
      bestBlock = block;
    }
  }

  return bestBlock;
}

function inferScoresFromSupportBlock(
  block: string,
  metricNames: string[],
): Record<string, number | string> {
  const numbers = (block.match(/-?\d+(?:\.\d+)?/g) ?? [])
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value));

  if (metricNames.length === 1 && numbers.length >= 1) {
    return { [metricNames[0]]: numbers[0] };
  }

  if (metricNames.length >= 2 && metricNames.length <= 4 && numbers.length === metricNames.length) {
    return Object.fromEntries(metricNames.map((metricName, index) => [metricName, numbers[index]]));
  }

  return {};
}

function normalizeSupportText(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9가-힣\s-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

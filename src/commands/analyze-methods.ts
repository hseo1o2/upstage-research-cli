import { promises as fs } from 'node:fs';
import path from 'node:path';

import { UpstageClient, type JsonSchema } from '../client';
import { resolveApiKey } from '../config';
import {
  buildAnchoredEvidenceRecords,
  buildEvidenceRecords,
  buildFocusedEvidenceRecords,
  mergeEvidenceRecordGroups,
  type EvidenceRecord,
} from '../evidence';
import { normalizeOutputFormat, resolvePrimaryOutputPath, writeOutputFile } from '../output';
import { mapWithConcurrency } from '../parallel';

export interface AnalyzeMethodsOptions {
  context?: string;
  format?: string;
  out?: string;
  saveReport?: string;
  cacheOnly?: boolean;
}

interface PaperMethodSummary {
  title?: string;
  model_architecture?: string;
  training_strategy?: string;
  datasets?: string[];
  main_contribution?: string;
  limitations?: string;
  evaluation_metrics?: string[];
}

interface ProcessedPaper {
  paperId: string;
  filePath: string;
  fileName: string;
  parseMode: 'sync' | 'async';
  summary: PaperMethodSummary;
  evidenceRecords: EvidenceRecord[];
}

interface MethodApplicationPoint {
  title: string;
  rationale: string;
  actions: string[];
  paper_ids: string[];
}

interface MethodPaperConnection {
  paper_id: string;
  relevance: string;
  adaptation: string;
  caution: string;
}

interface MethodReferencePriority {
  paper_id: string;
  rank: number;
  rationale: string;
}

interface MethodSynthesis {
  overall_takeaway: string;
  application_points: MethodApplicationPoint[];
  paper_connections: MethodPaperConnection[];
  reference_priority: MethodReferencePriority[];
}

const DEFAULT_SOLAR_MODEL = process.env.UPSTAGE_SOLAR_MODEL?.trim() || 'solar-pro3';

export async function analyzeMethodsCommand(
  papers: string[],
  options: AnalyzeMethodsOptions,
): Promise<void> {
  if (papers.length === 0) {
    throw new Error('Provide at least one paper file.');
  }

  const apiKey = await resolveApiKey();
  if (!apiKey) {
    throw new Error(
      'UPSTAGE_API_KEY is not set. Export it or run `upstage-research install --skills` to store it in config.',
    );
  }

  const language = pickLanguage([options.context ?? '', ...papers]);
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
      'paper-method-analyzer',
      'references',
      'method-schema.json',
    ),
  );

  const processed = await mapWithConcurrency(papers, resolveConcurrency(papers.length), async (paperPath, index) => {
    const inputPath = path.resolve(paperPath);
    const fileName = path.basename(inputPath);
    const paperId = `paper_${index + 1}`;

    console.error(`[${index + 1}/${papers.length}] Parsing ${fileName}`);
    const parsed = await client.parseDocument(inputPath);
    const methodExcerpt = selectMethodExcerpt(parsed.markdown);

    console.error(`[${index + 1}/${papers.length}] Extracting methodology fields from ${fileName}`);
    const extractedSummary = await client.informationExtract<PaperMethodSummary>({
      schemaName: 'paper_method_summary',
      schema,
      filePath: inputPath,
      instruction:
        'Extract methodology metadata from this academic paper. Focus on title, model architecture, training strategy, datasets, main contribution, limitations, and evaluation metrics. Use concise factual text. If a field is missing, return an empty string or an empty array.',
      contextText:
        `File name: ${fileName}\n\n` +
        `Prior document-parse excerpt:\n${clipText(methodExcerpt, 30_000)}`,
    });
    const summary = await maybeRepairMethodSummary(
      client,
      extractedSummary,
      schema,
      methodExcerpt,
      fileName,
      language,
    );

    return {
      paperId,
      filePath: inputPath,
      fileName,
      parseMode: parsed.mode,
      summary,
      evidenceRecords: buildMethodEvidenceRecords(summary, methodExcerpt, parsed.anchors),
    };
  });

  console.error('Generating structured cross-paper synthesis');
  const synthesis = await synthesizeMethodInsights(client, processed, options.context, language);
  const markdown = renderAnalysisMarkdown(processed, synthesis, options.context, language);
  const outputFormat = normalizeOutputFormat(options.format);
  const primaryOutput = outputFormat === 'json'
    ? `${JSON.stringify({
      context: options.context?.trim() || '',
      papers: processed,
      synthesis,
      markdown,
    }, null, 2)}\n`
    : `${markdown}\n`;
  const reportPath = resolvePrimaryOutputPath(options.out, options.saveReport);
  if (reportPath) {
    await writeOutputFile(reportPath, primaryOutput);
    console.error(`Wrote report to ${reportPath}`);
  }

  process.stdout.write(primaryOutput);
}

async function loadSchema(schemaPath: string): Promise<JsonSchema> {
  const raw = await fs.readFile(schemaPath, 'utf8');
  return JSON.parse(raw) as JsonSchema;
}

function renderAnalysisMarkdown(
  papers: ProcessedPaper[],
  synthesis: MethodSynthesis,
  context: string | undefined,
  language: 'ko' | 'en',
): string {
  const sections: string[] = [];
  const connectionById = new Map(
    synthesis.paper_connections.map((connection) => [connection.paper_id, connection]),
  );

  if (context?.trim()) {
    sections.push(language === 'ko' ? `> 연구 컨텍스트: ${context.trim()}` : `> Research context: ${context.trim()}`);
    sections.push('');
  }

  sections.push(language === 'ko' ? '## 방법론 비교표' : '## Methodology Comparison');
  sections.push(buildComparisonTable(papers, language));
  sections.push('');
  sections.push(language === 'ko' ? '## 각 논문 상세 분석' : '## Per-Paper Analysis');
  sections.push('');

  for (const paper of papers) {
    const title = paper.summary.title?.trim() || paper.fileName;
    const connection = connectionById.get(paper.paperId);
    sections.push(`### ${title}`);
    sections.push(
      language === 'ko'
        ? `- 핵심 아이디어: ${fallbackText(paper.summary.main_contribution, '추출 실패')}`
        : `- Core idea: ${fallbackText(paper.summary.main_contribution, 'Not extracted')}`,
    );
    sections.push(
      language === 'ko'
        ? `- 모델 구조: ${fallbackText(paper.summary.model_architecture, '추출 실패')}`
        : `- Model architecture: ${fallbackText(paper.summary.model_architecture, 'Not extracted')}`,
    );
    sections.push(
      language === 'ko'
        ? `- 학습 전략: ${fallbackText(paper.summary.training_strategy, '추출 실패')}`
        : `- Training strategy: ${fallbackText(paper.summary.training_strategy, 'Not extracted')}`,
    );
    sections.push(
      language === 'ko'
        ? `- 데이터셋: ${formatList(paper.summary.datasets, '확인 필요')}`
        : `- Datasets: ${formatList(paper.summary.datasets, 'Needs review')}`,
    );
    sections.push(
      language === 'ko'
        ? `- 평가 지표: ${formatList(paper.summary.evaluation_metrics, '확인 필요')}`
        : `- Evaluation metrics: ${formatList(paper.summary.evaluation_metrics, 'Needs review')}`,
    );
    sections.push(
      language === 'ko'
        ? `- 한계: ${fallbackText(paper.summary.limitations, '명시되지 않음')}`
        : `- Limitations: ${fallbackText(paper.summary.limitations, 'Not stated')}`,
    );
    if (connection) {
      sections.push(
        language === 'ko'
          ? `- 내 연구와의 관련성: ${fallbackText(connection.relevance, '추가 맥락 필요')}`
          : `- Relevance to your research: ${fallbackText(connection.relevance, 'Needs more context')}`,
      );
      sections.push(
        language === 'ko'
          ? `- 적용 아이디어: ${fallbackText(connection.adaptation, '추가 설계 필요')}`
          : `- Adaptation idea: ${fallbackText(connection.adaptation, 'Needs further design')}`,
      );
      sections.push(
        language === 'ko'
          ? `- 주의점: ${fallbackText(connection.caution, '명시적 주의점 없음')}`
          : `- Caution: ${fallbackText(connection.caution, 'No explicit caution provided')}`,
      );
    }
    if (paper.evidenceRecords.length > 0) {
      sections.push(
        language === 'ko'
          ? `- 근거 발췌: ${paper.evidenceRecords.map((record) => formatEvidenceRecordInline(record, language)).join(' / ')}`
          : `- Evidence snippets: ${paper.evidenceRecords.map((record) => formatEvidenceRecordInline(record, language)).join(' / ')}`,
      );
    }
    sections.push(
      language === 'ko'
        ? `- 파싱 모드: \`${paper.parseMode}\``
        : `- Parse mode: \`${paper.parseMode}\``,
    );
    sections.push('');
  }

  if (synthesis.overall_takeaway.trim()) {
    sections.push(language === 'ko' ? '## 전체 요약' : '## Overall Takeaway');
    sections.push('');
    sections.push(synthesis.overall_takeaway.trim());
    sections.push('');
  }

  sections.push(language === 'ko' ? '## 내 연구에 적용 가능한 포인트' : '## Application Suggestions');
  sections.push('');
  const applicationPoints = synthesis.application_points.length > 0
    ? synthesis.application_points
    : buildFallbackApplicationPoints(papers, language);
  applicationPoints.forEach((point, index) => {
    const relatedTitles = point.paper_ids
      .map((paperId) => papers.find((paper) => paper.paperId === paperId)?.summary.title?.trim()
        || papers.find((paper) => paper.paperId === paperId)?.fileName
        || '')
      .filter(Boolean);
    const evidenceRefs = collectEvidenceRefs(papers, point.paper_ids);
    if (language === 'ko') {
      sections.push(`${index + 1}. ${point.title}: ${point.rationale}`);
      if (point.actions.length > 0) {
        sections.push(`실행 포인트: ${point.actions.join(' / ')}`);
      }
      if (relatedTitles.length > 0) {
        sections.push(`관련 논문: ${relatedTitles.join(', ')}`);
      }
      if (evidenceRefs.length > 0) {
        sections.push(`근거 참조: ${evidenceRefs.join(', ')}`);
      }
    } else {
      sections.push(`${index + 1}. ${point.title}: ${point.rationale}`);
      if (point.actions.length > 0) {
        sections.push(`Execution points: ${point.actions.join(' / ')}`);
      }
      if (relatedTitles.length > 0) {
        sections.push(`Related papers: ${relatedTitles.join(', ')}`);
      }
      if (evidenceRefs.length > 0) {
        sections.push(`Evidence refs: ${evidenceRefs.join(', ')}`);
      }
    }
  });
  sections.push('');

  sections.push(language === 'ko' ? '## 추천 레퍼런스 우선순위' : '## Reference Priority');
  sections.push('');
  const referencePriority = synthesis.reference_priority.length > 0
    ? synthesis.reference_priority
    : buildFallbackReferencePriority(papers);
  referencePriority.forEach((item, index) => {
    const paper = papers.find((candidate) => candidate.paperId === item.paper_id);
    const title = paper?.summary.title?.trim() || paper?.fileName || item.paper_id;
    sections.push(`${index + 1}. ${title}: ${item.rationale}`);
  });

  return sections.join('\n').trim();
}

async function synthesizeMethodInsights(
  client: UpstageClient,
  papers: ProcessedPaper[],
  context: string | undefined,
  language: 'ko' | 'en',
): Promise<MethodSynthesis> {
  const schema = buildMethodSynthesisSchema(papers.map((paper) => paper.paperId));
  const payload = papers.map((paper) => ({
    paper_id: paper.paperId,
    title: paper.summary.title?.trim() || paper.fileName,
    model_architecture: paper.summary.model_architecture ?? '',
    training_strategy: paper.summary.training_strategy ?? '',
    datasets: paper.summary.datasets ?? [],
    main_contribution: paper.summary.main_contribution ?? '',
    limitations: paper.summary.limitations ?? '',
    evaluation_metrics: paper.summary.evaluation_metrics ?? [],
    evidence_items: paper.evidenceRecords.map((record) => ({
      id: record.id,
      text: record.text,
    })),
  }));

  try {
    const synthesis = await client.chatStructured<MethodSynthesis>({
      model: DEFAULT_SOLAR_MODEL,
      schemaName: 'paper_method_synthesis',
      schema,
      messages: [
        {
          role: 'system',
          content:
            language === 'ko'
              ? '당신은 여러 논문의 방법론을 비교하는 연구 조교다. 제공된 paper_id만 사용하고, 입력에 없는 논문이나 데이터셋을 추천 우선순위 항목으로 추가하지 마라. 적용 아이디어는 evidence_items의 id/text, 데이터셋, 평가 지표, 학습 전략에 직접 근거해야 하며, 맥락 없이 과도한 교차 도메인 전이를 제안하지 마라.'
              : 'You are a research assistant comparing paper methodologies. Only use the provided paper_id values and never introduce papers or datasets that were not in the input. Keep adaptation ideas grounded in the evidence item ids/text, datasets, metrics, and training strategy. Avoid speculative cross-domain transfer unless the context clearly supports it.',
        },
        {
          role: 'user',
          content:
            `${language === 'ko' ? '연구 컨텍스트' : 'Research context'}:\n` +
            `${context?.trim() || (language === 'ko'
              ? '제공되지 않음. 일반적인 적용 포인트를 제안하되, 가정은 짧게 밝혀라.'
              : 'Not provided. Suggest general application points and state assumptions briefly.')}\n\n` +
            `${language === 'ko' ? '논문 요약 JSON' : 'Paper summary JSON'}:\n${JSON.stringify(payload, null, 2)}\n\n` +
            `${language === 'ko'
              ? '각 paper_id를 기준으로 적용 포인트, 논문별 관련성, 우선순위를 구조화해서 반환하라. 적용 포인트의 rationale과 actions에는 입력 JSON에 없는 새 모델/데이터셋/벤치마크를 도입하지 마라.'
              : 'Return structured application points, per-paper relevance, and reference priority keyed by paper_id. Do not introduce new models, datasets, or benchmarks that are not supported by the input JSON.'}`,
        },
      ],
    });

    return normalizeMethodSynthesis(papers, synthesis, context, language);
  } catch {
    return buildFallbackMethodSynthesis(papers, context, language);
  }
}

async function maybeRepairMethodSummary(
  client: UpstageClient,
  summary: PaperMethodSummary,
  schema: JsonSchema,
  methodExcerpt: string,
  fileName: string,
  language: 'ko' | 'en',
): Promise<PaperMethodSummary> {
  if (!shouldRepairMethodSummary(summary)) {
    return normalizeMethodSummary(summary);
  }

  const repaired = await client.chatStructured<PaperMethodSummary>({
    model: DEFAULT_SOLAR_MODEL,
    schemaName: 'paper_method_summary_repair',
    schema,
    temperature: 0.1,
    messages: [
      {
        role: 'system',
        content:
          language === 'ko'
            ? '당신은 논문 방법론 메타데이터를 복원하는 연구 조교다. 제공된 문서 발췌에서 확인 가능한 정보만 추출하고, 없으면 빈 문자열이나 빈 배열을 유지하라.'
            : 'You repair methodology metadata extraction from a parsed paper excerpt. Only use information supported by the excerpt. Leave fields empty when unsupported.',
      },
      {
        role: 'user',
        content:
          `File name: ${fileName}\n\n` +
          `Current extracted summary:\n${JSON.stringify(summary, null, 2)}\n\n` +
          `Method excerpt:\n${clipText(methodExcerpt, 30_000)}`,
      },
    ],
  });

  return normalizeMethodSummary(mergeMethodSummary(summary, repaired));
}

function buildMethodSynthesisSchema(paperIds: string[]): JsonSchema {
  return {
    type: 'object',
    additionalProperties: false,
    required: ['overall_takeaway', 'application_points', 'paper_connections', 'reference_priority'],
    properties: {
      overall_takeaway: {
        type: 'string',
      },
      application_points: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          required: ['title', 'rationale', 'actions', 'paper_ids'],
          properties: {
            title: { type: 'string' },
            rationale: { type: 'string' },
            actions: {
              type: 'array',
              items: { type: 'string' },
            },
            paper_ids: {
              type: 'array',
              items: {
                type: 'string',
                enum: paperIds,
              },
            },
          },
        },
      },
      paper_connections: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          required: ['paper_id', 'relevance', 'adaptation', 'caution'],
          properties: {
            paper_id: {
              type: 'string',
              enum: paperIds,
            },
            relevance: { type: 'string' },
            adaptation: { type: 'string' },
            caution: { type: 'string' },
          },
        },
      },
      reference_priority: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          required: ['paper_id', 'rank', 'rationale'],
          properties: {
            paper_id: {
              type: 'string',
              enum: paperIds,
            },
            rank: {
              type: 'integer',
              minimum: 1,
            },
            rationale: { type: 'string' },
          },
        },
      },
    },
  };
}

function normalizeMethodSynthesis(
  papers: ProcessedPaper[],
  synthesis: MethodSynthesis,
  context: string | undefined,
  language: 'ko' | 'en',
): MethodSynthesis {
  const allowedIds = new Set(papers.map((paper) => paper.paperId));
  const uniquePaperConnections = new Map<string, MethodPaperConnection>();
  for (const connection of synthesis.paper_connections ?? []) {
    if (!allowedIds.has(connection.paper_id) || uniquePaperConnections.has(connection.paper_id)) {
      continue;
    }

    uniquePaperConnections.set(connection.paper_id, {
      paper_id: connection.paper_id,
      relevance: connection.relevance?.trim() || (language === 'ko' ? '관련성 해설이 없습니다.' : 'No relevance note provided.'),
      adaptation: connection.adaptation?.trim() || (language === 'ko' ? '적용 아이디어가 없습니다.' : 'No adaptation idea provided.'),
      caution: connection.caution?.trim() || (language === 'ko' ? '주의점이 없습니다.' : 'No caution provided.'),
    });
  }

  for (const paper of papers) {
    if (!uniquePaperConnections.has(paper.paperId)) {
      uniquePaperConnections.set(paper.paperId, {
        paper_id: paper.paperId,
        relevance: language === 'ko' ? '입력 요약 기준으로 추가 검토가 필요합니다.' : 'Needs additional review from the extracted summary.',
        adaptation: language === 'ko' ? '핵심 기여를 직접 실험 설계에 매핑해 보세요.' : 'Map the main contribution directly into your experiment design.',
        caution: fallbackText(
          paper.summary.limitations,
          language === 'ko' ? '제한 사항을 먼저 검증하세요.' : 'Validate the limitations before reusing the method.',
        ),
      });
    }
  }

  const seenReferenceIds = new Set<string>();
  const normalizedReferencePriority = (synthesis.reference_priority ?? [])
    .filter((item) => allowedIds.has(item.paper_id))
    .sort((left, right) => left.rank - right.rank)
    .filter((item) => {
      if (seenReferenceIds.has(item.paper_id)) {
        return false;
      }

      seenReferenceIds.add(item.paper_id);
      return true;
    });

  for (const paper of buildFallbackReferencePriority(papers)) {
    if (!seenReferenceIds.has(paper.paper_id)) {
      normalizedReferencePriority.push(paper);
      seenReferenceIds.add(paper.paper_id);
    }
  }

  const normalizedApplicationPoints = (synthesis.application_points ?? [])
    .filter((point) => point.title?.trim() && point.rationale?.trim())
    .map((point) => ({
      title: point.title.trim(),
      rationale: point.rationale.trim(),
      actions: (point.actions ?? []).map((action) => action.trim()).filter(Boolean).slice(0, 3),
      paper_ids: (point.paper_ids ?? []).filter((paperId) => allowedIds.has(paperId)),
    }))
    .slice(0, 6);

  return stabilizeMethodSynthesis(papers, {
    overall_takeaway: synthesis.overall_takeaway?.trim() || '',
    application_points: normalizedApplicationPoints,
    paper_connections: Array.from(uniquePaperConnections.values()),
    reference_priority: normalizedReferencePriority,
  }, context, language);
}

function buildMethodEvidenceRecords(
  summary: PaperMethodSummary,
  methodExcerpt: string,
  anchors: import('../client').ParseAnchor[] = [],
): EvidenceRecord[] {
  return mergeEvidenceRecordGroups(
    [
      buildEvidenceRecords(
        methodExcerpt,
        [
          summary.title ?? '',
          summary.model_architecture ?? '',
          summary.training_strategy ?? '',
        ],
        {
          maxSnippets: 2,
          maxCharsPerSnippet: 190,
          minScore: 2,
          anchors,
        },
      ),
      buildFocusedEvidenceRecords(
        methodExcerpt,
        [
          ...(summary.datasets ?? []),
          ...(summary.evaluation_metrics ?? []),
        ],
        {
          maxSnippets: 2,
          maxPerQuery: 1,
          maxCharsPerSnippet: 200,
          minScore: 1,
          anchors,
        },
      ),
      buildEvidenceRecords(
        methodExcerpt,
        [
          summary.main_contribution ?? '',
          summary.limitations ?? '',
        ],
        {
          maxSnippets: 2,
          maxCharsPerSnippet: 190,
          minScore: 2,
          anchors,
        },
      ),
    ],
    {
      maxSnippets: 4,
      anchors,
    },
  );
}

function buildFallbackMethodSynthesis(
  papers: ProcessedPaper[],
  context: string | undefined,
  language: 'ko' | 'en',
): MethodSynthesis {
  const prominentDatasets = summarizeFrequentTerms(
    papers.flatMap((paper) => paper.summary.datasets ?? []),
    3,
  );
  const prominentMetrics = summarizeFrequentTerms(
    papers.flatMap((paper) => paper.summary.evaluation_metrics ?? []),
    3,
  );

  const overallTakeaway = language === 'ko'
    ? [
        context?.trim() ? `입력된 연구 컨텍스트(${context.trim()})를 기준으로` : '입력 논문들만 기준으로',
        `공통적으로 ${formatList(prominentDatasets, '주요 데이터셋')}와 ${formatList(prominentMetrics, '핵심 평가 지표')} 주변에서 비교 가치가 높습니다.`,
      ].join(' ')
    : [
        context?.trim() ? `Using the provided research context (${context.trim()}),` : 'Using only the supplied papers,',
        `the strongest comparison points cluster around ${formatList(prominentDatasets, 'the reported datasets')} and ${formatList(prominentMetrics, 'the core evaluation metrics')}.`,
      ].join(' ');

  return {
    overall_takeaway: overallTakeaway,
    application_points: buildFallbackApplicationPoints(papers, language),
    paper_connections: papers.map((paper) => ({
      paper_id: paper.paperId,
      relevance: language === 'ko'
        ? `${fallbackText(paper.summary.main_contribution, '핵심 기여')}가 현재 비교군에서 직접적인 차별점입니다.`
        : `${fallbackText(paper.summary.main_contribution, 'The main contribution')} is the main differentiator in this comparison set.`,
      adaptation: language === 'ko'
        ? `${formatList(paper.summary.datasets, '보고된 데이터셋')}와 ${formatList(paper.summary.evaluation_metrics, '보고된 지표')} 기준으로 재현 범위를 정리하세요.`
        : `Scope adaptation around ${formatList(paper.summary.datasets, 'the reported datasets')} and ${formatList(paper.summary.evaluation_metrics, 'the reported metrics')}.`,
      caution: fallbackText(
        paper.summary.limitations,
        language === 'ko' ? '제한 사항은 원문에서 다시 검토하세요.' : 'Review the limitations directly in the paper.',
      ),
    })),
    reference_priority: buildFallbackReferencePriority(papers),
  };
}

function stabilizeMethodSynthesis(
  papers: ProcessedPaper[],
  synthesis: MethodSynthesis,
  context: string | undefined,
  language: 'ko' | 'en',
): MethodSynthesis {
  const broadContext = isBroadResearchContext(context);
  const groundedConnections = new Map(
    papers.map((paper) => [paper.paperId, buildGroundedPaperConnection(paper, language)]),
  );

  const paperConnections = synthesis.paper_connections.map((connection) => {
    const grounded = groundedConnections.get(connection.paper_id);
    if (!grounded) {
      return connection;
    }

    return {
      ...connection,
      relevance:
        broadContext || shouldGroundAdaptationText(connection.relevance)
          ? grounded.relevance
          : connection.relevance,
      adaptation:
        broadContext || shouldGroundAdaptationText(connection.adaptation)
          ? grounded.adaptation
          : connection.adaptation,
      caution: connection.caution?.trim() || grounded.caution,
    };
  });

  const applicationPoints = broadContext
    ? buildGroundedApplicationPoints(papers, synthesis.reference_priority, language)
    : synthesis.application_points;
  const referencePriority = synthesis.reference_priority.map((item) => {
    const paper = papers.find((candidate) => candidate.paperId === item.paper_id);
    if (!paper) {
      return item;
    }

    return {
      ...item,
      rationale:
        broadContext || shouldGroundAdaptationText(item.rationale)
          ? buildGroundedReferenceRationale(paper, language)
          : item.rationale,
    };
  });

  return {
    ...synthesis,
    application_points: applicationPoints.length > 0
      ? applicationPoints
      : buildGroundedApplicationPoints(papers, synthesis.reference_priority, language),
    paper_connections: paperConnections,
    reference_priority: referencePriority,
  };
}

function isBroadResearchContext(context: string | undefined): boolean {
  const value = context?.trim().toLowerCase();
  if (!value) {
    return true;
  }

  return /(methodology patterns|reusable ideas|without overfitting|across experimental papers|compare methodology|general pattern|broad coverage|방법론 패턴|재사용 아이디어|도메인에 과적합|여러 실험 논문)/i.test(value);
}

function shouldGroundAdaptationText(value: string | undefined): boolean {
  const compact = value?.trim().toLowerCase() || '';
  if (!compact) {
    return true;
  }

  return /(could be adapted|could inspire|other domains|for example|e\.g\.|cross-domain|generalize to|similar approaches in other domains|broader domain)/i.test(compact);
}

function buildGroundedPaperConnection(
  paper: ProcessedPaper,
  language: 'ko' | 'en',
): MethodPaperConnection {
  const patternLabel = deriveMethodPatternLabel(paper, language);
  const datasets = formatList(
    paper.summary.datasets,
    language === 'ko' ? '보고된 데이터셋' : 'the reported datasets',
  );
  const metrics = formatList(
    paper.summary.evaluation_metrics,
    language === 'ko' ? '보고된 지표' : 'the reported metrics',
  );
  const trainingAnchor = shortenText(
    fallbackText(
      paper.summary.training_strategy,
      paper.summary.model_architecture || (language === 'ko' ? '보고된 학습/모델 설정' : 'the reported training and model setup'),
    ),
    140,
  );

  return {
    paper_id: paper.paperId,
    relevance:
      language === 'ko'
        ? `${patternLabel}이 이 논문에서 가장 재사용 가능한 방법론 단위입니다.`
        : `${patternLabel} is the most reusable methodological unit in this paper.`,
    adaptation:
      language === 'ko'
        ? `${datasets}에서 먼저 재현하고, ${metrics}를 초기 평가 계약으로 유지하세요. 그다음 ${trainingAnchor}을 보존한 상태에서 범위를 넓히는 편이 안전합니다.`
        : `Reproduce it first on ${datasets}, keep ${metrics} as the initial evaluation contract, and preserve ${trainingAnchor} before widening scope.`,
    caution: fallbackText(
      paper.summary.limitations,
      language === 'ko' ? '원문 한계부터 다시 확인하세요.' : 'Recheck the paper limitations before reuse.',
    ),
  };
}

function buildGroundedApplicationPoints(
  papers: ProcessedPaper[],
  referencePriority: MethodReferencePriority[],
  language: 'ko' | 'en',
): MethodApplicationPoint[] {
  const paperById = new Map(papers.map((paper) => [paper.paperId, paper]));
  const orderedPapers = referencePriority
    .map((priority) => paperById.get(priority.paper_id))
    .filter((paper): paper is ProcessedPaper => Boolean(paper))
    .slice(0, Math.min(5, papers.length));

  return orderedPapers.map((paper) => ({
    title:
      language === 'ko'
        ? `재사용 패턴: ${deriveMethodPatternLabel(paper, language)}`
        : `Reusable pattern: ${deriveMethodPatternLabel(paper, language)}`,
    rationale:
      fallbackText(
        paper.summary.main_contribution,
        language === 'ko'
          ? '핵심 기여를 재현 가능한 방법론 패턴으로 먼저 고정하세요.'
          : 'Treat the main contribution as a reusable methodological pattern first.',
      ),
    actions: buildGroundedApplicationActions(paper, language),
    paper_ids: [paper.paperId],
  }));
}

function buildGroundedReferenceRationale(
  paper: ProcessedPaper,
  language: 'ko' | 'en',
): string {
  const patternLabel = deriveMethodPatternLabel(paper, language);
  const metrics = formatList(
    paper.summary.evaluation_metrics,
    language === 'ko' ? '보고된 지표' : 'the reported metrics',
  );

  return language === 'ko'
    ? `${patternLabel}을 재현 가능한 단위로 잘라서 읽기 좋고, ${metrics} 기준으로 비교가 가능합니다.`
    : `${patternLabel} is easy to reuse as a bounded method pattern, and it stays comparable through ${metrics}.`;
}

function buildGroundedApplicationActions(
  paper: ProcessedPaper,
  language: 'ko' | 'en',
): string[] {
  const actions: string[] = [];

  actions.push(
    language === 'ko'
      ? `${formatList(paper.summary.datasets, '보고된 데이터셋')} 기준으로 1차 재현 범위를 고정하기`
      : `Lock the first reproduction scope to ${formatList(paper.summary.datasets, 'the reported datasets')}`,
  );
  actions.push(
    language === 'ko'
      ? `${formatList(paper.summary.evaluation_metrics, '보고된 지표')}를 그대로 유지해 비교 가능성 확보하기`
      : `Keep ${formatList(paper.summary.evaluation_metrics, 'the reported metrics')} unchanged for comparability`,
  );
  actions.push(
    language === 'ko'
      ? `학습/모델 설정: ${shortenText(fallbackText(paper.summary.training_strategy, paper.summary.model_architecture || '원문 설정 확인'), 120)}`
      : `Preserve the reported training/model setup: ${shortenText(fallbackText(paper.summary.training_strategy, paper.summary.model_architecture || 'review the paper setup'), 120)}`,
  );

  return actions;
}

function deriveMethodPatternLabel(
  paper: ProcessedPaper,
  language: 'ko' | 'en',
): string {
  const corpus = [
    paper.summary.model_architecture ?? '',
    paper.summary.training_strategy ?? '',
    paper.summary.main_contribution ?? '',
  ].join(' ').toLowerCase();

  if (/\b(synthetic|judge|judges|preference validation|ppi|query generation)\b/.test(corpus)) {
    return language === 'ko' ? '합성 데이터와 경량 판정기 결합' : 'synthetic data with lightweight judges';
  }
  if (/\b(inverse folding|antibody|esm-if1|cdr|paratope|epitope|affinity prediction)\b/.test(corpus)) {
    return language === 'ko' ? '구조 조건부 단백질/항체 서열 설계' : 'structure-conditioned protein or antibody sequence design';
  }
  if (/\b(set prediction|bipartite matching|object queries|transformer encoder-decoder)\b/.test(corpus)) {
    return language === 'ko' ? '직접 집합 예측과 매칭 손실' : 'direct set prediction with matching loss';
  }
  if (/\b(graph attention|masked self-attention|attention coefficients|attentional layers|gat)\b/.test(corpus)) {
    return language === 'ko' ? '주의집중 기반 이웃 가중 집계' : 'attention-weighted neighborhood aggregation';
  }
  if (/\b(weisfeiler|isomorphism network|sum aggregator|injective aggregation|gin)\b/.test(corpus)) {
    return language === 'ko' ? 'WL 수준 그래프 판별력 확보' : 'weisfeiler-lehman level graph discrimination';
  }
  if (/\b(graph convolution|graph convolutions|localized spectral|message passing|renormalization|gcn)\b/.test(corpus)) {
    return language === 'ko' ? '국소 그래프 메시지 패싱' : 'localized graph message passing';
  }
  if (/\b(inductive|neighbor|neighbour|aggregation|graphsage)\b/.test(corpus)) {
    return language === 'ko' ? '유도형 이웃 집계' : 'inductive neighborhood aggregation';
  }
  if (/\b(policy optimization|clipped surrogate|ppo|trust-region|advantage estimation)\b/.test(corpus)) {
    return language === 'ko' ? '안정성 중심 정책 업데이트' : 'stability-focused policy updates';
  }
  if (/\b(soft actor-critic|maximum entropy|off-policy actor-critic|entropy regularization|temperature tuning|sac)\b/.test(corpus)) {
    return language === 'ko' ? '엔트로피 정규화 오프폴리시 액터-크리틱' : 'entropy-regularized off-policy actor-critic';
  }
  if (/\b(cloze objective|bidirectional self-attention|sequential recommendation|user behavior sequences|bert4rec)\b/.test(corpus)) {
    return language === 'ko' ? '양방향 시퀀스 모델링 기반 추천' : 'bidirectional sequence modeling for recommendation';
  }
  if (/\b(contrastive language-image pre-training|image-text pairs|zero-shot transfer|clip)\b/.test(corpus)) {
    return language === 'ko' ? '대조적 멀티모달 사전학습' : 'contrastive multimodal pretraining';
  }
  if (/\b(io-aware exact attention|flashattention|memory-efficient attention|fewer memory accesses|tiling)\b/.test(corpus)) {
    return language === 'ko' ? 'IO 최적화 정확 어텐션' : 'io-aware exact attention optimization';
  }
  if (/\b(image patches|vision transformer|patch embeddings|patch sequence|vit)\b/.test(corpus)) {
    return language === 'ko' ? '패치 시퀀스 기반 비전 트랜스포머' : 'patch-sequence vision transformer';
  }
  if (/\b(pre-train|pretrain|masked language modeling|germline|domain-specific objective|pre-training)\b/.test(corpus)) {
    return language === 'ko' ? '도메인 인식 사전학습 목표' : 'domain-aware pretraining objectives';
  }
  if (/\b(auto-correlation|series decomposition|long-term series forecasting|forecasting)\b/.test(corpus)) {
    return language === 'ko' ? '시계열 구조를 위한 아키텍처 유도편향' : 'architectural inductive bias for temporal structure';
  }
  if (/\b(contextual embeddings|embedding similarity|cosine similarity|greedy matching|evaluation metric)\b/.test(corpus)) {
    return language === 'ko' ? '임베딩 기반 자동 평가' : 'embedding-based automatic evaluation';
  }
  if (/\b(benchmark|suite|collection of tasks|long context|long-context|evaluation benchmark)\b/.test(corpus)) {
    return language === 'ko' ? '범용 평가 벤치마크 설계' : 'general-purpose evaluation benchmark design';
  }
  if (/\b(contrastive|retrieval|ranking|embedding)\b/.test(corpus)) {
    return language === 'ko' ? '표현 학습과 비교 기반 평가' : 'representation learning with comparison-based evaluation';
  }

  return language === 'ko' ? '보고된 핵심 방법론' : 'the reported core method';
}

function summarizeFrequentTerms(values: string[], limit: number): string[] {
  const counts = new Map<string, { label: string; count: number }>();
  for (const value of values) {
    const normalized = value.trim();
    if (!normalized) {
      continue;
    }

    const key = normalized.toLowerCase();
    const current = counts.get(key);
    if (current) {
      current.count += 1;
    } else {
      counts.set(key, { label: normalized, count: 1 });
    }
  }

  return [...counts.values()]
    .sort((left, right) => right.count - left.count || left.label.localeCompare(right.label))
    .slice(0, limit)
    .map((entry) => entry.label);
}

function buildFallbackApplicationPoints(
  papers: ProcessedPaper[],
  language: 'ko' | 'en',
): MethodApplicationPoint[] {
  return papers.slice(0, 3).map((paper) => ({
    title:
      language === 'ko'
        ? `${paper.summary.title?.trim() || paper.fileName}의 핵심 기여를 재현해 보기`
        : `Adapt the main contribution from ${paper.summary.title?.trim() || paper.fileName}`,
    rationale:
      paper.summary.main_contribution?.trim()
      || (language === 'ko' ? '핵심 기여를 기준으로 후속 실험을 설계할 수 있습니다.' : 'Use the main contribution as the starting point for follow-up experiments.'),
    actions: [
      language === 'ko'
        ? `데이터셋 ${formatList(paper.summary.datasets, '미확인')} 기준으로 재현 범위를 정의하기`
        : `Define a reproduction scope around ${formatList(paper.summary.datasets, 'the reported datasets')}`,
      language === 'ko'
        ? `평가 지표 ${formatList(paper.summary.evaluation_metrics, '미확인')}를 동일하게 맞추기`
        : `Align the evaluation setup to ${formatList(paper.summary.evaluation_metrics, 'the reported metrics')}`,
    ],
    paper_ids: [paper.paperId],
  }));
}

function buildFallbackReferencePriority(papers: ProcessedPaper[]): MethodReferencePriority[] {
  return [...papers]
    .sort((left, right) => {
      const leftScore = (left.summary.datasets?.length ?? 0) + (left.summary.evaluation_metrics?.length ?? 0);
      const rightScore = (right.summary.datasets?.length ?? 0) + (right.summary.evaluation_metrics?.length ?? 0);
      return rightScore - leftScore;
    })
    .map((paper, index) => ({
      paper_id: paper.paperId,
      rank: index + 1,
      rationale:
        paper.summary.main_contribution?.trim()
        || 'Provides a concrete starting point for follow-up reading.',
    }));
}

function buildComparisonTable(papers: ProcessedPaper[], language: 'ko' | 'en'): string {
  const header =
    language === 'ko'
      ? '| 논문 | 핵심 방법론 | 학습 전략 | 데이터셋 | 주요 기여 | 한계 |\n| --- | --- | --- | --- | --- | --- |'
      : '| Paper | Core methodology | Training strategy | Datasets | Main contribution | Limitations |\n| --- | --- | --- | --- | --- | --- |';

  const rows = papers.map((paper) => {
    const title = escapeTableCell(paper.summary.title?.trim() || paper.fileName);
    const architecture = escapeTableCell(shortenText(fallbackText(paper.summary.model_architecture, '-'), 160));
    const training = escapeTableCell(shortenText(fallbackText(paper.summary.training_strategy, '-'), 140));
    const datasets = escapeTableCell(formatLimitedList(paper.summary.datasets, '-', 4));
    const contribution = escapeTableCell(shortenText(fallbackText(paper.summary.main_contribution, '-'), 160));
    const limitations = escapeTableCell(shortenText(fallbackText(paper.summary.limitations, '-'), 140));
    return `| ${title} | ${architecture} | ${training} | ${datasets} | ${contribution} | ${limitations} |`;
  });

  return [header, ...rows].join('\n');
}

function selectMethodExcerpt(markdown: string): string {
  const section = extractSectionByHeadings(markdown, {
    startPatterns: [
      /^#+\s*(abstract|introduction|preliminaries|background|related work|problem formulation|approach|framework|method|methods|methodology|model|training|experiments|evaluation|limitations?)\b/i,
      /^#+\s*(초록|서론|배경|관련 연구|문제 정의|방법|방법론|프레임워크|모델|학습|실험|평가|한계)\b/i,
    ],
    stopPatterns: [
      /^#+\s*(references?|appendix|acknowledg(e)?ments?|supplementary)\b/i,
      /^#+\s*(참고문헌|부록|감사의 글)\b/i,
    ],
    limit: 28_000,
  });

  if (section.trim()) {
    return section;
  }

  return markdown.slice(0, 28_000);
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

function formatLimitedList(values: string[] | undefined, fallback: string, maxItems: number): string {
  if (!values || values.length === 0) {
    return fallback;
  }

  const normalized = normalizeStringList(values, maxItems + 2, 80);
  if (normalized.length === 0) {
    return fallback;
  }

  const visible = normalized.slice(0, maxItems);
  const remainder = normalized.length - visible.length;
  return remainder > 0 ? `${visible.join(', ')} +${remainder} more` : visible.join(', ');
}

function fallbackText(value: string | undefined, fallback: string): string {
  return value?.trim() || fallback;
}

function escapeTableCell(value: string): string {
  return value.replace(/\|/g, '\\|').replace(/\n+/g, ' ');
}

function shortenText(value: string, maxChars: number): string {
  const compact = value.replace(/\s+/g, ' ').trim();
  if (compact.length <= maxChars) {
    return compact;
  }

  return `${compact.slice(0, Math.max(0, maxChars - 3)).trimEnd()}...`;
}

function shouldRepairMethodSummary(summary: PaperMethodSummary): boolean {
  const filledCount = [
    summary.title,
    summary.model_architecture,
    summary.training_strategy,
    summary.main_contribution,
    summary.limitations,
  ].filter((value) => value?.trim()).length
    + ((summary.datasets?.length ?? 0) > 0 ? 1 : 0)
    + ((summary.evaluation_metrics?.length ?? 0) > 0 ? 1 : 0);

  return filledCount < 3;
}

function mergeMethodSummary(primary: PaperMethodSummary, fallback: PaperMethodSummary): PaperMethodSummary {
  return {
    title: primary.title?.trim() || fallback.title?.trim() || '',
    model_architecture: primary.model_architecture?.trim() || fallback.model_architecture?.trim() || '',
    training_strategy: primary.training_strategy?.trim() || fallback.training_strategy?.trim() || '',
    datasets: [...(primary.datasets ?? []), ...(fallback.datasets ?? [])],
    main_contribution: primary.main_contribution?.trim() || fallback.main_contribution?.trim() || '',
    limitations: primary.limitations?.trim() || fallback.limitations?.trim() || '',
    evaluation_metrics: [...(primary.evaluation_metrics ?? []), ...(fallback.evaluation_metrics ?? [])],
  };
}

function normalizeMethodSummary(summary: PaperMethodSummary): PaperMethodSummary {
  return {
    title: normalizeFreeText(summary.title, 220),
    model_architecture: normalizeFreeText(summary.model_architecture, 420),
    training_strategy: normalizeFreeText(summary.training_strategy, 360),
    datasets: normalizeStringList(summary.datasets ?? [], 10, 80),
    main_contribution: normalizeFreeText(summary.main_contribution, 420),
    limitations: normalizeFreeText(summary.limitations, 360),
    evaluation_metrics: normalizeStringList(summary.evaluation_metrics ?? [], 10, 80),
  };
}

function normalizeFreeText(value: string | undefined, maxChars: number): string {
  if (!value) {
    return '';
  }

  const compact = value
    .replace(/\s+/g, ' ')
    .replace(/^[•*-]\s*/, '')
    .trim();
  if (!compact || /^(n\/a|none|null|not specified|unknown|-|확인 필요|추출 실패)$/i.test(compact)) {
    return '';
  }

  return shortenText(compact, maxChars);
}

function normalizeStringList(values: string[], maxItems: number, maxChars: number): string[] {
  const seen = new Set<string>();
  const normalized: string[] = [];

  for (const rawValue of values) {
    const value = normalizeFreeText(rawValue, maxChars);
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

function collectEvidenceRefs(papers: ProcessedPaper[], paperIds: string[]): string[] {
  return paperIds
    .flatMap((paperId) => {
      const paper = papers.find((candidate) => candidate.paperId === paperId);
      if (!paper) {
        return [];
      }

      return paper.evidenceRecords.slice(0, 2).map((record) => `${paper.paperId}:${record.id}`);
    })
    .filter(Boolean);
}

function formatEvidenceRecordInline(record: EvidenceRecord, language: 'ko' | 'en'): string {
  const locator = formatEvidenceLocator(record, language);
  return locator ? `[${record.id} ${locator}] ${record.text}` : `[${record.id}] ${record.text}`;
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

function resolveConcurrency(itemCount: number): number {
  const raw = Number.parseInt(process.env.UPSTAGE_RESEARCH_CONCURRENCY ?? '', 10);
  if (Number.isFinite(raw) && raw > 0) {
    return Math.max(1, Math.min(raw, itemCount));
  }

  return Math.max(1, Math.min(3, itemCount));
}

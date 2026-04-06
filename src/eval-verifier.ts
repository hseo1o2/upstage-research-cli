import { expandEvidenceQueries, normalizeQueryPhrase, type EvidenceRecord } from './evidence';
import type { MetricPlan, PaperEvaluationSummary } from './eval-types';
import { datasetsEquivalent, expandDatasetSurfaceForms } from './eval-generalization';

export interface VerificationCheck {
  label: string;
  status: 'pass' | 'warn';
  detail: string;
}

export interface VerificationSubscore {
  label: string;
  status: 'pass' | 'warn';
  passed: number;
  total: number;
  ratio: number;
  detail: string;
}

export interface EvaluationVerification {
  score: number;
  checks: VerificationCheck[];
  subscores: {
    metric_fidelity: VerificationSubscore;
    dataset_fidelity: VerificationSubscore;
    baseline_fidelity: VerificationSubscore;
    evidence_fidelity: VerificationSubscore;
    protocol_fidelity: VerificationSubscore;
  };
}

export function buildEvaluationVerification(
  summary: PaperEvaluationSummary,
  metricPlans: MetricPlan[],
  evidenceRecords: EvidenceRecord[],
  language: 'ko' | 'en',
): EvaluationVerification {
  const extractedMetrics = summary.evaluation_metrics ?? [];
  const resolvedMetrics = metricPlans.filter((plan) =>
    plan.kind !== 'custom_placeholder' && plan.kind !== 'feature_coefficient_placeholder',
  );
  const unresolvedMetrics = metricPlans.filter((plan) =>
    plan.kind === 'custom_placeholder' || plan.kind === 'feature_coefficient_placeholder',
  );
  const evidenceText = evidenceRecords.map((record) => record.text.toLowerCase()).join('\n');
  const datasetMentions = (summary.datasets ?? []).filter((dataset) => dataset.trim());
  const metricMentions = extractedMetrics
    .map((metric) => metric.name?.trim() ?? '')
    .filter(Boolean);

  const coveredDatasets = datasetMentions.filter((dataset) =>
    containsDatasetEvidence(evidenceText, dataset, datasetMentions),
  );
  const coveredMetrics = metricMentions.filter((metric) => containsMetricEvidence(evidenceText, metric, summary.task_type ?? ''));
  const baselineRows = summary.baselines ?? [];
  const baselineScoreRows = baselineRows.filter((baseline) => Object.keys(baseline.scores ?? {}).length > 0);
  const groundedBaselineRows = baselineRows.filter((baseline) =>
    evidenceRecords.some((record) => baselineGroundedInEvidence(record, baseline)),
  );
  const anchoredEvidenceRows = evidenceRecords.filter((record) => record.page !== undefined && Boolean(record.elementId));

  const subscores = {
    metric_fidelity: buildSubscore(
      language === 'ko' ? '지표 충실도' : 'Metric fidelity',
      resolvedMetrics.length,
      metricPlans.length,
      language === 'ko'
        ? `${resolvedMetrics.length}/${metricPlans.length || 1}개 metric이 안전한 실행 family에 정착했습니다.`
        : `${resolvedMetrics.length}/${metricPlans.length || 1} metrics landed on safe executable families.`,
      ({ ratio, total }) => total === 0 || ratio >= 1,
    ),
    dataset_fidelity: buildSubscore(
      language === 'ko' ? '데이터셋 근거도' : 'Dataset fidelity',
      coveredDatasets.length,
      datasetMentions.length,
      language === 'ko'
        ? `evidence snippets에서 dataset ${coveredDatasets.length}/${datasetMentions.length || 1}개를 다시 확인했습니다.`
        : `Reconfirmed ${coveredDatasets.length}/${datasetMentions.length || 1} datasets in evidence snippets.`,
      ({ ratio, total }) => total === 0 || ratio >= 0.5,
    ),
    baseline_fidelity: buildSubscore(
      language === 'ko' ? '베이스라인 충실도' : 'Baseline fidelity',
      baselineRows.length === 0 ? 1 : baselineScoreRows.length + groundedBaselineRows.length,
      baselineRows.length === 0 ? 1 : baselineRows.length * 2,
      baselineRows.length === 0
        ? (language === 'ko'
          ? '베이스라인이 없어 추가 베이스라인 검증이 필요하지 않습니다.'
          : 'No baselines were extracted, so no baseline-specific verification was required.')
        : (language === 'ko'
          ? `${baselineScoreRows.length}/${baselineRows.length}개 row가 scores를 포함하고 ${groundedBaselineRows.length}/${baselineRows.length}개 row가 evidence에 고정됩니다.`
          : `${baselineScoreRows.length}/${baselineRows.length} rows include scores and ${groundedBaselineRows.length}/${baselineRows.length} rows are grounded in evidence.`),
      ({ ratio, total }) => total === 0 || ratio >= 0.75,
    ),
    evidence_fidelity: buildSubscore(
      language === 'ko' ? '근거 추적도' : 'Evidence fidelity',
      anchoredEvidenceRows.length,
      evidenceRecords.length,
      language === 'ko'
        ? `${anchoredEvidenceRows.length}/${evidenceRecords.length || 1}개 evidence snippet이 page/anchor provenance를 포함합니다.`
        : `${anchoredEvidenceRows.length}/${evidenceRecords.length || 1} evidence snippets include page/anchor provenance.`,
      ({ ratio, total }) => total === 0 || ratio >= 0.75,
    ),
    protocol_fidelity: buildSubscore(
      language === 'ko' ? '프로토콜 충실도' : 'Protocol fidelity',
      Math.max(0, extractedMetrics.length - unresolvedMetrics.length),
      extractedMetrics.length,
      unresolvedMetrics.length === 0
        ? (language === 'ko'
          ? '현재 출력에는 명시적 placeholder metric이 없습니다.'
          : 'No explicit placeholder metrics remain in the current artifact.')
        : (language === 'ko'
          ? `논문 전용 절차 검토가 필요한 metric: ${unresolvedMetrics.map((plan) => plan.metricName).join(', ')}`
          : `Paper-specific protocol review still needed for: ${unresolvedMetrics.map((plan) => plan.metricName).join(', ')}`),
      ({ ratio, total }) => total > 0 && ratio >= 1,
    ),
  };

  const checks: VerificationCheck[] = [
    {
      label: language === 'ko' ? '지표 커버리지' : 'Metric coverage',
      status: subscores.metric_fidelity.status,
      detail: subscores.metric_fidelity.detail,
    },
    {
      label: language === 'ko' ? '근거 고정도' : 'Evidence grounding',
      status:
        coveredMetrics.length >= Math.max(1, Math.ceil(metricMentions.length * 0.6))
        && subscores.dataset_fidelity.status === 'pass'
        && subscores.evidence_fidelity.status === 'pass'
          ? 'pass'
          : 'warn',
      detail:
        language === 'ko'
          ? `evidence snippets에서 metric ${coveredMetrics.length}/${metricMentions.length || 1}, dataset ${coveredDatasets.length}/${datasetMentions.length || 1}개를 다시 확인했습니다.`
          : `Reconfirmed ${coveredMetrics.length}/${metricMentions.length || 1} metrics and ${coveredDatasets.length}/${datasetMentions.length || 1} datasets inside evidence snippets.`,
    },
    {
      label: language === 'ko' ? '베이스라인 지원도' : 'Baseline support',
      status: subscores.baseline_fidelity.status,
      detail: subscores.baseline_fidelity.detail,
    },
    {
      label: language === 'ko' ? '프로토콜 리스크' : 'Protocol risk',
      status: subscores.protocol_fidelity.status,
      detail: subscores.protocol_fidelity.detail,
    },
  ];

  const score = checks.reduce((total, check) => total + (check.status === 'pass' ? 1 : 0), 0);
  return { score, checks, subscores };
}

function baselineGroundedInEvidence(record: EvidenceRecord, baseline: { model_name?: string; scores?: Record<string, number | string | null> }): boolean {
  const baselineName = baseline.model_name ?? '';
  const matchedQuery = record.matchedQuery ?? '';
  if (containsLooseEvidence(record.text.toLowerCase(), baselineName) || containsLooseEvidence(matchedQuery.toLowerCase(), baselineName)) {
    return true;
  }

  const scoreEntries = Object.entries(baseline.scores ?? {}).filter(([, value]) => value !== null && value !== undefined && `${value}`.trim());
  if (scoreEntries.length === 0) {
    return false;
  }

  const evidenceText = record.text.toLowerCase();
  const matchedMetricNames = scoreEntries.filter(([scoreName]) => containsLooseEvidence(evidenceText, scoreName)).length;
  const matchedValues = scoreEntries.filter(([, value]) => containsLooseEvidence(evidenceText, String(value))).length;

  return matchedMetricNames >= Math.min(2, scoreEntries.length) && matchedValues >= 1;
}

export function renderVerificationSection(
  verification: EvaluationVerification,
  language: 'ko' | 'en',
): string {
  const lines: string[] = [];
  lines.push(language === 'ko' ? '## 검증 리포트' : '## Verification Report');
  lines.push('');
  lines.push(
    language === 'ko'
      ? `- Verification score: ${verification.score}/${verification.checks.length}`
      : `- Verification score: ${verification.score}/${verification.checks.length}`,
  );
  lines.push(
    language === 'ko'
      ? `- Metric fidelity: ${formatSubscore(verification.subscores.metric_fidelity)}`
      : `- Metric fidelity: ${formatSubscore(verification.subscores.metric_fidelity)}`,
  );
  lines.push(
    language === 'ko'
      ? `- Dataset fidelity: ${formatSubscore(verification.subscores.dataset_fidelity)}`
      : `- Dataset fidelity: ${formatSubscore(verification.subscores.dataset_fidelity)}`,
  );
  lines.push(
    language === 'ko'
      ? `- Baseline fidelity: ${formatSubscore(verification.subscores.baseline_fidelity)}`
      : `- Baseline fidelity: ${formatSubscore(verification.subscores.baseline_fidelity)}`,
  );
  lines.push(
    language === 'ko'
      ? `- Evidence fidelity: ${formatSubscore(verification.subscores.evidence_fidelity)}`
      : `- Evidence fidelity: ${formatSubscore(verification.subscores.evidence_fidelity)}`,
  );
  lines.push(
    language === 'ko'
      ? `- Protocol fidelity: ${formatSubscore(verification.subscores.protocol_fidelity)}`
      : `- Protocol fidelity: ${formatSubscore(verification.subscores.protocol_fidelity)}`,
  );

  for (const check of verification.checks) {
    const marker = check.status === 'pass' ? 'PASS' : 'WARN';
    lines.push(`- ${marker} ${check.label}: ${check.detail}`);
  }

  return lines.join('\n');
}

function containsLooseEvidence(evidenceText: string, rawValue: string): boolean {
  const normalizedEvidence = normalizeQueryPhrase(evidenceText);
  if (!normalizedEvidence) {
    return false;
  }

  const variants = expandEvidenceQueries([rawValue]);
  for (const variant of variants) {
    const normalizedVariant = normalizeQueryPhrase(variant);
    if (!normalizedVariant) {
      continue;
    }

    const paddedEvidence = ` ${normalizedEvidence} `;
    const paddedVariant = ` ${normalizedVariant} `;
    if (paddedEvidence.includes(paddedVariant) || normalizedEvidence.includes(normalizedVariant)) {
      return true;
    }

    const parts = normalizedVariant
      .split(/\s+/)
      .filter((part) => part.length >= 2 || /[가-힣]{2,}/.test(part));
    if (parts.length === 0) {
      continue;
    }

    const matched = parts.filter((part) => paddedEvidence.includes(` ${part} `) || normalizedEvidence.includes(part)).length;
    if (matched >= Math.max(1, Math.ceil(parts.length / 2))) {
      return true;
    }
  }

  return false;
}

function containsMetricEvidence(evidenceText: string, metricName: string, taskType: string): boolean {
  if (containsLooseEvidence(evidenceText, metricName)) {
    return true;
  }

  const normalizedMetric = normalizeQueryPhrase(metricName);
  const normalizedTaskType = normalizeQueryPhrase(taskType);
  const isRankingLike = /\branking\b|recommendation|retrieval/.test(normalizedTaskType);

  if (
    isRankingLike
    && /^recall(?:\s+\d+)?$/.test(normalizedMetric)
    && (
      containsLooseEvidence(evidenceText, 'hit rate')
      || containsLooseEvidence(evidenceText, 'hit ratio')
      || /\bhr@\s*\d+/i.test(evidenceText)
      || /\bhit@\s*\d+/i.test(evidenceText)
    )
  ) {
    return true;
  }

  return false;
}

function containsDatasetEvidence(
  evidenceText: string,
  datasetName: string,
  peerDatasets: string[],
): boolean {
  if (containsLooseEvidence(evidenceText, datasetName)) {
    return true;
  }

  for (const variant of expandDatasetSurfaceForms(datasetName, peerDatasets)) {
    if (containsLooseEvidence(evidenceText, variant)) {
      return true;
    }
  }

  const normalizedEvidence = normalizeQueryPhrase(evidenceText);
  if (!normalizedEvidence) {
    return false;
  }

  const evidenceSegments = normalizedEvidence
    .split(/\n+/)
    .map((segment) => segment.trim())
    .filter(Boolean);

  return evidenceSegments.some((segment) => datasetsEquivalent(segment, datasetName, peerDatasets));
}

function buildSubscore(
  label: string,
  passed: number,
  total: number,
  detail: string,
  isPass: (input: { passed: number; total: number; ratio: number }) => boolean,
): VerificationSubscore {
  const safeTotal = total > 0 ? total : 0;
  const ratio = safeTotal > 0 ? passed / safeTotal : 1;
  return {
    label,
    passed,
    total: safeTotal,
    ratio,
    detail,
    status: isPass({ passed, total: safeTotal, ratio }) ? 'pass' : 'warn',
  };
}

function formatSubscore(subscore: VerificationSubscore): string {
  return `${Math.round(subscore.ratio * 100)}% (${subscore.passed}/${subscore.total})`;
}

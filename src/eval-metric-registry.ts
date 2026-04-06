import type { MetricPlan, MetricPlanKind, PaperEvaluationSummary, TaskMode } from './eval-types';
import { getEvalDiscoveryRegistry, getMetricKindRegistry } from './registry-config';

export function inferTaskMode(summary: PaperEvaluationSummary): TaskMode {
  const corpus = [
    summary.task_type ?? '',
    ...(summary.evaluation_metrics ?? []).map((metric) => metric.name ?? ''),
    summary.implementation_details ?? '',
  ]
    .join(' ')
    .toLowerCase();

  for (const rule of getEvalDiscoveryRegistry().taskModeRules) {
    if (rule.mode && new RegExp(rule.pattern).test(corpus)) {
      return rule.mode as TaskMode;
    }
  }

  return 'prediction_reference';
}

export function inferDomain(summary: PaperEvaluationSummary): string {
  const corpus = [
    summary.task_type ?? '',
    ...(summary.datasets ?? []),
    ...(summary.evaluation_metrics ?? []).map((metric) => metric.name ?? ''),
  ]
    .join(' ')
    .toLowerCase();

  const registry = getEvalDiscoveryRegistry();

  for (const rule of registry.domainRules) {
    if (rule.domain && new RegExp(rule.pattern).test(corpus)) {
      return rule.domain;
    }
  }

  for (const rule of registry.domainFallbackRules) {
    if (!rule.domain) {
      continue;
    }

    if (new RegExp(rule.pattern).test(corpus) && (!rule.requires || new RegExp(rule.requires).test(corpus))) {
      return rule.domain;
    }
  }

  return 'general';
}

export function libraryHintsByDomain(domain: string): string {
  return getEvalDiscoveryRegistry().libraryHints[domain] ?? getEvalDiscoveryRegistry().libraryHints.general ?? 'numpy, pandas';
}

export function buildMetricPlans(summary: PaperEvaluationSummary, taskMode: TaskMode): MetricPlan[] {
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

    if (/spearman('|’)?s?( rank)? correlation/.test(lower) || /spearman/.test(lower) || (/\bsr\b/.test(lower) && /spearman|rank correlation/.test(formulaLower))) {
      return { metricName, outputKey: slug, functionName: `compute_${slug}`, kind: 'spearman' };
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

    if (/edit sim|edit similarity|\banls\b|average normalized levenshtein similarity/.test(lower) || /levenshtein similarity/.test(formulaLower)) {
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

    const ruleBasedPlan = buildRuleBasedMetricPlan(metricName, slug, lower);
    if (ruleBasedPlan) {
      return ruleBasedPlan;
    }

    const generalizedPlan = buildGeneralizedMetricPlan(metricName, slug, lower, formulaLower, taskCorpus);
    if (generalizedPlan) {
      return generalizedPlan;
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

export function isRankingMetricKind(kind: MetricPlanKind): boolean {
  return getMetricKindRegistry().rankingKinds.includes(kind);
}

export function isDirectPredictionMetricKind(kind: MetricPlanKind): boolean {
  return getMetricKindRegistry().directPredictionKinds.includes(kind);
}

export function fallbackMetricNameForTaskMode(taskMode: TaskMode): string {
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

function slugifyMetricName(metricName: string): string {
  return metricName
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    || 'metric';
}

function fallbackText(value: string | undefined, fallback: string): string {
  return value?.trim() || fallback;
}

function buildRuleBasedMetricPlan(
  metricName: string,
  slug: string,
  lowerMetricName: string,
): MetricPlan | null {
  for (const rule of getMetricKindRegistry().rules) {
    const includePattern = new RegExp(rule.metricPattern);
    if (!includePattern.test(lowerMetricName)) {
      continue;
    }

    if (rule.excludeMetricPattern && new RegExp(rule.excludeMetricPattern).test(lowerMetricName)) {
      continue;
    }

    return {
      metricName,
      outputKey: slug,
      functionName: `compute_${slug}`,
      kind: rule.kind as MetricPlanKind,
    };
  }

  return null;
}

function buildGeneralizedMetricPlan(
  metricName: string,
  slug: string,
  lowerMetricName: string,
  lowerFormula: string,
  taskCorpus: string,
): MetricPlan | null {
  const combined = `${lowerMetricName} ${lowerFormula}`.trim();
  const rankingLike = /(ranking|retrieval|recommend|recommender|sequential recommendation|user-item|session)/.test(taskCorpus);
  const visionLike = /(vision|image|object detection|segmentation|bbox|bounding box|coco|imagenet)/.test(taskCorpus);
  const bioStructureLike = /(protein|structure|fold|residue|antibody|bioinformatics|molecule|docking)/.test(taskCorpus);
  const clusteringLike = /(cluster|clustering|integration|single-cell|single cell|batch effect|embedding)/.test(taskCorpus);

  if (/dcg|idcg|discounted cumulative gain/.test(combined)) {
    return buildMetricPlan(metricName, slug, 'ndcg');
  }
  if (/reciprocal rank/.test(combined)) {
    return buildMetricPlan(metricName, slug, 'mrr');
  }
  if (/hit ratio|hit rate/.test(combined) || (rankingLike && /\bhr\b/.test(lowerMetricName))) {
    return buildMetricPlan(metricName, slug, 'hit_rate');
  }
  if (/@\s*\d+/.test(combined) || /top[- ]?k|top[- ]?\d+/.test(combined)) {
    if (/precision/.test(combined)) {
      return buildMetricPlan(metricName, slug, 'precision_at_k');
    }
    if (/recall/.test(combined)) {
      return buildMetricPlan(metricName, slug, 'recall_at_k');
    }
    if (/pass/.test(combined)) {
      return buildMetricPlan(metricName, slug, 'pass_at_k');
    }
    if (/accuracy|acc\b/.test(combined)) {
      return buildMetricPlan(metricName, slug, 'top_k_accuracy');
    }
  }
  if (/exp\s*\(\s*-\s*1\s*\/\s*n|exp\s*\(\s*-\s*\\frac\{1\}\{n\}|per token log probability/.test(combined)) {
    return buildMetricPlan(metricName, slug, 'perplexity');
  }
  if (/mean absolute error|\|\s*y|absolute percentage error|symmetric mean absolute percentage error/.test(combined)) {
    if (/symmetric/.test(combined)) {
      return buildMetricPlan(metricName, slug, 'smape');
    }
    if (/percentage/.test(combined)) {
      return buildMetricPlan(metricName, slug, 'mape');
    }
    return buildMetricPlan(metricName, slug, 'mae');
  }
  if (/root mean square|sqrt/.test(combined) && /squared error/.test(combined)) {
    return buildMetricPlan(metricName, slug, 'rmse');
  }
  if (/mean squared error|squared error/.test(combined)) {
    return buildMetricPlan(metricName, slug, 'mse');
  }
  if (/pearson|covariance|linear correlation/.test(combined)) {
    return buildMetricPlan(metricName, slug, 'pearson');
  }
  if (/spearman|rank correlation/.test(combined)) {
    return buildMetricPlan(metricName, slug, 'spearman');
  }
  if (/kendall/.test(combined)) {
    return buildMetricPlan(metricName, slug, 'kendall_tau');
  }
  if (/intersection over union|overlap area/.test(combined)) {
    return buildMetricPlan(metricName, slug, 'iou');
  }
  if (visionLike && /average precision|\bap\d*\b/.test(combined) && !/\bmape\b/.test(combined)) {
    return buildMetricPlan(metricName, slug, /mask|segm|segmentation/.test(combined) ? 'mask_map' : 'map');
  }
  if (/frechet inception distance/.test(combined)) {
    return buildMetricPlan(metricName, slug, 'fid');
  }
  if (/signal[- ]to[- ]noise/.test(combined)) {
    return buildMetricPlan(metricName, slug, 'psnr');
  }
  if (/structural similarity/.test(combined)) {
    return buildMetricPlan(metricName, slug, 'ssim');
  }
  if (/word error rate/.test(combined)) {
    return buildMetricPlan(metricName, slug, 'wer');
  }
  if (/character error rate/.test(combined)) {
    return buildMetricPlan(metricName, slug, 'cer');
  }
  if (/bleu|rouge|bertscore|cider|spice/.test(combined)) {
    return buildMetricPlan(metricName, slug, 'library_metric');
  }
  if (/rmsd|root mean square deviation/.test(combined) || (bioStructureLike && /distance between structures|atomic deviation/.test(combined))) {
    return buildMetricPlan(metricName, slug, 'rmsd');
  }
  if (clusteringLike) {
    if (/silhouette/.test(combined)) {
      return buildMetricPlan(metricName, slug, 'silhouette_score');
    }
    if (/normalized mutual information/.test(combined)) {
      return buildMetricPlan(metricName, slug, 'normalized_mutual_info');
    }
    if (/adjusted rand/.test(combined)) {
      return buildMetricPlan(metricName, slug, 'adjusted_rand_index');
    }
  }

  return null;
}

function buildMetricPlan(metricName: string, slug: string, kind: MetricPlanKind): MetricPlan {
  return {
    metricName,
    outputKey: slug,
    functionName: `compute_${slug}`,
    kind,
  };
}

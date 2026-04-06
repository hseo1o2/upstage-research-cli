export interface EvaluationMetric {
  name?: string;
  formula?: string;
  value?: number | string | null;
}

export interface Baseline {
  model_name?: string;
  scores?: Record<string, number | string | null>;
}

export interface PaperEvaluationSummary {
  task_type?: string;
  evaluation_metrics?: EvaluationMetric[];
  datasets?: string[];
  baselines?: Baseline[];
  implementation_details?: string;
}

export type TaskMode =
  | 'pairwise_preference'
  | 'prediction_reference'
  | 'ranking'
  | 'forecasting'
  | 'classification'
  | 'vision';

export interface EvaluationGuidanceCriterion {
  name: string;
  description: string;
}

export interface EvaluationGuidance {
  reproduction_focus: string;
  implementation_cautions: string[];
  judge_criteria: EvaluationGuidanceCriterion[];
  judge_failure_conditions: string[];
}

export type MetricPlanKind =
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

export interface MetricPlan {
  metricName: string;
  outputKey: string;
  functionName: string;
  kind: MetricPlanKind;
  formula?: string;
  note?: string;
  resultKey?: string;
}

export const METRIC_PLAN_KIND_VALUES: MetricPlanKind[] = [
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

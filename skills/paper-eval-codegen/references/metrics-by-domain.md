# Metrics by Domain

## NLP
- Libraries: `evaluate`, `nltk`, `sacrebleu`
- Common metrics: Accuracy, F1, BLEU, ROUGE, Exact Match, BERTScore, perplexity
- Typical inputs: predictions, references, optional tokenized text

## Computer Vision
- Libraries: `torchmetrics`, `pycocotools`
- Common metrics: Top-1 accuracy, mAP, IoU, mIoU, FID, PSNR, SSIM
- Typical inputs: image predictions, labels, bounding boxes, masks

## Reinforcement Learning
- Libraries: `gymnasium`
- Common metrics: average return, success rate, cumulative reward
- Typical inputs: environment rollouts, reward logs, episode statistics

## Recommendation
- Libraries: `recbole`
- Common metrics: NDCG, Recall@K, HitRate@K, AUC, MRR
- Typical inputs: ranked items, user-item interactions, ground truth items

## Time Series
- Libraries: `tsmetric`
- Common metrics: MAE, RMSE, MAPE, sMAPE, MASE, precision/recall for anomaly detection
- Typical inputs: predictions, targets, timestamps, anomaly labels

## General Guidance
- Prefer exact paper metrics when stated.
- If the paper omits a metric definition, use the canonical library implementation and mark the assumption.
- Preserve reported baseline numbers in checklist output.

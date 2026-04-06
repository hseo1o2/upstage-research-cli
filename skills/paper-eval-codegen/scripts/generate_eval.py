#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

BASE_URL = "https://api.upstage.ai/v1"
DEFAULT_SOLAR_MODEL = os.getenv("UPSTAGE_SOLAR_MODEL", "solar-pro3")
DEFAULT_IE_MODEL = os.getenv("UPSTAGE_INFORMATION_EXTRACTION_MODEL", "information-extract")
DEFAULT_IE_FALLBACK_MODEL = os.getenv("UPSTAGE_INFORMATION_EXTRACT_FALLBACK_MODEL", DEFAULT_SOLAR_MODEL)
ROOT_DIR = Path(__file__).resolve().parents[3]
SCHEMA_PATH = ROOT_DIR / "skills" / "paper-eval-codegen" / "references" / "eval-schema.json"
METRICS_PATH = ROOT_DIR / "skills" / "paper-eval-codegen" / "references" / "metrics-by-domain.md"
PROMPT_TEMPLATES_PATH = ROOT_DIR / "skills" / "paper-eval-codegen" / "references" / "prompt-templates.md"
CONFIG_PATH = Path.home() / ".config" / "upstage-research" / "config.json"


def load_api_key() -> str:
    env_api_key = os.getenv("UPSTAGE_API_KEY", "").strip()
    if env_api_key:
        return env_api_key

    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        api_key = str(config.get("apiKey", "")).strip()
        if api_key:
            return api_key

    raise SystemExit(
        "UPSTAGE_API_KEY is not set. Export it or run `upstage-research install --skills` first."
    )


def build_openai_client(api_key: str) -> OpenAI:
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=BASE_URL)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_markdown_input(path_value: str) -> str:
    if path_value == "-":
        return sys.stdin.read()

    return Path(path_value).expanduser().resolve().read_text(encoding="utf-8")


def extract_eval_summary(client: OpenAI, markdown: str) -> dict[str, Any]:
    schema = json.loads(load_text(SCHEMA_PATH))
    model_candidates = unique_strings([DEFAULT_IE_MODEL, DEFAULT_IE_FALLBACK_MODEL])
    last_error: Exception | None = None

    for index, model_name in enumerate(model_candidates):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Extract evaluation setup details from academic paper experiment markdown. Use concise factual text. Return empty strings or empty arrays when evidence is missing.",
                    },
                    {
                        "role": "user",
                        "content": f"Experiment markdown:\n{clip_text(markdown, 28000)}",
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "paper_evaluation_summary",
                        "strict": True,
                        "schema": schema,
                    },
                },
            )
            content = response.choices[0].message.content or ""
            return parse_json_loose(content)
        except Exception as exc:
            last_error = exc
            if (
                index < len(model_candidates) - 1
                and is_unsupported_model_error(exc)
            ):
                print(
                    f"[warn] structured extraction model `{model_name}` unavailable; retrying with `{model_candidates[index + 1]}`",
                    file=sys.stderr,
                )
                continue
            raise

    if last_error is not None:
        raise last_error

    raise RuntimeError("evaluation summary extraction failed")


def generate_code(
    client: OpenAI,
    markdown: str,
    summary: dict[str, Any],
    requested_language: str,
    requested_framework: str,
    model: str,
) -> str:
    metrics_by_domain = load_text(METRICS_PATH)
    domain = infer_domain(summary)
    libraries = library_hints_by_domain(domain)

    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": "You write practical evaluation code for research reproduction. Return only one fenced code block and no commentary. Use only the paper metrics when they are available. Do not add generic benchmark metrics unless the paper is missing them.",
            },
            {
                "role": "user",
                "content": (
                    f"Generate runnable {requested_language} evaluation code for this paper.\n\n"
                    f"Requested framework: {requested_framework}\n"
                    f"Inferred domain: {domain}\n"
                    f"Recommended libraries: {libraries}\n\n"
                    f"Structured extraction:\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n\n"
                    f"Reference notes:\n{metrics_by_domain}\n\n"
                    f"Experiment markdown excerpt:\n{clip_text(markdown, 18000)}\n\n"
                    "Requirements:\n"
                    "- Include imports.\n"
                    "- Define a main evaluation function.\n"
                    "- Include placeholders for predictions, references, or a dataloader hook.\n"
                    "- Use paper metrics and datasets when available.\n"
                    "- If the paper already defines evaluation metrics, do not invent BLEU, ROUGE, BERTScore, or similar generic metrics.\n"
                    "- Every referenced variable, helper, and import must be defined inside the code block or clearly passed as a function argument.\n"
                    "- Prefer simple, explicit placeholder implementations over undefined pseudocode symbols.\n"
                    "- If a metric formula is custom, implement a TODO stub and place the formula in a comment.\n"
                    "- Return only one fenced code block."
                ),
            },
        ],
    )
    return ensure_code_fence(response.choices[0].message.content or "", requested_language)


def generate_judge_prompt(client: OpenAI, markdown: str, summary: dict[str, Any], model: str) -> str:
    prompt_templates = load_text(PROMPT_TEMPLATES_PATH)
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "You create rigorous LLM-as-judge prompts for paper reproduction. Return the final prompt text only in markdown. Do not explain your reasoning, do not describe what you are about to do, and do not mention the user request.",
            },
            {
                "role": "user",
                "content": (
                    "Create an LLM-as-judge prompt for this paper.\n\n"
                    f"Structured extraction:\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n\n"
                    f"Prompt templates:\n{prompt_templates}\n\n"
                    f"Experiment markdown excerpt:\n{clip_text(markdown, 12000)}\n\n"
                    "Return a prompt that an evaluator model can directly use.\n"
                    "Requirements:\n"
                    "- Start the prompt itself immediately.\n"
                    "- Include scoring criteria and failure conditions.\n"
                    "- Reference the paper metrics, datasets, and baselines when they are available.\n"
                    "- Do not include analysis, planning notes, or meta-commentary outside the prompt."
                ),
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def render_output(
    code_block: str,
    summary: dict[str, Any],
    requested_framework: str,
    include_prompt: bool,
    judge_prompt: str,
    language: str,
) -> str:
    domain = infer_domain(summary)
    sections: list[str] = []

    sections.append("## 평가 코드" if language == "ko" else "## Evaluation Code")
    sections.append("")
    sections.append(code_block.strip())
    sections.append("")
    sections.append(build_checklist_section(summary, domain, requested_framework, language))
    sections.append("")
    sections.append(build_metrics_section(summary, language))

    if include_prompt:
        sections.append("")
        sections.append("## LLM-as-judge 프롬프트" if language == "ko" else "## LLM-as-Judge Prompt")
        sections.append("")
        sections.append(judge_prompt.strip())

    return "\n".join(sections).strip()


def build_checklist_section(summary: dict[str, Any], domain: str, framework: str, language: str) -> str:
    if language == "ko":
        lines = [
            "## 실험 재현 체크리스트",
            "",
            f"- [ ] 태스크 유형: {fallback_text(summary.get('task_type'), '확인 필요')}",
            f"- [ ] 추론된 분야: {domain}",
            f"- [ ] 프레임워크: {framework}",
            f"- [ ] 데이터셋: {format_list(summary.get('datasets'), '확인 필요')}",
            f"- [ ] 평가지표: {format_metric_checklist(summary.get('evaluation_metrics'), '확인 필요')}",
            f"- [ ] 베이스라인: {format_baselines(summary.get('baselines'), '확인 필요')}",
            f"- [ ] 구현 세부사항: {fallback_text(summary.get('implementation_details'), '저자 구현 또는 부록 확인 필요')}",
        ]
    else:
        lines = [
            "## Reproduction Checklist",
            "",
            f"- [ ] Task type: {fallback_text(summary.get('task_type'), 'Needs review')}",
            f"- [ ] Inferred domain: {domain}",
            f"- [ ] Framework: {framework}",
            f"- [ ] Datasets: {format_list(summary.get('datasets'), 'Needs review')}",
            f"- [ ] Metrics: {format_metric_checklist(summary.get('evaluation_metrics'), 'Needs review')}",
            f"- [ ] Baselines: {format_baselines(summary.get('baselines'), 'Needs review')}",
            f"- [ ] Implementation details: {fallback_text(summary.get('implementation_details'), 'Review appendix or author code')}",
        ]
    return "\n".join(lines)


def build_metrics_section(summary: dict[str, Any], language: str) -> str:
    metrics = summary.get("evaluation_metrics") if isinstance(summary.get("evaluation_metrics"), list) else []
    baselines = summary.get("baselines") if isinstance(summary.get("baselines"), list) else []

    lines: list[str] = []
    lines.append("## 보고된 지표" if language == "ko" else "## Reported Metrics")
    lines.append("")

    if not metrics:
        lines.append("- 추출된 지표가 없습니다." if language == "ko" else "- No metrics were extracted.")
    else:
        lines.append("| Metric | Formula | Value |")
        lines.append("| --- | --- | --- |")
        for metric in metrics:
            metric_name = fallback_text(metric.get("name"), "-") if isinstance(metric, dict) else "-"
            formula = fallback_text(metric.get("formula"), "-") if isinstance(metric, dict) else "-"
            value = format_metric_value(metric.get("value")) if isinstance(metric, dict) else "-"
            lines.append(
                f"| {escape_table_cell(metric_name)} | {escape_table_cell(formula)} | {escape_table_cell(value)} |"
            )

    lines.append("")
    lines.append("## 베이스라인 스냅샷" if language == "ko" else "## Baseline Snapshot")
    lines.append("")

    if not baselines:
        lines.append("- 추출된 베이스라인이 없습니다." if language == "ko" else "- No baselines were extracted.")
    else:
        lines.append("| Baseline | Scores |")
        lines.append("| --- | --- |")
        for baseline in baselines:
            if not isinstance(baseline, dict):
                continue
            model_name = fallback_text(baseline.get("model_name"), "-")
            scores = baseline.get("scores")
            if isinstance(scores, dict):
                score_text = ", ".join(
                    f"{metric}={format_metric_value(value)}" for metric, value in scores.items()
                )
            else:
                score_text = "-"
            lines.append(
                f"| {escape_table_cell(model_name)} | {escape_table_cell(score_text or '-')} |"
            )

    return "\n".join(lines)


def infer_domain(summary: dict[str, Any]) -> str:
    task_type = str(summary.get("task_type", ""))
    datasets = summary.get("datasets") if isinstance(summary.get("datasets"), list) else []
    metrics = summary.get("evaluation_metrics") if isinstance(summary.get("evaluation_metrics"), list) else []
    metric_names = [str(metric.get("name", "")) for metric in metrics if isinstance(metric, dict)]

    corpus = " ".join([task_type, *map(str, datasets), *metric_names]).lower()

    if re.search(r"(translation|summarization|classification|retrieval|qa|question answering|text|rouge|bleu|bertscore)", corpus):
        return "nlp"
    if re.search(r"(image|vision|segmentation|detection|coco|imagenet|miou|fid|psnr|ssim)", corpus):
        return "cv"
    if re.search(r"(reinforcement|policy|reward|gym|episode|return)", corpus):
        return "rl"
    if re.search(r"(recommend|ranking|ctr|auc|ndcg|hr@|mrr)", corpus):
        return "recommendation"
    if re.search(r"(time series|forecast|forecasting|anomaly|temporal|mae|mase|smape)", corpus):
        return "time-series"
    return "general"


def library_hints_by_domain(domain: str) -> str:
    if domain == "nlp":
        return "evaluate, nltk, sacrebleu"
    if domain == "cv":
        return "torchmetrics, pycocotools"
    if domain == "rl":
        return "gymnasium"
    if domain == "recommendation":
        return "recbole"
    if domain == "time-series":
        return "tsmetric"
    return "numpy, pandas"


def ensure_code_fence(content: str, language: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        return stripped
    return f"```{language}\n{stripped}\n```"


def parse_json_loose(content: str) -> dict[str, Any]:
    stripped = content.strip()
    candidates = [
        stripped,
        re.sub(r"^```json\s*", "", stripped, flags=re.IGNORECASE),
    ]
    candidates = [re.sub(r"\s*```$", "", candidate) for candidate in candidates]

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    first_curly = stripped.find("{")
    first_square = stripped.find("[")
    indexes = [index for index in [first_curly, first_square] if index >= 0]
    if indexes:
        start_index = min(indexes)
        end_index = max(stripped.rfind("}"), stripped.rfind("]"))
        if end_index > start_index:
            return json.loads(stripped[start_index : end_index + 1])

    raise RuntimeError("failed to parse JSON from information extraction response")


def clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n\n[truncated]"


def pick_language(candidates: list[str]) -> str:
    return "ko" if any(re.search(r"[가-힣]", candidate) for candidate in candidates) else "en"


def fallback_text(value: Any, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or fallback


def format_list(values: Any, fallback: str) -> str:
    if not isinstance(values, list):
        return fallback
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    return ", ".join(cleaned) if cleaned else fallback


def format_metric_checklist(metrics: Any, fallback: str) -> str:
    if not isinstance(metrics, list) or not metrics:
        return fallback

    items: list[str] = []
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        name = fallback_text(metric.get("name"), "unnamed metric")
        value = format_metric_value(metric.get("value"))
        items.append(name if value == "-" else f"{name}={value}")
    return ", ".join(items) if items else fallback


def format_baselines(baselines: Any, fallback: str) -> str:
    if not isinstance(baselines, list) or not baselines:
        return fallback

    items: list[str] = []
    for baseline in baselines:
        if not isinstance(baseline, dict):
            continue
        model_name = fallback_text(baseline.get("model_name"), "unnamed baseline")
        scores = baseline.get("scores")
        if isinstance(scores, dict):
            score_text = ", ".join(
                f"{metric}={format_metric_value(value)}" for metric, value in scores.items()
            )
            items.append(f"{model_name} ({score_text})" if score_text else model_name)
        else:
            items.append(model_name)
    return "; ".join(items) if items else fallback


def format_metric_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip() or "-"
    if value is None:
        return "-"
    return str(value)


def escape_table_cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ")


def is_unsupported_model_error(error: Exception) -> bool:
    message = str(error).lower()
    return "model is invalid" in message or "no longer supported" in message


def unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        trimmed = value.strip()
        if not trimmed or trimmed in seen:
            continue
        seen.add(trimmed)
        result.append(trimmed)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation code from experiment markdown using Upstage APIs")
    parser.add_argument(
        "experiment_markdown",
        nargs="?",
        default="-",
        help="Path to extracted experiment markdown. Use '-' to read from stdin.",
    )
    parser.add_argument("--lang", default="python", help="Code language to generate")
    parser.add_argument("--framework", default="auto", help="Framework hint for the generated code")
    parser.add_argument("--include-prompt", action="store_true", help="Include an LLM-as-judge prompt")
    parser.add_argument("--model", default=DEFAULT_SOLAR_MODEL, help="Solar model to use for generation")
    args = parser.parse_args()

    markdown = read_markdown_input(args.experiment_markdown)
    api_key = load_api_key()
    client = build_openai_client(api_key)
    language = pick_language([args.experiment_markdown])

    print("[extract] extracting evaluation summary", file=sys.stderr)
    summary = extract_eval_summary(client, markdown)

    print("[generate] generating evaluation code", file=sys.stderr)
    code_block = generate_code(
        client,
        markdown,
        summary,
        args.lang,
        args.framework,
        args.model,
    )

    judge_prompt = ""
    if args.include_prompt:
        print("[generate] generating llm-as-judge prompt", file=sys.stderr)
        judge_prompt = generate_judge_prompt(client, markdown, summary, args.model)

    print(
        render_output(
            code_block,
            summary,
            args.framework,
            args.include_prompt,
            judge_prompt,
            language,
        )
    )


if __name__ == "__main__":
    main()

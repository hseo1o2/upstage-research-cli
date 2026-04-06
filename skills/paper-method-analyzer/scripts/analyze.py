#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

BASE_URL = "https://api.upstage.ai/v1"
DEFAULT_SOLAR_MODEL = os.getenv("UPSTAGE_SOLAR_MODEL", "solar-pro3")
DEFAULT_IE_MODEL = os.getenv("UPSTAGE_INFORMATION_EXTRACTION_MODEL", "information-extract")
DEFAULT_IE_FALLBACK_MODEL = os.getenv("UPSTAGE_INFORMATION_EXTRACT_FALLBACK_MODEL", DEFAULT_SOLAR_MODEL)
ASYNC_SIZE_THRESHOLD_BYTES = 15 * 1024 * 1024
POLL_INTERVAL_SECONDS = 5
MAX_POLLS = 120
ROOT_DIR = Path(__file__).resolve().parents[3]
SCHEMA_PATH = ROOT_DIR / "skills" / "paper-method-analyzer" / "references" / "method-schema.json"
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


def parse_document(file_path: Path, api_key: str) -> tuple[str, str]:
    import requests

    prefer_async = file_path.stat().st_size >= ASYNC_SIZE_THRESHOLD_BYTES

    if prefer_async:
        try:
            return parse_document_async(file_path, api_key)
        except requests.RequestException:
            print(
                f"[warn] async document parse unavailable for {file_path.name}; retrying sync",
                file=sys.stderr,
            )

    try:
        return parse_document_sync(file_path, api_key)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code in {408, 413, 422, 429, 500, 502, 503, 504}:
            print(
                f"[warn] sync document parse rejected for {file_path.name}; retrying async",
                file=sys.stderr,
            )
            return parse_document_async(file_path, api_key)
        raise


def parse_document_sync(file_path: Path, api_key: str) -> tuple[str, str]:
    import requests

    with file_path.open("rb") as handle:
        response = requests.post(
            f"{BASE_URL}/document-digitization",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"document": (file_path.name, handle, infer_mime_type(file_path))},
            data={"model": "document-parse", "ocr": "auto", "output_formats": "['markdown']"},
            timeout=300,
        )
    response.raise_for_status()
    payload = response.json()
    return extract_markdown(payload), "sync"


def parse_document_async(file_path: Path, api_key: str) -> tuple[str, str]:
    import requests

    with file_path.open("rb") as handle:
        response = requests.post(
            f"{BASE_URL}/document-digitization/async",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"document": (file_path.name, handle, infer_mime_type(file_path))},
            data={"model": "document-parse", "ocr": "auto", "output_formats": "['markdown']"},
            timeout=300,
        )
    response.raise_for_status()
    payload = response.json()
    request_id = payload.get("request_id") or payload.get("id")
    if not request_id:
        return extract_markdown(payload), "async"

    for _ in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL_SECONDS)
        status_response = requests.get(
            f"{BASE_URL}/document-digitization/requests/{request_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )
        status_response.raise_for_status()
        status_payload = status_response.json()
        status = str(status_payload.get("status", "")).lower()
        completed_pages = status_payload.get("completed_pages", 0)
        total_pages = status_payload.get("total_pages", "?")
        print(
            f"[parse] {file_path.name}: {status or 'processing'} ({completed_pages}/{total_pages})",
            file=sys.stderr,
        )

        if status in {"completed", "succeeded", "done"}:
            return materialize_async_parse(status_payload, api_key), "async"
        if status in {"failed", "error", "cancelled"}:
            message = status_payload.get("failure_message") or status_payload.get("message") or "unknown failure"
            raise RuntimeError(f"async document parse failed: {message}")

    raise TimeoutError(f"timed out waiting for async parse of {file_path.name}")


def materialize_async_parse(status_payload: dict[str, Any], api_key: str) -> str:
    import requests

    markdown = try_extract_markdown(status_payload)
    if markdown:
        return markdown

    download_urls: list[str] = []
    top_level_url = status_payload.get("download_url")
    if isinstance(top_level_url, str) and top_level_url.strip():
        download_urls.append(top_level_url.strip())

    batches = status_payload.get("batches")
    if isinstance(batches, list):
        for batch in sorted(batches, key=lambda item: int(item.get("start_page", 0))):
            download_url = batch.get("download_url")
            if isinstance(download_url, str) and download_url.strip():
                download_urls.append(download_url.strip())

    chunks: list[str] = []
    for download_url in download_urls:
        response = requests.get(download_url, timeout=120)
        if response.status_code in {401, 403}:
            response = requests.get(
                download_url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=120,
            )
        response.raise_for_status()

        try:
            payload = response.json()
        except ValueError:
            payload = response.text

        chunk = try_extract_markdown(payload) if isinstance(payload, dict) else str(payload)
        if chunk.strip():
            chunks.append(chunk.strip())

    if not chunks:
        raise RuntimeError("async document parse completed without downloadable markdown")

    return "\n\n".join(chunks)


def extract_method_summary(client: OpenAI, file_name: str, markdown: str) -> dict[str, Any]:
    with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)

    model_candidates = unique_strings([DEFAULT_IE_MODEL, DEFAULT_IE_FALLBACK_MODEL])
    last_error: Exception | None = None

    for index, model_name in enumerate(model_candidates):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Extract methodology metadata from academic paper markdown. Use concise factual text. Return empty strings or empty arrays when evidence is missing.",
                    },
                    {
                        "role": "user",
                        "content": f"File name: {file_name}\n\nPaper markdown:\n{clip_text(select_method_excerpt(markdown), 30000)}",
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "paper_method_summary",
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

    raise RuntimeError("method summary extraction failed")


def generate_recommendations(
    client: OpenAI,
    papers: list[dict[str, Any]],
    context: str,
    language: str,
    model: str,
) -> str:
    if language == "ko":
        user_prompt = (
            f"연구 컨텍스트:\n{context or '제공되지 않음. 필요한 가정을 짧게 밝힌 뒤 일반적인 적용 포인트를 제안하세요.'}\n\n"
            f"논문 요약 JSON:\n{json.dumps(papers, ensure_ascii=False, indent=2)}\n\n"
            "마크다운으로만 답하세요. 아래 두 섹션만 정확히 반환하세요.\n"
            "## 내 연구에 적용 가능한 포인트\n"
            "번호 목록으로 작성하세요.\n\n"
            "## 추천 레퍼런스 우선순위\n"
            "번호 목록으로 작성하고 각 항목마다 우선순위 이유를 한 줄씩 쓰세요."
        )
        system_prompt = "당신은 실험 중심 연구 논문의 방법론을 비교해 사용자의 연구 설계로 연결해 주는 연구 조교다."
    else:
        user_prompt = (
            f"Research context:\n{context or 'Not provided. State missing assumptions briefly, then give generic but actionable suggestions.'}\n\n"
            f"Paper summaries JSON:\n{json.dumps(papers, ensure_ascii=False, indent=2)}\n\n"
            "Return markdown only. Return exactly these two sections:\n"
            "## Application Suggestions\n"
            "Use a numbered list.\n\n"
            "## Reference Priority\n"
            "Use a numbered list and explain the priority in one sentence per item."
        )
        system_prompt = "You are a research assistant who compares experimental papers and turns them into actionable next steps."

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def render_markdown(papers: list[dict[str, Any]], recommendations: str, context: str, language: str) -> str:
    lines: list[str] = []
    if context.strip():
        if language == "ko":
            lines.append(f"> 연구 컨텍스트: {context.strip()}")
        else:
            lines.append(f"> Research context: {context.strip()}")
        lines.append("")

    if language == "ko":
        lines.append("## 방법론 비교표")
        lines.append("| 논문 | 핵심 방법론 | 학습 전략 | 데이터셋 | 주요 기여 | 한계 |")
    else:
        lines.append("## Methodology Comparison")
        lines.append("| Paper | Core methodology | Training strategy | Datasets | Main contribution | Limitations |")
    lines.append("| --- | --- | --- | --- | --- | --- |")

    for paper in papers:
        summary = paper["summary"]
        lines.append(
            "| {title} | {architecture} | {training} | {datasets} | {contribution} | {limitations} |".format(
                title=escape_table_cell(summary.get("title") or paper["file_name"]),
                architecture=escape_table_cell(fallback_text(summary.get("model_architecture"), "-")),
                training=escape_table_cell(fallback_text(summary.get("training_strategy"), "-")),
                datasets=escape_table_cell(format_list(summary.get("datasets"), "-")),
                contribution=escape_table_cell(fallback_text(summary.get("main_contribution"), "-")),
                limitations=escape_table_cell(fallback_text(summary.get("limitations"), "-")),
            )
        )

    lines.append("")
    lines.append("## 각 논문 상세 분석" if language == "ko" else "## Per-Paper Analysis")
    lines.append("")

    for paper in papers:
        summary = paper["summary"]
        lines.append(f"### {summary.get('title') or paper['file_name']}")
        if language == "ko":
            lines.append(f"- 핵심 아이디어: {fallback_text(summary.get('main_contribution'), '추출 실패')}")
            lines.append(f"- 모델 구조: {fallback_text(summary.get('model_architecture'), '추출 실패')}")
            lines.append(f"- 학습 전략: {fallback_text(summary.get('training_strategy'), '추출 실패')}")
            lines.append(f"- 데이터셋: {format_list(summary.get('datasets'), '확인 필요')}")
            lines.append(f"- 평가 지표: {format_list(summary.get('evaluation_metrics'), '확인 필요')}")
            lines.append(f"- 한계: {fallback_text(summary.get('limitations'), '명시되지 않음')}")
            lines.append(f"- 파싱 모드: `{paper['parse_mode']}`")
        else:
            lines.append(f"- Core idea: {fallback_text(summary.get('main_contribution'), 'Not extracted')}")
            lines.append(f"- Model architecture: {fallback_text(summary.get('model_architecture'), 'Not extracted')}")
            lines.append(f"- Training strategy: {fallback_text(summary.get('training_strategy'), 'Not extracted')}")
            lines.append(f"- Datasets: {format_list(summary.get('datasets'), 'Needs review')}")
            lines.append(f"- Evaluation metrics: {format_list(summary.get('evaluation_metrics'), 'Needs review')}")
            lines.append(f"- Limitations: {fallback_text(summary.get('limitations'), 'Not stated')}")
            lines.append(f"- Parse mode: `{paper['parse_mode']}`")
        lines.append("")

    lines.append(recommendations.strip())
    return "\n".join(lines).strip()


def select_method_excerpt(markdown: str) -> str:
    section = extract_section_by_headings(
        markdown,
        start_patterns=[
            r"^#+\s*(abstract|introduction|approach|method|methods|methodology|model|training|experiments|evaluation|limitations?)\b",
            r"^#+\s*(초록|서론|방법|방법론|모델|학습|실험|평가|한계)\b",
        ],
        stop_patterns=[
            r"^#+\s*(references?|appendix|acknowledg(e)?ments?|supplementary)\b",
            r"^#+\s*(참고문헌|부록|감사의 글)\b",
        ],
        char_limit=24000,
    )
    return section or markdown[:24000]


def extract_section_by_headings(
    markdown: str,
    start_patterns: list[str],
    stop_patterns: list[str],
    char_limit: int,
) -> str:
    selected: list[str] = []
    collecting = False
    compiled_start = [re.compile(pattern, re.IGNORECASE) for pattern in start_patterns]
    compiled_stop = [re.compile(pattern, re.IGNORECASE) for pattern in stop_patterns]

    for line in markdown.splitlines():
        if any(pattern.search(line) for pattern in compiled_start):
            collecting = True

        if collecting:
            if selected and any(pattern.search(line) for pattern in compiled_stop):
                break

            selected.append(line)
            if len("\n".join(selected)) >= char_limit:
                break

    return "\n".join(selected).strip()


def extract_markdown(payload: dict[str, Any]) -> str:
    markdown = try_extract_markdown(payload)
    if markdown:
        return markdown
    raise RuntimeError("document parse response did not include markdown")


def try_extract_markdown(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    content = payload.get("content")
    if isinstance(content, dict):
        markdown = content.get("markdown")
        if isinstance(markdown, str):
            return markdown.strip()
    return ""


def infer_mime_type(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if suffix == ".pptx":
        return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    if suffix == ".xlsx":
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if suffix == ".hwp":
        return "application/x-hwp"
    if suffix == ".hwpx":
        return "application/x-hwp+zip"
    return "application/octet-stream"


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


def format_list(values: Any, fallback: str) -> str:
    if not isinstance(values, list):
        return fallback
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    return ", ".join(cleaned) if cleaned else fallback


def fallback_text(value: Any, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or fallback


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
    parser = argparse.ArgumentParser(description="Analyze research paper methodologies with Upstage APIs")
    parser.add_argument("papers", nargs="+", help="Paper PDF paths")
    parser.add_argument("--context", default="", help="User research context")
    parser.add_argument("--model", default=DEFAULT_SOLAR_MODEL, help="Solar model to use for synthesis")
    args = parser.parse_args()

    api_key = load_api_key()
    client = build_openai_client(api_key)
    language = pick_language([args.context, *args.papers])

    processed_papers: list[dict[str, Any]] = []

    for index, paper in enumerate(args.papers, start=1):
        file_path = Path(paper).expanduser().resolve()
        print(f"[{index}/{len(args.papers)}] parsing {file_path.name}", file=sys.stderr)
        markdown, parse_mode = parse_document(file_path, api_key)
        print(f"[{index}/{len(args.papers)}] extracting method summary for {file_path.name}", file=sys.stderr)
        summary = extract_method_summary(client, file_path.name, markdown)
        processed_papers.append(
            {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "parse_mode": parse_mode,
                "summary": summary,
            }
        )

    print("[final] generating application suggestions", file=sys.stderr)
    recommendations = generate_recommendations(client, processed_papers, args.context, language, args.model)
    print(render_markdown(processed_papers, recommendations, args.context, language))


if __name__ == "__main__":
    main()

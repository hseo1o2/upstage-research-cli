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
ASYNC_SIZE_THRESHOLD_BYTES = 15 * 1024 * 1024
POLL_INTERVAL_SECONDS = 5
MAX_POLLS = 120
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


def select_experiment_excerpt(markdown: str) -> str:
    section = extract_section_by_headings(
        markdown,
        start_patterns=[
            r"^#+\s*(experiments?|experimental setup|evaluation|results?|benchmarks?|implementation details?)\b",
            r"^#+\s*(실험|평가|결과|벤치마크|구현)\b",
        ],
        stop_patterns=[
            r"^#+\s*(conclusion|discussion|limitations?|future work|references?|appendix)\b",
            r"^#+\s*(결론|토론|한계|향후 연구|참고문헌|부록)\b",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract experiment sections from a paper using Upstage Document Parse")
    parser.add_argument("paper", help="Paper PDF path")
    parser.add_argument("--output", help="Optional output file path for the extracted markdown")
    args = parser.parse_args()

    file_path = Path(args.paper).expanduser().resolve()
    api_key = load_api_key()

    print(f"[parse] parsing {file_path.name}", file=sys.stderr)
    markdown, parse_mode = parse_document(file_path, api_key)
    excerpt = select_experiment_excerpt(markdown)
    print(f"[parse] selected experiment excerpt via `{parse_mode}` parse mode", file=sys.stderr)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(excerpt, encoding="utf-8")
        print(f"[write] saved experiment markdown to {output_path}", file=sys.stderr)
    else:
        print(excerpt)


if __name__ == "__main__":
    main()

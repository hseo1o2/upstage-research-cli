# upstage-research-cli

`upstage-research-cli` is a bilingual research-paper CLI built on Upstage APIs.
It ships with two bundled Agent Skills:

- `paper-method-analyzer`
- `paper-eval-codegen`

The package is published on npm as [`upstage-research-cli`](https://www.npmjs.com/package/upstage-research-cli).

---

## 한국어 소개

`upstage-research-cli`는 논문 PDF를 바로 연구 실행 흐름으로 연결하기 위한 CLI입니다.

핵심은 단순 요약이 아닙니다.

- 여러 논문을 비교해서 방법론을 정리하고
- 한 논문의 실험 섹션을 읽어 평가 코드를 만들고
- 그 결과를 Agent Skill로 Claude, Codex, Cursor 같은 에이전트 환경에 설치할 수 있게 하는 것

### 포함된 스킬

#### 1. `paper-method-analyzer`

여러 논문 PDF를 비교 분석해서:

- 방법론 비교표
- 논문별 핵심 기여 / 한계 / 데이터셋 / 평가 지표
- 내 연구에 적용 가능한 포인트
- 참고 우선순위

를 생성합니다.

#### 2. `paper-eval-codegen`

한 논문의 실험 섹션을 읽고:

- 실행 가능한 평가 코드
- 재현 체크리스트
- baseline snapshot
- evidence snippet
- optional LLM-as-judge prompt

를 생성합니다.

### 왜 Upstage 기반인가

학술 논문 PDF는 일반 PDF 파서로 깨지기 쉽습니다.

- 2단 레이아웃
- 수식
- 알고리즘 박스
- 압축된 결과 테이블

이 프로젝트는 Upstage Document Parse를 먼저 사용해서, 이런 구조를 최대한 보존한 markdown을 기반으로 후속 추출과 생성을 진행합니다.

### 현재 상태

- `paper-method-analyzer`: 실사용 가능
- `paper-eval-codegen`: 일반 experimental paper 기준 강한 실사용 단계
- `protein design / structure generation` 같이 protocol-heavy metric이 많은 논문은 별도 challenge 영역으로 분리해 추적 중

### 설치

전역 설치:

```bash
npm install -g upstage-research-cli
```

로컬 개발 설치:

```bash
npm install
npm run build
```

### 인증

API 키 우선순위:

1. `UPSTAGE_API_KEY`
2. `~/.config/upstage-research/config.json`

스킬 설치 시 키가 없으면 저장을 유도합니다.

### 빠른 시작

논문 방법론 비교:

```bash
upstage-research analyze-methods paper1.pdf paper2.pdf \
  --context "나는 transformer 기반 시계열 이상탐지 연구를 하고 있다"
```

평가 코드 생성:

```bash
upstage-research eval-codegen paper.pdf \
  --lang python \
  --framework pytorch \
  --include-prompt
```

스킬 설치:

```bash
upstage-research install --skills
```

Claude / Codex / Cursor용으로 함께 설치:

```bash
upstage-research install --skills --all-targets
```

### 주요 기능

- Upstage `Document Parse -> Information Extract -> Solar` 파이프라인
- evidence snippet + provenance 기반 결과 추적
- deterministic evaluation artifact rendering
- metric / dataset / baseline / evidence / protocol verification
- cache, retry, async parse fallback
- regression batch / release gate / challenge set 지원

---

## English Overview

`upstage-research-cli` is a research-paper workflow CLI built on Upstage APIs.
It is designed for papers with real experiment sections, not generic PDF summarization.

The tool turns academic PDFs into:

- grounded methodology comparisons
- reproducible evaluation artifacts
- bundled Agent Skills for coding agents

### Bundled skills

#### 1. `paper-method-analyzer`

Compares multiple paper PDFs and produces:

- a methodology comparison table
- per-paper summaries
- evidence-backed application suggestions
- reading priority across the provided papers

#### 2. `paper-eval-codegen`

Reads one paper's experiment section and produces:

- runnable evaluation code
- a reproduction checklist
- a baseline snapshot
- evidence snippets with provenance
- an optional LLM-as-judge prompt

### Why this tool uses Upstage first

Academic PDFs are structurally hostile to generic parsers.

- two-column layouts
- tables with compressed headers
- equations inside narrative sections
- algorithm boxes and structured figure/table captions

This project starts with Upstage Document Parse so downstream extraction and generation can work from structure-preserving markdown instead of flattened text.

### Current quality position

- `paper-method-analyzer`: production-usable
- `paper-eval-codegen`: strong for mainstream experimental papers
- protein-design / structure-generation papers remain tracked in a separate challenge lane because they often require protocol-heavy evaluators or external scoring tools

---

## Install

Global install:

```bash
npm install -g upstage-research-cli
```

Local build:

```bash
npm install
npm run build
```

CLI binary:

```bash
upstage-research
```

---

## Authentication

The CLI resolves the Upstage API key in this order:

1. `UPSTAGE_API_KEY`
2. `~/.config/upstage-research/config.json`

---

## Commands

### Analyze methodologies

```bash
upstage-research analyze-methods paper1.pdf paper2.pdf \
  --context "I am working on transformer-based time-series anomaly detection" \
  --format markdown \
  --save-report tmp/method-analysis.md
```

Output includes:

- comparison table
- per-paper analysis
- evidence snippets
- application suggestions
- reference priority

Useful options:

- `--format json`
- `--out`
- `--save-report`
- `--cache-only`

### Generate evaluation code

```bash
upstage-research eval-codegen paper.pdf \
  --lang python \
  --framework pytorch \
  --include-prompt \
  --format markdown \
  --save-code tmp/eval.py \
  --save-report tmp/eval.md
```

Output includes:

- runnable evaluation code
- reproduction checklist
- verification report
- reported metrics table
- baseline snapshot
- evidence snippets
- optional judge prompt

Useful options:

- `--format json`
- `--verify-only`
- `--save-code`
- `--out`
- `--save-report`
- `--cache-only`

### Install bundled skills

Install into the current project:

```bash
upstage-research install --skills
```

Install into multiple agent-specific targets:

```bash
upstage-research install --skills --all-targets
```

Or explicitly:

```bash
upstage-research install --skills --targets claude,codex,cursor
```

### Run regression batches

```bash
upstage-research regression-batch tmp/papers \
  --mode both \
  --limit 20 \
  --concurrency 3 \
  --out-dir tmp/regression-batch \
  --resume
```

Fixed fixture example:

```bash
upstage-research regression-batch \
  --manifest fixtures/regression/public-hardening-v10.json \
  --out-dir tmp/regression-public-hardening-ci \
  --assert \
  --cache-only
```

Release-gate vs challenge split:

```bash
upstage-research regression-batch \
  --manifest fixtures/regression/release-gate-main-47.json \
  --out-dir tmp/regression-release-gate-main \
  --assert \
  --cache-only

upstage-research regression-batch \
  --manifest fixtures/regression/bio-challenge-3.json \
  --out-dir tmp/regression-bio-challenge \
  --cache-only
```

### Clear cache

```bash
upstage-research cache-clear document-parse --dry-run
```

---

## Pipeline

### `analyze-methods`

1. Document Parse turns each paper into structured markdown.
2. Information Extract fills the method schema.
3. Solar performs structured synthesis for comparison and adaptation.
4. Local evidence selection grounds the output.
5. Local stabilization rewrites overly broad suggestions into more defensible reuse patterns.

### `eval-codegen`

1. Document Parse extracts experiment-oriented markdown.
2. Information Extract fills the evaluation schema.
3. Solar provides structured guidance.
4. Local metric-family resolution and deterministic rendering generate the Python artifact.
5. Local verification scores fidelity across metric / dataset / baseline / evidence / protocol dimensions.

---

## Technical highlights

- Node.js 18+
- `commander`-based CLI
- native `fetch` client
- official Upstage v2 file-upload plus `responses` flow for IE
- deterministic rendering for common evaluation families
- evidence provenance from parse elements
- local cache under `.upstage-research-cache/`
- retry / timeout / async parse fallback
- resumable regression batches
- registry-driven metric and evidence normalization

---

## Validation

Local static validation:

```bash
npm run check
npm run build
npm run regression:public-hardening:assert
```

Live Upstage validation should be run with a real API key when you want to verify extraction quality against actual PDFs.

---

## Package contents

This npm package contains:

- the CLI runtime
- both bundled skills
- config registries
- regression fixtures
- Python helper scripts used by the skills

So this is one npm package that distributes two Agent Skills plus the CLI that powers them.

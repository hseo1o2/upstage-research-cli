# upstage-research-cli

`upstage-research-cli` is a Node.js CLI for research-paper workflows powered by Upstage APIs. It ships with two Agent Skills that turn paper PDFs into structured methodology analysis or evaluation-code generation.

## What it does

- `analyze-methods`: compares multiple papers and suggests how to apply them to the user's own research.
- `eval-codegen`: extracts experiment details from one paper and generates runnable evaluation code, a reproduction checklist, and an optional LLM-as-judge prompt.
- `install --skills`: saves the API key if needed and installs both skills into `.claude/skills/upstage-research` for the current project.
- `install --skills`: saves the API key if needed and installs the skill bundle into one or more agent-specific project paths such as Claude, Codex, or Cursor.
- `regression-batch`: reruns `eval-codegen` and `analyze-methods` over a paper directory and writes batch summaries for regression tracking.
- `cache-clear`: clears the local parse / extraction / Solar cache when you need to invalidate stale artifacts.

## Why Upstage Document Parse

Academic PDFs are layout-heavy. Two-column pages, equations, algorithm boxes, and dense result tables are where generic PDF parsers collapse. This project uses Upstage Document Parse first so the downstream extraction and generation steps start from structure-preserving markdown instead of flattened text.

## Project layout

```text
upstage-research-cli/
├── config/
│   ├── eval-discovery-registry.json
│   ├── evidence-alias-registry.json
│   └── metric-kind-registry.json
├── fixtures/
│   └── regression/
│       ├── bio-challenge-3.json
│       ├── public-hardening-v10.json
│       ├── public-hardening-v10.eval.tsv
│       ├── public-hardening-v10.analyze.tsv
│       └── release-gate-main-47.json
├── skills/
│   ├── paper-method-analyzer/
│   │   ├── SKILL.md
│   │   ├── scripts/
│   │   │   └── analyze.py
│   │   └── references/
│   │       └── method-schema.json
│   └── paper-eval-codegen/
│       ├── SKILL.md
│       ├── scripts/
│       │   ├── parse_experiments.py
│       │   └── generate_eval.py
│       └── references/
│           ├── eval-schema.json
│           ├── metrics-by-domain.md
│           └── prompt-templates.md
├── src/
│   ├── cli.ts
│   ├── client.ts
│   ├── config.ts
│   └── commands/
│       ├── analyze-methods.ts
│       ├── eval-codegen.ts
│       └── install.ts
├── package.json
├── tsconfig.json
└── README.md
```

## Install

```bash
npm install
npm run build
```

The CLI binary is:

```bash
upstage-research
```

If you install globally from the package output, the `bin` entry points to `dist/cli.js`.

## Authentication

The CLI resolves the API key in this order:

1. `UPSTAGE_API_KEY`
2. `~/.config/upstage-research/config.json`

`upstage-research install --skills` prompts for the key only when the environment variable is missing, then saves it to the config file.

## Commands

### Analyze methodologies

```bash
upstage-research analyze-methods paper1.pdf paper2.pdf \
  --context "I am working on transformer-based time-series anomaly detection" \
  --format markdown \
  --save-report tmp/method-analysis.md
```

Stdout is markdown with:

- A comparison table
- Per-paper analysis
- Evidence snippets pulled from the parsed paper text, with stable snippet ids plus page/section/anchor provenance when available
- Application suggestions
- Reference priority

Useful options:

- `--format json` for machine-readable output
- `--out` or `--save-report` to persist the rendered report
- `--cache-only` to fail on cache misses instead of calling Upstage APIs

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

Stdout is markdown with:

- Runnable evaluation code
- A reproduction checklist
- A verification report that scores metric coverage, evidence grounding, and protocol risk
- A reported-metrics table
- A baseline snapshot
- Evidence snippets from the experiment section, with stable snippet ids plus page/section/anchor provenance when available
- An optional LLM-as-judge prompt
 - Verification subscores for metric, dataset, baseline, evidence, and protocol fidelity

Useful options:

- `--format json` for machine-readable output
- `--verify-only` to emit only verification plus evidence instead of the full report
- `--save-code` to persist the generated evaluation code
- `--out` or `--save-report` to persist the rendered report
- `--cache-only` to fail on cache misses instead of calling Upstage APIs

Both commands support `--format json`. The JSON payload includes the structured summary, evidence records with provenance, verification object, and rendered markdown.

### Install skills into a project

```bash
upstage-research install --skills
```

This copies the bundled skills to:

```text
.claude/skills/upstage-research/
```

To install into multiple agent-specific project paths:

```bash
upstage-research install --skills --all-targets
```

Or choose explicit targets:

```bash
upstage-research install --skills --targets claude,codex,cursor
```

You can also preview the install without copying:

```bash
upstage-research install --skills --all-targets --dry-run
```

### Run a regression batch

```bash
upstage-research regression-batch tmp/papers \
  --mode both \
  --limit 20 \
  --concurrency 3 \
  --out-dir tmp/regression-batch \
  --resume
```

This writes:

- `eval/summary.tsv` with compile/run/manual-review/verification counts plus fidelity subscores
- `analyze/summary.tsv` with application-point counts, evidence-ref counts, and anchored-evidence counts
- Per-paper markdown/python artifacts and per-batch methodology reports
- Per-paper / per-batch JSON artifacts that can be reused with `--resume`

For a fixed fixture set with golden assertions:

```bash
upstage-research regression-batch \
  --manifest fixtures/regression/public-hardening-v10.json \
  --out-dir tmp/regression-public-hardening-ci \
  --assert \
  --cache-only
```

This also writes `summary.json` for CI parsing and fails with a non-zero exit code if thresholds or golden snapshots do not match.

To separate release readiness from hard protein-geometry generalization:

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

`release-gate-main-47` excludes the three bad corpus slots that turned out not to be target biology papers.
`bio-challenge-3` keeps real protein-design and structure-generation papers in a separate challenge set so they improve long-term generalization without distorting launch gates.

### Clear cache

```bash
upstage-research cache-clear document-parse --dry-run
```

Omit the namespace to clear the entire local cache directory.

## API pipeline

### `analyze-methods`

1. Document Parse converts each paper PDF into structured markdown.
2. Information Extract fills `skills/paper-method-analyzer/references/method-schema.json`.
3. Solar-backed structured synthesis generates application suggestions, per-paper relevance notes, and reading priority limited to the provided papers.
4. Local evidence-snippet selection surfaces supporting passages from the parsed markdown.
5. Local stabilization rewrites overly broad adaptation text into dataset/metric/training-grounded reuse patterns for mixed-domain batches.
6. If structured synthesis fails, the CLI falls back to heuristic comparison and priority generation instead of returning nothing.

### `eval-codegen`

1. Document Parse extracts the experiment-oriented markdown.
2. Information Extract fills `skills/paper-eval-codegen/references/eval-schema.json`.
3. Solar-backed structured guidance and metric-family resolution feed a deterministic Python evaluation artifact generator and an optional judge prompt renderer.
4. A local semantic verifier scores metric-family coverage, evidence grounding, and baseline extraction quality.
5. The verifier also emits metric / dataset / baseline / evidence / protocol subscores for release gating and regression tracking.
6. Local evidence-snippet selection exposes supporting experiment passages with page / section / anchor provenance, and repeated API calls are cached.

## Implementation notes

- Node.js 18+ compatible
- Uses `commander` for the CLI
- Uses native `fetch` in the TypeScript client
- Uses the Python `openai` SDK in skill scripts for Information Extract and Solar calls
- Retries Document Parse with the async endpoint for larger files or sync failures
- Sends progress logs to stderr so stdout stays clean markdown
- Uses the official Upstage v2 file-upload plus `responses` flow for Information Extract
- Caches Document Parse, Information Extract, and Solar outputs in `.upstage-research-cache/` by default
- Supports `--cache-only` so regression or smoke runs can prove they never leave the local cache
- Includes request timeout and transient retry handling for parse / extract / Solar calls
- Deduplicates in-flight API requests inside the same process so concurrent batches do not refetch identical work
- Supports resumable regression batches by reusing existing JSON artifacts with `--resume`
- Uses bounded concurrency for multi-paper methodology analysis; tune with `UPSTAGE_RESEARCH_CONCURRENCY`
- Externalizes task/domain discovery and evidence alias registries into `config/*.json` so new paper families can be tuned without patching core TypeScript logic
- Externalizes most name-driven metric-family mapping rules into `config/metric-kind-registry.json`, leaving only formula-sensitive or task-sensitive edge cases in code
- Attaches parse-anchor provenance from Document Parse raw elements to evidence records, including page, section, category, element id, and normalized coordinates when available
- Uses deterministic rendering for the main Python evaluation artifact so generated code is less likely to contain undefined helpers or irrelevant generic metrics
- Includes broader built-in coverage for common experimental metric families such as AP/mAP, ranking and retrieval metrics, forecasting metrics, calibration metrics, pass@k, WER/CER, PSNR/SSIM, correlation metrics, edit similarity, normalized scores, entropy-style RL metrics, and protein-sequence recovery
- Includes approximation families for paper-shaped metrics such as `ITER`, `RUSE`, `YiSi-1`, and DQN-style maximum state-value monitoring when the extracted formula supports them
- Uses Solar conservatively to re-map unresolved metrics onto safe built-in metric families when the paper evidence is strong enough
- Keeps paper-specific metrics explicit when the paper defines a protocol that cannot be recovered safely from text alone

## Skill scripts

The bundled skill scripts are useful when an agent wants to debug one stage directly:

```bash
python3 skills/paper-method-analyzer/scripts/analyze.py paper1.pdf paper2.pdf
python3 skills/paper-eval-codegen/scripts/parse_experiments.py paper.pdf --output experiment.md
python3 skills/paper-eval-codegen/scripts/generate_eval.py experiment.md --include-prompt
```

Python script dependencies:

```bash
pip install openai requests
```

## Validation status

Static validation is available locally with:

```bash
npm run check
npm run build
npm run regression:public-hardening:assert
```

Live Upstage validation should be run with a real API key when you want to verify extraction quality against actual PDFs.

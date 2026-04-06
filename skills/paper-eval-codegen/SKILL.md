---
name: paper-eval-codegen
description: >-
  Use this skill aggressively when the user asks for 평가 코드, evaluation code, 논문 재현, reproduce,
  평가지표, metric, benchmark, 실험 설정, or experimental setup, especially when a paper PDF and code
  generation are mentioned together. Trigger it before hand-writing eval scaffolding because it extracts paper
  metrics, baselines, datasets, and implementation details from the experiment section, then generates runnable
  evaluation code plus an optional LLM-as-judge prompt.
compatibility:
  tools:
    - bash
---

# Overview

`paper-eval-codegen` reads a paper's experiment section and turns it into research reproduction artifacts:

1. Document Parse preserves the experiment section and result tables from the source PDF.
2. Information Extract pulls structured metrics, datasets, baselines, and implementation details.
3. Solar-backed structured guidance and metric-family resolution feed a deterministic Python code template, and the command optionally renders an LLM-as-judge prompt from the same extracted evidence.

# When to use

Use this skill when:

- The user wants evaluation code or eval code from a paper.
- The goal is 논문 재현 or benchmark reproduction.
- The user asks how to implement the paper's metrics or experimental setup.
- A paper PDF is present and the request includes code generation or metric implementation.

Do not use this skill for unrelated training scripts or paper summarization without reproduction intent.

# Step-by-step instructions

1. Confirm the target paper PDF exists locally.
2. Run the CLI command:

```bash
upstage-research eval-codegen paper.pdf --lang python --framework pytorch --include-prompt
```

3. Omit `--include-prompt` if the user only wants runnable evaluation code and checklist output.
4. Review the generated markdown and call out any paper ambiguities instead of silently inventing hidden assumptions.

The CLI path is the highest-quality path. The bundled Python helper scripts are useful for lower-level debugging, but the main `upstage-research eval-codegen` command applies the strongest post-processing and output shaping.

For lower-level debugging, the fallback script pair is:

```bash
python3 skills/paper-eval-codegen/scripts/parse_experiments.py paper.pdf
python3 skills/paper-eval-codegen/scripts/generate_eval.py experiment.md --lang python --framework pytorch --include-prompt
```

# Output format

The command returns markdown with:

- A runnable code block
- A reproduction checklist
- A reported-metrics table
- A baseline snapshot
- Evidence snippets from the parsed experiment section, with page/section/anchor provenance when available
- An optional LLM-as-judge prompt
- Verification subscores for metric, dataset, baseline, evidence, and protocol fidelity
- For Python output, the code is template-shaped to avoid undefined helpers and other free-form generation drift
- The CLI caches repeated parse / extract / Solar calls locally so iteration is faster on the same paper

Progress logs go to stderr. Stdout stays markdown-only.

# Edge cases

- Metrics missing from the paper: the tool keeps the gap explicit and uses research-area hints only as a soft fallback.
- Custom metrics: common families such as AP/mAP, ranking and retrieval metrics, pass@k, calibration metrics, correlations, WER/CER, PSNR/SSIM, edit similarity, normalized scores, entropy-style RL metrics, and amino-acid recovery are handled directly. Solar also tries to map unresolved metrics onto these safe families when the evidence is strong. Truly paper-specific protocols still stay explicit instead of being hallucinated.
- Multiple result tables: the experiment-section extraction prefers sections labeled experiments, evaluation, results, benchmark, or implementation details.
- Large PDFs: Document Parse retries with the async endpoint automatically when sync parsing is too slow or rejected.

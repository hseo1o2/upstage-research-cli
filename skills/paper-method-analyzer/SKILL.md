---
name: paper-method-analyzer
description: >-
  Use this skill aggressively when the user asks for 선행연구 분석, 논문 비교, 방법론 정리, related work,
  literature review, 레퍼 찾기, or mentions multiple PDFs that are clearly research papers. Trigger it before
  doing ad-hoc summarization when the goal is to compare methods, extract datasets and evaluation metrics, or
  connect papers back to the user's own research direction. This skill is especially useful when academic PDFs
  contain equations, tables, or algorithm boxes that normal PDF parsers destroy.
compatibility:
  tools:
    - bash
---

# Overview

`paper-method-analyzer` compares one or more research paper PDFs with an Upstage pipeline:

1. Document Parse converts each PDF into structured markdown that preserves paper layout better than generic parsers.
2. Information Extract pulls the method schema fields needed for cross-paper comparison.
3. Solar-backed structured synthesis turns those summaries into application suggestions, per-paper relevance notes, and reading priority limited to the provided PDFs.
4. If the synthesis step fails, the CLI still emits a heuristic comparison with evidence snippets and reference priority instead of dropping the analysis.

# When to use

Use this skill when:

- The user wants 선행연구 분석, 논문 비교, 방법론 정리, related work, or literature review help.
- Multiple PDFs are mentioned and they are clearly papers.
- The user asks which references to read first or how to adapt prior work to their own topic.
- The papers contain tables, equations, or algorithm blocks that should not be flattened by a generic PDF parser.

Do not reach for this skill when the task is only bibliography formatting or citation style cleanup.

# Step-by-step instructions

1. Confirm the target PDFs exist locally.
2. Run the CLI command:

```bash
upstage-research analyze-methods paper1.pdf paper2.pdf --context "my research topic"
```

3. If the user has no explicit research context, omit `--context` and let the tool produce generic application ideas.
4. Review the markdown output and pass it through unchanged unless the user asked for an extra summary layer.

For lower-level debugging, the bundled fallback script is:

```bash
python3 skills/paper-method-analyzer/scripts/analyze.py paper1.pdf paper2.pdf --context "my research topic"
```

# Output format

The command returns markdown with:

- A methodology comparison table
- Per-paper analysis sections
- Evidence snippets tied to the extracted methodology fields, with page/section/anchor provenance when available
- Application suggestions tied to the user's research context
- Reference reading priority restricted to the input papers only

Progress logs go to stderr. Stdout stays markdown-only so coding agents can parse it cleanly.

# Edge cases

- Single paper: still use the skill, but treat it as a deep method breakdown plus application guidance.
- Korean and English papers mixed: the tool keeps paper titles as-is and chooses Korean output when the prompt context contains Korean text.
- Missing fields: Information Extract returns empty strings or arrays, and the markdown marks those spots as review-needed.
- Large PDFs: Document Parse retries with the async endpoint automatically when sync parsing is too slow or rejected.
- Repeated runs on the same files: the CLI caches parse and extraction results locally, so follow-up refinement passes are faster.

# Prompt Templates

## Generic LLM-as-Judge
You are an expert evaluator for the target task. Score the candidate output using the rubric below. Return a short rationale before the final score.

## Scoring Format
- Score each criterion on a 1-5 scale.
- Penalize hallucinations, unsupported claims, or malformed outputs.
- If a required field is missing, explain the failure explicitly.

## Suggested Criteria
- Task correctness
- Coverage of required information
- Faithfulness to the input or reference
- Output format compliance
- Safety or error handling, when relevant

## Reproduction-Oriented Notes
- Reuse the paper's metric names and experimental constraints when available.
- If the paper uses human preference or qualitative judgment, convert that into concrete rubric items.
- Include explicit failure conditions so another model can judge consistently.

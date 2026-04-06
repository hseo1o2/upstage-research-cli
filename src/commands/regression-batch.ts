import { promises as fs, readFileSync } from 'node:fs';
import path from 'node:path';
import { execFile as execFileCallback } from 'node:child_process';
import { promisify } from 'node:util';

import { mapWithConcurrency } from '../parallel';

const execFile = promisify(execFileCallback);

export interface RegressionBatchOptions {
  mode?: 'eval' | 'analyze' | 'both';
  concurrency?: string;
  limit?: string;
  outDir?: string;
  context?: string;
  manifest?: string;
  assert?: boolean;
  cacheOnly?: boolean;
  resume?: boolean;
}

interface EvalBatchResult {
  paper: string;
  compile: 'ok' | 'fail';
  run: 'ok' | 'fail';
  manualReviewItems: number;
  todoMentions: number;
  verificationScore: string;
  metricFidelity: string;
  datasetFidelity: string;
  baselineFidelity: string;
  evidenceFidelity: string;
  protocolFidelity: string;
  resumed: 'yes' | 'no';
  error: string;
}

interface AnalyzeBatchResult {
  batchName: string;
  paperCount: number;
  applicationPoints: number;
  evidenceRefs: number;
  anchoredEvidenceRefs: number;
  resumed: 'yes' | 'no';
  error: string;
}

interface EvalCommandPayload {
  markdown: string;
  evaluation_code: string;
  verification?: {
    score?: number;
    checks?: unknown[];
    subscores?: Record<string, { ratio?: number }>;
  };
}

interface AnalyzeCommandPayload {
  markdown: string;
  papers?: Array<{
    evidenceRecords?: Array<{ page?: number; elementId?: string }>;
  }>;
}

interface RegressionManifest {
  name?: string;
  paperDir?: string;
  papers?: string[];
  mode?: 'eval' | 'analyze' | 'both';
  context?: string;
  golden?: {
    evalSummary?: string;
    analyzeSummary?: string;
  };
  thresholds?: {
    eval?: {
      minCompileOk?: number;
      minRunOk?: number;
      maxManualReviewItems?: number;
      maxTodoMentions?: number;
      minVerificationScore?: string;
    };
    analyze?: {
      expectedBatches?: number;
      minApplicationPointsPerBatch?: number;
      minEvidenceRefsPerBatch?: number;
    };
  };
}

export async function regressionBatchCommand(
  paperDir: string | undefined,
  options: RegressionBatchOptions,
): Promise<void> {
  const projectRoot = resolveProjectRoot();
  const manifest = await loadRegressionManifest(options.manifest, projectRoot);
  const mode = options.mode ?? manifest?.mode ?? 'both';
  const concurrency = resolveBatchConcurrency(options.concurrency);
  const resolvedPaperDir = resolvePaperDir(paperDir, manifest, projectRoot);
  const papers = await listPdfPapers(resolvedPaperDir, options.limit, manifest?.papers);
  if (papers.length === 0) {
    throw new Error('No PDF papers found for regression batch.');
  }

  const outputDir = path.resolve(
    options.outDir?.trim()
      || (manifest?.name ? path.resolve(process.cwd(), 'tmp', manifest.name) : path.resolve(process.cwd(), 'tmp', 'regression-batch')),
  );
  await fs.mkdir(outputDir, { recursive: true });
  const shouldAssert = Boolean(options.assert);

  let evalResults: EvalBatchResult[] = [];
  if (mode === 'eval' || mode === 'both') {
    const evalDir = path.join(outputDir, 'eval');
    await fs.mkdir(evalDir, { recursive: true });
    evalResults = await mapWithConcurrency(papers, concurrency, async (paperPath) =>
      await runEvalRegression(paperPath, evalDir, Boolean(options.cacheOnly), Boolean(options.resume)),
    );
    await fs.writeFile(path.join(evalDir, 'summary.tsv'), buildEvalSummaryTsv(evalResults), 'utf8');
    console.error(`Wrote eval regression summary to ${path.join(evalDir, 'summary.tsv')}`);
  }

  let analyzeResults: AnalyzeBatchResult[] = [];
  if (mode === 'analyze' || mode === 'both') {
    const analyzeDir = path.join(outputDir, 'analyze');
    await fs.mkdir(analyzeDir, { recursive: true });
    analyzeResults = await runAnalyzeRegressionBatches(
      papers,
      analyzeDir,
      options.context?.trim()
        || manifest?.context
        || 'I want to compare methodology patterns across experimental papers and identify reusable ideas without overfitting to one domain.',
      Boolean(options.cacheOnly),
      Boolean(options.resume),
    );
    await fs.writeFile(path.join(analyzeDir, 'summary.tsv'), buildAnalyzeSummaryTsv(analyzeResults), 'utf8');
    console.error(`Wrote analyze regression summary to ${path.join(analyzeDir, 'summary.tsv')}`);
  }

  await fs.writeFile(
    path.join(outputDir, 'summary.json'),
    JSON.stringify(
      {
        manifest: manifest?.name ?? null,
        paperDir: resolvedPaperDir,
        paperCount: papers.length,
        eval: summarizeEvalResults(evalResults),
        analyze: summarizeAnalyzeResults(analyzeResults),
      },
      null,
      2,
    ),
    'utf8',
  );

  if (shouldAssert) {
    assertRegressionOutputs({ manifest, evalResults, analyzeResults, outputDir, projectRoot, mode });
  }
}

async function listPdfPapers(
  paperDir: string,
  limit?: string,
  explicitPapers?: string[],
): Promise<string[]> {
  const resolvedDir = path.resolve(paperDir);
  const maxItems = limit?.trim() ? Math.max(1, Number.parseInt(limit, 10)) : Number.POSITIVE_INFINITY;

  if (explicitPapers && explicitPapers.length > 0) {
    return explicitPapers
      .map((paper) => path.join(resolvedDir, paper))
      .slice(0, maxItems);
  }

  const entries = await fs.readdir(resolvedDir, { withFileTypes: true });
  return entries
    .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith('.pdf'))
    .map((entry) => path.join(resolvedDir, entry.name))
    .sort((left, right) => path.basename(left).localeCompare(path.basename(right)))
    .slice(0, maxItems);
}

async function runEvalRegression(
  paperPath: string,
  outDir: string,
  cacheOnly: boolean,
  resume: boolean,
): Promise<EvalBatchResult> {
  const paper = path.basename(paperPath, '.pdf');
  const jsonPath = path.join(outDir, `${paper}-eval.json`);
  const markdownPath = path.join(outDir, `${paper}-eval.md`);
  const pythonPath = path.join(outDir, `${paper}-eval.py`);
  const reused = resume && await exists(jsonPath);
  try {
    const payload = reused
      ? await readJsonFile<EvalCommandPayload>(jsonPath)
      : await runEvalCommand(paperPath, cacheOnly);

    if (!reused) {
      await fs.writeFile(jsonPath, JSON.stringify(payload, null, 2), 'utf8');
    }

    const markdown = payload.markdown ?? '';
    const pythonCode = (payload.evaluation_code ?? '').trim();
    await fs.writeFile(markdownPath, `${markdown.trimEnd()}\n`, 'utf8');
    await fs.writeFile(pythonPath, `${pythonCode.trimEnd()}\n`, 'utf8');

    const compile = await runProcess('python3', ['-m', 'py_compile', pythonPath]).then(() => 'ok' as const).catch(() => 'fail' as const);
    const run = await runProcess('python3', [pythonPath]).then(() => 'ok' as const).catch(() => 'fail' as const);

    return {
      paper,
      compile,
      run,
      manualReviewItems: countMatches(markdown, /^- \[ \] Manual review metric:/gm),
      todoMentions: countMatches(markdown, /TODO:/g),
      verificationScore: buildVerificationScore(payload.verification),
      metricFidelity: formatSubscoreRatio(payload.verification?.subscores?.metric_fidelity?.ratio),
      datasetFidelity: formatSubscoreRatio(payload.verification?.subscores?.dataset_fidelity?.ratio),
      baselineFidelity: formatSubscoreRatio(payload.verification?.subscores?.baseline_fidelity?.ratio),
      evidenceFidelity: formatSubscoreRatio(payload.verification?.subscores?.evidence_fidelity?.ratio),
      protocolFidelity: formatSubscoreRatio(payload.verification?.subscores?.protocol_fidelity?.ratio),
      resumed: reused ? 'yes' : 'no',
      error: '',
    };
  } catch (error) {
    return {
      paper,
      compile: 'fail',
      run: 'fail',
      manualReviewItems: 0,
      todoMentions: 0,
      verificationScore: '-',
      metricFidelity: '-',
      datasetFidelity: '-',
      baselineFidelity: '-',
      evidenceFidelity: '-',
      protocolFidelity: '-',
      resumed: reused ? 'yes' : 'no',
      error: compactErrorMessage(error),
    };
  }
}

async function runAnalyzeRegressionBatches(
  papers: string[],
  outDir: string,
  context: string,
  cacheOnly: boolean,
  resume: boolean,
): Promise<AnalyzeBatchResult[]> {
  const batches = chunkArray(papers, 5);
  const results: AnalyzeBatchResult[] = [];

  for (let index = 0; index < batches.length; index += 1) {
    const batch = batches[index];
    const batchName = `analyze-batch-${index + 1}`;
    const markdownPath = path.join(outDir, `${batchName}.md`);
    const jsonPath = path.join(outDir, `${batchName}.json`);
    const reused = resume && await exists(jsonPath);

    try {
      const payload = reused
        ? await readJsonFile<AnalyzeCommandPayload>(jsonPath)
        : await runAnalyzeCommand(batch, context, cacheOnly);

      if (!reused) {
        await fs.writeFile(jsonPath, JSON.stringify(payload, null, 2), 'utf8');
      }

      const markdown = payload.markdown ?? '';
      await fs.writeFile(markdownPath, `${markdown.trimEnd()}\n`, 'utf8');

      const anchoredEvidenceRefs = (payload.papers ?? [])
        .flatMap((paper) => paper.evidenceRecords ?? [])
        .filter((record) => record.page !== undefined && Boolean(record.elementId))
        .length;

      results.push({
        batchName,
        paperCount: batch.length,
        applicationPoints: countMatches(markdown, /^[0-9]+\. /gm),
        evidenceRefs: countMatches(markdown, /Evidence refs:|근거 참조:/g),
        anchoredEvidenceRefs,
        resumed: reused ? 'yes' : 'no',
        error: '',
      });
    } catch (error) {
      results.push({
        batchName,
        paperCount: batch.length,
        applicationPoints: 0,
        evidenceRefs: 0,
        anchoredEvidenceRefs: 0,
        resumed: reused ? 'yes' : 'no',
        error: compactErrorMessage(error),
      });
    }
  }

  return results;
}

async function loadRegressionManifest(
  manifestPath: string | undefined,
  projectRoot: string,
): Promise<RegressionManifest | null> {
  if (!manifestPath?.trim()) {
    return null;
  }

  const resolvedPath = path.resolve(manifestPath.trim());
  const raw = await fs.readFile(resolvedPath, 'utf8');
  const manifest = JSON.parse(raw) as RegressionManifest;
  if (manifest.paperDir && !path.isAbsolute(manifest.paperDir)) {
    manifest.paperDir = path.resolve(projectRoot, manifest.paperDir);
  }
  if (manifest.golden?.evalSummary && !path.isAbsolute(manifest.golden.evalSummary)) {
    manifest.golden.evalSummary = path.resolve(projectRoot, manifest.golden.evalSummary);
  }
  if (manifest.golden?.analyzeSummary && !path.isAbsolute(manifest.golden.analyzeSummary)) {
    manifest.golden.analyzeSummary = path.resolve(projectRoot, manifest.golden.analyzeSummary);
  }

  return manifest;
}

function resolvePaperDir(
  paperDir: string | undefined,
  manifest: RegressionManifest | null,
  projectRoot: string,
): string {
  const selected = paperDir?.trim() || manifest?.paperDir?.trim();
  if (!selected) {
    throw new Error('Provide <paperDir> or --manifest with paperDir.');
  }

  return path.isAbsolute(selected) ? selected : path.resolve(projectRoot, selected);
}

function resolveProjectRoot(): string {
  return path.resolve(__dirname, '..', '..');
}

function buildEvalSummaryTsv(results: EvalBatchResult[]): string {
  return [
    'paper\tcompile\trun\tmanual_review_items\ttodo_mentions\tverification_score\tmetric_fidelity\tdataset_fidelity\tbaseline_fidelity\tevidence_fidelity\tprotocol_fidelity\tresumed\terror',
    ...results.map((result) =>
      [
        result.paper,
        result.compile,
        result.run,
        String(result.manualReviewItems),
        String(result.todoMentions),
        result.verificationScore,
        result.metricFidelity,
        result.datasetFidelity,
        result.baselineFidelity,
        result.evidenceFidelity,
        result.protocolFidelity,
        result.resumed,
        sanitizeTsv(result.error),
      ].join('\t')),
  ].join('\n');
}

function buildAnalyzeSummaryTsv(results: AnalyzeBatchResult[]): string {
  return [
    'batch\tpapers\tapplication_points\tevidence_refs\tanchored_evidence_refs\tresumed\terror',
    ...results.map((result) =>
      [
        result.batchName,
        String(result.paperCount),
        String(result.applicationPoints),
        String(result.evidenceRefs),
        String(result.anchoredEvidenceRefs),
        result.resumed,
        sanitizeTsv(result.error),
      ].join('\t')),
  ].join('\n');
}

function summarizeEvalResults(results: EvalBatchResult[]): Record<string, number> | null {
  if (results.length === 0) {
    return null;
  }

  return {
    papers: results.length,
    compile_ok: results.filter((result) => result.compile === 'ok').length,
    run_ok: results.filter((result) => result.run === 'ok').length,
    manual_review_items: results.reduce((total, result) => total + result.manualReviewItems, 0),
    todo_mentions: results.reduce((total, result) => total + result.todoMentions, 0),
    verification_4_4: results.filter((result) => result.verificationScore === '4/4').length,
    errors: results.filter((result) => result.error).length,
    avg_metric_fidelity: averageNumericString(results.map((result) => result.metricFidelity)),
    avg_dataset_fidelity: averageNumericString(results.map((result) => result.datasetFidelity)),
    avg_baseline_fidelity: averageNumericString(results.map((result) => result.baselineFidelity)),
    avg_evidence_fidelity: averageNumericString(results.map((result) => result.evidenceFidelity)),
    avg_protocol_fidelity: averageNumericString(results.map((result) => result.protocolFidelity)),
  };
}

function summarizeAnalyzeResults(results: AnalyzeBatchResult[]): Record<string, number> | null {
  if (results.length === 0) {
    return null;
  }

  return {
    batches: results.length,
    papers: results.reduce((total, result) => total + result.paperCount, 0),
    application_points: results.reduce((total, result) => total + result.applicationPoints, 0),
    evidence_refs: results.reduce((total, result) => total + result.evidenceRefs, 0),
    anchored_evidence_refs: results.reduce((total, result) => total + result.anchoredEvidenceRefs, 0),
    errors: results.filter((result) => result.error).length,
  };
}

function assertRegressionOutputs(input: {
  manifest: RegressionManifest | null;
  evalResults: EvalBatchResult[];
  analyzeResults: AnalyzeBatchResult[];
  outputDir: string;
  projectRoot: string;
  mode: 'eval' | 'analyze' | 'both';
}): void {
  const { manifest, evalResults, analyzeResults, outputDir, mode } = input;
  if (!manifest) {
    throw new Error('Assertions require --manifest so the command knows which golden summaries and thresholds to enforce.');
  }

  const failures: string[] = [];
  if ((mode === 'eval' || mode === 'both') && evalResults.length > 0) {
    assertEvalThresholds(evalResults, manifest, failures);
    compareGoldenSummary(path.join(outputDir, 'eval', 'summary.tsv'), manifest.golden?.evalSummary, 'eval', failures);
  }

  if ((mode === 'analyze' || mode === 'both') && analyzeResults.length > 0) {
    assertAnalyzeThresholds(analyzeResults, manifest, failures);
    compareGoldenSummary(path.join(outputDir, 'analyze', 'summary.tsv'), manifest.golden?.analyzeSummary, 'analyze', failures);
  }

  if (failures.length > 0) {
    throw new Error(`Regression assertions failed:\n- ${failures.join('\n- ')}`);
  }
}

function assertEvalThresholds(
  results: EvalBatchResult[],
  manifest: RegressionManifest,
  failures: string[],
): void {
  const thresholds = manifest.thresholds?.eval;
  if (!thresholds) {
    return;
  }

  const compileOk = results.filter((result) => result.compile === 'ok').length;
  const runOk = results.filter((result) => result.run === 'ok').length;
  const manualReviewItems = results.reduce((total, result) => total + result.manualReviewItems, 0);
  const todoMentions = results.reduce((total, result) => total + result.todoMentions, 0);
  const minVerificationScore = thresholds.minVerificationScore;

  if (thresholds.minCompileOk !== undefined && compileOk < thresholds.minCompileOk) {
    failures.push(`compile_ok ${compileOk} < required ${thresholds.minCompileOk}`);
  }
  if (thresholds.minRunOk !== undefined && runOk < thresholds.minRunOk) {
    failures.push(`run_ok ${runOk} < required ${thresholds.minRunOk}`);
  }
  if (thresholds.maxManualReviewItems !== undefined && manualReviewItems > thresholds.maxManualReviewItems) {
    failures.push(`manual_review_items ${manualReviewItems} > allowed ${thresholds.maxManualReviewItems}`);
  }
  if (thresholds.maxTodoMentions !== undefined && todoMentions > thresholds.maxTodoMentions) {
    failures.push(`todo_mentions ${todoMentions} > allowed ${thresholds.maxTodoMentions}`);
  }
  if (minVerificationScore) {
    const failingPapers = results
      .filter((result) => compareVerificationScores(result.verificationScore, minVerificationScore) < 0)
      .map((result) => `${result.paper}=${result.verificationScore}`);
    if (failingPapers.length > 0) {
      failures.push(`verification_score below ${minVerificationScore}: ${failingPapers.join(', ')}`);
    }
  }
}

function assertAnalyzeThresholds(
  results: AnalyzeBatchResult[],
  manifest: RegressionManifest,
  failures: string[],
): void {
  const thresholds = manifest.thresholds?.analyze;
  if (!thresholds) {
    return;
  }

  if (thresholds.expectedBatches !== undefined && results.length !== thresholds.expectedBatches) {
    failures.push(`analyze batch count ${results.length} != expected ${thresholds.expectedBatches}`);
  }
  if (thresholds.minApplicationPointsPerBatch !== undefined) {
    const failing = results
      .filter((result) => result.applicationPoints < thresholds.minApplicationPointsPerBatch!)
      .map((result) => `${result.batchName}=${result.applicationPoints}`);
    if (failing.length > 0) {
      failures.push(`application_points below ${thresholds.minApplicationPointsPerBatch}: ${failing.join(', ')}`);
    }
  }
  if (thresholds.minEvidenceRefsPerBatch !== undefined) {
    const failing = results
      .filter((result) => result.evidenceRefs < thresholds.minEvidenceRefsPerBatch!)
      .map((result) => `${result.batchName}=${result.evidenceRefs}`);
    if (failing.length > 0) {
      failures.push(`evidence_refs below ${thresholds.minEvidenceRefsPerBatch}: ${failing.join(', ')}`);
    }
  }
}

function compareGoldenSummary(
  actualPath: string,
  goldenPath: string | undefined,
  label: string,
  failures: string[],
): void {
  if (!goldenPath) {
    return;
  }

  const actual = normalizeSummaryForGolden(readFileSync(actualPath, 'utf8'));
  const golden = normalizeSummaryForGolden(readFileSync(goldenPath, 'utf8'));
  if (actual !== golden) {
    failures.push(`${label} summary does not match golden snapshot ${goldenPath}`);
  }
}

function compareVerificationScores(left: string, right: string): number {
  const parse = (value: string): [number, number] => {
    const match = value.match(/(\d+)\/(\d+)/);
    return match ? [Number.parseInt(match[1], 10), Number.parseInt(match[2], 10)] : [0, 0];
  };

  const [leftPassed, leftTotal] = parse(left);
  const [rightPassed, rightTotal] = parse(right);
  return leftPassed / Math.max(1, leftTotal) - rightPassed / Math.max(1, rightTotal);
}

async function runEvalCommand(paperPath: string, cacheOnly: boolean): Promise<EvalCommandPayload> {
  const stdout = await runCliCommand([
    'eval-codegen',
    paperPath,
    '--lang',
    'python',
    '--framework',
    'pytorch',
    '--include-prompt',
    '--format',
    'json',
    ...(cacheOnly ? ['--cache-only'] : []),
  ]);

  return JSON.parse(stdout) as EvalCommandPayload;
}

async function runAnalyzeCommand(
  papers: string[],
  context: string,
  cacheOnly: boolean,
): Promise<AnalyzeCommandPayload> {
  const stdout = await runCliCommand([
    'analyze-methods',
    ...papers,
    '--context',
    context,
    '--format',
    'json',
    ...(cacheOnly ? ['--cache-only'] : []),
  ]);

  return JSON.parse(stdout) as AnalyzeCommandPayload;
}

async function runCliCommand(args: string[]): Promise<string> {
  const cliPath = path.resolve(__dirname, '..', 'cli.js');
  const { stdout, stderr } = await execFile(process.execPath, [cliPath, ...args], {
    maxBuffer: 50 * 1024 * 1024,
  });
  if (stderr.trim()) {
    process.stderr.write(stderr);
  }
  return stdout;
}

async function runProcess(command: string, args: string[]): Promise<void> {
  await execFile(command, args, {
    maxBuffer: 50 * 1024 * 1024,
  });
}

async function readJsonFile<T>(filePath: string): Promise<T> {
  const raw = await fs.readFile(filePath, 'utf8');
  return JSON.parse(raw) as T;
}

async function exists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function extractPythonBlock(markdown: string): string {
  const match = markdown.match(/```python\s+([\s\S]*?)```/i);
  if (!match) {
    throw new Error('Markdown output did not include a Python code block.');
  }

  return match[1].trim();
}

function countMatches(input: string, pattern: RegExp): number {
  const matches = input.match(pattern);
  return matches ? matches.length : 0;
}

function buildVerificationScore(
  verification: EvalCommandPayload['verification'] | undefined,
): string {
  const score = verification?.score;
  const total = Array.isArray(verification?.checks) ? verification.checks.length : 0;
  return Number.isFinite(score) && total > 0 ? `${score}/${total}` : '-';
}

function formatSubscoreRatio(value: number | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(3) : '-';
}

function compactErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message.replace(/\s+/g, ' ').trim();
  }

  return String(error).replace(/\s+/g, ' ').trim();
}

function sanitizeTsv(value: string): string {
  return value.replace(/\t/g, ' ').replace(/\r?\n/g, ' ').trim();
}

function averageNumericString(values: string[]): number {
  const numeric = values
    .map((value) => Number.parseFloat(value))
    .filter((value) => Number.isFinite(value));
  if (numeric.length === 0) {
    return 0;
  }

  return Number((numeric.reduce((total, value) => total + value, 0) / numeric.length).toFixed(3));
}

function normalizeSummaryForGolden(input: string): string {
  const lines = input.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) {
    return '';
  }

  const headers = lines[0].split('\t');
  const keepIndexes = headers
    .map((header, index) => ({ header, index }))
    .filter(({ header }) => !['resumed', 'error'].includes(header))
    .map(({ index }) => index);

  return lines
    .map((line) => {
      const columns = line.split('\t');
      return keepIndexes.map((index) => columns[index] ?? '').join('\t');
    })
    .join('\n');
}

function chunkArray<T>(items: T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let index = 0; index < items.length; index += size) {
    chunks.push(items.slice(index, index + size));
  }
  return chunks;
}

function resolveBatchConcurrency(rawValue: string | undefined): number {
  const parsed = Number.parseInt(rawValue?.trim() ?? '', 10);
  if (Number.isFinite(parsed) && parsed > 0) {
    return Math.max(1, parsed);
  }

  return 3;
}

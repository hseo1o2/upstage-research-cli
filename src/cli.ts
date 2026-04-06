#!/usr/bin/env node

import { Command } from 'commander';

import { analyzeMethodsCommand } from './commands/analyze-methods';
import { cacheClearCommand } from './commands/cache-clear';
import { evalCodegenCommand } from './commands/eval-codegen';
import { installCommand } from './commands/install';
import { regressionBatchCommand } from './commands/regression-batch';

const program = new Command();

program
  .name('upstage-research')
  .description('Upstage-powered research paper CLI and agent skill installer')
  .version('0.1.0');

program
  .command('analyze-methods')
  .description('Analyze one or more research papers and compare their methodologies')
  .argument('<papers...>', 'PDF papers to analyze')
  .option('--context <text>', 'research context to tailor the application suggestions')
  .option('--format <format>', 'markdown or json output', 'markdown')
  .option('--out <path>', 'write the primary output to a file as well as stdout')
  .option('--save-report <path>', 'explicit path for the rendered report file')
  .option('--cache-only', 'fail instead of calling Upstage APIs when a cache entry is missing', false)
  .action(wrapCommand(analyzeMethodsCommand));

program
  .command('eval-codegen')
  .description('Generate evaluation code and reproduction notes from a paper experiment section')
  .argument('<paper>', 'paper PDF to analyze')
  .option('--lang <language>', 'output language for the runnable code block', 'python')
  .option('--framework <framework>', 'target framework or leave as auto', 'auto')
  .option('--include-prompt', 'include an LLM-as-judge prompt in the markdown output', false)
  .option('--format <format>', 'markdown or json output', 'markdown')
  .option('--out <path>', 'write the primary output to a file as well as stdout')
  .option('--save-report <path>', 'explicit path for the rendered report file')
  .option('--save-code <path>', 'write the generated evaluation code to a separate file')
  .option('--verify-only', 'emit only verification output instead of the full report', false)
  .option('--cache-only', 'fail instead of calling Upstage APIs when a cache entry is missing', false)
  .action(wrapCommand(evalCodegenCommand));

program
  .command('install')
  .description('Install Upstage research skills into the current project')
  .option('--skills', 'install the bundled agent skills into .claude/skills/upstage-research', true)
  .option('--targets <targets>', 'comma-separated install targets: claude,codex,cursor')
  .option('--all-targets', 'install to claude, codex, and cursor project paths', false)
  .option('--dest <path>', 'custom install destination for the skills bundle')
  .option('--dry-run', 'show planned install destinations without copying files', false)
  .action(wrapCommand(installCommand));

program
  .command('regression-batch')
  .description('Run batched eval/analyze regression checks over a paper directory')
  .argument('[paperDir]', 'directory containing paper PDFs')
  .option('--mode <mode>', 'eval, analyze, or both', 'both')
  .option('--concurrency <n>', 'max concurrent eval jobs', '3')
  .option('--limit <n>', 'limit the number of PDFs')
  .option('--out-dir <path>', 'output directory for regression artifacts')
  .option('--context <text>', 'context used for batched analyze-methods runs')
  .option('--manifest <path>', 'fixture manifest with fixed paper set, thresholds, and golden summaries')
  .option('--assert', 'fail when thresholds or golden snapshots do not match', false)
  .option('--cache-only', 'fail instead of calling Upstage APIs when a cache entry is missing', false)
  .option('--resume', 'reuse existing per-paper/per-batch JSON artifacts when present', false)
  .action(wrapCommand(regressionBatchCommand));

program
  .command('cache-clear')
  .description('Clear the local Upstage research cache directory or a specific namespace')
  .argument('[namespace]', 'optional cache namespace such as document-parse or chat-structured')
  .option('--dry-run', 'show the target without deleting anything', false)
  .action(wrapCommand(cacheClearCommand));

void program.parseAsync(process.argv);

function wrapCommand<TArgs extends unknown[]>(
  command: (...args: TArgs) => Promise<void>,
): (...args: TArgs) => Promise<void> {
  return async (...args: TArgs) => {
    try {
      await command(...args);
    } catch (error) {
      console.error(`Error: ${formatError(error)}`);
      process.exitCode = 1;
    }
  };
}

function formatError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }

  return String(error);
}

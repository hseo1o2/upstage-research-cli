import { promises as fs } from 'node:fs';
import path from 'node:path';

export type OutputFormat = 'markdown' | 'json';

export function normalizeOutputFormat(format: string | undefined): OutputFormat {
  return (format?.trim().toLowerCase() || 'markdown') === 'json' ? 'json' : 'markdown';
}

export function resolvePrimaryOutputPath(out: string | undefined, saveReport: string | undefined): string | undefined {
  const value = saveReport?.trim() || out?.trim();
  return value ? path.resolve(value) : undefined;
}

export async function writeOutputFile(filePath: string, content: string): Promise<void> {
  const resolvedPath = path.resolve(filePath);
  await fs.mkdir(path.dirname(resolvedPath), { recursive: true });
  await fs.writeFile(resolvedPath, content, 'utf8');
}

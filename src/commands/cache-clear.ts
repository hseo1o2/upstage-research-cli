import { promises as fs } from 'node:fs';
import path from 'node:path';

import { resolveDefaultCacheDir } from '../client';

export interface CacheClearOptions {
  dryRun?: boolean;
}

export async function cacheClearCommand(
  namespace: string | undefined,
  options: CacheClearOptions,
): Promise<void> {
  const cacheDir = resolveDefaultCacheDir(process.cwd());
  const trimmedNamespace = namespace?.trim();
  const targetPath = trimmedNamespace
    ? path.resolve(cacheDir, trimmedNamespace)
    : cacheDir;

  if (!targetPath.startsWith(cacheDir)) {
    throw new Error('Cache namespace must stay within the configured cache directory.');
  }

  const exists = await pathExists(targetPath);
  const lines = ['# Cache Clear'];
  lines.push('');
  lines.push(`- Cache dir: \`${cacheDir}\``);
  lines.push(`- Target: \`${targetPath}\``);

  if (!exists) {
    lines.push('- Status: no-op (target does not exist)');
    process.stdout.write(`${lines.join('\n')}\n`);
    return;
  }

  if (options.dryRun) {
    lines.push('- Status: dry-run only');
    process.stdout.write(`${lines.join('\n')}\n`);
    return;
  }

  await fs.rm(targetPath, { recursive: true, force: true });
  lines.push('- Status: cleared');
  process.stdout.write(`${lines.join('\n')}\n`);
}

async function pathExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';

export interface UpstageResearchConfig {
  apiKey?: string;
}

export function getConfigDir(): string {
  return path.join(os.homedir(), '.config', 'upstage-research');
}

export function getConfigPath(): string {
  return path.join(getConfigDir(), 'config.json');
}

export async function loadConfig(): Promise<UpstageResearchConfig> {
  try {
    const raw = await fs.readFile(getConfigPath(), 'utf8');
    const parsed = JSON.parse(raw) as UpstageResearchConfig;
    return typeof parsed === 'object' && parsed !== null ? parsed : {};
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return {};
    }

    throw new Error(`Failed to read config from ${getConfigPath()}: ${(error as Error).message}`);
  }
}

export async function saveConfig(config: UpstageResearchConfig): Promise<void> {
  await fs.mkdir(getConfigDir(), { recursive: true });
  await fs.writeFile(getConfigPath(), `${JSON.stringify(config, null, 2)}\n`, 'utf8');
}

export async function resolveApiKey(override?: string): Promise<string | undefined> {
  if (process.env.UPSTAGE_API_KEY?.trim()) {
    return process.env.UPSTAGE_API_KEY.trim();
  }

  if (override?.trim()) {
    return override.trim();
  }

  const config = await loadConfig();
  return config.apiKey?.trim() || undefined;
}

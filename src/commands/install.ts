import { promises as fs } from 'node:fs';
import path from 'node:path';
import readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';

import { getConfigPath, loadConfig, saveConfig } from '../config';

const INSTALL_TARGETS = {
  claude: ['.claude', 'skills', 'upstage-research'],
  codex: ['.codex', 'skills', 'upstage-research'],
  cursor: ['.cursor', 'skills', 'upstage-research'],
} as const;

export interface InstallOptions {
  skills?: boolean;
  targets?: string;
  allTargets?: boolean;
  dest?: string;
  dryRun?: boolean;
}

export async function installCommand(options: InstallOptions): Promise<void> {
  const config = await loadConfig();
  const envApiKey = process.env.UPSTAGE_API_KEY?.trim();
  let storedApiKey = config.apiKey?.trim();
  let keyStatus = '';

  if (envApiKey) {
    keyStatus = 'Using `UPSTAGE_API_KEY` from the environment. Config file was left unchanged.';
  } else if (storedApiKey) {
    keyStatus = `Using saved API key from \`${getConfigPath()}\`.`;
  } else {
    if (!input.isTTY || !output.isTTY) {
      throw new Error(
        'UPSTAGE_API_KEY is not set and no TTY is available for prompting. Export the variable and retry.',
      );
    }

    const rl = readline.createInterface({ input, output });
    try {
      storedApiKey = (await rl.question('Enter UPSTAGE_API_KEY: ')).trim();
    } finally {
      rl.close();
    }

    if (!storedApiKey) {
      throw new Error('UPSTAGE_API_KEY cannot be empty.');
    }

    await saveConfig({
      ...config,
      apiKey: storedApiKey,
    });
    keyStatus = `Saved API key to \`${getConfigPath()}\`.`;
  }

  const sourceDir = path.resolve(__dirname, '..', '..', 'skills');
  const destinations = resolveInstallDestinations(options);
  const installedSkills = (
    await fs.readdir(sourceDir, {
      withFileTypes: true,
    })
  )
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();

  for (const destination of destinations) {
    if (options.skills === false) {
      continue;
    }

    if (options.dryRun) {
      continue;
    }

    await fs.mkdir(path.dirname(destination.path), { recursive: true });
    await fs.cp(sourceDir, destination.path, {
      recursive: true,
      force: true,
    });
  }

  const lines = [
    '# Upstage Research Skills Installed',
    '',
    `- ${keyStatus}`,
    `- Installed skills: ${installedSkills.map((skill) => `\`${skill}\``).join(', ')}`,
  ];

  if (options.dryRun) {
    lines.push('- Status: dry-run only');
  }

  for (const destination of destinations) {
    lines.push(`- ${destination.label}: \`${destination.path}\``);
  }

  process.stdout.write(`${lines.join('\n')}\n`);
}

function resolveInstallDestinations(options: InstallOptions): Array<{ label: string; path: string }> {
  if (options.dest?.trim()) {
    return [
      {
        label: 'custom',
        path: path.resolve(options.dest.trim()),
      },
    ];
  }

  const requestedTargets = options.allTargets
    ? Object.keys(INSTALL_TARGETS)
    : parseRequestedTargets(options.targets);

  return requestedTargets.map((target) => ({
    label: target,
    path: path.resolve(process.cwd(), ...INSTALL_TARGETS[target as keyof typeof INSTALL_TARGETS]),
  }));
}

function parseRequestedTargets(rawTargets: string | undefined): string[] {
  if (!rawTargets?.trim()) {
    return ['claude'];
  }

  const normalized = rawTargets
    .split(',')
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean);
  const invalid = normalized.filter((value) => !(value in INSTALL_TARGETS));
  if (invalid.length > 0) {
    throw new Error(`Unsupported install target(s): ${invalid.join(', ')}`);
  }

  return Array.from(new Set(normalized));
}

import { readFileSync } from 'node:fs';
import path from 'node:path';

interface DiscoveryRule {
  mode?: string;
  domain?: string;
  pattern: string;
  requires?: string;
}

interface EvalDiscoveryRegistry {
  taskModeRules: DiscoveryRule[];
  domainRules: DiscoveryRule[];
  domainFallbackRules: DiscoveryRule[];
  libraryHints: Record<string, string>;
}

interface EvidenceAliasRule {
  pattern: string;
  values: string[];
}

interface EvidenceAliasRegistry {
  aliases: EvidenceAliasRule[];
}

interface MetricKindRule {
  kind: string;
  metricPattern: string;
  excludeMetricPattern?: string;
}

interface MetricKindRegistry {
  rules: MetricKindRule[];
  rankingKinds: string[];
  directPredictionKinds: string[];
}

let discoveryRegistryCache: EvalDiscoveryRegistry | null = null;
let evidenceAliasRegistryCache: EvidenceAliasRegistry | null = null;
let metricKindRegistryCache: MetricKindRegistry | null = null;

export function getEvalDiscoveryRegistry(): EvalDiscoveryRegistry {
  if (!discoveryRegistryCache) {
    discoveryRegistryCache = readJsonConfig<EvalDiscoveryRegistry>('eval-discovery-registry.json');
  }

  return discoveryRegistryCache;
}

export function getEvidenceAliasRegistry(): EvidenceAliasRegistry {
  if (!evidenceAliasRegistryCache) {
    evidenceAliasRegistryCache = readJsonConfig<EvidenceAliasRegistry>('evidence-alias-registry.json');
  }

  return evidenceAliasRegistryCache;
}

export function getMetricKindRegistry(): MetricKindRegistry {
  if (!metricKindRegistryCache) {
    metricKindRegistryCache = readJsonConfig<MetricKindRegistry>('metric-kind-registry.json');
  }

  return metricKindRegistryCache;
}

function readJsonConfig<T>(fileName: string): T {
  const configPath = path.resolve(__dirname, '..', 'config', fileName);
  return JSON.parse(readFileSync(configPath, 'utf8')) as T;
}

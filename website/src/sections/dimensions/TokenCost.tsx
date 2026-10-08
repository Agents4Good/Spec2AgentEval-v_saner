import { useEffect, useMemo, useState } from 'react';
import type { Cell, MetricRow, SelectedCell, SubRow } from './shared/metric-table';
import {
  LlmFilter,
  MetricMatrix,
  SidePanelShell,
} from './shared/metric-table';
import InfoTooltip from '../../components/InfoTooltip';

interface TokenBreakdown {
  input: number;
  output: number;
  cache_read: number;
  cache_created: number;
  reasoning: number;
  total: number;
  cost_usd: number | null;
}

interface ResourceRunResult {
  run: number;
  model: string;
  agent_folder: string;
  tokens: TokenBreakdown | null;
}

interface ResourceResult {
  model: string;
  agent_folder: string;
  tokens: TokenBreakdown | null;
  runs?: ResourceRunResult[];
  runs_count?: number;
}

interface ResourceData {
  stage: string;
  results: ResourceResult[];
}

const formatTokens = (n: number | null | undefined): string => {
  if (n === null || n === undefined) return '—';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return `${Math.round(n)}`;
};
const formatCost = (n: number | null | undefined): string =>
  n === null || n === undefined ? '—' : `$${n.toFixed(2)}`;

interface Props { experiment: string }

const TokenCost = ({ experiment }: Props) => {
  const [data, setData] = useState<ResourceData | null>(null);
  const [selected, setSelected] = useState<SelectedCell | null>(null);
  const [activeModels, setActiveModels] = useState<string[]>([]);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}${experiment}__01_resource_aggregated.json`).then(r => r.json()).then(setData);
  }, [experiment]);

  const index = useMemo(() => {
    const map = new Map<string, Map<string, ResourceResult>>();
    for (const r of data?.results || []) {
      const m = r.model.toLowerCase();
      if (!map.has(m)) map.set(m, new Map());
      map.get(m)!.set(r.agent_folder, r);
    }
    return map;
  }, [data]);

  const allModels = useMemo(() => Array.from(index.keys()).sort(), [index]);

  useEffect(() => {
    if (allModels.length && activeModels.length === 0) setActiveModels(allModels);
  }, [allModels, activeModels.length]);

  const activeModelList = useMemo(() => allModels.filter(m => activeModels.includes(m)), [allModels, activeModels]);

  const runCountByAgent = useMemo(() => {
    const map = new Map<string, number>();
    for (const perAgent of index.values()) {
      for (const [name, r] of perAgent.entries()) {
        const n = r.runs?.length ?? 0;
        if (n > (map.get(name) ?? 0)) map.set(name, n);
      }
    }
    return map;
  }, [index]);

  const agentRows: MetricRow[] = useMemo(() => {
    const names = new Set<string>();
    for (const perAgent of index.values()) for (const a of perAgent.keys()) names.add(a);
    return Array.from(names).sort().map(a => {
      const n = runCountByAgent.get(a) ?? 0;
      const subRows: SubRow[] = n > 1
        ? Array.from({ length: n }, (_, i) => ({ id: `run-${i + 1}`, label: `run ${i + 1}` }))
        : [];
      return {
        id: a,
        label: (
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <InfoTooltip agentName={a} />
            <span>{a}</span>
          </div>
        ),
        subRows,
      };
    });
  }, [index, runCountByAgent]);

  const totals = useMemo(() => {
    const out: number[] = [];
    for (const perAgent of index.values()) {
      for (const r of perAgent.values()) if (r.tokens?.total) out.push(r.tokens.total);
    }
    return out;
  }, [index]);
  const maxTotal = totals.length ? Math.max(...totals) : 0;
  const sumTotal = totals.reduce((a, b) => a + b, 0);
  const sumCost = useMemo(() => {
    let s = 0;
    for (const perAgent of index.values()) {
      for (const r of perAgent.values()) if (r.tokens?.cost_usd) s += r.tokens.cost_usd;
    }
    return s;
  }, [index]);

  const toneForTotal = (total: number): 'muted' | 'success' | 'warning' | 'danger' => {
    if (total === 0) return 'muted';
    const ratio = maxTotal > 0 ? total / maxTotal : 0;
    return ratio <= 0.33 ? 'success' : ratio <= 0.66 ? 'warning' : 'danger';
  };

  const getCell = (row: MetricRow, model: string, subRowId?: string): Cell | null => {
    const r = index.get(model)?.get(row.id);
    if (!r) return null;
    if (subRowId) {
      const runNum = Number(subRowId.replace('run-', ''));
      const run = r.runs?.find(x => x.run === runNum);
      if (!run || !run.tokens) return null;
      const total = run.tokens.total ?? 0;
      const costPart = run.tokens.cost_usd !== null ? ` · ${formatCost(run.tokens.cost_usd)}` : '';
      return {
        value: `${formatTokens(total)}${costPart}`,
        tone: toneForTotal(total),
        hint: `${total} tokens`,
      };
    }
    if (!r.tokens) return null;
    const total = r.tokens.total ?? 0;
    const runs = r.runs ?? [];
    const costPart = r.tokens.cost_usd !== null ? ` · ${formatCost(r.tokens.cost_usd)}` : '';
    const label = runs.length > 1
      ? `${formatTokens(total)}${costPart} (avg of ${runs.length})`
      : `${formatTokens(total)}${costPart}`;
    return {
      value: label,
      tone: toneForTotal(total),
      hint: `${total} tokens`,
    };
  };

  if (!data) return null;

  const selectedAgent = selected ? index.get(selected.model)?.get(selected.rowId) : null;
  const selectedRun = selected && selectedAgent && selected.subRowId
    ? selectedAgent.runs?.find(r => r.run === Number(selected.subRowId!.replace('run-', ''))) ?? null
    : null;
  const panelTokens = selectedRun?.tokens ?? selectedAgent?.tokens ?? null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <p className="panel-description" style={{ margin: 0 }}>
        Token consumption and reported cost per agent during stage 01. Lower is cheaper.
      </p>

      <div style={{ display: 'flex', gap: 12 }}>
        <div className="card" style={{ flex: 1 }}>
          <p className="card-label">Total tokens</p>
          <p className="card-value">{formatTokens(sumTotal || null)}</p>
          <p className="card-sub">across {totals.length} runs</p>
        </div>
        <div className="card" style={{ flex: 1 }}>
          <p className="card-label">Most expensive</p>
          <p className="card-value">{formatTokens(maxTotal || null)}</p>
        </div>
        <div className="card" style={{ flex: 1 }}>
          <p className="card-label">Reported cost</p>
          <p className="card-value">{formatCost(sumCost || null)}</p>
        </div>
      </div>

      <LlmFilter models={allModels} active={activeModels} onChange={setActiveModels} />

      <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start' }}>
        <div
          style={{
            flex: 1,
            background: 'var(--color-background-primary)',
            border: '1px solid var(--color-border-tertiary)',
            borderRadius: 14,
            padding: 16,
          }}
        >
          <MetricMatrix
            rows={agentRows}
            models={activeModelList}
            getCell={getCell}
            selected={selected}
            onCellClick={(row, model, subRowId) => {
              const same = selected?.rowId === row.id && selected?.model === model && selected?.subRowId === subRowId;
              setSelected(same ? null : { rowId: row.id, model, subRowId });
            }}
            rowHeaderLabel="Agent"
          />
        </div>

        {selected && selectedAgent && (
          <SidePanelShell
            model={selected.model}
            title={selectedAgent.agent_folder}
            subtitle={selected.subRowId
              ? `Token cost · stage 01 · run ${selected.subRowId.replace('run-', '')}`
              : 'Token cost · stage 01'}
            rightLabel="Total tokens"
            rightValue={formatTokens(panelTokens?.total ?? null)}
            onClose={() => setSelected(null)}
          >
            <Row label="Input" value={formatTokens(panelTokens?.input)} />
            <Row label="Output" value={formatTokens(panelTokens?.output)} />
            <Row label="Cache read" value={formatTokens(panelTokens?.cache_read)} />
            <Row label="Cache created" value={formatTokens(panelTokens?.cache_created)} />
            <Row label="Reasoning" value={formatTokens(panelTokens?.reasoning)} />
            <Row label="Total" value={formatTokens(panelTokens?.total)} />
            <Row label="Cost (USD)" value={formatCost(panelTokens?.cost_usd)} />

          </SidePanelShell>
        )}
      </div>
    </div>
  );
};

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        padding: '8px 10px',
        borderRadius: 6,
        background: 'rgba(0,0,0,0.02)',
        border: '1px solid var(--color-border-tertiary)',
        fontSize: 12,
      }}
    >
      <span style={{ color: 'var(--color-text-tertiary)' }}>{label}</span>
      <span style={{ fontWeight: 600 }}>{value}</span>
    </div>
  );
}

export default TokenCost;

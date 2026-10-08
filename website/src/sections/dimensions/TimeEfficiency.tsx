import { useEffect, useMemo, useState } from 'react';
import type { Cell, MetricRow, SelectedCell, SubRow } from './shared/metric-table';
import {
  LlmFilter,
  MetricMatrix,
  SidePanelShell,
} from './shared/metric-table';
import InfoTooltip from '../../components/InfoTooltip';

interface ResourceRunResult {
  run: number;
  model: string;
  agent_folder: string;
  status: string | null;
  execution_time: string | null;
  execution_seconds: number | null;
  requests: number | null;
}

interface ResourceResult {
  model: string;
  agent_folder: string;
  execution_time: string | null;
  execution_seconds: number | null;
  requests: number | null;
  runs?: ResourceRunResult[];
  runs_count?: number;
}

interface ResourceData {
  stage: string;
  results: ResourceResult[];
}

const formatSeconds = (s: number | null): string => {
  if (s === null) return '—';
  if (s < 60) return `${s.toFixed(2)}s`;
  const minutes = Math.floor(s / 60);
  const rem = Math.round(s - minutes * 60);
  return `${minutes}m ${rem}s`;
};

interface Props { experiment: string }

const TimeEfficiency = ({ experiment }: Props) => {
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

  const timings = useMemo(() => {
    const out: number[] = [];
    for (const perAgent of index.values()) {
      for (const r of perAgent.values()) {
        if (r.execution_seconds !== null && r.execution_seconds > 0) out.push(r.execution_seconds);
      }
    }
    return out;
  }, [index]);
  const minT = timings.length ? Math.min(...timings) : 0;
  const maxT = timings.length ? Math.max(...timings) : 0;
  const avgT = timings.length ? timings.reduce((a, b) => a + b, 0) / timings.length : null;

  const toneForSeconds = (s: number): 'success' | 'warning' | 'danger' => {
    const ratio = maxT > minT ? (s - minT) / (maxT - minT) : 0;
    return ratio <= 0.33 ? 'success' : ratio <= 0.66 ? 'warning' : 'danger';
  };

  const getCell = (row: MetricRow, model: string, subRowId?: string): Cell | null => {
    const r = index.get(model)?.get(row.id);
    if (!r) return null;
    if (subRowId) {
      const runNum = Number(subRowId.replace('run-', ''));
      const run = r.runs?.find(x => x.run === runNum);
      if (!run) return null;
      const s = run.execution_seconds;
      if (s === null) return { value: '—', tone: 'muted', hint: 'No time recorded' };
      return {
        value: formatSeconds(s),
        tone: toneForSeconds(s),
        hint: `${run.execution_time ?? '—'} · ${run.requests ?? 0} requests`,
      };
    }
    const seconds = r.execution_seconds;
    if (seconds === null) return { value: '—', tone: 'muted', hint: 'No time recorded' };
    const runs = r.runs ?? [];
    const label = runs.length > 1
      ? `${formatSeconds(seconds)} (avg of ${runs.length})`
      : formatSeconds(seconds);
    return {
      value: label,
      tone: toneForSeconds(seconds),
      hint: `${r.execution_time ?? '—'} · ${r.requests ?? 0} requests`,
    };
  };

  if (!data) return null;

  const selectedAgent = selected ? index.get(selected.model)?.get(selected.rowId) : null;
  const selectedRun = selected && selectedAgent && selected.subRowId
    ? selectedAgent.runs?.find(r => r.run === Number(selected.subRowId!.replace('run-', ''))) ?? null
    : null;

  const panelExecTime = selectedRun?.execution_time ?? selectedAgent?.execution_time ?? '—';
  const panelExecSeconds = selectedRun?.execution_seconds ?? selectedAgent?.execution_seconds ?? null;
  const panelRequests = selectedRun?.requests ?? selectedAgent?.requests ?? null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <p className="panel-description" style={{ margin: 0 }}>
        Wall-clock time spent generating each agent during stage 01. Shorter is better.
      </p>

      {avgT !== null && (
        <div style={{ display: 'flex', gap: 12 }}>
          <div className="card" style={{ flex: 1 }}>
            <p className="card-label">Average</p>
            <p className="card-value">{formatSeconds(avgT)}</p>
            <p className="card-sub">across {timings.length} runs</p>
          </div>
          <div className="card" style={{ flex: 1 }}>
            <p className="card-label">Fastest</p>
            <p className="card-value">{formatSeconds(minT)}</p>
          </div>
          <div className="card" style={{ flex: 1 }}>
            <p className="card-label">Slowest</p>
            <p className="card-value">{formatSeconds(maxT)}</p>
          </div>
        </div>
      )}

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
              ? `Time efficiency · stage 01 · run ${selected.subRowId.replace('run-', '')}`
              : 'Time efficiency · stage 01'}
            rightLabel="Execution time"
            rightValue={panelExecTime}
            onClose={() => setSelected(null)}
          >
            <Row label="Requests" value={String(panelRequests ?? '—')} />
            <Row label="Raw time" value={panelExecTime} />
            {maxT > 0 && panelExecSeconds !== null && (
              <Row label="Relative to slowest" value={`${Math.round((panelExecSeconds / maxT) * 100)}%`} />
            )}

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

export default TimeEfficiency;

import { useEffect, useMemo, useState } from 'react';
import type { Cell, MetricRow, SelectedCell, SubRow } from './shared/metric-table';
import {
  LlmFilter,
  MetricMatrix,
  SidePanelShell,
  TONE_COLORS,
} from './shared/metric-table';
import InfoTooltip from '../../components/InfoTooltip';

type CheckKey = 'file_exists' | 'syntax_valid' | 'dependency_error' | 'has_entrypoint' | 'entrypoint_callable';

interface AgentRunResult {
  run: number;
  model: string;
  agent_folder: string;
  file_exists: boolean;
  syntax_valid: boolean;
  dependency_error: boolean;
  has_entrypoint: boolean;
  entrypoint_callable: boolean;
  final_status: string;
  _source_error?: string | null;
}

interface AgentResult {
  model: string;
  agent_folder: string;
  file_exists: number | boolean;
  syntax_valid: number | boolean;
  dependency_error: number | boolean;
  has_entrypoint: number | boolean;
  entrypoint_callable: number | boolean;
  final_status: string;
  _source_error?: string | null;
  runs?: AgentRunResult[];
  runs_count?: number;
}

interface ExecutabilityAggregated {
  stage: string;
  results: AgentResult[];
}

const CHECKS: Array<{ key: CheckKey; label: string; description: string }> = [
  { key: 'file_exists', label: 'File exists', description: 'agent.py exists in the expected location.' },
  { key: 'syntax_valid', label: 'Syntax valid', description: 'Python file parses without syntax errors.' },
  { key: 'dependency_error', label: 'Imports resolve', description: 'Module imports resolve without runtime errors.' },
  { key: 'has_entrypoint', label: 'Entrypoint present', description: 'The declared agent() entrypoint is present.' },
  { key: 'entrypoint_callable', label: 'Entrypoint callable', description: 'The entrypoint can be invoked.' },
];

const checkPassed = (v: number | boolean | undefined): boolean => {
  if (typeof v === 'boolean') return v;
  if (typeof v === 'number') return v >= 0.5;
  return false;
};

const passedCountForRun = (run: AgentRunResult): number =>
  CHECKS.filter(c => Boolean(run[c.key])).length;

const passedCountForAgent = (r: AgentResult): number =>
  CHECKS.filter(c => checkPassed(r[c.key])).length;

interface Props { experiment: string }

const ExecutabilityPanel = ({ experiment }: Props) => {
  const [data, setData] = useState<ExecutabilityAggregated | null>(null);
  const [selected, setSelected] = useState<SelectedCell | null>(null);
  const [activeModels, setActiveModels] = useState<string[]>([]);

  useEffect(() => {
    fetch(`/${experiment}__02_executability_aggregated.json`).then(r => r.json()).then(setData);
  }, [experiment]);

  const index = useMemo(() => {
    const map = new Map<string, Map<string, AgentResult>>();
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

  const getCell = (row: MetricRow, model: string, subRowId?: string): Cell | null => {
    const r = index.get(model)?.get(row.id);
    if (!r) return null;
    if (subRowId) {
      const runNum = Number(subRowId.replace('run-', ''));
      const run = r.runs?.find(x => x.run === runNum);
      if (!run) return null;
      const passed = passedCountForRun(run);
      const ok = run.final_status === 'SUCCESS';
      return {
        value: `${run.final_status} · ${passed}/${CHECKS.length}`,
        tone: ok ? 'success' : 'danger',
        hint: ok ? 'All checks passed' : 'Executability failed',
      };
    }
    const runs = r.runs ?? [];
    if (runs.length > 1) {
      const successRuns = runs.filter(x => x.final_status === 'SUCCESS').length;
      const rate = successRuns / runs.length;
      const avgChecks = runs.reduce((a, x) => a + passedCountForRun(x), 0) / runs.length;
      const tone: 'success' | 'warning' | 'danger' =
        rate >= 1 ? 'success' : rate >= 0.5 ? 'warning' : 'danger';
      return {
        value: `${Math.round(rate * 100)}% · ${successRuns}/${runs.length} runs (avg of ${runs.length})`,
        tone,
        hint: `avg ${avgChecks.toFixed(1)}/${CHECKS.length} checks per run`,
      };
    }
    const passed = passedCountForAgent(r);
    const ok = r.final_status === 'SUCCESS';
    return {
      value: `${r.final_status} · ${passed}/${CHECKS.length}`,
      tone: ok ? 'success' : 'danger',
      hint: ok ? 'All checks passed' : 'Executability failed',
    };
  };

  if (!data) return null;

  const selectedAgent = selected ? index.get(selected.model)?.get(selected.rowId) : null;
  const selectedRun = selected && selectedAgent && selected.subRowId
    ? selectedAgent.runs?.find(r => r.run === Number(selected.subRowId!.replace('run-', ''))) ?? null
    : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
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
              ? `Executability · stage 02 · run ${selected.subRowId.replace('run-', '')}`
              : 'Executability · stage 02'}
            rightLabel={selected.subRowId ? 'Run status' : 'Final'}
            rightValue={selectedRun ? selectedRun.final_status : selectedAgent.final_status}
            rightValueColor={(selectedRun ? selectedRun.final_status : selectedAgent.final_status) === 'SUCCESS' ? '#047857' : '#B91C1C'}
            onClose={() => setSelected(null)}
          >
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {CHECKS.map((check, i) => {
                const runs = selectedAgent.runs ?? [];
                const isAggregate = !selectedRun && runs.length > 1;
                const passedRuns = isAggregate
                  ? runs.filter(r => Boolean(r[check.key])).length
                  : 0;
                const passed = selectedRun
                  ? Boolean(selectedRun[check.key])
                  : isAggregate
                    ? passedRuns === runs.length
                    : checkPassed(selectedAgent[check.key]);
                const prevAllPassed = CHECKS.slice(0, i).every(c => {
                  if (selectedRun) return Boolean(selectedRun[c.key]);
                  if (isAggregate) return runs.every(r => Boolean(r[c.key]));
                  return checkPassed(selectedAgent[c.key]);
                });
                const skipped = !prevAllPassed;
                const tone = skipped
                  ? TONE_COLORS.muted
                  : passed
                    ? TONE_COLORS.success
                    : isAggregate && passedRuns > 0
                      ? TONE_COLORS.warning
                      : TONE_COLORS.danger;
                const label = skipped
                  ? 'Skipped'
                  : isAggregate
                    ? `${passedRuns}/${runs.length} runs`
                    : passed ? 'Passed' : 'Failed';
                return (
                  <div
                    key={check.key}
                    style={{
                      padding: 10,
                      borderRadius: 8,
                      background: 'var(--color-background-primary)',
                      border: `1px solid ${tone.border}`,
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'baseline' }}>
                      <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--color-text-primary)' }}>
                        {check.label}
                      </span>
                      <span
                        style={{
                          fontSize: 10,
                          fontWeight: 800,
                          color: tone.fg,
                          background: tone.bg,
                          padding: '2px 6px',
                          borderRadius: 4,
                          textTransform: 'uppercase',
                        }}
                      >
                        {label}
                      </span>
                    </div>
                    <p style={{ fontSize: 11, color: 'var(--color-text-tertiary)', margin: '6px 0 0', lineHeight: 1.45 }}>
                      {check.description}
                    </p>
                  </div>
                );
              })}
            </div>

            {(selectedRun?._source_error ?? selectedAgent._source_error) && (
              <div
                style={{
                  padding: 12,
                  borderRadius: 10,
                  background: 'rgba(239, 68, 68, 0.05)',
                  border: '1px solid rgba(239, 68, 68, 0.3)',
                  fontSize: 12,
                  fontFamily: 'ui-monospace, SFMono-Regular, monospace',
                  color: '#B91C1C',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                }}
              >
                {selectedRun?._source_error ?? selectedAgent._source_error}
              </div>
            )}
          </SidePanelShell>
        )}
      </div>
    </div>
  );
};

export default ExecutabilityPanel;

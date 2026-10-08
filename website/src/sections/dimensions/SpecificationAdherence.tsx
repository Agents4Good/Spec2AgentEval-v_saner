import { useEffect, useMemo, useState } from 'react';
import type { Cell, MetricRow, SelectedCell, SubRow } from './shared/metric-table';
import {
  LlmFilter,
  MetricMatrix,
  SidePanelShell,
  TONE_COLORS,
} from './shared/metric-table';
import InfoTooltip from '../../components/InfoTooltip';

interface SpecDetail { key: string; score: number; }

interface SpecRunResult {
  run: number;
  model: string;
  agent_folder: string;
  score_percent: number | null;
  total_score: number | null;
  max_score: number | null;
  passed: boolean | null;
  summary: string;
  details: SpecDetail[];
}

interface SpecResult {
  model: string;
  agent_folder: string;
  score_percent: number | null;
  total_score: number | null;
  max_score: number | null;
  passed: number | boolean | null;
  summary: string;
  details: SpecDetail[];
  runs?: SpecRunResult[];
  runs_count?: number;
}

interface SpecData {
  stage: string;
  results: SpecResult[];
}

const toneFromPercent = (p: number | null): keyof typeof TONE_COLORS => {
  if (p === null) return 'muted';
  if (p >= 80) return 'success';
  if (p >= 50) return 'warning';
  return 'danger';
};

interface Props { experiment: string }

const SpecificationAdherence = ({ experiment }: Props) => {
  const [data, setData] = useState<SpecData | null>(null);
  const [selected, setSelected] = useState<SelectedCell | null>(null);
  const [activeModels, setActiveModels] = useState<string[]>([]);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}${experiment}__05_spec_adherence_aggregated.json`).then(r => r.json()).then(setData);
  }, [experiment]);

  const index = useMemo(() => {
    const map = new Map<string, Map<string, SpecResult>>();
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
      return {
        value: run.score_percent !== null ? `${run.score_percent}% · ${run.total_score ?? '—'}/${run.max_score ?? '—'}` : '—',
        tone: toneFromPercent(run.score_percent),
        hint: run.passed ? 'Judge gate passed' : 'Judge gate failed',
      };
    }
    const runs = r.runs ?? [];
    const label = runs.length > 1 && r.score_percent !== null
      ? `${r.score_percent}% (avg of ${runs.length})`
      : r.score_percent !== null
        ? `${r.score_percent}% · ${r.total_score ?? '—'}/${r.max_score ?? '—'}`
        : '—';
    return {
      value: label,
      tone: toneFromPercent(r.score_percent),
      hint: runs.length > 1
        ? `${runs.filter(x => x.passed).length}/${runs.length} runs passed gate`
        : r.passed ? 'Judge gate passed' : 'Judge gate failed',
    };
  };

  if (!data) return null;

  const selectedAgent = selected ? index.get(selected.model)?.get(selected.rowId) : null;
  const selectedRun = selected && selectedAgent && selected.subRowId
    ? selectedAgent.runs?.find(r => r.run === Number(selected.subRowId!.replace('run-', ''))) ?? null
    : null;

  const panelDetails = selectedRun?.details ?? selectedAgent?.details ?? [];
  const panelSummary = selectedRun?.summary ?? selectedAgent?.summary ?? '';
  const panelScorePct = selectedRun?.score_percent ?? selectedAgent?.score_percent ?? null;
  const panelTotal = selectedRun?.total_score ?? selectedAgent?.total_score ?? null;
  const panelMax = selectedRun?.max_score ?? selectedAgent?.max_score ?? null;
  const panelPassed = selectedRun ? Boolean(selectedRun.passed) : Boolean(selectedAgent?.passed);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <p className="panel-description" style={{ margin: 0 }}>
        LLM-as-judge scoring from stage 05. Each cell is the overall adherence score for that
        agent on that LLM. Click a cell for the judge summary and per-criterion breakdown.
      </p>

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
              ? `Specification adherence · stage 05 · run ${selected.subRowId.replace('run-', '')}`
              : 'Specification adherence · stage 05'}
            rightLabel={`${panelTotal ?? '—'}/${panelMax ?? '—'} criteria`}
            rightValue={panelScorePct !== null ? `${panelScorePct}%` : '—'}
            rightValueColor={TONE_COLORS[toneFromPercent(panelScorePct)].fg}
            onClose={() => setSelected(null)}
          >
            {panelSummary && (
              <div
                style={{
                  fontSize: 12,
                  padding: 12,
                  background: 'rgba(0,0,0,0.03)',
                  borderRadius: 8,
                  color: 'var(--color-text-secondary)',
                  lineHeight: 1.5,
                  borderLeft: `4px solid ${panelPassed ? '#10B981' : '#EF4444'}`,
                }}
              >
                <strong style={{ display: 'block', marginBottom: 4, color: 'var(--color-text-primary)' }}>
                  Judge summary
                </strong>
                {panelSummary}
              </div>
            )}

            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <p style={sectionLabel}>Criteria breakdown</p>
              {panelDetails.map(item => {
                const passed = item.score >= (selectedRun ? 1 : 0.5);
                const tone = passed ? TONE_COLORS.success : TONE_COLORS.danger;
                const scoreLabel = selectedRun
                  ? (item.score === 1 ? 'Passed' : 'Failed')
                  : `${Math.round(item.score * 100)}%`;
                return (
                  <div
                    key={item.key}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      padding: '8px 10px',
                      borderRadius: 6,
                      background: 'var(--color-background-primary)',
                      border: `1px solid ${tone.border}`,
                    }}
                  >
                    <span style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
                      {item.key.replace(/_/g, ' ')}
                    </span>
                    <span
                      style={{
                        fontSize: 10,
                        fontWeight: 800,
                        color: tone.fg,
                        background: tone.bg,
                        padding: '2px 8px',
                        borderRadius: 999,
                        textTransform: 'uppercase',
                      }}
                    >
                      {scoreLabel}
                    </span>
                  </div>
                );
              })}
            </div>
          </SidePanelShell>
        )}
      </div>
    </div>
  );
};

const sectionLabel: React.CSSProperties = {
  margin: 0,
  fontSize: 11,
  fontWeight: 800,
  textTransform: 'uppercase',
  letterSpacing: '0.06em',
  color: 'var(--color-text-tertiary)',
};

export default SpecificationAdherence;

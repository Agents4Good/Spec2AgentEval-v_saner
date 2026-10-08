import { useEffect, useMemo, useState } from 'react';
import type { Cell, MetricRow, SelectedCell, SubRow } from './shared/metric-table';
import {
  LlmFilter,
  MetricMatrix,
  SidePanelShell,
  TONE_COLORS,
} from './shared/metric-table';
import InfoTooltip from '../../components/InfoTooltip';

type StageKey = '07_deepeval' | '08_trace';

interface DeepevalMetric {
  name: string;
  score: number | null;
  threshold: number | null;
  reason: string;
  error: string | null;
  passed: boolean;
}

interface TestEntry {
  name: string;
  qualified_name: string;
  status: 'passed' | 'failed' | 'error' | 'skipped' | 'xfail' | 'xpass';
  passed: boolean;
  reason: string;
  metrics?: DeepevalMetric[];
  traceback?: string;
}

interface PerRun {
  run: number;
  totals: Record<string, number>;
  rate: number | null;
  passed: number;
  total: number;
  tests: TestEntry[];
}

interface AgentRun {
  stage: string;
  model: string;
  agent_folder: string;
  test_file: string | null;
  returncode: number | null;
  passed: boolean | null;
  tests: TestEntry[];
  totals: Record<string, number>;
  runs?: PerRun[];
  runs_count?: number;
  rate_mean?: number | null;
}

interface BehavioralData {
  stages: Record<StageKey, AgentRun[]>;
}

const STAGE_LABELS: Record<StageKey, string> = {
  '07_deepeval': '07 · DeepEval',
  '08_trace': '08 · Trace',
};

type VerificationCategory = 'Output' | 'Tool' | 'Order' | 'Trajectory';
const CATEGORY_ORDER: VerificationCategory[] = ['Output', 'Tool', 'Order', 'Trajectory'];

const classifyTest = (name: string): VerificationCategory => {
  const n = name.toLowerCase();
  if (n.includes('trajectory') || n.includes('trace')) return 'Trajectory';
  if (n.includes('order') || n.includes('sequence')) return 'Order';
  if (n.includes('tool') || n.includes('mcp')) return 'Tool';
  return 'Output';
};

const STATUS_TONES: Record<TestEntry['status'], keyof typeof TONE_COLORS> = {
  passed: 'success',
  failed: 'danger',
  error: 'danger',
  skipped: 'muted',
  xfail: 'muted',
  xpass: 'warning',
};

interface Props { experiment: string }

const BehavioralPanel = ({ experiment }: Props) => {
  const [data, setData] = useState<BehavioralData | null>(null);
  const [activeStage, setActiveStage] = useState<StageKey>('07_deepeval');
  const [selected, setSelected] = useState<SelectedCell | null>(null);
  const [activeModels, setActiveModels] = useState<string[]>([]);

  useEffect(() => {
    fetch(`/${experiment}__behavioral_aggregated.json`).then(r => r.json()).then(setData);
  }, [experiment]);

  const stageRuns: AgentRun[] = useMemo(() => (data ? data.stages[activeStage] || [] : []), [data, activeStage]);

  // index[model][agent_folder] = AgentRun
  const index = useMemo(() => {
    const map = new Map<string, Map<string, AgentRun>>();
    for (const run of stageRuns) {
      const m = run.model.toLowerCase();
      if (!map.has(m)) map.set(m, new Map());
      map.get(m)!.set(run.agent_folder, run);
    }
    return map;
  }, [stageRuns]);

  const allModels = useMemo(() => Array.from(index.keys()).sort(), [index]);

  useEffect(() => {
    if (allModels.length && activeModels.length === 0) setActiveModels(allModels);
  }, [allModels, activeModels.length]);

  const activeModelList = useMemo(() => allModels.filter(m => activeModels.includes(m)), [allModels, activeModels]);

  const runCountByAgent = useMemo(() => {
    const map = new Map<string, number>();
    for (const perAgent of index.values()) {
      for (const [name, run] of perAgent.entries()) {
        const n = run.runs?.length ?? 0;
        if (n > (map.get(name) ?? 0)) map.set(name, n);
      }
    }
    return map;
  }, [index]);

  // rows = agent folders (union across models)
  const agentRows: MetricRow[] = useMemo(() => {
    const names = new Set<string>();
    for (const perAgent of index.values()) for (const a of perAgent.keys()) names.add(a);
    return Array.from(names).sort().map(a => {
      const n = runCountByAgent.get(a) ?? 0;
      const subRows: SubRow[] = n > 1
        ? Array.from({ length: n }, (_, i) => ({ id: `run-${i + 1}`, label: `run ${i + 1}` }))
        : [];
      return { id: a, label: <AgentLabel agent={a} />, subRows };
    });
  }, [index, runCountByAgent]);

  const getCell = (row: MetricRow, model: string, subRowId?: string): Cell | null => {
    const run = index.get(model)?.get(row.id);
    if (!run || run.tests.length === 0) return null;
    const perRun = run.runs ?? [];

    if (subRowId) {
      const runNum = Number(subRowId.replace('run-', ''));
      const pr = perRun.find(x => x.run === runNum);
      if (!pr) return null;
      const rate = pr.rate ?? (pr.total > 0 ? (pr.passed / pr.total) * 100 : 0);
      const tone = rate >= 80 ? 'success' : rate >= 50 ? 'warning' : 'danger';
      return {
        value: `${Math.round(rate)}% · ${pr.passed}/${pr.total}`,
        tone,
        hint: `run ${pr.run}`,
      };
    }

    // Prefer server-computed rate_mean when multiple runs exist
    if (perRun.length > 1 && run.rate_mean !== null && run.rate_mean !== undefined) {
      const rate = run.rate_mean / 100;
      const tone = rate >= 0.8 ? 'success' : rate >= 0.5 ? 'warning' : 'danger';
      const perRunSummary = perRun.map(r => `run ${r.run}: ${r.passed}/${r.total}`).join(' · ');
      return {
        value: `${run.rate_mean.toFixed(0)}% (avg of ${perRun.length})`,
        tone,
        hint: perRunSummary,
      };
    }

    const passed = run.totals.passed || 0;
    const failed = (run.totals.failed || 0) + (run.totals.error || 0) + (run.totals.errors || 0);
    const skipped = (run.totals.skipped || 0) + (run.totals.xfail || 0) + (run.totals.xpass || 0);
    const total = passed + failed + skipped;
    const rate = total === 0 ? 0 : passed / total;
    const tone = rate >= 0.8 ? 'success' : rate >= 0.5 ? 'warning' : 'danger';
    const hintParts = [`${passed} passed`, `${failed} failed/errored`];
    if (skipped) hintParts.push(`${skipped} skipped`);
    return {
      value: `${Math.round(rate * 100)}% · ${passed}/${total}`,
      tone,
      hint: hintParts.join(', '),
    };
  };

  if (!data) return null;

  const selectedRun = selected ? index.get(selected.model)?.get(selected.rowId) : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'flex', gap: 0, borderBottom: '1px solid var(--color-border-tertiary)' }}>
        {(Object.keys(STAGE_LABELS) as StageKey[]).map(stage => (
          <button
            key={stage}
            onClick={() => { setActiveStage(stage); setSelected(null); }}
            style={{
              padding: '10px 20px', fontSize: 13, fontWeight: 600, cursor: 'pointer', border: 'none',
              borderBottom: activeStage === stage ? '2px solid var(--color-text-primary)' : '2px solid transparent',
              background: 'transparent',
              color: activeStage === stage ? 'var(--color-text-primary)' : 'var(--color-text-tertiary)',
              marginBottom: '-1px',
            }}
          >
            {STAGE_LABELS[stage]}
          </button>
        ))}
      </div>

      <p className="panel-description" style={{ margin: 0 }}>
        Behavioral tests from stage {activeStage === '07_deepeval' ? '07 (DeepEval)' : '08 (Trace)'}.
        Each cell is the share of tests the LLM-generated agent passed. Click a cell for the full breakdown.
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

        {selected && selectedRun && (
          <BehavioralSidePanel
            run={selectedRun}
            stage={activeStage}
            focusRun={selected.subRowId ? Number(selected.subRowId.replace('run-', '')) : null}
            onClose={() => setSelected(null)}
          />
        )}
      </div>
    </div>
  );
};

function AgentLabel({ agent }: { agent: string }) {
  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
      <InfoTooltip agentName={agent} />
      <span>{agent}</span>
    </div>
  );
}

function TestCard({ test }: { test: TestEntry }) {
  const [showTrace, setShowTrace] = useState(false);
  const tone = TONE_COLORS[STATUS_TONES[test.status]];
  const hasMetrics = test.metrics && test.metrics.length > 0;

  return (
    <div
      style={{
        padding: 10,
        borderRadius: 8,
        background: 'var(--color-background-primary)',
        border: `1px solid ${tone.border}`,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'flex-start' }}>
        <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--color-text-primary)', wordBreak: 'break-word' }}>
          {test.name}
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
            whiteSpace: 'nowrap',
          }}
        >
          {test.status}
        </span>
      </div>

      {test.reason && !hasMetrics && (
        <div
          style={{
            fontSize: 11,
            color: 'var(--color-text-secondary)',
            marginTop: 6,
            lineHeight: 1.45,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            background: 'rgba(0,0,0,0.03)',
            padding: 8,
            borderRadius: 6,
          }}
        >
          {test.reason}
        </div>
      )}

      {hasMetrics && (
        <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 8 }}>
          {test.metrics!.map((m, i) => <DeepevalMetricCard key={i} metric={m} />)}
        </div>
      )}

      {test.traceback && (
        <>
          <button
            type="button"
            onClick={() => setShowTrace(v => !v)}
            style={{
              marginTop: 8,
              padding: '3px 8px',
              fontSize: 10,
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: '0.04em',
              borderRadius: 999,
              background: 'rgba(0,0,0,0.04)',
              border: '1px solid var(--color-border-tertiary)',
              color: 'var(--color-text-secondary)',
              cursor: 'pointer',
            }}
          >
            {showTrace ? 'Hide traceback' : 'Show traceback'}
          </button>
          {showTrace && (
            <pre
              style={{
                marginTop: 6,
                fontSize: 10,
                background: 'rgba(0,0,0,0.04)',
                padding: 10,
                borderRadius: 6,
                maxHeight: 300,
                overflow: 'auto',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                color: 'var(--color-text-secondary)',
                lineHeight: 1.4,
                fontFamily: 'ui-monospace, SFMono-Regular, monospace',
              }}
            >
              {test.traceback}
            </pre>
          )}
        </>
      )}
    </div>
  );
}

function DeepevalMetricCard({ metric }: { metric: DeepevalMetric }) {
  const tone = TONE_COLORS[metric.passed ? 'success' : 'danger'];
  const scorePct = metric.score !== null ? Math.round(metric.score * 100) : null;
  const threshPct = metric.threshold !== null ? Math.round(metric.threshold * 100) : null;
  return (
    <div
      style={{
        padding: 10,
        borderRadius: 8,
        background: 'rgba(0,0,0,0.025)',
        border: `1px solid ${tone.border}`,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'baseline' }}>
        <span style={{ fontSize: 11, fontWeight: 800, color: 'var(--color-text-primary)' }}>
          {metric.name}
        </span>
        {scorePct !== null && threshPct !== null && (
          <span style={{ fontSize: 10, fontWeight: 700, color: tone.fg }}>
            {scorePct}% / {threshPct}%
          </span>
        )}
      </div>
      {scorePct !== null && (
        <div style={{ height: 4, background: 'rgba(0,0,0,0.08)', borderRadius: 2, marginTop: 6, overflow: 'hidden' }}>
          <div style={{ height: '100%', width: `${scorePct}%`, background: tone.fg }} />
        </div>
      )}
      {metric.reason && (
        <div
          style={{
            marginTop: 8,
            fontSize: 11,
            color: 'var(--color-text-secondary)',
            lineHeight: 1.5,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          <strong style={{ color: 'var(--color-text-primary)' }}>Reason: </strong>
          {metric.reason}
        </div>
      )}
      {metric.error && (
        <div style={{ marginTop: 6, fontSize: 10, color: '#B91C1C' }}>
          <strong>Error:</strong> {metric.error}
        </div>
      )}
    </div>
  );
}

function BehavioralSidePanel({
  run,
  stage,
  focusRun,
  onClose,
}: {
  run: AgentRun;
  stage: StageKey;
  focusRun: number | null;
  onClose: () => void;
}) {
  const perRun = run.runs ?? [];
  const focusedPerRun = focusRun !== null ? perRun.find(r => r.run === focusRun) ?? null : null;
  const showPerRun = perRun.length > 1 && !focusedPerRun;

  const testsForView: TestEntry[] = focusedPerRun ? focusedPerRun.tests : run.tests;
  const grouped: Record<VerificationCategory, TestEntry[]> = {
    Output: [], Tool: [], Order: [], Trajectory: [],
  };
  for (const t of testsForView) grouped[classifyTest(t.name)].push(t);

  const passed = focusedPerRun ? focusedPerRun.passed : (run.totals.passed || 0);
  const total = focusedPerRun
    ? focusedPerRun.total
    : (run.totals.passed || 0) + (run.totals.failed || 0) + (run.totals.error || 0) + (run.totals.errors || 0) + (run.totals.skipped || 0) + (run.totals.xfail || 0) + (run.totals.xpass || 0);
  const rate = focusedPerRun
    ? Math.round(focusedPerRun.rate ?? (total > 0 ? (passed / total) * 100 : 0))
    : (total ? Math.round((passed / total) * 100) : 0);

  const headerLabel = focusedPerRun
    ? `run ${focusedPerRun.run} · ${passed}/${total} tests`
    : perRun.length > 1
      ? `avg of ${perRun.length} runs`
      : `${passed}/${total} tests`;
  const headerValue = !focusedPerRun && perRun.length > 1 && run.rate_mean !== null && run.rate_mean !== undefined
    ? `${run.rate_mean.toFixed(0)}%`
    : `${rate}%`;

  return (
    <SidePanelShell
      model={run.model}
      title={run.agent_folder}
      subtitle={`${STAGE_LABELS[stage]}${focusedPerRun ? ` · run ${focusedPerRun.run}` : ''} · ${run.test_file ?? ''}`}
      rightLabel={headerLabel}
      rightValue={headerValue}
      onClose={onClose}
    >
      {showPerRun && (
        <p style={{ fontSize: 11, color: 'var(--color-text-tertiary)', margin: 0 }}>
          Per-test detail below reflects the latest run (run {perRun[perRun.length - 1].run}).
          Expand the row for per-run numbers.
        </p>
      )}
      {CATEGORY_ORDER.map(cat => {
        const tests = grouped[cat];
        if (tests.length === 0) return null;
        return (
          <section
            key={cat}
            style={{
              padding: 12,
              borderRadius: 10,
              background: 'rgba(0,0,0,0.02)',
              border: '1px solid var(--color-border-tertiary)',
            }}
          >
            <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
              <h5 style={sectionHeading}>{cat} verification</h5>
              <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>
                {tests.filter(t => t.passed).length}/{tests.length} passed
              </span>
            </header>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {tests.map(t => <TestCard key={t.qualified_name} test={t} />)}
            </div>
          </section>
        );
      })}
    </SidePanelShell>
  );
}

const sectionHeading: React.CSSProperties = {
  margin: 0,
  fontSize: 12,
  fontWeight: 800,
  textTransform: 'uppercase',
  letterSpacing: '0.06em',
  color: 'var(--color-text-primary)',
};


export default BehavioralPanel;

import { useEffect, useMemo, useState } from 'react';
import type { Cell, MetricRow, SelectedCell, SubRow } from './shared/metric-table';
import {
  LlmFilter,
  MetricMatrix,
  SidePanelShell,
  TONE_COLORS,
} from './shared/metric-table';
import InfoTooltip from '../../components/InfoTooltip';

interface RuffSample { code: string; message: string; row: number | null; }
interface PylintSample { symbol: string; message: string; line: number | null; type: string; }
interface RadonCCItem { file: string; name: string; rank: string; complexity: number; }
interface RadonMIFile { file: string; mi: number; rank: string; }

interface LintSummary {
  ruff?: { total: number; by_code: Record<string, number>; sample: RuffSample[] };
  pylint?: { total: number; by_type: Record<string, number>; sample: PylintSample[] };
  radon_cc?: { total: number; top: RadonCCItem[] };
  radon_mi?: { files: RadonMIFile[] };
}

interface LintRunResult {
  run: number;
  model: string;
  agent: string;
  passed: boolean | null;
  summary: LintSummary;
}

interface LintResult {
  model: string;
  agent: string;
  passed: number | boolean | null;
  summary: LintSummary;
  runs?: LintRunResult[];
  runs_count?: number;
}

interface LintData {
  stage: string;
  results: LintResult[];
}

const totalIssues = (s: LintSummary): number => (s.ruff?.total ?? 0) + (s.pylint?.total ?? 0);

interface Props { experiment: string }

const LintPanel = ({ experiment }: Props) => {
  const [data, setData] = useState<LintData | null>(null);
  const [selected, setSelected] = useState<SelectedCell | null>(null);
  const [activeModels, setActiveModels] = useState<string[]>([]);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}${experiment}__03_lint_aggregated.json`).then(r => r.json()).then(setData);
  }, [experiment]);

  const index = useMemo(() => {
    const map = new Map<string, Map<string, LintResult>>();
    for (const r of data?.results || []) {
      const m = r.model.toLowerCase();
      if (!map.has(m)) map.set(m, new Map());
      map.get(m)!.set(r.agent, r);
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
      const issues = totalIssues(run.summary);
      return {
        value: run.passed ? 'PASS' : `FAIL · ${issues} issues`,
        tone: run.passed ? 'success' : issues < 10 ? 'warning' : 'danger',
        hint: `${run.summary.ruff?.total ?? 0} ruff + ${run.summary.pylint?.total ?? 0} pylint`,
      };
    }
    const runs = r.runs ?? [];
    const passRate = runs.length > 0
      ? runs.filter(x => x.passed).length / runs.length
      : (typeof r.passed === 'number' ? r.passed : r.passed ? 1 : 0);
    const avgIssues = runs.length > 0
      ? runs.reduce((a, x) => a + totalIssues(x.summary), 0) / runs.length
      : totalIssues(r.summary);
    const allPass = passRate >= 1;
    const label = runs.length > 1
      ? allPass
        ? `PASS (avg of ${runs.length})`
        : `${Math.round(passRate * 100)}% pass · ${avgIssues.toFixed(1)} issues (avg of ${runs.length})`
      : allPass ? 'PASS' : `FAIL · ${Math.round(avgIssues)} issues`;
    return {
      value: label,
      tone: allPass ? 'success' : avgIssues < 10 ? 'warning' : 'danger',
      hint: runs.length > 1
        ? `${runs.filter(x => x.passed).length}/${runs.length} runs passed`
        : `${r.summary.ruff?.total ?? 0} ruff + ${r.summary.pylint?.total ?? 0} pylint`,
    };
  };

  if (!data) return null;

  const selectedAgent = selected ? index.get(selected.model)?.get(selected.rowId) : null;
  const selectedRun = selected && selectedAgent && selected.subRowId
    ? selectedAgent.runs?.find(r => r.run === Number(selected.subRowId!.replace('run-', ''))) ?? null
    : null;
  const panelSummary: LintSummary | null = selectedRun?.summary ?? selectedAgent?.summary ?? null;
  const panelPassed = selectedRun ? Boolean(selectedRun.passed) : Boolean(selectedAgent?.passed);

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

        {selected && selectedAgent && panelSummary && (
          <SidePanelShell
            model={selected.model}
            title={selectedAgent.agent}
            subtitle={selected.subRowId
              ? `Lint · stage 03 · run ${selected.subRowId.replace('run-', '')}`
              : 'Lint · stage 03'}
            rightLabel="Gate"
            rightValue={panelPassed ? 'PASS' : 'FAIL'}
            rightValueColor={panelPassed ? '#047857' : '#B91C1C'}
            onClose={() => setSelected(null)}
          >
            {!selected.subRowId && selectedAgent.runs && selectedAgent.runs.length > 1 && (
              <section
                style={{
                  padding: 12,
                  borderRadius: 10,
                  background: 'rgba(0,0,0,0.02)',
                  border: '1px solid var(--color-border-tertiary)',
                }}
              >
                <h5 style={perRunHeading}>Per run</h5>
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr>
                      <th style={perRunThStyle}>Run</th>
                      <th style={perRunThStyle}>Gate</th>
                      <th style={perRunThStyle}>Ruff</th>
                      <th style={perRunThStyle}>Pylint</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedAgent.runs.map(rr => (
                      <tr key={rr.run}>
                        <td style={perRunTdStyle}><strong>run {rr.run}</strong></td>
                        <td style={perRunTdStyle}>{rr.passed ? 'PASS' : 'FAIL'}</td>
                        <td style={perRunTdStyle}>{rr.summary.ruff?.total ?? 0}</td>
                        <td style={perRunTdStyle}>{rr.summary.pylint?.total ?? 0}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
            )}

            <LintSection
              title={`Ruff · ${panelSummary.ruff?.total ?? 0} violations`}
              tone={(panelSummary.ruff?.total ?? 0) === 0 ? 'success' : (panelSummary.ruff?.total ?? 0) < 10 ? 'warning' : 'danger'}
            >
              <GroupedCounts by={panelSummary.ruff?.by_code} label="code" />
              <SampleList items={panelSummary.ruff?.sample?.map(x => ({ label: x.code, detail: `${x.message} (line ${x.row ?? '—'})` }))} />
            </LintSection>

            <LintSection
              title={`Pylint · ${panelSummary.pylint?.total ?? 0} messages`}
              tone={(panelSummary.pylint?.total ?? 0) === 0 ? 'success' : (panelSummary.pylint?.total ?? 0) < 10 ? 'warning' : 'danger'}
            >
              <GroupedCounts by={panelSummary.pylint?.by_type} label="category" />
              <SampleList items={panelSummary.pylint?.sample?.map(x => ({ label: x.symbol, detail: `${x.message} (line ${x.line ?? '—'}, ${x.type})` }))} />
            </LintSection>

            <LintSection
              title="Most complex functions"
              tone={(panelSummary.radon_cc?.top ?? []).some(f => f.complexity > 10) ? 'danger' : 'success'}
            >
              <SampleList items={panelSummary.radon_cc?.top?.map(f => ({ label: `${f.name} · cc ${f.complexity}`, detail: `${f.file} · rank ${f.rank}` }))} />
            </LintSection>

            <LintSection
              title="Maintainability per file"
              tone={(panelSummary.radon_mi?.files ?? []).some(f => f.rank !== 'A') ? 'warning' : 'success'}
            >
              <SampleList items={panelSummary.radon_mi?.files?.map(f => ({ label: f.file, detail: `MI ${f.mi?.toFixed(1)} · rank ${f.rank}` }))} />
            </LintSection>
          </SidePanelShell>
        )}
      </div>
    </div>
  );
};

function LintSection({ title, tone, children }: { title: string; tone: keyof typeof TONE_COLORS; children: React.ReactNode }) {
  const t = TONE_COLORS[tone];
  return (
    <section style={{ padding: 12, borderRadius: 10, border: `1px solid ${t.border}`, background: 'rgba(0,0,0,0.02)' }}>
      <h5
        style={{
          margin: '0 0 8px',
          fontSize: 12,
          fontWeight: 800,
          textTransform: 'uppercase',
          letterSpacing: '0.06em',
          color: t.fg,
        }}
      >
        {title}
      </h5>
      {children}
    </section>
  );
}

function GroupedCounts({ by, label }: { by?: Record<string, number>; label: string }) {
  if (!by || Object.keys(by).length === 0) return null;
  return (
    <div style={{ marginBottom: 10 }}>
      <p style={{ fontSize: 11, color: 'var(--color-text-tertiary)', margin: '0 0 6px' }}>Top {label}s</p>
      {Object.entries(by)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([k, v]) => (
          <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, padding: '2px 0' }}>
            <span style={{ fontFamily: 'ui-monospace, monospace', color: 'var(--color-text-secondary)' }}>{k}</span>
            <span style={{ fontWeight: 700 }}>{v}</span>
          </div>
        ))}
    </div>
  );
}

function SampleList({ items }: { items?: Array<{ label: string; detail: string }> }) {
  if (!items || items.length === 0) return null;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {items.slice(0, 6).map((s, i) => (
        <div
          key={i}
          style={{
            fontSize: 11,
            padding: 8,
            borderRadius: 6,
            background: 'var(--color-background-primary)',
            border: '1px solid var(--color-border-tertiary)',
          }}
        >
          <div style={{ fontFamily: 'ui-monospace, monospace', fontWeight: 700 }}>{s.label}</div>
          <div style={{ color: 'var(--color-text-tertiary)', marginTop: 2 }}>{s.detail}</div>
        </div>
      ))}
    </div>
  );
}

const perRunHeading: React.CSSProperties = {
  margin: '0 0 8px', fontSize: 12, fontWeight: 800,
  textTransform: 'uppercase', letterSpacing: '0.06em',
  color: 'var(--color-text-primary)',
};
const perRunThStyle: React.CSSProperties = {
  padding: '6px 8px', textAlign: 'left', fontSize: 11, fontWeight: 800,
  textTransform: 'uppercase', letterSpacing: '0.04em',
  color: 'var(--color-text-tertiary)',
  borderBottom: '1px solid var(--color-border-tertiary)',
};
const perRunTdStyle: React.CSSProperties = {
  padding: '6px 8px', fontSize: 12, color: 'var(--color-text-primary)',
  borderBottom: '1px solid rgba(0,0,0,0.04)',
};

export default LintPanel;

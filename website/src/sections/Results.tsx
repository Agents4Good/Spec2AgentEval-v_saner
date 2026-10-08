import React, { useMemo, useState } from 'react';
import InfoTooltip from '../components/InfoTooltip';

interface RunEntry {
  run: number;
  executability: number | null;
  structure: number | null;
  spec_graph: number | null;
  spec_prompt: number | null;
  behavioral: number | null;
  behavioral_passed?: number | null;
  behavioral_total?: number | null;
  lint: number | null;
}

interface AgentEntry {
  agent: string;
  model: string;
  executability: number | null;
  structure: number | null;
  spec_graph: number | null;
  spec_prompt: number | null;
  behavioral: number | null;
  behavioral_passed?: number | null;
  behavioral_total?: number | null;
  lint: number | null;
  runs?: RunEntry[];
  runs_count?: number;
  _errors?: Record<string, string>;
}

interface ResultsProps {
  data: AgentEntry[];
  selectedMetrics: string[];
  selectedModels: string[];
  selectedAgents: string[];
  metrics: { id: string; label: string }[];
  TOOL_COLORS: Record<string, string>;
}

function isValidValue(v: number | null | undefined): v is number {
  return v !== null && v !== undefined && !Number.isNaN(Number(v));
}

function behavioralValue(item: AgentEntry): number | null {
  if (!isValidValue(item.behavioral)) return null;
  const total  = item.behavioral_total  ?? null;
  const passed = item.behavioral_passed ?? null;
  if (total !== null && passed !== null && total > 0) return (passed / total) * 100;
  return Number(item.behavioral);
}

function metricValue(item: AgentEntry, id: string): number | null {
  if (id === 'behavioral') return behavioralValue(item);
  const v = item[id as keyof AgentEntry] as number | null | undefined;
  return isValidValue(v) ? v : null;
}

function runMetricValue(run: RunEntry, id: string): number | null {
  if (id === 'behavioral') {
    const passed = run.behavioral_passed ?? null;
    const total = run.behavioral_total ?? null;
    if (passed !== null && total !== null && total > 0) return (passed / total) * 100;
    return isValidValue(run.behavioral) ? Number(run.behavioral) : null;
  }
  const v = run[id as keyof RunEntry] as number | null | undefined;
  return isValidValue(v) ? v : null;
}

const Results = ({ data, selectedMetrics, selectedModels, selectedAgents, metrics }: ResultsProps) => {
  const [expandedKey, setExpandedKey] = useState<string | null>(null);

  const sortedData = useMemo(() => {
    let filtered = data.filter(d => {
      const matchesFilters = (selectedModels.length === 0 || selectedModels.includes(d.model)) &&
                             (selectedAgents.length === 0 || selectedAgents.includes(d.agent));
      const hasValidMetrics = selectedMetrics.every(m => metricValue(d, m) !== null);
      return matchesFilters && hasValidMetrics;
    });

    if (selectedMetrics.length === 0) {
      return [...filtered].sort((a, b) =>
        a.agent.localeCompare(b.agent, undefined, { numeric: true })
      );
    }

    return [...filtered].sort((a, b) => {
      for (const filter of selectedMetrics) {
        let diff = 0;
        const valA = metricValue(a, filter) ?? 0;
        const valB = metricValue(b, filter) ?? 0;

        if (filter === 'behavioral') {
          const valA = metricValue(a, 'behavioral') ?? 0;
          const valB = metricValue(b, 'behavioral') ?? 0;
          diff = valB - valA;
        } else {
          diff = valB - valA;
        }

        if (diff !== 0) return diff;
      }
      return a.agent.localeCompare(b.agent, undefined, { numeric: true });
    });
  }, [data, selectedMetrics, selectedModels, selectedAgents]);

  const itemsWithRank = useMemo(() => {
    let currentRank = 1;
    return sortedData.map((agent, index) => {
      if (index > 0 && selectedMetrics.length > 0) {
        const isTied = selectedMetrics.every(f => {
          const valPrev = metricValue(sortedData[index - 1], f);
          const valCurr = metricValue(agent, f);
          return valPrev === valCurr;
        });

        if (!isTied) currentRank++;
      }

      return {
        ...agent,
        displayRank: selectedMetrics.length > 0 ? currentRank : index + 1
      };
    });
  }, [sortedData, selectedMetrics]);

  return (
    <div style={{ marginBottom: '48px' }}>
      <div style={{ marginBottom: '24px' }}>
        <h2 style={{ fontSize: 24, fontWeight: 800, marginBottom: '16px' }}>Comparative Analysis</h2>
        <p className="panel-description" style={{ margin: 0 }}>
          This panel utilizes a <strong>Dense Ranking</strong> system, ensuring that identical performances share the same position.
          <br /><br />
          The sorting logic follows a <strong>selection-based hierarchy</strong>: the first metric selected establishes the primary rank,
          while subsequent selections serve as tie-breaking criteria to refine the comparison.
          <br /><br />
          When no metrics are selected, results are displayed in a default alphabetical order.
        </p>
      </div>

      <div style={tableContainerStyle}>
        <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '2px', borderStyle: 'hidden'}}>
          <thead>
            <tr style={{ background: 'var(--color-background-secondary)' }}>
              {selectedMetrics.length > 0 && <th style={{ ...thStyle, textAlign: 'center', width: '50px' }}>#</th>}
              <th style={thStyle}>Agent</th>
              <th style={thStyle}>Tool</th>
              {metrics.map(m => (
                <th key={m.id} style={{
                  ...thStyle,
                  textAlign: 'center',
                  color: selectedMetrics.includes(m.id) ? '#059669' : 'inherit',
                  borderBottom: selectedMetrics.includes(m.id) ? '2px solid #059669' : 'none'
                }}>
                  {m.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {itemsWithRank.map((agent) => {
              const rowKey = `${agent.model}-${agent.agent}`;
              const runs = agent.runs ?? [];
              const hasRuns = runs.length > 0;
              const isExpanded = expandedKey === rowKey;
              const colSpan = (selectedMetrics.length > 0 ? 1 : 0) + 2 + metrics.length;
              return (
                <React.Fragment key={rowKey}>
                  <tr
                    onClick={() => hasRuns && setExpandedKey(isExpanded ? null : rowKey)}
                    style={{
                      borderBottom: '1px solid var(--color-border-tertiary)',
                      cursor: hasRuns ? 'pointer' : 'default',
                      background: isExpanded ? 'rgba(5,150,105,0.04)' : undefined,
                    }}
                    title={hasRuns ? `Click to see ${runs.length} runs` : undefined}
                  >
                    {selectedMetrics.length > 0 && (
                      <td style={{ ...centerTd, fontWeight: 800, color: '#185FA5' }}>
                        {agent.displayRank}
                      </td>
                    )}
                    <td style={tdStyle}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <InfoTooltip agentName={agent.agent} />
                        {agent.agent}
                        {hasRuns && (
                          <span style={expandBadgeStyle}>
                            {isExpanded ? '▾' : '▸'} {runs.length} runs
                          </span>
                        )}
                      </div>
                    </td>
                    <td style={tdStyle}>
                      <span style={modelBadgeStyle}>{agent.model.toUpperCase()}</span>
                    </td>
                    <td style={centerTd}>{metricValue(agent, 'executability') ?? '—'}%</td>
                    <td style={centerTd}>{metricValue(agent, 'structure') ?? '—'}%</td>
                    <td style={centerTd}>{metricValue(agent, 'spec_graph') ?? '—'}%</td>
                    <td style={centerTd}>{metricValue(agent, 'spec_prompt') ?? '—'}%</td>
                    <td style={centerTd}>
                      {metricValue(agent, 'behavioral') !== null
                        ? `${metricValue(agent, 'behavioral')?.toFixed(0)}%`
                        : '—'}
                    </td>
                    <td style={centerTd}>
                      {metricValue(agent, 'lint') !== null ? (
                        <b style={{ color: (agent.lint ?? 0) >= 80 ? '#059669' : '#ff791f' }}>
                          {(agent.lint ?? 0) >= 100 ? 'A' : (agent.lint ?? 0) >= 80 ? 'B' : 'C'}
                        </b>
                      ) : '—'}
                    </td>
                  </tr>
                  {isExpanded && hasRuns && (
                    <tr style={{ background: 'rgba(5,150,105,0.04)' }}>
                      <td colSpan={colSpan} style={{ padding: '12px 16px 16px' }}>
                        <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--color-text-secondary)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                          Per-run breakdown · average over {runs.length} run{runs.length > 1 ? 's' : ''}
                        </div>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                          <thead>
                            <tr>
                              <th style={runHeadStyle}>Run</th>
                              {metrics.map(m => (
                                <th key={m.id} style={{ ...runHeadStyle, textAlign: 'center' }}>{m.label}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {runs.map(r => (
                              <tr key={r.run}>
                                <td style={{ ...runCellStyle, fontWeight: 700 }}>run {r.run}</td>
                                {metrics.map(m => {
                                  const v = runMetricValue(r, m.id);
                                  return (
                                    <td key={m.id} style={{ ...runCellStyle, textAlign: 'center' }}>
                                      {v === null ? '—' : m.id === 'lint'
                                        ? ((v >= 100 ? 'A' : v >= 80 ? 'B' : 'C'))
                                        : `${v.toFixed(1)}%`}
                                    </td>
                                  );
                                })}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const thStyle: React.CSSProperties = {
  padding: '16px 12px',
  fontSize: 13,
  fontWeight: 800,
  textAlign: 'left',
  textTransform: 'uppercase',
  color: 'var(--color-text-secondary)',
  borderBottom: '1px solid var(--color-border-tertiary)'
};
const tdStyle: React.CSSProperties = { padding: '16px 12px', fontSize: 13, color: 'var(--color-text-primary)' };
const centerTd: React.CSSProperties = { ...tdStyle, textAlign: 'center', fontWeight: 600 };
const modelBadgeStyle: React.CSSProperties = {
  padding: '2px 8px',
  borderRadius: '4px',
  fontSize: 13,
  fontWeight: 800,
  background: 'var(--color-background-secondary)',
  color: 'var(--color-text-secondary)'
};
const tableContainerStyle: React.CSSProperties = {
  background: 'var(--color-background-primary)',
  border: '1px solid var(--color-border-tertiary)',
  borderRadius: '12px',
  overflow: 'visible',
  padding: '0px 2px 2px 2px',
};

const expandBadgeStyle: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 700,
  color: 'var(--color-text-secondary)',
  background: 'rgba(0,0,0,0.04)',
  border: '1px solid var(--color-border-tertiary)',
  borderRadius: 4,
  padding: '1px 6px',
};

const runHeadStyle: React.CSSProperties = {
  padding: '6px 8px',
  fontSize: 11,
  fontWeight: 800,
  textTransform: 'uppercase',
  color: 'var(--color-text-secondary)',
  letterSpacing: '0.04em',
  borderBottom: '1px solid var(--color-border-tertiary)',
  textAlign: 'left',
};

const runCellStyle: React.CSSProperties = {
  padding: '6px 8px',
  fontSize: 12,
  color: 'var(--color-text-primary)',
  borderBottom: '1px solid rgba(0,0,0,0.04)',
};

export default Results;
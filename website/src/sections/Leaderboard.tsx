import React, { useEffect, useState, useMemo, useDeferredValue } from 'react';
import Results from './Results';
import Analytics from './Analytics';
import { MODEL_COLORS } from './dimensions/shared/metric-table';


export interface RunEntry {
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

interface AgentData {
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

const METRIC_IDS = ['executability', 'structure', 'behavioral', 'lint'] as const;

const TOOL_COLORS: Record<string, string> = MODEL_COLORS;

function isValidValue(v: number | null | undefined): v is number {
  return v !== null && v !== undefined && !Number.isNaN(Number(v));
}

function behavioralValue(item: AgentData): number | null {
  if (!isValidValue(item.behavioral)) return null;
  const total  = item.behavioral_total  ?? null;
  const passed = item.behavioral_passed ?? null;
  if (total !== null && passed !== null && total > 0) return (passed / total) * 100;
  return Number(item.behavioral);
}

function metricValue(item: AgentData, id: string): number | null {
  if (id === 'behavioral') return behavioralValue(item);
  const v = item[id as keyof AgentData] as number | null | undefined;
  return isValidValue(v) ? v : null;
}

const Leaderboard = () => {
  const [data, setData]                       = useState<AgentData[]>([]);
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([]);
  const [selectedModels, setSelectedModels]   = useState<string[]>([]);
  const [selectedAgents, setSelectedAgents]   = useState<string[]>([]);
  const [agentSearch, setAgentSearch]         = useState('');
  const [showWarnings, setShowWarnings]       = useState(false);

  const deferredAgents = useDeferredValue(selectedAgents);
  const deferredModels = useDeferredValue(selectedModels);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}leaderboard_data.json`)
      .then(res => res.json())
      .then((json: AgentData[]) => setData(json));
  }, []);

  const availableModels = useMemo(() =>
    Array.from(new Set(data.map(d => d.model))),
  [data]);

  const COLORS = useMemo(() => {
    const map: Record<string, string> = {};
    availableModels.forEach(m => { map[m] = TOOL_COLORS[m] || '#888'; });
    return map;
  }, [availableModels]);

  const allAgents = useMemo(() =>
    Array.from(new Set(data.map(d => d.agent)))
      .sort((a, b) => a.localeCompare(b, undefined, { numeric: true })),
  [data]);

  const visibleAgents = useMemo(() =>
    allAgents.filter(a => a.toLowerCase().includes(agentSearch.toLowerCase())),
  [allAgents, agentSearch]);

  const warningItems = useMemo(() => {
    const items: {
      agent: string;
      model: string;
      missing: { metric: string; error: string | null }[];
    }[] = [];

    allAgents.forEach(agentName => {
      availableModels.forEach(modelName => {
        const item = data.find(d => d.agent === agentName && d.model === modelName);
        const missing = METRIC_IDS
          .filter(id => {
            const val = item ? metricValue(item, id) : null;
            return val === null;
          })
          .map(id => ({ metric: id, error: item?._errors?.[id] ?? null }));
        if (missing.length > 0) {
          items.push({ agent: agentName, model: modelName, missing });
        }
      });
    });

    return items.sort((a, b) => {
      const agentComp = a.agent.localeCompare(b.agent, undefined, { numeric: true });
      if (agentComp !== 0) return agentComp;
      return a.model.localeCompare(b.model);
    });
  }, [data, allAgents, availableModels]);

  const clearAll = () => {
    setSelectedAgents([]);
    setSelectedModels([]);
    setSelectedMetrics([]);
    setAgentSearch('');
  };

  const metrics = [
    { id: 'executability', label: 'Executability' },
    { id: 'structure',     label: 'Structure' },
    { id: 'spec_graph',    label: 'Spec: Graph & Integration' },
    { id: 'spec_prompt',   label: 'Spec: Agent Prompt' },
    { id: 'behavioral',    label: 'Behavioral' },
    { id: 'lint',          label: 'Lint' },
  ];

  return (
    <section style={{ padding: '24px' }}>

      {/* GLOBAL WARNING BANNER */}
      {warningItems.length > 0 && (
        <div style={warningBannerStyle}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 16 }}>⚠️</span>
              <span style={{ fontWeight: 700, fontSize: 13 }}>
                {warningItems.length} agent-model record{warningItems.length > 1 ? 's' : ''} have
                missing metrics - excluded only from the affected charts and when those metrics are used.
              </span>
            </div>
            <button onClick={() => setShowWarnings(v => !v)} style={warningToggleStyle}>
              {showWarnings ? 'Hide details ▲' : 'Show details ▼'}
            </button>
          </div>

          {showWarnings && (
            <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 6 }}>
              <div style={{
                ...warningRowStyle,
                fontWeight: 700, fontSize: 13,
                color: 'var(--color-text-secondary)',
                borderBottom: '1px solid rgba(0,0,0,0.08)',
                paddingBottom: 6, marginBottom: 2,
              }}>
                <span style={{ minWidth: 200 }}>AGENT</span>
                <span style={{ minWidth: 90 }}>MODEL</span>
                <span>MISSING METRIC · ERROR</span>
              </div>

              {warningItems.map((w, i) => (
                <div key={i} style={warningRowStyle}>
                  <span style={{ fontSize: 13, fontWeight: 600, minWidth: 200 }}>{w.agent}</span>
                  <span style={{ fontSize: 13, minWidth: 90, color: 'var(--color-text-secondary)' }}>{w.model}</span>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4, flex: 1 }}>
                    {w.missing.map(({ metric, error }) => (
                      <div key={metric} style={{ display: 'flex', alignItems: 'flex-start', gap: 6, flexWrap: 'wrap' }}>
                        <span style={missingBadgeStyle}>{metric}</span>
                        {error && (
                          <span style={errorInlineStyle} title={error}>
                            {error.length > 80 ? error.slice(0, 80) + '…' : error}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* SHARED FILTERS */}
      <div style={filtersContainerStyle}>

        <div style={filterRowStyle}>
          <span style={filterLabelStyle}>AGENTS:</span>
          <div style={{ flex: 1 }}>
            <input
              placeholder="Search agents..."
              value={agentSearch}
              onChange={e => setAgentSearch(e.target.value)}
              style={searchInputStyle}
              onFocus={e => {
                e.currentTarget.style.borderColor = '#059669';
                e.currentTarget.style.boxShadow = '0 0 0 3px rgba(5,150,105,0.1)';
              }}
              onBlur={e => {
                e.currentTarget.style.borderColor = 'var(--color-border-tertiary)';
                e.currentTarget.style.boxShadow = 'none';
              }}
            />
            <div style={agentBoxStyle}>
              {visibleAgents.map(agent => {
                const active = selectedAgents.includes(agent);
                return (
                  <button
                    key={agent}
                    onClick={() => {
                      setSelectedAgents(prev => {
                        const next = prev.includes(agent)
                          ? prev.filter(a => a !== agent)
                          : [...prev, agent];
                        return next.length === allAgents.length ? [] : next;
                      });
                      setAgentSearch('');
                    }}
                    style={{
                      ...chipStyle,
                      background:  active ? 'rgba(5,150,105,0.15)' : 'transparent',
                      borderColor: active ? '#059669' : 'var(--color-border-tertiary)',
                    }}
                  >
                    {agent}
                  </button>
                );
              })}
              {visibleAgents.length === 0 && (
                <span style={{ fontSize: 13, color: 'var(--color-text-secondary)', padding: '4px 8px' }}>
                  No agents match your search.
                </span>
              )}
            </div>
          </div>
        </div>

        <div style={filterRowStyle}>
          <span style={filterLabelStyle}>MODELS:</span>
          <div style={chipsContainerStyle}>
            {availableModels.map(m => {
              const active = selectedModels.includes(m);
              const activeColor = COLORS[m] || '#185FA5';
              return (
                <button
                  key={m}
                  onClick={() =>
                    setSelectedModels(prev => {
                      const next = prev.includes(m) ? prev.filter(x => x !== m) : [...prev, m];
                      return next.length === availableModels.length ? [] : next;
                    })
                  }
                  style={{
                    ...chipStyle,
                    borderColor: active ? activeColor : 'var(--color-border-tertiary)',
                    background:  active ? activeColor : 'transparent',
                    color:       active ? 'white' : 'var(--color-text-primary)',
                  }}
                >
                  {m.toUpperCase()}
                </button>
              );
            })}
          </div>
        </div>

        <div style={filterRowStyle}>
          <span style={filterLabelStyle}>METRICS:</span>
          <div style={chipsContainerStyle}>
            {metrics.map(m => {
              const order = selectedMetrics.indexOf(m.id) + 1;
              return (
                <button
                  key={m.id}
                  onClick={() =>
                    setSelectedMetrics(prev => {
                      const next = prev.includes(m.id) ? prev.filter(x => x !== m.id) : [...prev, m.id];
                      return next.length === metrics.length ? [] : next;
                    })
                  }
                  style={{
                    ...chipStyle,
                    background:  order > 0 ? '#059669' : 'transparent',
                    borderColor: order > 0 ? '#059669' : 'var(--color-border-tertiary)',
                    color:       order > 0 ? 'white' : 'var(--color-text-primary)',
                  }}
                >
                  {order > 0 && <span style={{ marginRight: 6 }}>{order}.</span>}
                  {m.label}
                </button>
              );
            })}
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <button onClick={clearAll} style={clearAllStyle}>CLEAR ALL</button>
        </div>
      </div>

      {/* RESULTS TABLE */}
      <div style={{ marginTop: '20px', overflow: 'visible', position: 'relative' }}>
        <Results
          data={data}
          selectedMetrics={selectedMetrics}
          selectedModels={selectedModels}
          selectedAgents={selectedAgents}
          metrics={metrics}
          TOOL_COLORS={TOOL_COLORS}
        />
      </div>

      {/* ANALYTICS CHARTS */}
      <div style={{ 
        display: 'flex', 
        flexDirection: 'column', // Força um abaixo do outro
        gap: '32px',             // Espaço entre um gráfico e outro
        width: '100%' 
      }}>
        <Analytics
          data={data}
          selectedMetrics={selectedMetrics}
          selectedModels={deferredModels}
          selectedAgents={deferredAgents}
          availableModels={availableModels}
          COLORS={COLORS}
        />
      </div>
    </section>
  );
};

const warningBannerStyle: React.CSSProperties = {
  background: 'rgba(234, 179, 8, 0.08)',
  border: '1px solid rgba(234, 179, 8, 0.35)',
  borderRadius: '12px',
  padding: '14px 18px',
  marginBottom: '24px',
  color: 'var(--color-text-primary)',
};

const warningToggleStyle: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 700,
  background: 'transparent',
  border: 'none',
  cursor: 'pointer',
  color: 'var(--color-text-secondary)',
};

const warningRowStyle: React.CSSProperties = {
  display: 'flex',
  gap: 12,
  alignItems: 'flex-start',
  padding: '4px 8px',
  borderRadius: 6,
};

const missingBadgeStyle: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 700,
  color: '#B91C1C',
  background: 'rgba(239,68,68,0.1)',
  border: '1px solid rgba(239,68,68,0.25)',
  borderRadius: 4,
  padding: '1px 6px',
  whiteSpace: 'nowrap',
};

const errorInlineStyle: React.CSSProperties = {
  fontSize: 13,
  color: '#92400E',
  fontFamily: 'monospace',
  background: 'rgba(234,179,8,0.08)',
  borderRadius: 4,
  padding: '1px 6px',
  maxWidth: 400,
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
  cursor: 'help',
};

const filtersContainerStyle: React.CSSProperties = {
  background: 'var(--color-background-secondary)',
  padding: '20px',
  borderRadius: '16px',
  border: '1px solid var(--color-border-tertiary)',
  display: 'flex',
  flexDirection: 'column',
  gap: '16px',
  marginBottom: '30px',
};

const filterRowStyle: React.CSSProperties = {
  display: 'flex',
  gap: '12px',
  alignItems: 'flex-start',
};

const filterLabelStyle: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 800,
  color: 'var(--color-text-secondary)',
  textTransform: 'uppercase',
  minWidth: '80px',
  marginTop: '8px',
};

const chipsContainerStyle: React.CSSProperties = {
  display: 'flex',
  gap: '8px',
  flexWrap: 'wrap',
};

const chipStyle: React.CSSProperties = {
  padding: '6px 14px',
  borderRadius: '20px',
  fontSize: 13,
  fontWeight: 700,
  cursor: 'pointer',
  border: '1px solid var(--color-border-tertiary)',
};

const agentBoxStyle: React.CSSProperties = {
  maxHeight: '120px',
  overflowY: 'auto',
  display: 'flex',
  flexWrap: 'wrap',
  gap: '6px',
  padding: '8px',
  border: '1px solid var(--color-border-tertiary)',
  borderRadius: '8px',
  marginTop: '8px',
};

const searchInputStyle: React.CSSProperties = {
  width: '300px',
  padding: '10px 12px',
  borderRadius: '8px',
  border: '1px solid rgba(5, 150, 105, 0.3)',
  fontSize: 13,
  outline: 'none',
  transition: 'all 0.2s ease',
  backgroundColor: 'var(--color-background-primary)',
};

const clearAllStyle: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 800,
  color: '#EF4444',
  background: 'transparent',
  border: '1px solid #EF4444',
  padding: '6px 12px',
  borderRadius: '8px',
  cursor: 'pointer',
};

export default Leaderboard;
import React, { useState, useMemo } from 'react';
import InfoTooltip from '../components/InfoTooltip';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Legend, CartesianGrid,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';

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
  _errors?: Record<string, string>;
}

const METRIC_IDS = ['executability', 'structure', 'spec_graph', 'spec_prompt', 'behavioral', 'lint'] as const;
type MetricId = typeof METRIC_IDS[number];

interface AnalyticsProps {
  data: AgentData[];
  selectedMetrics: string[];
  selectedModels: string[];
  selectedAgents: string[];
  availableModels: string[];
  COLORS: Record<string, string>;
}

// helpers

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

function metricValue(item: AgentData, id: MetricId): number | null {
  if (id === 'behavioral') return behavioralValue(item);
  const v = item[id] as number | null | undefined;
  return isValidValue(v) ? v : null;
}

const CustomYAxisTick = (props: any) => {
  const { x, y, payload } = props;
  
  return (
    <g transform={`translate(${x},${y})`}>
      <text
        x={-30}
        y={0}
        dy={4}
        textAnchor="end"
        fill="var(--color-text-secondary)"
        style={{ fontSize: '12px' }}
      >
        {payload.value}
      </text>
      <foreignObject x={-25} y={-8} width="20" height="20">
        <div style={{ display: 'flex', alignItems: 'center' }}>
           <InfoTooltip agentName={payload.value} />
        </div>
      </foreignObject>
    </g>
  );
};

// MetricBarChart

const MetricBarChart = ({
  metric,
  filteredData,
  availableModels,
  COLORS,
}: {
  metric: { id: MetricId; label: string };
  filteredData: AgentData[];
  availableModels: string[];
  COLORS: Record<string, string>;
}) => {
  const { chartData, skippedEntries } = useMemo(() => {
    const agentMap: Record<string, Record<string, number>> = {};
    const skipped: { agent: string; model: string; error: string | null }[] = [];

    const uniqueAgents = Array.from(new Set(filteredData.map(d => d.agent)));

    uniqueAgents.forEach(agentName => {
      availableModels.forEach(modelName => {
        const item = filteredData.find(d => d.agent === agentName && d.model === modelName);
        const val = item ? metricValue(item, metric.id) : null;

        if (val === null) {
          skipped.push({ agent: agentName, model: modelName, error: item?._errors?.[metric.id] ?? null });
        } else {
          if (!agentMap[agentName]) agentMap[agentName] = {};
          agentMap[agentName][modelName] = Number(val.toFixed(1));
        }
      });
    });

    const sortedSkipped = skipped.sort((a, b) => {
      const agentComp = a.agent.localeCompare(b.agent, undefined, { numeric: true });
      if (agentComp !== 0) return agentComp;
      return a.model.localeCompare(b.model);
    });

    const sortedChartData = Object.entries(agentMap)
      .map(([agent, vals]) => ({ agent, ...vals }))
      .sort((a, b) => a.agent.localeCompare(b.agent, undefined, { numeric: true }));

    return { chartData: sortedChartData, skippedEntries: sortedSkipped };
  }, [filteredData, metric, availableModels]);

  if (chartData.length === 0) {
    return (
      <div style={tableContainerStyle}>
        <h4 style={{ padding: '18px', margin: 0, fontSize: '16px' }}>
          {metric.label} by Agent
        </h4>
        <p style={{ padding: '0 18px 18px', color: 'var(--color-text-secondary)', fontSize: 13 }}>
          No valid data available for this metric.
        </p>
      </div>
    );
  }

  return (
    <div style={tableContainerStyle}>
      <div style={{ padding: '18px 18px 0', display: 'flex', alignItems: 'flex-start', gap: 10, flexWrap: 'wrap' }}>
        <h4 style={{ margin: 0, fontSize: '16px', flex: 1 }}>{metric.label} by Agent</h4>
        {skippedEntries.length > 0 && (
          <SkippedTooltip entries={skippedEntries} />
        )}
      </div>

      <div style={{ width: '100%', height: Math.max(400, chartData.length * 50), padding: '10px' }}>
        <ResponsiveContainer>
          <BarChart data={chartData} layout="vertical" margin={{ right: 40, left: 50 }} barGap={0}>
            <CartesianGrid strokeDasharray="3 3" horizontal vertical={false} />
            <XAxis
              type="number"
              domain={[0, 100]}
              fontSize={12}
              ticks={[0, 25, 50, 75, 100]}
              tickFormatter={(value: number) => {
                if (metric.id === 'lint') {
                  const labels: Record<number, string> = {
                    100: '100% (A)', 75: '75% (B)', 50: '50% (C)', 25: '25% (D)', 0: '0% (E)',
                  };
                  return labels[value] ?? `${value}%`;
                }
                return `${value}%`;
              }}
            />
            <YAxis dataKey="agent" type="category" width={200} fontSize={12} tick={<CustomYAxisTick />} />
            <Tooltip
              contentStyle={tooltipStyle}
              formatter={(value: any) => {
                const numValue = Number(value);
                if (metric.id !== 'lint') return [`${numValue}%`, metric.label];
                let letter = 'E';
                if (numValue >= 90) letter = 'A';
                else if (numValue >= 70) letter = 'B';
                else if (numValue >= 45) letter = 'C';
                else if (numValue >= 20) letter = 'D';
                return [`${numValue}% (${letter})`, metric.label];
              }}
            />
            <Legend iconSize={12} wrapperStyle={{ fontSize: 13 }} />
            {availableModels.map(m => (
              <Bar key={m} dataKey={m} fill={COLORS[m] || '#ccc'} radius={[0, 4, 4, 0]} barSize={12} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// SkippedTooltip - clickable warning with scrollable panel

const SkippedTooltip = ({
  entries,
}: {
  entries: { agent: string; model: string; error: string | null }[];
}) => {
  const [open, setOpen] = useState(false);

  return (
    <div style={{ position: 'relative' }}>
      <button
        style={{
          ...inlineWarningStyle,
          backgroundColor: open ? 'rgba(234,179,8,0.2)' : 'rgba(234,179,8,0.12)',
          transition: 'background-color 0.2s ease-in-out',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          outline: 'none',
          border: open ? '1px solid rgba(234,179,8,0.5)' : '1px solid rgba(234,179,8,0.3)',
        }}
        onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = 'rgba(234,179,8,0.25)'; }}
        onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = open ? 'rgba(234,179,8,0.2)' : 'rgba(234,179,8,0.12)'; }}
        onClick={() => setOpen(!open)}
      >
        <span style={{ fontSize: '12px' }}>⚠</span>
        <span>{entries.length} record{entries.length > 1 ? 's' : ''} excluded</span>
      </button>

      {open && (
        <>
          <div onClick={() => setOpen(false)} style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, zIndex: 49 }} />
          <div style={{ ...floatingPanelStyle, maxHeight: '280px', overflowY: 'auto' }}>
            <p style={{ position: 'sticky', top: 0, background: 'var(--color-background-primary)', margin: '0 0 12px', paddingBottom: '8px', borderBottom: '1px solid var(--color-border-tertiary)', fontWeight: 700, fontSize: 13, zIndex: 1 }}>
              Excluded due to missing/invalid data:
            </p>
            {entries.map((e, i) => (
              <div key={i} style={{ marginBottom: 10 }}>
                <div style={{ fontSize: 13 }}>
                  <strong>{e.agent}</strong> · <span style={{ color: 'var(--color-text-secondary)' }}>{e.model}</span>
                </div>
                {e.error && <div style={errorMsgStyle}>{e.error}</div>}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
};

// main component
const Analytics = ({ data, selectedMetrics, selectedModels, selectedAgents, availableModels, COLORS }: AnalyticsProps) => {

  const metrics: { id: MetricId; label: string }[] = useMemo(() => [
    { id: 'executability', label: 'Executability' },
    { id: 'structure',     label: 'Structure' },
    { id: 'spec_graph',    label: 'Spec: Graph & Integration' },
    { id: 'spec_prompt',   label: 'Spec: Agent Prompt' },
    { id: 'behavioral',    label: 'Behavioral' },
    { id: 'lint',          label: 'Lint' },
  ], []);

  const activeMetrics = selectedMetrics.length === 0
    ? metrics
    : metrics.filter(m => selectedMetrics.includes(m.id));

  const filteredData = useMemo(() =>
    data
      .filter(d =>
        (selectedModels.length === 0 || selectedModels.includes(d.model)) &&
        (selectedAgents.length === 0 || selectedAgents.includes(d.agent))
      )
      .sort((a, b) => a.agent.localeCompare(b.agent, undefined, { numeric: true })),
  [data, selectedModels, selectedAgents]);

  const activeModels = selectedModels.length > 0 ? selectedModels : availableModels;

  const dimensionAxes: { label: string; components: MetricId[] }[] = useMemo(() => [
    { label: 'Behavioral Correctness', components: ['behavioral'] },
    { label: 'Specification Adherence', components: ['spec_graph', 'spec_prompt'] },
    { label: 'Structural Correctness', components: ['executability', 'structure', 'lint'] },
  ], []);

  const radarData = useMemo(() =>
    dimensionAxes.map(dim => {
      const entry: Record<string, any> = { subject: dim.label };
      activeModels.forEach(model => {
        const perAgent = filteredData
          .filter(d => d.model === model)
          .map(d => {
            const vals = dim.components
              .map(c => metricValue(d, c))
              .filter((v): v is number => v !== null);
            return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
          })
          .filter((v): v is number => v !== null);
        if (perAgent.length > 0) {
          entry[model] = Number((perAgent.reduce((a, b) => a + b, 0) / perAgent.length).toFixed(1));
        }
      });
      return entry;
    }),
  [filteredData, activeModels, dimensionAxes]);

  return (
    <div>
      <h2 style={{ fontSize: 24, fontWeight: 800, marginBottom: '32px' }}>
        Performance Analytics
      </h2>

      {/* DIMENSION RADAR */}
      <div style={{ marginBottom: '40px' }}>
        <div style={{ ...tableContainerStyle, padding: '20px' }}>
          <h4 style={{ margin: '0 0 6px 0', fontSize: 16 }}>Dimension Balance</h4>
          <p style={{ margin: '0 0 16px 0', fontSize: 12, color: 'var(--color-text-secondary)' }}>
            Average score per evaluation dimension. Time Efficiency and Token Cost pending data.
          </p>
          <div style={{ width: '100%', height: 400 }}>
            <ResponsiveContainer>
              <RadarChart cx="50%" cy="50%" outerRadius="75%" data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="subject" tick={{ fontSize: 13 }} />
                <PolarRadiusAxis domain={[0, 100]} />
                {activeModels.map(m => (
                  <Radar key={m} name={m.toUpperCase()} dataKey={m} stroke={COLORS[m]} fill={COLORS[m]} fillOpacity={0.3} />
                ))}
                <Legend />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* BAR CHARTS */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '32px',
        width: '100%'
      }}>
        {activeMetrics.map(metric => (
          <MetricBarChart
            key={metric.id}
            metric={metric}
            filteredData={filteredData}
            availableModels={activeModels}
            COLORS={COLORS}
          />
        ))}
      </div>
    </div>
  );
};

const inlineWarningStyle: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 600,
  color: '#92400E',
  background: 'rgba(234,179,8,0.12)',
  border: '1px solid rgba(234,179,8,0.3)',
  borderRadius: 6,
  padding: '2px 8px',
  cursor: 'pointer',
  whiteSpace: 'nowrap',
};

const floatingPanelStyle: React.CSSProperties = {
  position: 'absolute',
  top: 'calc(100% + 6px)',
  right: 0,
  zIndex: 50,
  background: 'var(--color-background-primary)',
  border: '1px solid var(--color-border-tertiary)',
  borderRadius: 10,
  padding: '12px 14px',
  minWidth: 340,
  maxWidth: 520,
  boxShadow: '0 8px 24px rgba(0,0,0,0.12)',
};

const errorMsgStyle: React.CSSProperties = {
  fontSize: 13,
  color: '#B91C1C',
  background: 'rgba(239,68,68,0.06)',
  border: '1px solid rgba(239,68,68,0.15)',
  borderRadius: 4,
  padding: '3px 7px',
  marginTop: 2,
  wordBreak: 'break-word',
  fontFamily: 'monospace',
};

const tableContainerStyle: React.CSSProperties = {
  background: 'var(--color-background-primary)',
  border: '1px solid var(--color-border-tertiary)',
  borderRadius: '12px',
};

const tooltipStyle = {
  borderRadius: '8px',
  fontSize: 13,
  border: '1px solid var(--color-border-tertiary)',
  background: 'var(--color-background-primary)',
};

export default Analytics;
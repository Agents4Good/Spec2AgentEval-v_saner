import { Fragment, useState } from 'react';
import type { CSSProperties, ReactNode } from 'react';

export const MODEL_COLORS: Record<string, string> = {
  gemini: '#185FA5',
  copilot: '#3B6D11',
  claude: '#534AB7',
  codex: '#111827',
  opencode: '#0E7490',
  gpt: '#854F0B',

  // By Claude Version experiment — distinct shades from oldest to newest
  'claude-opus-4-6-low': '#A78BFA',
  'claude-opus-4-8-low': '#7C3AED',
  'claude-opus-5-5-low': '#4C1D95',
};

export const getModelColor = (model: string): string =>
  MODEL_COLORS[model.toLowerCase()] || '#5F5E5A';

export type CellTone = 'success' | 'danger' | 'warning' | 'neutral' | 'muted';

export const TONE_COLORS: Record<CellTone, { fg: string; bg: string; border: string }> = {
  success: { fg: '#047857', bg: 'rgba(16, 185, 129, 0.12)', border: 'rgba(16, 185, 129, 0.35)' },
  danger: { fg: '#B91C1C', bg: 'rgba(239, 68, 68, 0.12)', border: 'rgba(239, 68, 68, 0.35)' },
  warning: { fg: '#B45309', bg: 'rgba(245, 158, 11, 0.14)', border: 'rgba(245, 158, 11, 0.4)' },
  neutral: { fg: '#1F2937', bg: 'rgba(0, 0, 0, 0.04)', border: 'rgba(0, 0, 0, 0.1)' },
  muted: { fg: '#9CA3AF', bg: 'rgba(0, 0, 0, 0.02)', border: 'rgba(0, 0, 0, 0.08)' },
};

export interface SubRow {
  id: string;
  label: ReactNode;
}

export interface MetricRow {
  id: string;
  label: ReactNode;
  hint?: ReactNode;
  subRows?: SubRow[];
}

export interface Cell {
  value: ReactNode;
  tone: CellTone;
  hint?: string;
}

export interface SelectedCell {
  rowId: string;
  model: string;
  subRowId?: string;
}

export interface MetricMatrixProps<Row extends MetricRow> {
  rows: Row[];
  models: string[];
  getCell: (row: Row, model: string, subRowId?: string) => Cell | null;
  selected?: SelectedCell | null;
  onCellClick: (row: Row, model: string, subRowId?: string) => void;
  rowHeaderLabel?: string;
}

export function MetricMatrix<Row extends MetricRow>({
  rows,
  models,
  getCell,
  selected,
  onCellClick,
  rowHeaderLabel = 'Metric',
}: MetricMatrixProps<Row>) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const toggleExpanded = (rowId: string) => {
    setExpanded(prev => {
      const next = new Set(prev);
      if (next.has(rowId)) next.delete(rowId);
      else next.add(rowId);
      return next;
    });
  };

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={tableStyle}>
        <thead>
          <tr>
            <th style={{ ...headerStyle, textAlign: 'left', minWidth: 220 }}>{rowHeaderLabel}</th>
            {models.map(m => (
              <th key={m} style={{ ...headerStyle, textAlign: 'center' }}>
                <span
                  style={{
                    display: 'inline-block',
                    padding: '2px 10px',
                    borderRadius: 999,
                    fontSize: 11,
                    fontWeight: 700,
                    textTransform: 'uppercase',
                    letterSpacing: '0.06em',
                    background: `${getModelColor(m)}18`,
                    color: getModelColor(m),
                  }}
                >
                  {m}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(row => {
            const hasSubRows = Boolean(row.subRows && row.subRows.length > 0);
            const isExpanded = hasSubRows && expanded.has(row.id);
            return (
              <Fragment key={row.id}>
                <tr className="metric-row">
                  <td
                    style={{
                      ...rowHeaderStyle,
                      cursor: hasSubRows ? 'pointer' : 'default',
                      userSelect: 'none',
                    }}
                    onClick={() => hasSubRows && toggleExpanded(row.id)}
                    title={hasSubRows ? (isExpanded ? 'Hide runs' : 'Show runs') : undefined}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      {hasSubRows && (
                        <span
                          aria-hidden
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            width: 16,
                            height: 16,
                            fontSize: 10,
                            fontWeight: 700,
                            color: 'var(--color-text-tertiary)',
                            transition: 'transform 0.12s ease',
                            transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
                          }}
                        >
                          ▸
                        </span>
                      )}
                      <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--color-text-primary)' }}>
                        {row.label}
                      </div>
                      {hasSubRows && (
                        <span style={subRowBadgeStyle}>
                          {row.subRows!.length} runs
                        </span>
                      )}
                    </div>
                    {row.hint && (
                      <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginTop: 2 }}>
                        {row.hint}
                      </div>
                    )}
                  </td>
                  {models.map(m => {
                    const cell = getCell(row, m);
                    const isSelected =
                      selected?.rowId === row.id &&
                      selected?.model === m &&
                      !selected?.subRowId;
                    return (
                      <td
                        key={m}
                        style={cellTdStyle}
                        onClick={e => {
                          e.stopPropagation();
                          if (cell) onCellClick(row, m);
                        }}
                      >
                        {cell ? <Chip cell={cell} isSelected={isSelected} /> : <EmptyChip />}
                      </td>
                    );
                  })}
                </tr>
                {isExpanded && row.subRows!.map(sub => (
                  <tr key={`${row.id}::${sub.id}`} className="metric-subrow">
                    <td style={subRowHeaderStyle}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, paddingLeft: 28 }}>
                        <span style={subRowDotStyle} />
                        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--color-text-secondary)' }}>
                          {sub.label}
                        </span>
                      </div>
                    </td>
                    {models.map(m => {
                      const cell = getCell(row, m, sub.id);
                      const isSelected =
                        selected?.rowId === row.id &&
                        selected?.model === m &&
                        selected?.subRowId === sub.id;
                      return (
                        <td
                          key={m}
                          style={cellTdStyle}
                          onClick={e => {
                            e.stopPropagation();
                            if (cell) onCellClick(row, m, sub.id);
                          }}
                        >
                          {cell ? <Chip cell={cell} isSelected={isSelected} /> : <EmptyChip />}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </Fragment>
            );
          })}
        </tbody>
      </table>
      <style>{`
        tr.metric-row td { background: transparent; transition: background 0.12s ease; }
        tr.metric-row:hover td { background: rgba(0, 0, 0, 0.025); }
        tr.metric-row td:first-child { border-radius: 10px 0 0 10px; }
        tr.metric-row td:last-child { border-radius: 0 10px 10px 0; }
        tr.metric-row td { border-bottom: 1px solid rgba(0, 0, 0, 0.04); }
        tr.metric-subrow td { background: rgba(0, 0, 0, 0.015); transition: background 0.12s ease; }
        tr.metric-subrow:hover td { background: rgba(0, 0, 0, 0.035); }
        tr.metric-subrow td:first-child { border-radius: 8px 0 0 8px; }
        tr.metric-subrow td:last-child { border-radius: 0 8px 8px 0; }
        .metric-chip { cursor: pointer; }
        .metric-chip:hover { filter: brightness(0.96); }
      `}</style>
    </div>
  );
}

function Chip({ cell, isSelected }: { cell: Cell; isSelected: boolean }) {
  const tone = TONE_COLORS[cell.tone];
  return (
    <span
      className="metric-chip"
      title={cell.hint}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        minWidth: 60,
        padding: '4px 10px',
        borderRadius: 8,
        fontSize: 12,
        fontWeight: 700,
        color: tone.fg,
        background: tone.bg,
        border: `1px solid ${isSelected ? tone.fg : tone.border}`,
        boxShadow: isSelected ? `0 0 0 2px ${tone.border}` : 'none',
      }}
    >
      {cell.value}
    </span>
  );
}

function EmptyChip() {
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        minWidth: 60,
        padding: '4px 10px',
        borderRadius: 8,
        fontSize: 12,
        fontWeight: 500,
        color: 'var(--color-text-tertiary)',
        background: 'rgba(0,0,0,0.02)',
        border: '1px dashed rgba(0,0,0,0.08)',
      }}
    >
      —
    </span>
  );
}

const tableStyle: CSSProperties = {
  width: '100%',
  borderCollapse: 'separate',
  borderSpacing: '0 6px',
};

const headerStyle: CSSProperties = {
  fontSize: 11,
  fontWeight: 700,
  color: 'var(--color-text-tertiary)',
  textTransform: 'uppercase',
  letterSpacing: '0.06em',
  padding: '8px 12px',
  background: 'transparent',
  border: 'none',
};

const rowHeaderStyle: CSSProperties = {
  padding: '12px 12px',
  background: 'var(--color-background-primary)',
  verticalAlign: 'middle',
};

const cellTdStyle: CSSProperties = {
  padding: '12px 12px',
  background: 'var(--color-background-primary)',
  textAlign: 'center',
};

const subRowHeaderStyle: CSSProperties = {
  padding: '8px 12px',
  background: 'var(--color-background-primary)',
  verticalAlign: 'middle',
};

const subRowBadgeStyle: CSSProperties = {
  fontSize: 10,
  fontWeight: 700,
  color: 'var(--color-text-tertiary)',
  background: 'rgba(0,0,0,0.04)',
  border: '1px solid var(--color-border-tertiary)',
  borderRadius: 4,
  padding: '1px 6px',
  textTransform: 'uppercase',
  letterSpacing: '0.04em',
};

const subRowDotStyle: CSSProperties = {
  display: 'inline-block',
  width: 6,
  height: 6,
  borderRadius: 999,
  background: 'rgba(0,0,0,0.25)',
};

// -----------------------------------------------------------------------------
// LLM filter chips (hide/show columns)
// -----------------------------------------------------------------------------

interface LlmFilterProps {
  models: string[];
  active: string[];
  onChange: (next: string[]) => void;
}

export function LlmFilter({ models, active, onChange }: LlmFilterProps) {
  const allActive = active.length === models.length;
  const toggle = (model: string) => {
    if (allActive) {
      onChange([model]);
      return;
    }
    const next = active.includes(model) ? active.filter(m => m !== model) : [...active, model];
    onChange(next.length === 0 ? models : next);
  };
  return (
    <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
      <button
        type="button"
        onClick={() => onChange(models)}
        style={{
          padding: '4px 10px',
          fontSize: 11,
          fontWeight: 700,
          textTransform: 'uppercase',
          letterSpacing: '0.04em',
          borderRadius: 999,
          border: '1px solid var(--color-border-tertiary)',
          background: allActive ? 'var(--color-text-primary)' : 'transparent',
          color: allActive ? 'white' : 'var(--color-text-secondary)',
          cursor: 'pointer',
        }}
      >
        All
      </button>
      {models.map(m => {
        const color = getModelColor(m);
        const isActive = !allActive && active.includes(m);
        return (
          <button
            type="button"
            key={m}
            onClick={() => toggle(m)}
            style={{
              padding: '4px 10px',
              fontSize: 11,
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: '0.04em',
              borderRadius: 999,
              border: `1px solid ${isActive ? color : 'var(--color-border-tertiary)'}`,
              background: isActive ? color : 'transparent',
              color: isActive ? 'white' : 'var(--color-text-secondary)',
              cursor: 'pointer',
            }}
          >
            {m}
          </button>
        );
      })}
    </div>
  );
}

// -----------------------------------------------------------------------------
// Side panel shell
// -----------------------------------------------------------------------------

export function SidePanelShell({
  model,
  title,
  subtitle,
  rightLabel,
  rightValue,
  rightValueColor,
  onClose,
  children,
}: {
  model: string;
  title: ReactNode;
  subtitle?: ReactNode;
  rightLabel?: string;
  rightValue?: ReactNode;
  rightValueColor?: string;
  onClose?: () => void;
  children: ReactNode;
}) {
  const color = getModelColor(model);
  return (
    <aside
      style={{
        width: 420,
        flexShrink: 0,
        position: 'sticky',
        top: 90,
        maxHeight: '82vh',
        display: 'flex',
        flexDirection: 'column',
        gap: 14,
        background: 'var(--color-background-primary)',
        border: '1px solid var(--color-border-tertiary)',
        borderRadius: 16,
        padding: 20,
        boxShadow: '0 1px 2px rgba(0, 0, 0, 0.04)',
      }}
    >
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 10 }}>
          <span
            style={{
              fontSize: 11,
              background: `${color}18`,
              color,
              padding: '3px 10px',
              borderRadius: 999,
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: '0.04em',
            }}
          >
            {model}
          </span>
          {rightValue !== undefined && (
            <div style={{ textAlign: 'right' }}>
              <div
                style={{
                  fontSize: 20,
                  fontWeight: 800,
                  color: rightValueColor ?? color,
                  lineHeight: 1,
                }}
              >
                {rightValue}
              </div>
              {rightLabel && (
                <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginTop: 4, fontWeight: 700 }}>
                  {rightLabel}
                </div>
              )}
            </div>
          )}
        </div>
        <h4 style={{ margin: '10px 0 0', fontSize: 17, fontWeight: 700 }}>{title}</h4>
        {subtitle && (
          <div style={{ fontSize: 12, color: 'var(--color-text-tertiary)', marginTop: 2 }}>{subtitle}</div>
        )}
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            style={{
              marginTop: 10,
              padding: '4px 10px',
              fontSize: 11,
              borderRadius: 999,
              background: 'rgba(0, 0, 0, 0.04)',
              border: '1px solid var(--color-border-tertiary)',
              color: 'var(--color-text-secondary)',
              cursor: 'pointer',
              fontWeight: 600,
            }}
          >
            Close
          </button>
        )}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, overflowY: 'auto', paddingRight: 4 }}>
        {children}
      </div>
    </aside>
  );
}

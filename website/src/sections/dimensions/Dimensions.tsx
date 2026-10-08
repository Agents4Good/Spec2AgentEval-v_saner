import { useState } from 'react';
import './Dimensions.css';
import BehavioralPanel from './Behavioral';
import SpecificationAdherencePanel from './SpecificationAdherence';
import StructuralCorrectness from './StructuralCorrectness';
import TimeEfficiency from './TimeEfficiency';
import TokenCost from './TokenCost';

const DIMENSIONS = [
  {
    id: 'behavioral',
    label: 'Behavioral Correctness',
    hint: 'Flows, tools, order & trajectory (07, 08)',
  },
  {
    id: 'specification',
    label: 'Specification Adherence',
    hint: 'LLM-as-judge against the contract (05)',
  },
  {
    id: 'structural',
    label: 'Structural Correctness',
    hint: 'Executability & lint (02, 03)',
  },
  {
    id: 'time',
    label: 'Time Efficiency',
    hint: 'Wall-clock time (01)',
  },
  {
    id: 'cost',
    label: 'Token Cost',
    hint: 'Tokens & reported cost (01)',
  },
] as const;

type DimensionId = typeof DIMENSIONS[number]['id'];

interface DimensionsProps {
  experiment: string;
  title: string;
}

const Dimensions = ({ experiment, title }: DimensionsProps) => {
  const [active, setActive] = useState<DimensionId>('behavioral');
  const activeMeta = DIMENSIONS.find(d => d.id === active)!;

  return (
    <div className="dimensions-page">
      <div className="dimensions-container">
        <header className="dimensions-header">
          <h1 className="dimensions-title">{title}</h1>
          <p className="dimensions-subtitle">
            Five dimensions, each built from one or more pipeline stages, scored over the same agent runs.
          </p>
        </header>

        <nav className="dimensions-tabs" role="tablist">
          {DIMENSIONS.map(d => (
            <button
              key={d.id}
              role="tab"
              aria-selected={active === d.id}
              className={`dimensions-tab ${active === d.id ? 'is-active' : ''}`}
              onClick={() => setActive(d.id)}
            >
              <span className="dimensions-tab-label">{d.label}</span>
              <span className="dimensions-tab-hint">{d.hint}</span>
            </button>
          ))}
        </nav>

        <section className="dimensions-content">
          <header className="dimensions-content-header">
            <span className="dimensions-breadcrumb">{title} ·</span>
            <h2 className="dimensions-active-title">{activeMeta.label}</h2>
          </header>

          <div className="dimensions-panel">
            {active === 'behavioral' && <BehavioralPanel experiment={experiment} />}
            {active === 'specification' && <SpecificationAdherencePanel experiment={experiment} />}
            {active === 'structural' && <StructuralCorrectness experiment={experiment} />}
            {active === 'time' && <TimeEfficiency experiment={experiment} />}
            {active === 'cost' && <TokenCost experiment={experiment} />}
          </div>
        </section>
      </div>
    </div>
  );
};

export default Dimensions;

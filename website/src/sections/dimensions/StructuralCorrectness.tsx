import { useState } from 'react';
import ExecutabilityPanel from './Executability';
import LintPanel from './Lints';

type SubDimension = 'executability' | 'lint';

const TAB_CONFIG: Record<SubDimension, { label: string; description: string }> = {
  executability: {
    label: 'Executability',
    description:
      'Sequential execution validation (stage 02): file existence, syntax, imports, entrypoint presence, and callability.',
  },
  lint: {
    label: 'Lint',
    description:
      'Static code quality (stage 03): ruff, pylint, radon complexity (cc) and maintainability index (mi).',
  },
};

interface Props { experiment: string }

const StructuralCorrectness = ({ experiment }: Props) => {
  const [active, setActive] = useState<SubDimension>('executability');

  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', gap: 0, borderBottom: '1px solid var(--color-border-tertiary)' }}>
        {(Object.keys(TAB_CONFIG) as SubDimension[]).map(tab => (
          <button
            key={tab}
            onClick={() => setActive(tab)}
            style={{
              padding: '10px 20px', fontSize: 13, fontWeight: 600, cursor: 'pointer', border: 'none',
              borderBottom: active === tab ? '2px solid var(--color-text-primary)' : '2px solid transparent',
              background: 'transparent',
              color: active === tab ? 'var(--color-text-primary)' : 'var(--color-text-tertiary)',
              marginBottom: '-1px',
            }}
          >
            {TAB_CONFIG[tab].label}
          </button>
        ))}
      </div>

      <div style={{ marginTop: 16, marginBottom: 16 }}>
        <p className="panel-description" style={{ margin: 0 }}>
          {TAB_CONFIG[active].description}
        </p>
      </div>

      {active === 'executability' && <ExecutabilityPanel experiment={experiment} />}
      {active === 'lint' && <LintPanel experiment={experiment} />}
    </div>
  );
};

export default StructuralCorrectness;

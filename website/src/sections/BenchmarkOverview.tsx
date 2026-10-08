import { useEffect, useState } from 'react';
import { AGENT_DESCRIPTIONS } from './utils/constants';
import { 
  Target, 
  Cpu, 
  Layers, 
  Zap, 
  BookOpen, 
  ShieldCheck,
  Code2,
  Activity,
  PlayCircle,
  FileCheck
} from 'lucide-react';


interface Stats {
  agentSpecifications: number;
  codeGenTools: number;
  evaluationFacets: number;
  totalGenerations: number;
}

const BenchmarkOverview = () => {
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => {
    fetch('/overviewStats.json')
      .then(res => res.json())
      .then(setStats)
      .catch(err => console.error('Failed to load stats:', err));
  }, []);

  return (
    <section id="overview" style={{ padding: '40px 24px', margin: '0 auto' }}>
      <div className="hero" style={{ marginBottom: '48px' }}>
        <h1 style={{ fontSize: '48px', fontWeight: 800, color: 'var(--color-text-primary)', marginBottom: '16px' }}>
            SPEC2AGENT
        </h1>
        <p className="subtitle" style={{ fontSize: '20px', color: 'var(--color-text-info)', fontWeight: 600, marginTop: '32px', marginBottom: '16px' }}>
          Evaluate Agent Generation by Code Generation Tools
        </p>
        <p className="description" style={{ fontSize: '16px', color: 'var(--color-text-secondary)', maxWidth: '850px', lineHeight: '1.6' }}>
          Spec2Agent is a benchmark designed to evaluate the generation of
          software agents from specifications...
        </p>
      </div>

      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: '0.8fr 1fr', 
        gap: '32px', 
        marginBottom: '48px',
        alignItems: 'stretch'
      }}>
        
        {/* Benchmark Overview */}
        <div style={{ 
          display: 'flex',
          flexDirection: 'column',
          height: '100%'
        }}>
          <h2 style={sectionTitleStyle}>Benchmark Overview</h2>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: '1fr 1fr', 
            gap: '16px',
            flex: 1
          }}>
            <StatCard label="Agent Specifications" value={stats?.agentSpecifications} icon={<Layers size={18}/>} />
            <StatCard label="Code Generation Tools" value={stats?.codeGenTools} icon={<Cpu size={18}/>} />
            <StatCard label="Evaluation Facets" value={stats?.evaluationFacets} icon={<Target size={18}/>} />
            <StatCard label="Total Generations" value={stats?.totalGenerations} icon={<Zap size={18}/>} />
          </div>
        </div>

        {/* Evaluation Dimensions */}
        <div style={{ 
          display: 'flex',
          flexDirection: 'column',
          height: '100%'
        }}>
          <h2 style={sectionTitleStyle}>Evaluation Dimensions</h2>
          <div style={{ 
            background: 'var(--color-background-primary)',
            padding: '20px',
            borderRadius: '12px',
            border: '1px solid var(--color-border-tertiary)',
            display: 'flex',
            flexDirection: 'column',
            gap: '16px',
            flex: 1
          }}>
            <DimensionItem title="Behavioral" desc="Correct decisions in scenarios" icon={<Activity size={16} color="#059669"/>} />
            <DimensionItem title="Structural" desc="Architecture conformity" icon={<ShieldCheck size={16} color="#185FA5"/>} />
            <DimensionItem title="Spec Compliance" desc="Adherence to specification" icon={<Code2 size={16} color="#7c3aed"/>} />
            <DimensionItem title="Execution" desc="Executability" icon={<PlayCircle size={16} color="#ea580c"/>} />
            <DimensionItem title="Lint" desc="Code quality" icon={<FileCheck size={16} color="#2563eb"/>} />
          </div>
        </div>
      </div>

      {/* Agent Catalog */}
      <div>
        <h2 style={sectionTitleStyle}>
          <BookOpen size={20} color="var(--color-text-info)" />
          Agent Catalog
        </h2>
        <div style={{ 
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
          gap: '20px'
        }}>
          {Object.entries(AGENT_DESCRIPTIONS).map(([name, description]) => (
            <div key={name} style={{
              padding: '20px',
              background: 'var(--color-background-primary)',
              borderRadius: '12px',
              border: '1px solid var(--color-border-tertiary)',
              display: 'flex',
              flexDirection: 'column',
              gap: '8px',
              boxShadow: '0 2px 4px rgba(0,0,0,0.02)'
            }}>
              <span style={{ 
                fontWeight: 800, 
                fontSize: '15px', 
                color: 'var(--color-text-info)', 
                textTransform: 'uppercase', 
                letterSpacing: '0.5px' 
              }}>
                {name}
              </span>
              <p style={{ 
                margin: 0, 
                fontSize: '13px', 
                color: 'var(--color-text-secondary)', 
                lineHeight: '1.5' 
              }}>
                {description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

/* --- Componentes Internos --- */
const StatCard = ({ label, value, icon }: { label: string, value: any, icon: React.ReactNode }) => (
  <div style={{
    padding: '20px',
    background: 'var(--color-background-primary)',
    borderRadius: '12px',
    border: '1px solid var(--color-border-tertiary)',
    display: 'flex',
    flexDirection: 'column',
    gap: '8px'
  }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--color-text-secondary)' }}>
      {icon}
      <span style={{ fontSize: '13px', fontWeight: 700, textTransform: 'uppercase' }}>{label}</span>
    </div>
    <span style={{ fontSize: '26px', fontWeight: 800, color: 'var(--color-text-primary)' }}>{value ?? '—'}</span>
  </div>
);

const DimensionItem = ({ title, desc, icon }: { title: string, desc: string, icon: React.ReactNode }) => (
  <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
    <div style={{ 
      minWidth: '32px', height: '32px', borderRadius: '8px', 
      background: 'var(--color-background-secondary)', 
      display: 'flex', alignItems: 'center', justifyContent: 'center' 
    }}>
      {icon}
    </div>
    <div>
      <div style={{ fontSize: '15px', fontWeight: 700, color: 'var(--color-text-primary)' }}>{title}</div>
      <div style={{ fontSize: '13px', color: 'var(--color-text-secondary)', lineHeight: '1.4' }}>{desc}</div>
    </div>
  </div>
);

const sectionTitleStyle: React.CSSProperties = {
  fontSize: '16px',
  fontWeight: 800,
  marginBottom: '20px',
  color: 'var(--color-text-primary)',
  textTransform: 'uppercase',
  letterSpacing: '1px',
  display: 'flex',
  alignItems: 'center',
  gap: '10px'
};

export default BenchmarkOverview;
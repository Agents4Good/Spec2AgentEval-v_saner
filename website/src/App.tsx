import { useState } from 'react';
import BenchmarkOverview from './sections/BenchmarkOverview';
import Dimensions from './sections/dimensions/Dimensions';
import './App.css';

type Tab = 'overview' | 'by_tool' | 'by_claude_version';

const App = () => {
  const [activeTab, setActiveTab] = useState<Tab>('overview');

  const renderContent = () => {
    switch (activeTab) {
      case 'overview': return <BenchmarkOverview />;
      case 'by_tool': return <Dimensions experiment="by_tool" title="By Tool" />;
      case 'by_claude_version': return <Dimensions experiment="by_claude_version" title="By Claude Version" />;
      default: return <BenchmarkOverview />;
    }
  };

  return (
    <div className="app-container">
      <nav className="top-nav">
        <div className="nav-tabs">
          <button
            className={activeTab === 'overview' ? 'active' : ''}
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </button>
          <button
            className={activeTab === 'by_tool' ? 'active' : ''}
            onClick={() => setActiveTab('by_tool')}
          >
            Cross Tools
          </button>
          <button
            className={activeTab === 'by_claude_version' ? 'active' : ''}
            onClick={() => setActiveTab('by_claude_version')}
          >
            Cross Claude Models
          </button>
        </div>
      </nav>

      <main className="page-content">
        {renderContent()}
      </main>
    </div>
  );
};

export default App;

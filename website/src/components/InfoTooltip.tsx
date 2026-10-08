import React from 'react';
import { getAgentDescription } from '../sections/utils/constants';

interface InfoTooltipProps {
  agentName?: string;
  text?: string;
}

const InfoTooltip: React.FC<InfoTooltipProps> = ({ agentName, text }) => {
  const tooltipContent = text || (agentName ? getAgentDescription(agentName) : "");

  if (!tooltipContent) return null;

  return (
    <div className="custom-tooltip-wrapper">
      <span className="info-icon-dot">i</span>
      <div className="tooltip-content-box">
        {tooltipContent}
      </div>
    </div>
  );
};

export default InfoTooltip;
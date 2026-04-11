import React from 'react';

const MetricBlock = ({ label, value, type = 'normal', suffix = '' }) => {
  let color = 'text-white';
  if (type === 'percent') {
    const num = parseFloat(value);
    if (num >= 80) color = 'text-emerald-500';
    else if (num >= 50) color = 'text-amber-500';
    else color = 'text-[#ea580c]'; // strict orange/red
  } else if (type === 'negative') {
    const num = parseFloat(value);
    if (num <= 20) color = 'text-emerald-500';
    else if (num <= 50) color = 'text-amber-500';
    else color = 'text-[#ea580c]';
  }

  return (
    <div className="flex flex-col border-l border-[#333] pl-5 first:border-0 first:pl-0">
      <span className="text-[11px] text-[#888] font-medium tracking-wide pb-1">{label}</span>
      <div className={`text-2xl font-mono tracking-tight ${color}`}>
        {value}<span className="text-sm ml-0.5 opacity-60">{suffix}</span>
      </div>
    </div>
  );
};

export default function ScoreBoard({ state }) {
  if (!state) return null;

  return (
    <div className="clean-card p-6 flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <div className="flex items-center justify-between">
          <h2 className="text-[17px] font-semibold text-white tracking-tight">System State</h2>
          <span className="text-[11px] font-mono text-[#666] border border-[#222] bg-black px-2 py-0.5 rounded">
            STEP {state.step_count} / {state.max_steps}
          </span>
        </div>
        <p className="text-[13px] text-[#888] leading-relaxed">
          Active Scenario: <span className="text-[#ededed] font-mono">{state.task_name}</span> ({state.task_variant}). 
          Monitor overall stability and financial limits below.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-6 pt-4 border-t border-[#222]">
        <MetricBlock 
          label="OVERALL HEALTH" 
          value={(state.system_health * 100).toFixed(1)} 
          type="percent" 
          suffix="%"
        />
        <MetricBlock 
          label="ON-CALL BUDGET" 
          value={state.remaining_budget} 
          type="normal"
        />
        <MetricBlock 
          label="IMPACT RATING" 
          value={state.customer_impact.toFixed(1)} 
          type="negative" 
          suffix="%"
        />
        <MetricBlock 
          label="SCORE" 
          value={((state.score || 0) * 100).toFixed(1)} 
          type="percent"
        />
      </div>

      {state.done && (
        <div className={`p-4 rounded-lg border text-[13px] font-mono mt-2 ${state.success ? 'bg-emerald-950/20 border-emerald-900 text-emerald-400' : 'bg-rose-950/20 border-rose-900 text-rose-400'}`}>
          <div className="flex items-center justify-between">
            <span className="font-semibold">
              [TERMINATED] {state.success ? 'SUCCESSFUL STABILIZATION' : 'SYSTEM FAILURE'}
            </span>
            <span className="opacity-80">BECAUSE: {state.reason}</span>
          </div>
        </div>
      )}
    </div>
  );
}

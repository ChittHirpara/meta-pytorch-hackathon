import React from 'react';

export default function ActionLog({ history = [] }) {
  // Show newest items at the top
  const displayHistory = [...history].reverse();

  return (
    <div className="clean-card mt-6 overflow-hidden flex flex-col h-[300px]">
      <div className="bg-[#111] border-b border-[#222] px-4 py-3 flex justify-between items-center z-10 shadow-sm relative">
        <span className="text-[11px] text-[#888] font-mono tracking-widest uppercase">Audit.Log // TTY</span>
      </div>
      
      <div className="p-4 flex-1 overflow-y-auto space-y-3 font-mono text-[13px]">
        {displayHistory.length === 0 && (
          <div className="text-[#666] italic text-[12px]">System booted. Listening for root interventions...</div>
        )}
        
        {displayHistory.map((log, i) => (
          <div key={i} className={`border-l pl-3 py-1 ${i === 0 ? 'border-white' : 'border-[#333]'}`}>
            <div className="text-[#666] mb-1 flex items-center justify-between text-[11px]">
              <span>[STEP {log.step}]</span>
              {log.reward !== undefined && (
                <span className={`${log.reward > 0 ? 'text-emerald-500' : log.reward < 0 ? 'text-[#ea580c]' : 'text-[#888]'}`}>
                  {log.reward > 0 ? '+' : ''}{log.reward.toFixed(4)}
                </span>
              )}
            </div>
            <div className={`text-[#ededed] ${i === 0 ? 'font-bold' : 'opacity-80'}`}>
              $ <span className={i === 0 ? 'text-white' : ''}>{log.action}</span>
              {log.target && <span className="text-[#888] ml-2">--target={log.target}</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

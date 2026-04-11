import React from 'react';

export default function AlertPanel({ alerts }) {
  if (!alerts || alerts.length === 0) {
    return (
      <div className="clean-card p-6 border-dashed text-center">
        <h2 className="text-[13px] font-semibold text-white tracking-widest uppercase mb-1">Incident Queue Empty</h2>
        <p className="text-[12px] text-[#666]">No unhandled routing anomalies or crash loops detected.</p>
      </div>
    );
  }

  return (
    <div className="clean-card p-6 flex flex-col">
      <div className="flex flex-col gap-1 mb-5">
        <h2 className="text-[17px] font-semibold text-white tracking-tight flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-[#ea580c] animate-pulse" /> Triggered Alerts
        </h2>
        <p className="text-[13px] text-[#888] leading-relaxed">
          The following symptoms have been isolated by Datadog/CloudWatch simulants.
        </p>
      </div>
      
      <div className="flex flex-col gap-2">
        {alerts.map((alert, i) => (
          <div 
            key={alert.id || i} 
            className={`p-4 rounded-xl border flex flex-col gap-2 ${alert.silenced ? 'bg-transparent border-[#222]' : 'bg-[#1a0a00] border-[#311]'}`}
          >
            <div className="flex justify-between items-center">
              <span className={`font-mono text-[12px] font-bold ${alert.silenced ? 'text-[#666]' : 'text-[#ea580c]'}`}>
                ALERT_{alert.id || i}
              </span>
              {alert.silenced && (
                <span className="text-[9px] font-bold bg-[#222] text-[#888] px-1.5 py-0.5 rounded border border-[#333] uppercase tracking-widest">ACKNOWLEDGED</span>
              )}
            </div>
            <div className={`text-[13px] leading-relaxed ${alert.silenced ? 'text-[#666]' : 'text-[#ededed]'}`}>
              {alert.symptom}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

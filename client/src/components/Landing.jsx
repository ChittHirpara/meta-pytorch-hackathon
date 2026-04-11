import React from 'react';

export default function Landing() {
  return (
    <div className="w-full max-w-6xl mx-auto px-6 pt-24 pb-16 flex flex-col items-center text-center">
      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-[#333] text-[11px] font-medium text-[#888] mb-8 uppercase tracking-widest hover:border-[#555] transition-colors cursor-default">
        Production Environment Ready
      </div>

      <h1 className="text-4xl md:text-6xl font-bold tracking-tight text-white mb-6">
        Incident Triage Interface
      </h1>

      <p className="text-[#888] text-sm md:text-base max-w-2xl mx-auto leading-relaxed">
        A deterministic environment for observing and mitigating distributed system failures. 
        Evaluate telemetry, resolve anomalies, and stabilize the architecture in real-time.
      </p>
    </div>
  );
}

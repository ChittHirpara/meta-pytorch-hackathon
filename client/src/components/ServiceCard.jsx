import React from 'react';

export default function ServiceCard({ service }) {
  const { name, health, error_rate, queue_depth, latency, inspected } = service;
  
  const healthPercent = Math.max(0, health * 100).toFixed(1);
  const isCritical = health < 0.5;

  return (
    <div className={`p-4 border rounded-xl bg-black transition-colors ${isCritical ? 'border-[#ea580c]' : 'border-[#333]'}`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className="font-mono text-[13px] text-white truncate pr-2" title={name}>
          <span className="text-[#666] mr-1">/</span>{name}
        </h3>
        <div className={`text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded-sm border ${inspected ? 'border-[#444] text-[#888]' : 'border-[#222] text-[#555]'}`}>
          {inspected ? 'SCANNED' : 'UNKNOWN'}
        </div>
      </div>

      <div className="space-y-4">
        <div>
          <div className="flex justify-between text-[11px] font-mono mb-1.5">
            <span className="text-[#666]">Health</span>
            <span className={isCritical ? 'text-[#ea580c]' : 'text-white'}>{healthPercent}%</span>
          </div>
          <div className="w-full bg-[#111] overflow-hidden rounded-full h-1">
            <div 
              className={`h-1 rounded-full ${isCritical ? 'bg-[#ea580c]' : 'bg-[#fff]'} transition-all duration-700 ease-out`}
              style={{ width: `${healthPercent}%` }}
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-x-4 gap-y-3">
          <div className="flex items-center justify-between col-span-1">
            <span className="text-[11px] text-[#666] font-mono">Queue</span>
            <span className={`font-mono text-[12px] ${queue_depth > 100 ? 'text-[#ea580c]' : 'text-[#ededed]'}`}>
              {queue_depth}
            </span>
          </div>
          <div className="flex items-center justify-between col-span-1">
            <span className="text-[11px] text-[#666] font-mono">Errs</span>
            <span className={`font-mono text-[12px] ${error_rate > 0.1 ? 'text-[#ea580c]' : 'text-[#ededed]'}`}>
              {(error_rate * 100).toFixed(1)}%
            </span>
          </div>
          <div className="flex items-center justify-between col-span-2">
            <span className="text-[11px] text-[#666] font-mono">Latency</span>
            <span className="font-mono text-[12px] text-[#ededed]">{latency} ms</span>
          </div>
        </div>
      </div>
    </div>
  );
}

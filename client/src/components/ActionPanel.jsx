import React, { useState } from 'react';
import { ACTIONS } from '../utils/constants';

export default function ActionPanel({ services = [], onStep, disabled }) {
  const [action, setAction] = useState('noop');
  const [target, setTarget] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    onStep({ action, target: action === 'noop' ? undefined : target });
  };

  return (
    <div className="clean-card p-6 mt-6">
      <div className="flex flex-col gap-1 mb-5">
        <h2 className="text-[17px] font-semibold text-white tracking-tight">2. Manual Escalation</h2>
        <p className="text-[13px] text-[#888] leading-relaxed">
          Issue triage commands directly to the backend resolver. You must act fast to preserve the system health budget.
        </p>
      </div>
      
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <div>
          <label className="block text-[11px] font-medium text-[#666] mb-1.5 uppercase tracking-wide">Select Operation</label>
          <select
            value={action}
            onChange={(e) => setAction(e.target.value)}
            disabled={disabled}
            className="w-full bg-black border border-[#333] text-[#ededed] text-[13px] rounded-lg px-4 py-2.5 outline-none hover:border-[#555] focus:border-white transition-colors font-mono"
          >
            {ACTIONS.map(a => <option key={a} value={a}>{a}</option>)}
          </select>
        </div>

        <div>
           <label className="block text-[11px] font-medium text-[#666] mb-1.5 uppercase tracking-wide">Specify Target Namespace</label>
          <select
            value={target}
            onChange={(e) => setTarget(e.target.value)}
            disabled={disabled || action === 'noop'}
            className="w-full bg-black border border-[#333] text-[#ededed] text-[13px] rounded-lg px-4 py-2.5 outline-none hover:border-[#555] focus:border-white transition-colors font-mono disabled:opacity-30 disabled:hover:border-[#333]"
          >
            <option value="">-- Require Selection --</option>
            {services.map(s => <option key={s.name} value={s.name}>{s.name}</option>)}
          </select>
        </div>

        <button
          type="submit"
          disabled={disabled || (action !== 'noop' && !target)}
          className="w-full mt-2 bg-white hover:bg-[#eaeaea] text-black text-[13px] font-semibold py-3 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed uppercase tracking-widest"
        >
          Execute Protocol
        </button>
      </form>
    </div>
  );
}

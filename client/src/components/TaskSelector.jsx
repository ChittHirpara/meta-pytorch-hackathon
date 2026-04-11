import React, { useEffect, useState } from 'react';
import { getTasks } from '../services/api';

export default function TaskSelector({ onSelectTask, isLoading }) {
  const [tasks, setTasks] = useState([]);
  const [selected, setSelected] = useState('');

  useEffect(() => {
    getTasks()
      .then((res) => {
        if (res.ok && res.tasks) {
          setTasks(res.tasks);
          if (res.tasks.length > 0) setSelected(res.tasks[0]);
        }
      })
      .catch((err) => console.error('Failed to load tasks', err));
  }, []);

  return (
    <div className="clean-card p-6 flex flex-col gap-5">
      <div className="flex flex-col gap-1">
        <h2 className="text-[17px] font-semibold text-white tracking-tight">1. Initialize Scenario</h2>
        <p className="text-[13px] text-[#888] leading-relaxed">
          Select an incident type. Once initialized, the backend spins up an isolated sandbox. This must be done before the NOC dashboard can observe telemetry.
        </p>
      </div>
      
      <div className="flex flex-col sm:flex-row gap-3">
        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          disabled={isLoading}
          className="flex-1 bg-black border border-[#333] text-[13px] text-[#ededed] rounded-lg px-4 py-3 outline-none hover:border-[#555] focus:border-white transition-colors appearance-none font-mono disabled:opacity-50"
        >
          {tasks.map((taskName) => (
            <option key={taskName} value={taskName}>
              {taskName.toUpperCase()}
            </option>
          ))}
        </select>
        
        <button
          onClick={() => selected && onSelectTask(selected)}
          disabled={!selected || isLoading}
          className="bg-white hover:bg-[#eaeaea] text-black text-[13px] font-semibold px-8 py-3 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed uppercase tracking-wide"
        >
          {isLoading ? 'Starting...' : 'Start Session'}
        </button>
      </div>
    </div>
  );
}

import React, { useState, useEffect } from 'react';
import TaskSelector from '../components/TaskSelector';
import ScoreBoard from '../components/ScoreBoard';
import ServiceCard from '../components/ServiceCard';
import AlertPanel from '../components/AlertPanel';
import ActionPanel from '../components/ActionPanel';
import ActionLog from '../components/ActionLog';
import { getState, resetEnv, stepEnv } from '../services/api';

export default function Dashboard() {
  const [envState, setEnvState] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isActive, setIsActive] = useState(false);

  // Less aggressive polling (5 seconds) to reduce network logs
  useEffect(() => {
    if (!isActive || envState?.done) return;
    
    const poll = async () => {
      try {
        const res = await getState();
        if (res.ok && res.state) {
          setEnvState(res.state);
        }
      } catch (e) {
        if (e.response && e.response.status === 400) {
          setIsActive(false);
          setEnvState(null);
        }
      }
    };
    
    poll();
    const interval = setInterval(poll, 5000);
    return () => clearInterval(interval);
  }, [isActive, envState?.done]);

  const handleReset = async (taskName) => {
    setLoading(true);
    try {
      const res = await resetEnv(taskName);
      if (res.ok) {
        const stateRes = await getState();
        setEnvState(stateRes.state);
        setIsActive(true);
      }
    } catch (e) {
      console.error(e);
      alert('Failed to reset environment');
    } finally {
      setLoading(false);
    }
  };

  const handleStep = async (actionPayload) => {
    setLoading(true);
    try {
      await stepEnv(actionPayload);
      const stateRes = await getState();
      setEnvState(stateRes.state);
    } catch (e) {
      console.error(e);
      alert('Failed to execute action');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      
      {/* Configuration Column */}
      <div className="lg:col-span-5 flex flex-col gap-6">
        <TaskSelector onSelectTask={handleReset} isLoading={loading} />
        {envState && (
          <>
            <ActionPanel 
              services={envState.services} 
              onStep={handleStep} 
              disabled={loading || envState.done} 
            />
            <ActionLog history={envState.action_history} />
          </>
        )}
      </div>

      {/* Observation Column */}
      <div className="lg:col-span-7 flex flex-col gap-6">
        {envState ? (
          <>
            <ScoreBoard state={envState} />
            <div className="clean-card p-6">
              <div className="flex flex-col gap-1 mb-6">
                <h2 className="text-[17px] font-semibold text-white tracking-tight">Active Topology</h2>
                <p className="text-[13px] text-[#888] leading-relaxed">
                  Real-time telemetry from dependent backend services. Wait for inspections to reveal unknown errors.
                </p>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {envState.services.map(s => (
                  <ServiceCard key={s.name} service={s} />
                ))}
              </div>
            </div>
            <AlertPanel alerts={envState.active_alerts} />
          </>
        ) : (
          <div className="clean-card p-12 flex flex-col items-center justify-center h-[500px] border-dashed text-center">
            <div className="w-12 h-12 border border-[#333] border-t-[#fff] rounded-full animate-spin mb-6" />
            <h2 className="text-[15px] font-semibold text-white mb-2 tracking-tight">Simulation Offline</h2>
            <p className="text-[13px] text-[#666] max-w-[250px] leading-relaxed">
              Complete Step 1 on the left side to compile the architecture and stream logs.
            </p>
          </div>
        )}
      </div>

    </div>
  );
}

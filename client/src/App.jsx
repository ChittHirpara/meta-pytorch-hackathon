import React from 'react';
import Dashboard from './pages/Dashboard';
import Landing from './components/Landing';

export default function App() {
  return (
    <div className="min-h-screen bg-black overflow-x-hidden selection:bg-white selection:text-black">
      {/* Navigation Bar */}
      <nav className="w-full border-b border-[#222] bg-[#000] sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-6 h-6 bg-white flex items-center justify-center rounded-sm">
              <div className="w-3 h-3 bg-black rounded-sm" />
            </div>
            <h1 className="text-sm font-semibold tracking-wide text-white">TRIAGE-X</h1>
          </div>
          
          <div className="hidden md:flex items-center gap-6 text-[13px] text-[#888]">
            <span className="cursor-pointer hover:text-white transition-colors">Documentation</span>
            <span className="cursor-pointer hover:text-white transition-colors">API Reference</span>
            <span className="cursor-pointer hover:text-white transition-colors">Support</span>
          </div>
          
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <span className="text-[11px] font-mono text-[#888] uppercase tracking-widest">NOC Active</span>
          </div>
        </div>
      </nav>

      {/* Hero Region */}
      <Landing />

      {/* Main Framework */}
      <main className="max-w-6xl mx-auto px-6 pb-24 relative z-20">
        <Dashboard />
      </main>

      {/* Footer */}
      <footer className="w-full border-t border-[#222] bg-black py-8 mt-12">
        <div className="max-w-6xl mx-auto px-6 flex justify-between items-center text-[11px] text-[#666]">
          <span>© 2026 Triage-X Systems</span>
          <div className="flex gap-4">
            <span className="cursor-pointer hover:text-[#ededed]">Terms</span>
            <span className="cursor-pointer hover:text-[#ededed]">Privacy</span>
            <span className="cursor-pointer hover:text-[#ededed]">Status</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

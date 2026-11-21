import React from 'react';
import { Sliders, Play, TrendingUp } from 'lucide-react';

export default function Controls({
  scenarios,
  selectedScenario,
  setSelectedScenario,
  snr,
  setSnr,
  speed,
  setSpeed,
  onRunEstimate,
  onRunSweep,
  loadingEstimate,
  loadingSweep,
  backendOnline,
}) {
  // Compute Doppler frequency: fd = v * fc / c at 3.5 GHz
  const fc = 3.5e9;
  const v_ms = speed / 3.6;
  const dopplerHz = (v_ms * fc / 3.0e8).toFixed(1);

  return (
    <div className="gw-panel border border-[#1E293B]/80 rounded-xl p-5 space-y-4">
      {/* Header Row */}
      <div className="flex items-center justify-between border-b border-[#1E293B]/70 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-[#1E293B]/70 border border-[#334155]/60 flex items-center justify-center text-[#60A5FA]">
            <Sliders size={14} strokeWidth={1.5} />
          </div>
          <div>
            <h2 className="text-xs font-semibold text-[#F8FAFC]">
              Simulation parameters
            </h2>
            <p className="text-[11px] text-[#94A3B8]">
              Channel settings for 64×64 massive MIMO array
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <span
            className={`w-2 h-2 rounded-full ${
              backendOnline ? 'bg-[#D4FF00] shadow-sm shadow-[#D4FF00]/50' : 'bg-rose-500'
            }`}
          />
          <span className="text-[11px] font-mono text-[#94A3B8]">
            {backendOnline ? 'FastAPI connected' : 'Connecting...'}
          </span>
        </div>
      </div>

      {/* 4-Column Horizontal Parameter Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 items-end">
        {/* Col 1: Scenario Selection */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-[#94A3B8] block">
            Channel scenario
          </label>
          <select
            value={selectedScenario}
            onChange={(e) => setSelectedScenario(e.target.value)}
            className="w-full bg-[#0B1120] border border-[#1E293B] focus:border-[#2563EB] rounded-lg px-3 py-1.5 text-xs text-[#F8FAFC] font-medium focus:outline-none transition cursor-pointer"
          >
            {scenarios.map((sc) => (
              <option key={sc.id} value={sc.id} className="bg-[#0B1120] text-[#F8FAFC]">
                {sc.name} ({sc.type})
              </option>
            ))}
          </select>
          <span className="text-[10px] text-[#64748B] font-mono block">
            Profile: {scenarios.find(s => s.id === selectedScenario)?.delay_spread || 'Standard TDL'}
          </span>
        </div>

        {/* Col 2: SNR Setting */}
        <div className="space-y-1.5">
          <div className="flex justify-between items-center">
            <label className="text-xs font-medium text-[#94A3B8]">
              SNR
            </label>
            <span className="text-xs font-mono font-medium text-[#D4FF00] bg-[#0B1120] border border-[#1E293B] px-2 py-0.5 rounded">
              {snr >= 0 ? `+${snr}` : snr} dB
            </span>
          </div>
          <input
            type="range"
            min="-15"
            max="20"
            step="1"
            value={snr}
            onChange={(e) => setSnr(parseFloat(e.target.value))}
            className="w-full accent-[#2563EB] cursor-pointer h-1.5 bg-[#0B1120] rounded border border-[#1E293B]"
          />
          {/* Quick presets */}
          <div className="flex gap-1 justify-between pt-0.5">
            {[-10, 0, 10, 20].map((v) => (
              <button
                key={v}
                type="button"
                onClick={() => setSnr(v)}
                className={`text-[10px] font-mono px-2 py-0.5 rounded border transition cursor-pointer ${
                  snr === v
                    ? 'bg-[#2563EB] text-white border-[#2563EB] shadow-xs'
                    : 'bg-[#0B1120] text-[#94A3B8] border-[#1E293B] hover:bg-[#1E293B] hover:text-[#F8FAFC]'
                }`}
              >
                {v >= 0 ? `+${v}` : v}
              </button>
            ))}
          </div>
        </div>

        {/* Col 3: Vehicular Speed & Doppler */}
        <div className="space-y-1.5">
          <div className="flex justify-between items-center">
            <label className="text-xs font-medium text-[#94A3B8]">
              User velocity
            </label>
            <span className="text-xs font-mono font-medium text-[#60A5FA] bg-[#0B1120] border border-[#1E293B] px-2 py-0.5 rounded">
              {speed} km/h
            </span>
          </div>
          <input
            type="range"
            min="3"
            max="500"
            step="10"
            value={speed}
            onChange={(e) => setSpeed(parseFloat(e.target.value))}
            className="w-full accent-[#2563EB] cursor-pointer h-1.5 bg-[#0B1120] rounded border border-[#1E293B]"
          />
          <div className="flex justify-between items-center text-[10px] text-[#64748B] font-mono pt-0.5">
            <span>Doppler: {dopplerHz} Hz</span>
            <span>Array: 64×64</span>
          </div>
        </div>

        {/* Col 4: Action Buttons */}
        <div className="flex flex-col gap-2">
          <button
            type="button"
            onClick={onRunEstimate}
            disabled={loadingEstimate || !backendOnline}
            className="w-full h-8 flex items-center justify-center gap-1.5 bg-[#2563EB] hover:bg-[#1D4ED8] disabled:opacity-40 text-white font-medium px-3 rounded-lg transition text-xs cursor-pointer border border-[#3B82F6] shadow-sm shadow-blue-500/20"
          >
            {loadingEstimate ? (
              <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <Play size={13} strokeWidth={2} />
            )}
            <span>{loadingEstimate ? 'Estimating...' : 'Run estimation'}</span>
          </button>

          <button
            type="button"
            onClick={onRunSweep}
            disabled={loadingSweep || !backendOnline}
            className="w-full h-8 flex items-center justify-center gap-1.5 bg-[#0B1120] hover:bg-[#1E293B] disabled:opacity-40 text-[#F8FAFC] border border-[#1E293B] hover:border-[#2563EB] font-medium px-3 rounded-lg transition text-xs cursor-pointer"
          >
            {loadingSweep ? (
              <div className="w-3.5 h-3.5 border-2 border-[#2563EB] border-t-transparent rounded-full animate-spin" />
            ) : (
              <TrendingUp size={13} strokeWidth={1.5} className="text-[#60A5FA]" />
            )}
            <span>{loadingSweep ? 'Sweeping...' : 'Run SNR sweep'}</span>
          </button>
        </div>
      </div>
    </div>
  );
}

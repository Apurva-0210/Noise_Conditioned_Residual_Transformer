import React from 'react';
import { Sliders, Zap, TrendingUp, Radio, Activity, Sparkles } from 'lucide-react';

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
  onOpenBackendSettings,
}) {
  // Compute Doppler frequency: fd = v * fc / c at 3.5 GHz
  const fc = 3.5e9;
  const v_ms = speed / 3.6;
  const dopplerHz = (v_ms * fc / 3.0e8).toFixed(1);

  return (
    <div className="bg-[#F2F3E2] border border-[#CCD3B5] rounded-xl p-4 shadow-subtle space-y-4">
      {/* Header Row */}
      <div className="flex items-center justify-between border-b border-[#CCD3B5]/70 pb-2.5">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-md bg-[#789575]/20 flex items-center justify-center border border-[#789575]/30">
            <Sliders size={14} className="text-[#263629]" />
          </div>
          <div>
            <h2 className="text-xs font-black uppercase tracking-wider text-[#263629]">
              3GPP Simulation Control Panel
            </h2>
            <p className="text-[10px] text-[#647B62] font-medium">
              Real-time parameter adjustment for 64×64 massive MIMO channel estimation
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={onOpenBackendSettings}
          className="flex items-center gap-2 text-xs px-2.5 py-1 rounded-md hover:bg-[#EEF0DC] border border-transparent hover:border-[#CCD3B5] transition cursor-pointer"
          title="Click to view or edit backend connection URL"
        >
          <span
            className={`w-2 h-2 rounded-full ${
              backendOnline ? 'bg-[#789575] animate-pulse' : 'bg-rose-500'
            }`}
          />
          <span className="text-[11px] font-mono text-[#647B62] font-semibold underline decoration-dotted">
            {backendOnline ? 'FastAPI Connected' : 'Offline (Click to Connect)'}
          </span>
        </button>
      </div>

      {/* 4-Column Horizontal Parameter Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 items-end">
        {/* Col 1: Scenario Selection */}
        <div className="space-y-1.5">
          <label className="text-[11px] font-bold text-[#647B62] block uppercase tracking-wider">
            Channel Scenario
          </label>
          <select
            value={selectedScenario}
            onChange={(e) => setSelectedScenario(e.target.value)}
            className="w-full bg-[#EEF0DC] border border-[#CCD3B5] focus:border-[#789575] rounded-lg px-2.5 py-1.5 text-xs text-[#263629] font-bold focus:outline-none transition shadow-sm cursor-pointer"
          >
            {scenarios.map((sc) => (
              <option key={sc.id} value={sc.id}>
                {sc.name} ({sc.type})
              </option>
            ))}
          </select>
          <span className="text-[9.5px] text-[#647B62] font-mono block">
            Profile: <strong>{scenarios.find(s => s.id === selectedScenario)?.delay_spread || 'Standard TDL'}</strong>
          </span>
        </div>

        {/* Col 2: SNR Setting */}
        <div className="space-y-1.5">
          <div className="flex justify-between items-center">
            <label className="text-[11px] font-bold text-[#647B62] flex items-center gap-1 uppercase tracking-wider">
              <Radio size={12} className="text-[#789575]" /> SNR Setting
            </label>
            <span className="text-xs font-bold text-[#263629] font-mono bg-[#EEF0DC] border border-[#CCD3B5] px-2 py-0.5 rounded shadow-inner">
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
            className="w-full accent-[#789575] cursor-pointer h-1.5 bg-[#EEF0DC] rounded border border-[#CCD3B5]"
          />
          {/* Quick presets */}
          <div className="flex gap-1 justify-between pt-0.5">
            {[-10, 0, 10, 20].map((v) => (
              <button
                key={v}
                onClick={() => setSnr(v)}
                className={`text-[9px] font-mono px-1.5 py-0.5 rounded border transition cursor-pointer ${
                  snr === v
                    ? 'bg-[#789575] text-[#F2F3E2] border-[#789575]'
                    : 'bg-[#EEF0DC] text-[#647B62] border-[#CCD3B5] hover:bg-[#E5E8CE]'
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
            <label className="text-[11px] font-bold text-[#647B62] flex items-center gap-1 uppercase tracking-wider">
              <Activity size={12} className="text-[#789575]" /> User Velocity
            </label>
            <span className="text-xs font-bold text-[#263629] font-mono bg-[#EEF0DC] border border-[#CCD3B5] px-2 py-0.5 rounded shadow-inner">
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
            className="w-full accent-[#789575] cursor-pointer h-1.5 bg-[#EEF0DC] rounded border border-[#CCD3B5]"
          />
          <div className="flex justify-between items-center text-[9.5px] text-[#647B62] font-mono pt-0.5">
            <span>Doppler: <strong className="text-[#263629]">{dopplerHz} Hz</strong></span>
            <span>Array: <strong className="text-[#263629]">64×64</strong></span>
          </div>
        </div>

        {/* Col 4: Action Buttons */}
        <div className="flex flex-col gap-1.5">
          <button
            onClick={onRunEstimate}
            disabled={loadingEstimate || !backendOnline}
            className="w-full flex items-center justify-center gap-1.5 bg-[#789575] hover:bg-[#668263] active:scale-[0.99] disabled:opacity-50 text-[#F2F3E2] font-bold py-2 px-3 rounded-lg shadow-sm transition text-xs cursor-pointer border border-[#789575]"
          >
            {loadingEstimate ? (
              <div className="w-3.5 h-3.5 border-2 border-[#F2F3E2] border-t-transparent rounded-full animate-spin" />
            ) : (
              <Zap size={14} className="text-[#F2F3E2]" />
            )}
            <span>{loadingEstimate ? 'Inferring...' : 'Run Real-Time Simulation'}</span>
          </button>

          <button
            onClick={onRunSweep}
            disabled={loadingSweep || !backendOnline}
            className="w-full flex items-center justify-center gap-1.5 bg-[#EEF0DC] hover:bg-[#E5E8CE] disabled:opacity-50 text-[#263629] border border-[#CCD3B5] hover:border-[#789575] font-bold py-1.5 px-3 rounded-lg transition text-xs cursor-pointer shadow-sm"
          >
            {loadingSweep ? (
              <div className="w-3.5 h-3.5 border-2 border-[#789575] border-t-transparent rounded-full animate-spin" />
            ) : (
              <TrendingUp size={13} className="text-[#789575]" />
            )}
            <span>{loadingSweep ? 'Running Sweep...' : 'Run Full SNR Sweep'}</span>
          </button>
        </div>
      </div>
    </div>
  );
}

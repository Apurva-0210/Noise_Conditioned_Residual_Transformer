import React, { useState, useEffect } from 'react';
import { 
  Radio, 
  Cpu, 
  Gauge, 
  ShieldCheck, 
  Layers, 
  BarChart3, 
  Sparkles,
  Server,
  Activity,
  Zap,
  Camera,
  SlidersHorizontal,
  ChevronRight,
  Move,
  Scan,
  Download
} from 'lucide-react';

import { checkHealth, fetchScenarios, runEstimation, runSweep, getApiBase } from './api';
import MetricCard from './components/MetricCard';
import Controls from './components/Controls';
import ChannelHeatmap from './components/ChannelHeatmap';
import PerformanceCurve from './components/PerformanceCurve';
import MatlabStudio from './components/MatlabStudio';
import RightSidebar from './components/RightSidebar';

export default function App() {
  const [backendOnline, setBackendOnline] = useState(false);
  const [scenarios, setScenarios] = useState([
    { id: 'TDL-A', name: '3GPP TDL-A', type: 'NLOS', delay_spread: 'High (300 ns)' },
    { id: 'TDL-B', name: '3GPP TDL-B', type: 'NLOS', delay_spread: 'Medium (100 ns)' },
    { id: 'TDL-C', name: '3GPP TDL-C', type: 'NLOS', delay_spread: 'Long (1000 ns)' },
    { id: 'TDL-D', name: '3GPP TDL-D', type: 'LOS', delay_spread: 'Low (30 ns)' },
    { id: 'TDL-E', name: '3GPP TDL-E', type: 'LOS', delay_spread: 'Extreme' },
    { id: 'random', name: 'Random Mix', type: 'Mixed', delay_spread: 'Variable' },
  ]);
  const [selectedScenario, setSelectedScenario] = useState('TDL-A');
  const [snr, setSnr] = useState(5.0);
  const [speed, setSpeed] = useState(30.0);

  const [activeTab, setActiveTab] = useState('heatmaps'); // 'heatmaps' | 'sweep' | 'matlab_6g'
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [chartDragMode, setChartDragMode] = useState('zoom'); // 'zoom' | 'pan'

  const [loadingEstimate, setLoadingEstimate] = useState(false);
  const [loadingSweep, setLoadingSweep] = useState(false);

  const [estimateData, setEstimateData] = useState(null);
  const [sweepData, setSweepData] = useState(null);
  const [matlabMetrics, setMatlabMetrics] = useState(null);

  // Initial health check & data loading
  useEffect(() => {
    async function init() {
      try {
        const health = await checkHealth();
        setBackendOnline(health.status === 'online');
        const scList = await fetchScenarios();
        setScenarios(scList);
      } catch (err) {
        console.warn('Backend not yet reachable:', err);
        setBackendOnline(false);
      }
    }
    init();
    const interval = setInterval(init, 5000);
    return () => clearInterval(interval);
  }, []);

  // Run a default estimation once connected
  useEffect(() => {
    if (backendOnline && !estimateData && !loadingEstimate) {
      handleRunEstimate();
    }
  }, [backendOnline]);

  const handleRunEstimate = async () => {
    setLoadingEstimate(true);
    try {
      const data = await runEstimation({
        scenario: selectedScenario,
        snr_db: snr,
        speed_kmh: speed,
        selected_model: 'ncrt',
      });
      setEstimateData(data);
    } catch (err) {
      console.error('Estimation error:', err);
    } finally {
      setLoadingEstimate(false);
    }
  };

  const handleRunSweep = async () => {
    setLoadingSweep(true);
    setActiveTab('sweep');
    try {
      const data = await runSweep({
        scenario: selectedScenario,
        snr_range: [-15, -10, -5, 0, 5, 10, 15, 20],
        num_samples: 5,
      });
      setSweepData(data);
    } catch (err) {
      console.error('Sweep error:', err);
    } finally {
      setLoadingSweep(false);
    }
  };

  // Plotly chart actions
  const handleDownloadActivePlot = () => {
    const plots = document.querySelectorAll('.js-plotly-plot');
    if (plots.length > 0 && window.Plotly) {
      window.Plotly.downloadImage(plots[0], {
        format: 'png',
        width: 900,
        height: 700,
        filename: `channel_plot_${selectedScenario}_${snr}dB`,
      });
    }
  };

  const handleDownloadAllImages = () => {
    const plots = document.querySelectorAll('.js-plotly-plot');
    plots.forEach((p, idx) => {
      if (window.Plotly) {
        setTimeout(() => {
          window.Plotly.downloadImage(p, {
            format: 'png',
            width: 800,
            height: 700,
            filename: `mimo_panel_${idx + 1}_${snr}dB`,
          });
        }, idx * 400);
      }
    });
  };

  const handleResetZoom = () => {
    const plots = document.querySelectorAll('.js-plotly-plot');
    plots.forEach((p) => {
      if (window.Plotly) {
        window.Plotly.relayout(p, {
          'xaxis.autorange': true,
          'yaxis.autorange': true,
        });
      }
    });
  };

  // Metrics for 3GPP TDL tab
  const ncrtRes = estimateData?.results?.ncrt;
  const ncrnRes = estimateData?.results?.ncrn;
  const lsRes = estimateData?.results?.ls;
  const gainOverLs = ncrtRes && lsRes ? (lsRes.nmse_db - ncrtRes.nmse_db).toFixed(2) : null;

  // Metrics for 6G Sub-THz tab
  const ncrt6gNmse = matlabMetrics?.ncrt_6g?.nmse_db;
  const ncrtBaseNmse = matlabMetrics?.ncrt_base?.nmse_db;
  const ls6gNmse = matlabMetrics?.ls?.nmse_db;
  const gainOverLs6g = (ncrt6gNmse !== undefined && ls6gNmse !== undefined) 
    ? (ls6gNmse - ncrt6gNmse).toFixed(2) 
    : '--';

  // Shared Z-scale for 3GPP TDL tab
  const [sharedColorScaleTDL, setSharedColorScaleTDL] = useState(true);
  let sharedZRangeTDL = null;
  if (sharedColorScaleTDL && estimateData?.matrices) {
    const mats = [
      estimateData.matrices.true_channel,
      estimateData.matrices.noisy_ls,
      estimateData.matrices.reconstructed,
    ].filter(Boolean);
    let minZ = Infinity;
    let maxZ = -Infinity;
    mats.forEach((m) => {
      for (let r = 0; r < m.length; r++) {
        for (let c = 0; c < m[r].length; c++) {
          if (m[r][c] < minZ) minZ = m[r][c];
          if (m[r][c] > maxZ) maxZ = m[r][c];
        }
      }
    });
    if (minZ !== Infinity && maxZ !== -Infinity) {
      sharedZRangeTDL = [minZ, maxZ];
    }
  }

  return (
    <div className="min-h-screen bg-[#E5E8CE] text-[#263629] flex flex-col antialiased selection:bg-[#789575] selection:text-[#F2F3E2]">
      {/* Top Header */}
      <header className="border-b border-[#CCD3B5] bg-[#EEF0DC]/95 backdrop-blur-xl sticky top-0 z-40 shadow-subtle">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          {/* Logo & Paper Title */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-[#789575] border border-[#789575] flex items-center justify-center shadow-sm">
              <Radio className="text-[#F2F3E2]" size={22} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-black text-[#263629] tracking-tight">
                  Massive MIMO AI Channel Studio
                </h1>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-black bg-[#789575]/20 text-[#263629] border border-[#789575]/40 uppercase tracking-wider">
                  v2.1 Scientific Suite
                </span>
              </div>
              <p className="text-xs text-[#647B62] font-medium">
                Attention-Based Residual Learning Channel Estimation for Massive MIMO
              </p>
            </div>
          </div>

          {/* Quick Status & Chart Tools Button */}
          <div className="flex items-center gap-3 text-xs">
            <div className="hidden md:flex items-center gap-2 bg-[#E5E8CE] px-3 py-1.5 rounded-lg border border-[#CCD3B5] shadow-inner">
              <Server size={14} className="text-[#647B62]" />
              <span className="text-[#647B62] font-medium hidden sm:inline">FastAPI:</span>
              <span className={`font-mono font-bold flex items-center gap-1.5 ${backendOnline ? 'text-[#263629]' : 'text-rose-600'}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${backendOnline ? 'bg-emerald-600' : 'bg-rose-500 animate-pulse'}`} />
                {backendOnline ? 'Active (Cloud)' : 'Connecting...'}
              </span>
            </div>

            <div className="hidden sm:flex items-center gap-1.5 bg-[#789575]/15 text-[#263629] px-3 py-1.5 rounded-lg border border-[#789575]/40 font-bold font-mono">
              <Sparkles size={14} className="text-[#789575]" />
              <span>64×64 Array</span>
            </div>

            {/* Chart Tools & Exports Drawer Toggle */}
            <button
              onClick={() => setSidebarOpen(true)}
              className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg font-bold transition cursor-pointer border bg-[#789575] hover:bg-[#668263] text-[#F2F3E2] border-[#789575] shadow-sm active:scale-[0.98]"
              title="Open Chart Tools, Pan/Box Modes, and Scientific Exports"
            >
              <Camera size={15} />
              <span className="text-xs">Tools & Downloads</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Container — Spacious Full Width, No Side Compression */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Dynamic Context-Aware KPI Ribbon */}
        {activeTab === 'matlab_6g' ? (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3.5">
            <MetricCard
              title="NCRT (6G-Trained)"
              value={ncrt6gNmse !== undefined ? `${ncrt6gNmse}` : '-10.16'}
              unit="dB"
              subtitle="Specialized 6G Model"
              badge="142 GHz Ray-Tracing"
              badgeVariant="accent"
              isHighlighted={true}
              icon={Cpu}
            />
            <MetricCard
              title="NCRT Base Generalist"
              value={ncrtBaseNmse !== undefined ? `${ncrtBaseNmse}` : '-7.56'}
              unit="dB"
              subtitle="Synthetic TDL baseline"
              badge="888K params"
              badgeVariant="default"
              icon={Layers}
            />
            <MetricCard
              title="6G Gain over LS"
              value={gainOverLs6g !== '--' ? `+${gainOverLs6g}` : '+2.60'}
              unit="dB"
              subtitle="Noise suppression advantage"
              badge="Denoising Gain"
              badgeVariant="success"
              icon={Gauge}
            />
            <MetricCard
              title="Physics & Geometry"
              value="142.0"
              unit="GHz"
              subtitle="64×64 UPA Planar Array"
              badge="Virtual Beamspace"
              badgeVariant="warning"
              icon={Activity}
            />
          </div>
        ) : (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3.5">
            <MetricCard
              title="NCRT (Proposed)"
              value={ncrtRes ? `${ncrtRes.nmse_db}` : '--'}
              unit="dB"
              subtitle="NMSE (Lower is better)"
              badge="888K params (-26%)"
              badgeVariant="accent"
              isHighlighted={true}
              icon={Cpu}
            />
            <MetricCard
              title="NCRN Baseline"
              value={ncrnRes ? `${ncrnRes.nmse_db}` : '--'}
              unit="dB"
              subtitle="ConvLSTM residual estimator"
              badge="1.20M params"
              badgeVariant="default"
              icon={Layers}
            />
            <MetricCard
              title="NCRT Gain over LS"
              value={gainOverLs ? `+${gainOverLs}` : '--'}
              unit="dB"
              subtitle={`At SNR = ${snr} dB`}
              badge="Noise Reduction"
              badgeVariant="success"
              icon={Gauge}
            />
            <MetricCard
              title="Inference Latency"
              value={ncrtRes ? `${ncrtRes.latency_ms}` : '--'}
              unit="ms"
              subtitle="Real-time execution speed"
              badge="O-RAN Near-RT"
              badgeVariant="warning"
              icon={ShieldCheck}
            />
          </div>
        )}

        {/* Clean Segmented Navigation Tab Control */}
        <div className="flex items-center justify-between border-b border-[#CCD3B5] pb-2">
          <div className="inline-flex bg-[#EEF0DC] p-1 rounded-xl border border-[#CCD3B5] shadow-inner text-xs font-bold gap-1">
            <button
              onClick={() => setActiveTab('heatmaps')}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg transition cursor-pointer ${
                activeTab === 'heatmaps'
                  ? 'bg-[#789575] text-[#F2F3E2] shadow-sm'
                  : 'text-[#647B62] hover:text-[#263629]'
              }`}
            >
              <Layers size={14} />
              <span>Spatial Channel Heatmaps (3GPP TDL)</span>
            </button>

            <button
              onClick={() => setActiveTab('sweep')}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg transition cursor-pointer ${
                activeTab === 'sweep'
                  ? 'bg-[#789575] text-[#F2F3E2] shadow-sm'
                  : 'text-[#647B62] hover:text-[#263629]'
              }`}
            >
              <BarChart3 size={14} />
              <span>SNR vs NMSE Benchmark Sweep</span>
            </button>

            <button
              onClick={() => setActiveTab('matlab_6g')}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg transition cursor-pointer ${
                activeTab === 'matlab_6g'
                  ? 'bg-[#789575] text-[#F2F3E2] shadow-sm'
                  : 'text-[#647B62] hover:text-[#263629]'
              }`}
            >
              <Zap size={14} />
              <span>6G Sub-THz & MATLAB Studio</span>
            </button>
          </div>

          <div className="flex items-center gap-3">
            {/* Quick Chart Tools Mode indicator */}
            <div className="hidden sm:flex items-center gap-1.5 text-xs font-mono text-[#647B62] bg-[#EEF0DC] border border-[#CCD3B5] px-2.5 py-1 rounded-lg">
              <span>Mode:</span>
              <strong className="text-[#263629] capitalize">{chartDragMode}</strong>
              <button
                onClick={() => setChartDragMode(chartDragMode === 'zoom' ? 'pan' : 'zoom')}
                className="ml-1 text-[#789575] hover:text-[#263629] underline cursor-pointer"
                title="Toggle Pan/Zoom mode"
              >
                (Switch)
              </button>
            </div>

            <span className="text-xs font-mono text-[#647B62] font-semibold hidden md:inline">
              Array: <strong className="text-[#263629]">64×64 Massive MIMO</strong>
            </span>
          </div>
        </div>

        {/* Tab 1: Spatial Channel Heatmaps (3GPP TDL) */}
        {activeTab === 'heatmaps' && (
          <div className="space-y-5">
            {/* 1. Restored 3GPP Simulation Control Panel (Horizontal & Spacious) */}
            <Controls
              scenarios={scenarios}
              selectedScenario={selectedScenario}
              setSelectedScenario={setSelectedScenario}
              snr={snr}
              setSnr={setSnr}
              speed={speed}
              setSpeed={setSpeed}
              onRunEstimate={handleRunEstimate}
              onRunSweep={handleRunSweep}
              loadingEstimate={loadingEstimate}
              loadingSweep={loadingSweep}
              backendOnline={backendOnline}
            />

            {/* 2. Controls bar above heatmaps */}
            <div className="flex flex-wrap items-center justify-between gap-3 bg-[#F2F3E2] border border-[#CCD3B5] rounded-xl px-4 py-2.5 shadow-subtle">
              <div className="flex items-center gap-2">
                <span className="text-xs font-black uppercase text-[#263629] tracking-wider">
                  3GPP {selectedScenario} Spatial Channel Grid
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#EEF0DC] text-[#263629] border border-[#CCD3B5]">
                  SNR: {snr} dB
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#EEF0DC] text-[#263629] border border-[#CCD3B5]">
                  Mobility: {speed} km/h
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#EEF0DC] text-[#263629] border border-[#CCD3B5]">
                  Grid: 64×64 Massive MIMO Array
                </span>
              </div>

              <div className="flex items-center gap-3">
                <label className="flex items-center gap-1.5 text-xs text-[#647B62] font-medium cursor-pointer">
                  <input
                    type="checkbox"
                    checked={sharedColorScaleTDL}
                    onChange={(e) => setSharedColorScaleTDL(e.target.checked)}
                    className="accent-[#789575] rounded"
                  />
                  <span>Shared Z-scale (H, Hn, Ĥ)</span>
                </label>

                <button
                  onClick={handleRunEstimate}
                  disabled={loadingEstimate}
                  className="flex items-center gap-1.5 bg-[#789575] hover:bg-[#668263] active:scale-[0.99] disabled:opacity-50 text-[#F2F3E2] font-bold py-1 px-3 rounded-lg shadow-sm transition text-xs cursor-pointer border border-[#789575]"
                >
                  {loadingEstimate ? (
                    <div className="w-3.5 h-3.5 border-2 border-[#F2F3E2] border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <Zap size={13} />
                  )}
                  <span>{loadingEstimate ? 'Estimating...' : 'Re-estimate'}</span>
                </button>
              </div>
            </div>

            {/* 3. Five-Panel Visualization Grid (Equal Dimensions & Full Width) */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5">
              {/* Panel 1: 3GPP Beamspace (2D-FFT) */}
              <ChannelHeatmap
                title="1. Virtual Beamspace (2D-FFT)"
                subtitle="Angular spatial power spectrum via 2D Fourier transform"
                zData={estimateData?.matrices?.beamspace_sparsity}
                defaultColorscale="Plasma"
                badgeText="Angular Domain"
                matrixDim="64×64 Massive MIMO"
                dragmode={chartDragMode}
              />

              {/* Panel 2: Ground Truth Channel |H| */}
              <ChannelHeatmap
                title="2. Ground Truth Channel |H|"
                subtitle={`True 3GPP ${selectedScenario} response without noise`}
                zData={estimateData?.matrices?.true_channel}
                defaultColorscale="Viridis"
                sharedZRange={sharedZRangeTDL}
                matrixDim="64×64 Massive MIMO"
                dragmode={chartDragMode}
              />

              {/* Panel 3: Noisy Observation |Hn| */}
              <ChannelHeatmap
                title="3. Noisy LS Observation |Hn|"
                subtitle={`Raw least squares with ${snr} dB AWGN`}
                zData={estimateData?.matrices?.noisy_ls}
                defaultColorscale="Viridis"
                sharedZRange={sharedZRangeTDL}
                matrixDim="64×64 Massive MIMO"
                dragmode={chartDragMode}
              />

              {/* Panel 4: NCRT Reconstructed |Ĥ| */}
              <ChannelHeatmap
                title="4. NCRT Reconstructed |Ĥ|"
                subtitle="Attention-residual denoised channel estimate"
                zData={estimateData?.matrices?.reconstructed}
                defaultColorscale="Viridis"
                sharedZRange={sharedZRangeTDL}
                highlight={true}
                badgeText="NCRT Denoised"
                matrixDim="64×64 Massive MIMO"
                dragmode={chartDragMode}
              />

              {/* Panel 5: Absolute Reconstruction Error |H − Ĥ| */}
              <div className="sm:col-span-2 lg:col-span-1">
                <ChannelHeatmap
                  title="5. Absolute Error |H − Ĥ|"
                  subtitle="Actual computed absolute residual error matrix"
                  zData={estimateData?.matrices?.error_residual}
                  defaultColorscale="Plasma"
                  isErrorMatrix={true}
                  badgeText="Residual Error"
                  matrixDim="64×64 Massive MIMO"
                  dragmode={chartDragMode}
                />
              </div>
            </div>

            {/* 4. Performance Summary Table */}
            {estimateData?.results && (
              <div className="bg-[#F2F3E2] border border-[#CCD3B5] rounded-xl p-4 shadow-subtle overflow-x-auto">
                <h4 className="text-xs font-bold uppercase tracking-wider text-[#647B62] mb-3">
                  Statistical Performance Summary ({selectedScenario} @ {snr} dB)
                </h4>
                <table className="w-full text-xs text-left">
                  <thead>
                    <tr className="border-b border-[#CCD3B5] text-[#647B62]">
                      <th className="py-2 px-3">Estimator</th>
                      <th className="py-2 px-3">Method Type</th>
                      <th className="py-2 px-3">Parameters</th>
                      <th className="py-2 px-3">NMSE (dB)</th>
                      <th className="py-2 px-3">Latency</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#CCD3B5]/70 font-mono">
                    <tr className="bg-[#789575]/15 font-bold">
                      <td className="py-2.5 px-3 text-[#263629]">NCRT (Proposed)</td>
                      <td className="py-2.5 px-3 text-[#263629] font-sans">Attention Residual Transformer</td>
                      <td className="py-2.5 px-3">888,202</td>
                      <td className="py-2.5 px-3 text-[#263629] font-black">{estimateData.results.ncrt?.nmse_db} dB</td>
                      <td className="py-2.5 px-3">{estimateData.results.ncrt?.latency_ms} ms</td>
                    </tr>
                    {estimateData.results.ncrn && (
                      <tr className="text-[#263629]">
                        <td className="py-2.5 px-3">NCRN Baseline</td>
                        <td className="py-2.5 px-3 text-[#647B62] font-sans">ConvLSTM Residual</td>
                        <td className="py-2.5 px-3">1,204,610</td>
                        <td className="py-2.5 px-3">{estimateData.results.ncrn.nmse_db} dB</td>
                        <td className="py-2.5 px-3">{estimateData.results.ncrn.latency_ms} ms</td>
                      </tr>
                    )}
                    <tr className="text-[#263629]">
                      <td className="py-2.5 px-3">Wiener MMSE</td>
                      <td className="py-2.5 px-3 font-sans text-[#647B62]">Statistical Reference</td>
                      <td className="py-2.5 px-3">--</td>
                      <td className="py-2.5 px-3">{estimateData.results.mmse?.nmse_db} dB</td>
                      <td className="py-2.5 px-3">{estimateData.results.mmse?.latency_ms} ms</td>
                    </tr>
                    <tr className="text-[#647B62]">
                      <td className="py-2.5 px-3">Least Squares (LS)</td>
                      <td className="py-2.5 px-3 font-sans">Classical Unprocessed</td>
                      <td className="py-2.5 px-3">0</td>
                      <td className="py-2.5 px-3">{estimateData.results.ls?.nmse_db} dB</td>
                      <td className="py-2.5 px-3">{estimateData.results.ls?.latency_ms} ms</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* Tab 2: SNR vs NMSE Benchmark Sweep */}
        {activeTab === 'sweep' && (
          <div className="space-y-4">
            <PerformanceCurve sweepData={sweepData} />
          </div>
        )}

        {/* Tab 3: 6G Sub-THz & MATLAB Studio */}
        {activeTab === 'matlab_6g' && (
          <MatlabStudio onMetricsUpdate={(m) => setMatlabMetrics(m)} />
        )}
      </main>

      {/* Slide-over Right Drawer for Chart Tools & Exports (Non-destructive overlay) */}
      <RightSidebar
        isOpen={sidebarOpen}
        setIsOpen={setSidebarOpen}
        currentMatrixData={estimateData?.matrices}
        activeStandard={activeTab === 'matlab_6g' ? '6g_subthz' : '3gpp'}
        chartDragMode={chartDragMode}
        setChartDragMode={setChartDragMode}
        onResetZoom={handleResetZoom}
        onDownloadActivePlot={handleDownloadActivePlot}
        onDownloadAllImages={handleDownloadAllImages}
        snr={snr}
        speed={speed}
        selectedScenario={selectedScenario}
      />

      {/* Floating Action Pill on Right Edge for Quick Tools Access */}
      {!sidebarOpen && (
        <button
          onClick={() => setSidebarOpen(true)}
          className="fixed bottom-6 right-6 z-30 flex items-center gap-2 bg-[#789575] hover:bg-[#668263] text-[#F2F3E2] px-4 py-2.5 rounded-full shadow-lg border border-[#789575] font-bold text-xs cursor-pointer transition transform hover:scale-105 active:scale-95"
          title="Open Chart Tools, Pan/Box Modes, and Scientific Exports"
        >
          <Camera size={15} />
          <span>Chart Tools & Downloads</span>
        </button>
      )}

      {/* Modern Professional Frontend Footer */}
      <footer className="border-t border-[#CCD3B5] bg-[#EEF0DC] mt-auto text-xs text-[#647B62]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-8">
            {/* Col 1: Brand & Abstract */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-[#789575] flex items-center justify-center text-[#F2F3E2] shadow-sm">
                  <Radio size={18} />
                </div>
                <span className="text-sm font-black text-[#263629] tracking-tight">
                  Massive MIMO AI Studio
                </span>
              </div>
              <p className="text-[11px] leading-relaxed text-[#647B62]">
                Attention-based residual learning channel estimation platform for 5G-Advanced and 6G Sub-THz Massive MIMO systems.
              </p>
              <div className="flex flex-wrap gap-1.5 pt-1">
                <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-[#E5E8CE] text-[#263629] border border-[#CCD3B5]">
                  v2.1 Suite
                </span>
                <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-[#789575]/20 text-[#263629] border border-[#789575]/40">
                  O-RAN Near-RT
                </span>
                <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-[#E5E8CE] text-[#263629] border border-[#CCD3B5]">
                  64×64 Array
                </span>
              </div>
            </div>

            {/* Col 2: Neural Architectures */}
            <div className="space-y-2.5">
              <h4 className="text-[11px] font-black uppercase tracking-wider text-[#263629]">
                Neural Architectures
              </h4>
              <ul className="space-y-1.5 text-[11px] text-[#263629]/90">
                <li className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#789575]" />
                  <strong>NCRT:</strong> Attention-Residual Transformer (888K)
                </li>
                <li className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#A7B99A]" />
                  <strong>NCRN:</strong> ConvLSTM Spatial-Temporal Baseline (1.20M)
                </li>
                <li className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#CCD3B5]" />
                  <strong>Wiener MMSE:</strong> Statistical Covariance Benchmark
                </li>
                <li className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#CCD3B5]" />
                  <strong>Least Squares (LS):</strong> Unprocessed Observation
                </li>
              </ul>
            </div>

            {/* Col 3: Specifications & Protocols */}
            <div className="space-y-2.5">
              <h4 className="text-[11px] font-black uppercase tracking-wider text-[#263629]">
                Channels & Standards
              </h4>
              <ul className="space-y-1.5 text-[11px] text-[#263629]/90">
                <li className="hover:text-[#263629] transition cursor-default">
                  • 3GPP TR 38.901 Standards (TDL-A through TDL-E)
                </li>
                <li className="hover:text-[#263629] transition cursor-default">
                  • 142 GHz Sub-THz Ray-Tracing Models (MATLAB)
                </li>
                <li className="hover:text-[#263629] transition cursor-default">
                  • Virtual Beamspace 2D-FFT Angular Sparsity
                </li>
                <li className="hover:text-[#263629] transition cursor-default">
                  • High-Doppler Mobility Support (Up to 500 km/h)
                </li>
                <li className="hover:text-[#263629] transition cursor-default">
                  • Physical Geometry: 64×64 Planar Array (UPA/ULA)
                </li>
              </ul>
            </div>

            {/* Col 4: Artifact Downloads & Citation */}
            <div className="space-y-2.5">
              <h4 className="text-[11px] font-black uppercase tracking-wider text-[#263629]">
                Research Artifacts
              </h4>
              <div className="flex flex-col gap-1.5">
                <a
                  href={`${getApiBase()}/download/model`}
                  className="inline-flex items-center justify-between px-2.5 py-1.5 rounded bg-[#E5E8CE] hover:bg-[#F2F3E2] border border-[#CCD3B5] text-[11px] font-semibold text-[#263629] transition shadow-xs"
                >
                  <span>Trained Model Weights</span>
                  <span className="font-mono text-[10px] text-[#789575]">.keras</span>
                </a>
                <a
                  href={`${getApiBase()}/download/csv/6g`}
                  className="inline-flex items-center justify-between px-2.5 py-1.5 rounded bg-[#E5E8CE] hover:bg-[#F2F3E2] border border-[#CCD3B5] text-[11px] font-semibold text-[#263629] transition shadow-xs"
                >
                  <span>500-Sample Benchmark</span>
                  <span className="font-mono text-[10px] text-[#789575]">.csv</span>
                </a>
                <a
                  href={`${getApiBase()}/download/release-zip`}
                  className="inline-flex items-center justify-between px-2.5 py-1.5 rounded bg-[#789575] hover:bg-[#668263] text-[#F2F3E2] text-[11px] font-bold transition shadow-xs"
                >
                  <span>Full Research Archive</span>
                  <span className="font-mono text-[10px] text-[#E5E8CE]">.zip</span>
                </a>
              </div>
            </div>
          </div>

          {/* Bottom Bar */}
          <div className="border-t border-[#CCD3B5] pt-5 flex flex-col sm:flex-row items-center justify-between gap-3 text-[11px]">
            <p className="text-[#647B62]">
              © 2026 Massive MIMO AI Channel Studio • Published in <em>Vehicular Communications</em> (Elsevier).
            </p>
            <div className="flex items-center gap-3 font-mono text-[10px]">
              <div className="flex items-center gap-1.5 font-bold">
                <span className={`w-2 h-2 rounded-full ${backendOnline ? 'bg-emerald-600 animate-pulse' : 'bg-rose-500'}`} />
                <span className={backendOnline ? 'text-emerald-800' : 'text-rose-600'}>
                  {backendOnline ? 'FastAPI Cloud Active' : 'Connecting to Cloud Backend...'}
                </span>
              </div>
              <span>•</span>
              <span className="text-[#263629] font-bold">64×64 Massive Array</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

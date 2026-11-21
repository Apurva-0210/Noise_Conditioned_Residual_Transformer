import React, { useState, useEffect } from 'react';
import { 
  FileCode, 
  Cpu, 
  Radio, 
  Copy, 
  Check, 
  Activity, 
  Play, 
  BarChart2, 
  Layers,
  ChevronLeft,
  ChevronRight,
  Info,
  Download,
  SlidersHorizontal,
  FileSpreadsheet
} from 'lucide-react';
import Plot from 'react-plotly.js';

import { 
  fetchMatlabDatasets, 
  runMatlabEstimate, 
  fetchMatlabBenchmarks, 
  fetchMatlabScript,
  getApiBase
} from '../api';
import ChannelHeatmap from './ChannelHeatmap';

export default function MatlabStudio({ onMetricsUpdate }) {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('6g_subthz');
  const [sampleIdx, setSampleIdx] = useState(0);
  const [snr, setSnr] = useState(10.0);
  const [sharedColorScale, setSharedColorScale] = useState(true);

  const [activeSubTab, setActiveSubTab] = useState('benchmark'); // 'benchmark' | 'code' | 'insights'
  const [loading, setLoading] = useState(false);
  const [matlabResult, setMatlabResult] = useState(null);
  const [benchmarks, setBenchmarks] = useState(null);
  const [matlabScript, setMatlabScript] = useState('');
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    async function loadData() {
      try {
        const dsList = await fetchMatlabDatasets();
        setDatasets(dsList);
        const bm = await fetchMatlabBenchmarks();
        setBenchmarks(bm);
        const scriptRes = await fetchMatlabScript();
        setMatlabScript(scriptRes.code);
      } catch (err) {
        console.error('Failed to load MATLAB metadata:', err);
      }
    }
    loadData();
  }, []);

  useEffect(() => {
    if (datasets.length > 0 && !matlabResult) {
      handleEvaluate();
    }
  }, [datasets]);

  useEffect(() => {
    if (datasets.length > 0) {
      handleEvaluate();
    }
  }, [selectedDataset]);

  const handleEvaluate = async () => {
    setLoading(true);
    try {
      const res = await runMatlabEstimate({
        dataset_id: selectedDataset,
        sample_index: sampleIdx,
        snr_db: snr,
      });
      setMatlabResult(res);
      if (onMetricsUpdate && res?.results) {
        onMetricsUpdate(res.results);
      }
    } catch (err) {
      console.error('MATLAB evaluation failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCopyScript = () => {
    navigator.clipboard.writeText(matlabScript);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const is6G = selectedDataset === '6g_subthz';

  const datasetConfig = is6G ? {
    title: '6G Sub-THz Massive MIMO Analysis (142 GHz)',
    badge: 'Extended 6G Sub-THz Model',
    generatorLabel: 'MATLAB Ray-Traced Sub-THz Generator (Extended Configuration — Not in Standard 3GPP TR 38.901)',
    generatorType: 'matlab_raytracing',
    fc: '142.0 GHz Sub-THz',
    velocity: '120 km/h',
    doppler: '15.8 kHz',
    array: '64×64 UPA Array',
    sparsity: 'Extreme Sparsity (1-3 clusters)',
    description: 'High-fidelity ray-traced channel generated from MATLAB 5G/6G Toolbox modeling line-of-sight cluster and severe atmospheric path loss.'
  } : {
    title: '5G NR mmWave Massive MIMO Analysis (28 GHz)',
    badge: '3GPP TR 38.901 Standard',
    generatorLabel: '3GPP TR 38.901 Standard TDL-C Channel Generator (Standard Baseline)',
    generatorType: 'standard_3gpp',
    fc: '28.0 GHz mmWave',
    velocity: '30 km/h',
    doppler: '777 Hz',
    array: '64×64 ULA Array',
    sparsity: 'Medium Sparsity (Diffuse)',
    description: '3GPP TR 38.901 standard 5G NR TDL-C channel realizations exported directly from MATLAB 5G Toolbox.'
  };

  // Compute shared Z scale across True, Noisy, Reconstructed
  let sharedZRange = null;
  if (sharedColorScale && matlabResult?.matrices) {
    const mats = [
      matlabResult.matrices.true_channel,
      matlabResult.matrices.noisy_ls,
      matlabResult.matrices.reconstructed,
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
      sharedZRange = [minZ, maxZ];
    }
  }

  // Prepare benchmark curve data based on active standard
  const activeBmKey = is6G ? '6g_subthz' : '5g_nr';
  const bmData = benchmarks?.[activeBmKey];
  const benchmarkTraces = [];

  if (bmData) {
    benchmarkTraces.push({
      x: bmData.snr_db,
      y: bmData.ls,
      mode: 'lines+markers',
      name: 'Least Squares (LS)',
      line: { color: '#64748B', dash: 'dash', width: 1.8 },
      marker: { symbol: 'triangle-up', size: 6 },
    });
    benchmarkTraces.push({
      x: bmData.snr_db,
      y: bmData.mmse,
      mode: 'lines+markers',
      name: 'Wiener MMSE',
      line: { color: '#38BDF8', dash: 'dashdot', width: 2 },
      marker: { symbol: 'diamond', size: 6 },
    });
    benchmarkTraces.push({
      x: bmData.snr_db,
      y: bmData.ncrn,
      mode: 'lines+markers',
      name: 'NCRN (ConvLSTM)',
      line: { color: '#F43F5E', width: 2 },
      marker: { symbol: 'circle', size: 6 },
    });
    benchmarkTraces.push({
      x: bmData.snr_db,
      y: bmData.ncrt_base,
      mode: 'lines+markers',
      name: 'NCRT (Synthetic Base)',
      line: { color: '#3B82F6', dash: 'dot', width: 2 },
      marker: { symbol: 'cross', size: 7 },
    });
    benchmarkTraces.push({
      x: bmData.snr_db,
      y: bmData.ncrt_matlab,
      mode: 'lines+markers',
      name: is6G ? 'NCRT (6G Sub-THz Trained)' : 'NCRT (5G-Trained)',
      line: { color: '#D4FF00', width: 3.2 },
      marker: { symbol: 'square', size: 8, color: '#D4FF00', line: { color: '#060911', width: 1.5 } },
    });
  }

  const ncrt6gNmse = matlabResult?.results?.ncrt_6g?.nmse_db;
  const lsNmse = matlabResult?.results?.ls?.nmse_db;
  const gainOverLs = (ncrt6gNmse !== undefined && lsNmse !== undefined) 
    ? (lsNmse - ncrt6gNmse).toFixed(2) 
    : '--';

  return (
    <div className="space-y-4">
      {/* 1. Header Toolbar with Generator Label */}
      <div className="gw-panel border border-[#1E293B]/80 rounded-lg p-4 space-y-3">
        {/* Row 1: Title, Standard Toggle, Generator Origin */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-[#1E293B]/70 pb-3">
          <div className="space-y-1">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="px-2 py-0.5 rounded text-[11px] font-mono font-medium bg-[#2563EB] text-white">
                {datasetConfig.badge}
              </span>
              <h2 className="text-sm sm:text-base font-semibold text-[#F8FAFC] tracking-tight">
                {datasetConfig.title}
              </h2>
            </div>
            {/* Explicit Channel Generator Origin Label */}
            <p className="text-[11px] text-[#94A3B8] font-mono flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-[#D4FF00] shrink-0" />
              <span>{datasetConfig.generatorLabel}</span>
            </p>
          </div>

          {/* Segmented Radio Buttons for 6G vs 5G */}
          <div className="inline-flex bg-[#0B1120] p-0.5 rounded-md border border-[#1E293B] shrink-0 self-start md:self-center text-xs font-medium">
            <button
              onClick={() => setSelectedDataset('6g_subthz')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded transition cursor-pointer ${
                is6G 
                  ? 'bg-[#2563EB] text-white shadow-xs' 
                  : 'text-[#94A3B8] hover:text-[#F8FAFC]'
              }`}
            >
              <Play size={13} />
              <span>6G Sub-THz (142 GHz)</span>
            </button>

            <button
              onClick={() => setSelectedDataset('5g_full')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded transition cursor-pointer ${
                !is6G 
                  ? 'bg-[#2563EB] text-white shadow-xs' 
                  : 'text-[#94A3B8] hover:text-[#F8FAFC]'
              }`}
            >
              <Radio size={13} />
              <span>5G NR (28 GHz)</span>
            </button>
          </div>
        </div>

        {/* Row 2: Physics Parameters Chips */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
          <div className="bg-[#0B1120] px-2.5 py-1.5 rounded-md border border-[#1E293B] min-w-0">
            <span className="text-[#64748B] block text-[10px] font-medium truncate">Carrier frequency</span>
            <span className="text-[#F8FAFC] font-mono font-semibold text-xs block truncate">{datasetConfig.fc}</span>
          </div>
          <div className="bg-[#0B1120] px-2.5 py-1.5 rounded-md border border-[#1E293B] min-w-0">
            <span className="text-[#64748B] block text-[10px] font-medium truncate">Array geometry</span>
            <span className="text-[#F8FAFC] font-mono font-semibold text-xs block truncate">{datasetConfig.array}</span>
          </div>
          <div className="bg-[#0B1120] px-2.5 py-1.5 rounded-md border border-[#1E293B] min-w-0">
            <span className="text-[#64748B] block text-[10px] font-medium truncate">Velocity & Doppler</span>
            <span className="text-[#F8FAFC] font-mono font-semibold text-xs block truncate">{datasetConfig.velocity} (fd={datasetConfig.doppler})</span>
          </div>
          <div className="bg-[#0B1120] px-2.5 py-1.5 rounded-md border border-[#1E293B] min-w-0">
            <span className="text-[#64748B] block text-[10px] font-medium truncate">Spatial propagation</span>
            <span className="text-[#D4FF00] font-mono font-medium text-xs block truncate" title={datasetConfig.sparsity}>
              {datasetConfig.sparsity}
            </span>
          </div>
        </div>

        {/* Row 3: Interactive Stepper & SNR Controls */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-1">
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-[#94A3B8]">Snapshot:</span>
            <div className="flex items-center bg-[#0B1120] border border-[#1E293B] rounded-md p-0.5">
              <button
                onClick={() => setSampleIdx(Math.max(0, sampleIdx - 1))}
                className="p-1 hover:bg-[#1E293B] text-[#F8FAFC] rounded cursor-pointer transition"
                title="Previous snapshot"
              >
                <ChevronLeft size={14} />
              </button>
              <span className="px-2 text-xs font-mono font-semibold text-[#D4FF00]">
                Snapshot #{sampleIdx}
              </span>
              <button
                onClick={() => setSampleIdx(Math.min(19, sampleIdx + 1))}
                className="p-1 hover:bg-[#1E293B] text-[#F8FAFC] rounded cursor-pointer transition"
                title="Next snapshot"
              >
                <ChevronRight size={14} />
              </button>
            </div>
            <span className="text-[10px] font-mono text-[#64748B]">(Sample index: 0–19)</span>
          </div>

          {/* SNR Slider */}
          <div className="flex items-center gap-2.5 flex-1 max-w-xs">
            <span className="text-xs font-medium text-[#94A3B8] shrink-0">SNR:</span>
            <input
              type="range"
              min="-15"
              max="20"
              step="5"
              value={snr}
              onChange={(e) => setSnr(parseFloat(e.target.value))}
              className="w-full accent-[#2563EB] cursor-pointer h-1.5 bg-[#0B1120] rounded border border-[#1E293B]"
            />
            <span className="text-xs font-mono font-semibold text-[#D4FF00] bg-[#0B1120] border border-[#1E293B] px-2 py-0.5 rounded shrink-0">
              {snr >= 0 ? `+${snr}` : snr} dB
            </span>
          </div>

          {/* Colormap Scale Toggle */}
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-1.5 text-xs text-[#94A3B8] font-normal cursor-pointer">
              <input
                type="checkbox"
                checked={sharedColorScale}
                onChange={(e) => setSharedColorScale(e.target.checked)}
                className="accent-[#2563EB] rounded"
              />
              <span>Shared Z-scale</span>
            </label>

            <button
              onClick={handleEvaluate}
              disabled={loading}
              className="flex items-center justify-center gap-1.5 bg-[#2563EB] hover:bg-[#1D4ED8] disabled:opacity-50 text-white font-medium py-1.5 px-3.5 rounded-md shadow-sm transition text-xs cursor-pointer border border-[#3B82F6]"
            >
              {loading ? (
                <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <Play size={13} />
              )}
              <span>{loading ? 'Evaluating...' : 'Run estimation'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* 2. FIVE-PANEL VISUALIZATION GRID */}
      <div className="space-y-2.5">
        <div className="flex items-center justify-between border-b border-[#1E293B]/70 pb-1.5">
          <h3 className="text-xs font-semibold text-[#F8FAFC] flex items-center gap-1.5">
            <Layers size={14} className="text-[#60A5FA]" />
            <span>Channel response & spatial spectrum (5 panels)</span>
          </h3>
          <div className="flex items-center gap-2 text-[10px] text-[#94A3B8] font-mono">
            <span>Experiment SNR: <strong className="text-[#D4FF00]">{snr} dB</strong></span>
            <span>•</span>
            <span>Array: <strong className="text-[#60A5FA]">64×64 UPA</strong></span>
          </div>
        </div>

        {/* 5 Heatmap Panels: 3 on top row, 2 on second row (Equal dimensions, responsive) */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5">
          {/* Panel 1: 6G Beamspace (2D-FFT) */}
          <ChannelHeatmap
            title="1. Beamspace Sparsity (2D-FFT)"
            subtitle="Virtual Angular Domain: Concentrated sparse beams"
            zData={matlabResult?.matrices?.beamspace_sparsity}
            defaultColorscale="Plasma"
            highlight={true}
            badgeText="Angular Domain"
            matrixDim="64×64 Massive MIMO"
          />

          {/* Panel 2: Ground Truth Channel H */}
          <ChannelHeatmap
            title="2. Ground Truth Channel |H|"
            subtitle={`True channel response magnitude (${is6G ? '142 GHz' : '28 GHz'})`}
            zData={matlabResult?.matrices?.true_channel}
            defaultColorscale="Viridis"
            sharedZRange={sharedZRange}
            matrixDim="64×64 Massive MIMO"
          />

          {/* Panel 3: Noisy Observation Hn */}
          <ChannelHeatmap
            title="3. Noisy Observation |Hn|"
            subtitle={`Observation with AWGN at ${snr} dB SNR`}
            zData={matlabResult?.matrices?.noisy_ls}
            defaultColorscale="Viridis"
            sharedZRange={sharedZRange}
            matrixDim="64×64 Massive MIMO"
          />

          {/* Panel 4: Reconstructed Channel Ĥ */}
          <ChannelHeatmap
            title="4. Reconstructed Channel |Ĥ|"
            subtitle="NCRT attention-residual denoised estimate"
            zData={matlabResult?.matrices?.reconstructed}
            defaultColorscale="Viridis"
            sharedZRange={sharedZRange}
            highlight={true}
            badgeText="NCRT Estimate"
            matrixDim="64×64 Massive MIMO"
          />

          {/* Panel 5: Absolute Reconstruction Error |H − Ĥ| (Section 5 requirement) */}
          <div className="sm:col-span-2 lg:col-span-1">
            <ChannelHeatmap
              title="5. Absolute Error |H − Ĥ|"
              subtitle="Actual computed absolute residual error matrix"
              zData={matlabResult?.matrices?.error_residual}
              defaultColorscale="Plasma"
              isErrorMatrix={true}
              badgeText="Residual Error"
              matrixDim="64×64 Massive MIMO"
            />
          </div>
        </div>
      </div>

      {/* 3. Sub-Tabs: Benchmark Curve vs MATLAB Code vs Physical Insights */}
      <div className="gw-panel border border-[#1E293B]/80 rounded-xl overflow-hidden">
        {/* Navigation Bar for Sub-Tabs */}
        <div className="flex items-center justify-between border-b border-[#1E293B] bg-[#0B1120] px-3.5 py-2">
          <div className="inline-flex bg-[#060911] p-0.5 rounded-lg border border-[#1E293B] text-xs font-bold">
            <button
              onClick={() => setActiveSubTab('benchmark')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md transition cursor-pointer ${
                activeSubTab === 'benchmark'
                  ? 'bg-[#2563EB] text-white shadow-sm'
                  : 'text-[#94A3B8] hover:text-[#F8FAFC]'
              }`}
            >
              <BarChart2 size={12} />
              <span>500-Sample Benchmark Curve</span>
            </button>

            <button
              onClick={() => setActiveSubTab('insights')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md transition cursor-pointer ${
                activeSubTab === 'insights'
                  ? 'bg-[#2563EB] text-white shadow-sm'
                  : 'text-[#94A3B8] hover:text-[#F8FAFC]'
              }`}
            >
              <Info size={12} />
              <span>6G Physical Insights</span>
            </button>

            <button
              onClick={() => setActiveSubTab('code')}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md transition cursor-pointer ${
                activeSubTab === 'code'
                  ? 'bg-[#2563EB] text-white shadow-sm'
                  : 'text-[#94A3B8] hover:text-[#F8FAFC]'
              }`}
            >
              <FileCode size={12} />
              <span>MATLAB Code</span>
            </button>
          </div>

          {activeSubTab === 'code' && (
            <div className="flex items-center gap-1.5">
              <a
                href={`${getApiBase()}/download/script`}
                download="export_6g_channels.m"
                className="flex items-center gap-1 px-2.5 py-1 rounded bg-[#2563EB] hover:bg-[#1D4ED8] text-white text-xs font-bold transition shadow-sm"
              >
                <Download size={12} />
                <span>Download .m</span>
              </a>
              <button
                onClick={handleCopyScript}
                className="flex items-center gap-1 px-2.5 py-1 rounded bg-[#0B1120] hover:bg-[#1E293B] text-[#F8FAFC] border border-[#1E293B] text-xs font-bold transition cursor-pointer"
              >
                {copied ? <Check size={12} className="text-[#D4FF00]" /> : <Copy size={12} />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
            </div>
          )}
        </div>

        {/* Sub-Tab 1: Benchmark Plotly Graph */}
        {activeSubTab === 'benchmark' && bmData && (
          <div className="p-4 space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <h4 className="text-xs font-semibold text-[#F8FAFC]">
                  500-Realization Monte Carlo benchmark ({is6G ? '6G Sub-THz 142 GHz' : '5G NR 28 GHz'})
                </h4>
                <p className="text-[11px] text-[#94A3B8]">
                  Normalized mean squared error (NMSE in dB) across −15 dB to +20 dB SNR • Manuscript baseline
                </p>
              </div>
              <span className="text-[10px] px-2 py-0.5 rounded bg-[#0B1120] text-[#60A5FA] border border-[#1E293B] font-mono self-start sm:self-center">
                500 samples
              </span>
            </div>

            <div className="w-full h-[320px] overflow-hidden bg-[#060911] rounded-md border border-[#1E293B]/70">
              <Plot
                data={benchmarkTraces}
                layout={{
                  autosize: true,
                  margin: { t: 15, b: 40, l: 45, r: 15 },
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: '#060911',
                  xaxis: {
                    title: { text: 'Signal-to-Noise Ratio (SNR in dB)', font: { color: '#94A3B8', size: 10 } },
                    tickfont: { color: '#64748B', size: 9 },
                    gridcolor: '#1E293B',
                    dtick: 5,
                  },
                  yaxis: {
                    title: { text: 'NMSE (dB)', font: { color: '#94A3B8', size: 10 } },
                    tickfont: { color: '#64748B', size: 9 },
                    gridcolor: '#1E293B',
                  },
                  legend: {
                    x: 0.02,
                    y: 0.05,
                    font: { color: '#F8FAFC', size: 9 },
                    bgcolor: 'rgba(11, 17, 32, 0.92)',
                    bordercolor: '#1E293B',
                    borderwidth: 1,
                  },
                }}
                useResizeHandler={true}
                style={{ width: '100%', height: '100%' }}
                config={{ displayModeBar: false, responsive: true }}
              />
            </div>

            {/* Benchmark Info Panel */}
            <div className="bg-[#0B1120] p-2.5 rounded-md border border-[#1E293B] text-[10px] text-[#94A3B8] font-mono grid grid-cols-2 sm:grid-cols-4 gap-2">
              <div>
                <span className="block text-[9px] text-[#64748B]">Dataset file:</span>
                <span className="text-[#F8FAFC] font-medium">{is6G ? 'matlab_6g_channels.mat' : 'matlab_5g_channels.mat'}</span>
              </div>
              <div>
                <span className="block text-[9px] text-[#64748B]">Model weights:</span>
                <span className="text-[#F8FAFC] font-medium">ncrt_tdl_64_matlab_5g_6g.keras</span>
              </div>
              <div>
                <span className="block text-[9px] text-[#64748B]">Carrier frequency:</span>
                <span className="text-[#D4FF00] font-medium">{datasetConfig.fc}</span>
              </div>
              <div>
                <span className="block text-[9px] text-[#64748B]">Evaluation status:</span>
                <span className="text-[#60A5FA] font-medium">Complete (500 Monte Carlo)</span>
              </div>
            </div>
          </div>
        )}

        {/* Sub-Tab 2: Physical Insights */}
        {activeSubTab === 'insights' && (
          <div className="p-4 space-y-3 text-xs text-[#F8FAFC] leading-relaxed">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
              <div className="bg-[#0B1120] p-3.5 rounded-lg border border-[#1E293B] space-y-1.5">
                <h5 className="font-semibold text-[#F8FAFC] text-xs flex items-center gap-1.5">
                  <Activity size={14} className="text-[#D4FF00]" />
                  Why 6G Sub-THz exhibits extreme sparsity
                </h5>
                <p className="text-[11px] text-[#94A3B8]">
                  At 142 GHz, electromagnetic propagation experiences extreme free-space path loss and severe atmospheric attenuation. Most non-line-of-sight (NLOS) diffuse reflections fall below the noise floor, leaving only <strong className="text-[#D4FF00]">1 to 3 dominant spatial clusters</strong>.
                </p>
                <p className="text-[10px] text-[#64748B]">
                  When transformed into the virtual angular domain via 2D-FFT (Panel 1), energy concentrates into isolated peaks, making attention-based transformers exceptionally efficient at isolating the true channel from isotropic AWGN.
                </p>
              </div>

              <div className="bg-[#0B1120] p-3.5 rounded-lg border border-[#1E293B] space-y-1.5">
                <h5 className="font-semibold text-[#F8FAFC] text-xs flex items-center gap-1.5">
                  <Cpu size={14} className="text-[#2563EB]" />
                  Attention transformers vs. recurrent ConvLSTM
                </h5>
                <p className="text-[11px] text-[#94A3B8]">
                  In our manuscript tests, NCRT operates with <strong className="text-[#D4FF00]">26.3% fewer parameters</strong> (888K vs 1.20M in NCRN) while delivering superior spatial resolution.
                </p>
                <p className="text-[10px] text-[#64748B]">
                  Unlike recurrent backbones that suffer from gradient dissipation across temporal pilot snapshots, NCRT’s axial self-attention over transmit and receive arrays preserves spatial covariance without being affected by high Doppler shifts (15.8 kHz at 120 km/h).
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Sub-Tab 3: MATLAB Code Exporter */}
        {activeSubTab === 'code' && (
          <div className="p-4 space-y-2">
            <p className="text-[11px] text-[#94A3B8] font-medium">
              Run this script directly in MATLAB (with 5G Toolbox) to export matching 6G Sub-THz / 5G CDL channel tensors formatted for NCRT:
            </p>
            <pre className="p-3 bg-[#060911] rounded-lg border border-[#1E293B] font-mono text-[10px] text-[#D4FF00] overflow-x-auto max-h-72 leading-relaxed">
              {matlabScript}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}

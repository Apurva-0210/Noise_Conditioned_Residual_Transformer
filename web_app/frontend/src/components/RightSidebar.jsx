import React, { useState } from 'react';
import { getApiBase } from '../api';
import { 
  Download, 
  FileCode, 
  Database, 
  Package, 
  Sliders, 
  X, 
  Check, 
  Copy, 
  Radio, 
  Activity, 
  Zap, 
  TrendingUp, 
  FileSpreadsheet,
  BookOpen,
  FileText,
  AlertTriangle,
  Move,
  Scan,
  RotateCcw,
  Share2,
  Image as ImageIcon,
  Camera,
  Maximize2
} from 'lucide-react';

export default function RightSidebar({
  isOpen,
  setIsOpen,
  currentMatrixData,
  activeStandard = '3gpp', // '3gpp' | '6g_subthz' | '5g_nr'
  chartDragMode = 'zoom',
  setChartDragMode,
  onResetZoom,
  onDownloadActivePlot,
  onDownloadAllImages,
  snr = 5.0,
  speed = 30.0,
  selectedScenario = 'TDL-A',
}) {
  const [activeTab, setActiveTab] = useState('chart_tools'); // 'chart_tools' | 'data_exports' | 'citation'
  const [copiedBibtex, setCopiedBibtex] = useState(false);
  const [copiedShare, setCopiedShare] = useState(false);
  const [exportedJson, setExportedJson] = useState(false);
  const [exportedConfig, setExportedConfig] = useState(false);
  const [downloadingImages, setDownloadingImages] = useState(false);

  const bibtexCode = `@article{kumar2026attention,
  title={Attention-Based Residual Learning Channel Estimation for Massive MIMO},
  author={Kumar, Apurva and Patel, Vibha and Kukade, Shweta},
  journal={Vehicular Communications},
  year={2026},
  publisher={Elsevier}
}`;

  const handleCopyBibtex = () => {
    navigator.clipboard.writeText(bibtexCode);
    setCopiedBibtex(true);
    setTimeout(() => setCopiedBibtex(false), 2000);
  };

  const handleShareChart = () => {
    const summary = `Massive MIMO AI Channel Studio V2.1
Scenario: ${selectedScenario} | SNR: ${snr} dB | Velocity: ${speed} km/h
Standard: ${activeStandard.toUpperCase()}
Platform URL: ${window.location.href}`;
    navigator.clipboard.writeText(summary);
    setCopiedShare(true);
    setTimeout(() => setCopiedShare(false), 2000);
  };

  const handleExportCurrentMatrix = () => {
    if (!currentMatrixData) return;
    const exportPayload = {
      timestamp: new Date().toISOString(),
      metadata: {
        platform: "Massive MIMO AI Channel Studio V2.1",
        manuscript: "Attention-Based Residual Learning Channel Estimation for Massive MIMO",
        scenario: selectedScenario,
        snr_db: snr,
        speed_kmh: speed,
        standard: activeStandard,
        array_size: "64x64 Massive MIMO Array",
      },
      channel_matrices: currentMatrixData,
    };
    const blob = new Blob([JSON.stringify(exportPayload, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `channel_matrix_${selectedScenario}_${snr}dB.json`;
    a.click();
    URL.revokeObjectURL(url);
    setExportedJson(true);
    setTimeout(() => setExportedJson(false), 2000);
  };

  const handleExportConfig = () => {
    const configPayload = {
      platform: "Massive MIMO AI Channel Studio V2.1",
      generator: activeStandard === '6g_subthz' ? 'MATLAB Ray-Tracer (142 GHz Sub-THz)' : '3GPP TR 38.901 TDL',
      scenario: selectedScenario,
      snr_db: snr,
      speed_kmh: speed,
      antenna_array: {
        physical_elements: [64, 64],
        resolution: "64x64 Full Massive MIMO Resolution",
        pilot_window_symbols: 10,
      },
      models: {
        proposed: "NCRT (Attention-Residual Transformer, 888K parameters)",
        baseline: "NCRN (ConvLSTM Residual, 1.20M parameters)",
      },
      timestamp: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(configPayload, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `simulation_config_${selectedScenario}.json`;
    a.click();
    URL.revokeObjectURL(url);
    setExportedConfig(true);
    setTimeout(() => setExportedConfig(false), 2000);
  };

  const handleDownloadAllImagesClick = () => {
    setDownloadingImages(true);
    if (onDownloadAllImages) {
      onDownloadAllImages();
    }
    setTimeout(() => setDownloadingImages(false), 1500);
  };

  if (!isOpen) return null;

  return (
    <>
      {/* Subtle backdrop overlay */}
      <div 
        className="fixed inset-0 bg-[#060911]/60 backdrop-blur-[2px] z-40 transition-opacity"
        onClick={() => setIsOpen(false)}
      />

      {/* Slide-over Right Drawer with Glass Material */}
      <aside className="fixed top-0 right-0 h-full w-88 sm:w-96 gw-glass border-l border-[#1E293B] shadow-2xl z-50 p-5 flex flex-col justify-between overflow-y-auto transform transition-transform duration-200 ease-in-out">
        <div className="space-y-4">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-[#1E293B] pb-3">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-md bg-[#2563EB] flex items-center justify-center text-white shadow-xs">
                <Camera size={14} />
              </div>
              <div>
                <h3 className="text-xs font-semibold text-[#F8FAFC]">
                  Chart tools & export hub
                </h3>
                <p className="text-[10px] text-[#94A3B8]">
                  Scientific actions & data artifacts
                </p>
              </div>
            </div>

            <button
              onClick={() => setIsOpen(false)}
              className="p-1 rounded-md hover:bg-[#1E293B] text-[#94A3B8] hover:text-[#F8FAFC] transition cursor-pointer border border-transparent hover:border-[#1E293B]"
              title="Close panel"
            >
              <X size={15} />
            </button>
          </div>

          {/* Sub-Tabs: Chart Tools | Data Exports | Citation */}
          <div className="grid grid-cols-3 bg-[#0B1120] p-0.5 rounded-md border border-[#1E293B] text-[11px] font-medium">
            <button
              onClick={() => setActiveTab('chart_tools')}
              className={`py-1.5 px-2 rounded transition text-center cursor-pointer ${
                activeTab === 'chart_tools'
                  ? 'bg-[#2563EB] text-white shadow-xs'
                  : 'text-[#94A3B8] hover:text-[#F8FAFC]'
              }`}
            >
              Tools
            </button>
            <button
              onClick={() => setActiveTab('data_exports')}
              className={`py-1.5 px-2 rounded transition text-center cursor-pointer ${
                activeTab === 'data_exports'
                  ? 'bg-[#2563EB] text-white shadow-xs'
                  : 'text-[#94A3B8] hover:text-[#F8FAFC]'
              }`}
            >
              Exports
            </button>
            <button
              onClick={() => setActiveTab('citation')}
              className={`py-1.5 px-2 rounded transition text-center cursor-pointer ${
                activeTab === 'citation'
                  ? 'bg-[#2563EB] text-white shadow-xs'
                  : 'text-[#94A3B8] hover:text-[#F8FAFC]'
              }`}
            >
              Citation
            </button>
          </div>

          {/* TAB 1: CHART TOOLS */}
          {activeTab === 'chart_tools' && (
            <div className="space-y-3">
              {/* Interactive Viewport Tools: Pan & Box Zoom */}
              <div className="bg-[#0B1120] border border-[#1E293B] rounded-md p-3 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-medium text-[#F8FAFC]">
                    Plot navigation mode
                  </span>
                  <span className="text-[10px] text-[#94A3B8] font-mono">
                    Active: <strong className="text-[#D4FF00] capitalize">{chartDragMode}</strong>
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => setChartDragMode && setChartDragMode('pan')}
                    className={`flex items-center justify-center gap-1.5 py-1.5 px-2.5 rounded-md border text-xs font-medium transition cursor-pointer ${
                      chartDragMode === 'pan'
                        ? 'bg-[#2563EB] text-white border-[#2563EB] shadow-xs'
                        : 'bg-[#060911] text-[#94A3B8] border-[#1E293B] hover:bg-[#1E293B] hover:text-[#F8FAFC]'
                    }`}
                  >
                    <Move size={13} />
                    <span>Pan</span>
                  </button>

                  <button
                    onClick={() => setChartDragMode && setChartDragMode('zoom')}
                    className={`flex items-center justify-center gap-1.5 py-1.5 px-2.5 rounded-md border text-xs font-medium transition cursor-pointer ${
                      chartDragMode === 'zoom'
                        ? 'bg-[#2563EB] text-white border-[#2563EB] shadow-xs'
                        : 'bg-[#060911] text-[#94A3B8] border-[#1E293B] hover:bg-[#1E293B] hover:text-[#F8FAFC]'
                    }`}
                  >
                    <Scan size={13} />
                    <span>Box</span>
                  </button>
                </div>

                {onResetZoom && (
                  <button
                    onClick={onResetZoom}
                    className="w-full flex items-center justify-center gap-1.5 py-1.5 px-3 rounded-lg bg-[#060911] hover:bg-[#1E293B] border border-[#1E293B] text-[11px] font-semibold text-[#94A3B8] hover:text-[#F8FAFC] transition cursor-pointer"
                  >
                    <RotateCcw size={12} />
                    <span>Reset Axes & Viewport</span>
                  </button>
                )}
              </div>

              {/* Visual Figure Downloads */}
              <div className="bg-[#0B1120] border border-[#1E293B] rounded-md p-3 space-y-2">
                <span className="text-[11px] font-medium text-[#F8FAFC] block">
                  Figure & image exports
                </span>

                {/* 1. Download Plot as PNG */}
                <button
                  onClick={onDownloadActivePlot}
                  className="w-full flex items-center justify-between py-1.5 px-2.5 rounded-md bg-[#060911] hover:bg-[#1E293B] border border-[#1E293B] text-xs font-medium text-[#F8FAFC] transition cursor-pointer"
                >
                  <div className="flex items-center gap-2">
                    <Camera size={13} className="text-[#2563EB]" />
                    <span>Download plot as PNG</span>
                  </div>
                  <span className="text-[10px] font-mono text-[#94A3B8] bg-[#0B1120] px-1.5 py-0.5 rounded border border-[#1E293B]">
                    .png
                  </span>
                </button>

                {/* 2. Download Images (Batch 5-Panel Figures) */}
                <button
                  onClick={handleDownloadAllImagesClick}
                  disabled={downloadingImages}
                  className="w-full flex items-center justify-between py-1.5 px-2.5 rounded-md bg-[#060911] hover:bg-[#1E293B] border border-[#1E293B] text-xs font-medium text-[#F8FAFC] transition cursor-pointer"
                >
                  <div className="flex items-center gap-2">
                    <ImageIcon size={13} className="text-[#2563EB]" />
                    <span>Download all panels</span>
                  </div>
                  <span className="text-[10px] font-mono text-[#94A3B8] bg-[#0B1120] px-1.5 py-0.5 rounded border border-[#1E293B]">
                    {downloadingImages ? 'Saving...' : '5 panels'}
                  </span>
                </button>

                {/* 3. Share Chart */}
                <button
                  onClick={handleShareChart}
                  className="w-full flex items-center justify-between py-1.5 px-2.5 rounded-md bg-[#060911] hover:bg-[#1E293B] border border-[#1E293B] text-xs font-medium text-[#F8FAFC] transition cursor-pointer"
                >
                  <div className="flex items-center gap-2">
                    {copiedShare ? (
                      <Check size={13} className="text-[#D4FF00]" />
                    ) : (
                      <Share2 size={13} className="text-[#2563EB]" />
                    )}
                    <span>Copy summary</span>
                  </div>
                  <span className="text-[10px] font-mono text-[#94A3B8] bg-[#0B1120] px-1.5 py-0.5 rounded border border-[#1E293B]">
                    {copiedShare ? 'Copied' : 'Clipboard'}
                  </span>
                </button>
              </div>
            </div>
          )}

          {/* TAB 2: DATA EXPORTS (.json, .csv, .keras, .zip) */}
          {activeTab === 'data_exports' && (
            <div className="space-y-3">
              <div className="bg-[#0B1120] border border-[#1E293B] rounded-md p-3 space-y-2">
                <span className="text-[11px] font-medium text-[#F8FAFC] block">
                  Scientific data & models
                </span>

                {/* Channel Matrix JSON */}
                <button
                  onClick={handleExportCurrentMatrix}
                  disabled={!currentMatrixData}
                  className="w-full flex items-center justify-between py-1.5 px-2.5 rounded-md bg-[#060911] hover:bg-[#1E293B] disabled:opacity-50 border border-[#1E293B] text-xs font-medium text-[#F8FAFC] transition cursor-pointer"
                >
                  <div className="flex items-center gap-2">
                    <FileCode size={13} className="text-[#2563EB]" />
                    <span>Channel matrix (.json)</span>
                  </div>
                  <span className="text-[10px] font-mono text-[#94A3B8] bg-[#0B1120] px-1.5 py-0.5 rounded border border-[#1E293B]">
                    {exportedJson ? 'Saved' : '64×64'}
                  </span>
                </button>

                {/* Simulation Config JSON */}
                <button
                  onClick={handleExportConfig}
                  className="w-full flex items-center justify-between py-1.5 px-2.5 rounded-md bg-[#060911] hover:bg-[#1E293B] border border-[#1E293B] text-xs font-medium text-[#F8FAFC] transition cursor-pointer"
                >
                  <div className="flex items-center gap-2">
                    <FileText size={13} className="text-[#2563EB]" />
                    <span>Simulation config (.json)</span>
                  </div>
                  <span className="text-[10px] font-mono text-[#94A3B8] bg-[#0B1120] px-1.5 py-0.5 rounded border border-[#1E293B]">
                    {exportedConfig ? 'Saved' : 'Params'}
                  </span>
                </button>

                {/* Benchmark CSV (MATLAB 6G / 5G) */}
                <a
                  href={`${getApiBase()}/download/csv/${activeStandard === '5g_nr' ? '5g' : '6g'}`}
                  target="_blank"
                  rel="noreferrer"
                  className="w-full flex items-center justify-between py-1.5 px-2.5 rounded-md bg-[#060911] hover:bg-[#1E293B] border border-[#1E293B] text-xs font-medium text-[#F8FAFC] transition cursor-pointer"
                >
                  <div className="flex items-center gap-2">
                    <FileSpreadsheet size={13} className="text-[#2563EB]" />
                    <span>Benchmark results (.csv)</span>
                  </div>
                  <span className="text-[10px] font-mono text-[#94A3B8] bg-[#0B1120] px-1.5 py-0.5 rounded border border-[#1E293B]">
                    500 samples
                  </span>
                </a>

                {/* Trained Model .keras */}
                <a
                  href={`${getApiBase()}/download/model`}
                  target="_blank"
                  rel="noreferrer"
                  className="w-full flex items-center justify-between py-1.5 px-2.5 rounded-md bg-[#060911] hover:bg-[#1E293B] border border-[#1E293B] text-xs font-medium text-[#F8FAFC] transition cursor-pointer"
                >
                  <div className="flex items-center gap-2">
                    <Database size={13} className="text-[#D4FF00]" />
                    <span>Trained weights (.keras)</span>
                  </div>
                  <span className="text-[10px] font-mono text-[#D4FF00] bg-[#0B1120] px-1.5 py-0.5 rounded border border-[#1E293B]">
                    888K
                  </span>
                </a>

                {/* Full Release ZIP */}
                <a
                  href={`${getApiBase()}/download/release-zip`}
                  target="_blank"
                  rel="noreferrer"
                  className="w-full flex items-center justify-between py-1.5 px-2.5 rounded-md bg-[#2563EB] hover:bg-[#1D4ED8] border border-[#3B82F6] text-xs font-medium text-white transition cursor-pointer shadow-sm"
                >
                  <div className="flex items-center gap-2">
                    <Package size={13} className="text-white" />
                    <span>Complete release (.zip)</span>
                  </div>
                  <span className="text-[10px] font-mono text-white bg-blue-700 px-1.5 py-0.5 rounded">
                    Bundle
                  </span>
                </a>
              </div>
            </div>
          )}

          {/* TAB 3: CITATION & METADATA */}
          {activeTab === 'citation' && (
            <div className="space-y-3">
              <div className="bg-[#0B1120] border border-[#1E293B] rounded-md p-3 space-y-2">
                <span className="text-[11px] font-medium text-[#F8FAFC] block">
                  Manuscript citation
                </span>
                <p className="text-xs text-[#94A3B8] font-normal leading-relaxed">
                  Apurva Kumar, Vibha Patel, and Shweta Kukade. "Attention-Based Residual Learning Channel Estimation for Massive MIMO." <em>Vehicular Communications</em>, Elsevier, 2026.
                </p>
                <div className="relative mt-2">
                  <pre className="p-2.5 rounded-md bg-[#060911] border border-[#1E293B] text-[10px] font-mono text-[#D4FF00] overflow-x-auto">
                    {bibtexCode}
                  </pre>
                  <button
                    onClick={handleCopyBibtex}
                    className="absolute top-2 right-2 flex items-center gap-1 text-[10px] font-medium bg-[#2563EB] text-white px-2 py-0.5 rounded hover:bg-[#1D4ED8] cursor-pointer"
                  >
                    {copiedBibtex ? <Check size={11} className="text-[#D4FF00]" /> : <Copy size={11} />}
                    <span>{copiedBibtex ? 'Copied' : 'Copy BibTeX'}</span>
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Sidebar Footer */}
        <div className="pt-3 border-t border-[#1E293B]/60 text-[10px] text-[#64748B] font-mono flex items-center justify-between">
          <span>Massive MIMO Studio V2.1</span>
          <span className="text-[#D4FF00]">Port 8001 Active</span>
        </div>
      </aside>
    </>
  );
}

import React, { useState } from 'react';
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
        className="fixed inset-0 bg-[#263629]/25 backdrop-blur-[1px] z-40 transition-opacity"
        onClick={() => setIsOpen(false)}
      />

      {/* Slide-over Right Drawer */}
      <aside className="fixed top-0 right-0 h-full w-88 sm:w-96 bg-[#F2F3E2] border-l border-[#CCD3B5] shadow-2xl z-50 p-5 flex flex-col justify-between overflow-y-auto transform transition-transform duration-300 ease-in-out">
        <div className="space-y-5">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-[#CCD3B5] pb-3">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-lg bg-[#789575] flex items-center justify-center text-[#F2F3E2] shadow-sm">
                <Camera size={15} />
              </div>
              <div>
                <h3 className="text-xs font-black uppercase tracking-wider text-[#263629]">
                  Chart Tools & Export Hub
                </h3>
                <p className="text-[10px] text-[#647B62] font-medium">
                  Scientific Plot Actions & Artifacts
                </p>
              </div>
            </div>

            <button
              onClick={() => setIsOpen(false)}
              className="p-1.5 rounded-lg hover:bg-[#EEF0DC] text-[#647B62] hover:text-[#263629] transition cursor-pointer border border-transparent hover:border-[#CCD3B5]"
              title="Close panel"
            >
              <X size={16} />
            </button>
          </div>

          {/* Sub-Tabs: Chart Tools | Data Exports | Citation */}
          <div className="grid grid-cols-3 bg-[#EEF0DC] p-1 rounded-lg border border-[#CCD3B5] text-[11px] font-bold">
            <button
              onClick={() => setActiveTab('chart_tools')}
              className={`py-1.5 px-2 rounded-md transition text-center cursor-pointer ${
                activeTab === 'chart_tools'
                  ? 'bg-[#789575] text-[#F2F3E2] shadow-sm'
                  : 'text-[#647B62] hover:text-[#263629]'
              }`}
            >
              Chart Tools
            </button>
            <button
              onClick={() => setActiveTab('data_exports')}
              className={`py-1.5 px-2 rounded-md transition text-center cursor-pointer ${
                activeTab === 'data_exports'
                  ? 'bg-[#789575] text-[#F2F3E2] shadow-sm'
                  : 'text-[#647B62] hover:text-[#263629]'
              }`}
            >
              Exports
            </button>
            <button
              onClick={() => setActiveTab('citation')}
              className={`py-1.5 px-2 rounded-md transition text-center cursor-pointer ${
                activeTab === 'citation'
                  ? 'bg-[#789575] text-[#F2F3E2] shadow-sm'
                  : 'text-[#647B62] hover:text-[#263629]'
              }`}
            >
              Citation
            </button>
          </div>

          {/* TAB 1: CHART TOOLS (Download Images, Download Plot as PNG, Share Chart, Pan, Box) */}
          {activeTab === 'chart_tools' && (
            <div className="space-y-4">
              {/* Interactive Viewport Tools: Pan & Box Zoom */}
              <div className="bg-[#EEF0DC] border border-[#CCD3B5] rounded-xl p-3.5 space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-[#263629]">
                    Plot Navigation Mode
                  </span>
                  <span className="text-[10px] text-[#647B62] font-mono">
                    Active: <strong className="text-[#263629] capitalize">{chartDragMode}</strong>
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => setChartDragMode && setChartDragMode('pan')}
                    className={`flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg border text-xs font-bold transition cursor-pointer ${
                      chartDragMode === 'pan'
                        ? 'bg-[#789575] text-[#F2F3E2] border-[#789575] shadow-sm'
                        : 'bg-[#F2F3E2] text-[#263629] border-[#CCD3B5] hover:bg-[#E5E8CE]'
                    }`}
                  >
                    <Move size={14} />
                    <span>Pan</span>
                  </button>

                  <button
                    onClick={() => setChartDragMode && setChartDragMode('zoom')}
                    className={`flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg border text-xs font-bold transition cursor-pointer ${
                      chartDragMode === 'zoom'
                        ? 'bg-[#789575] text-[#F2F3E2] border-[#789575] shadow-sm'
                        : 'bg-[#F2F3E2] text-[#263629] border-[#CCD3B5] hover:bg-[#E5E8CE]'
                    }`}
                  >
                    <Scan size={14} />
                    <span>Box</span>
                  </button>
                </div>

                {onResetZoom && (
                  <button
                    onClick={onResetZoom}
                    className="w-full flex items-center justify-center gap-1.5 py-1.5 px-3 rounded-lg bg-[#F2F3E2] hover:bg-[#E5E8CE] border border-[#CCD3B5] text-[11px] font-semibold text-[#647B62] hover:text-[#263629] transition cursor-pointer"
                  >
                    <RotateCcw size={12} />
                    <span>Reset Axes & Viewport</span>
                  </button>
                )}
              </div>

              {/* Visual Figure Downloads */}
              <div className="bg-[#EEF0DC] border border-[#CCD3B5] rounded-xl p-3.5 space-y-2.5">
                <span className="text-[11px] font-bold uppercase tracking-wider text-[#263629] block">
                  Figure & Image Exports
                </span>

                {/* 1. Download Plot as PNG */}
                <button
                  onClick={onDownloadActivePlot}
                  className="w-full flex items-center justify-between py-2 px-3 rounded-lg bg-[#F2F3E2] hover:bg-[#E5E8CE] border border-[#CCD3B5] text-xs font-bold text-[#263629] transition cursor-pointer shadow-sm group"
                >
                  <div className="flex items-center gap-2">
                    <Camera size={14} className="text-[#789575] group-hover:scale-110 transition" />
                    <span>Download Plot as PNG</span>
                  </div>
                  <span className="text-[10px] font-mono text-[#647B62] bg-[#EEF0DC] px-1.5 py-0.5 rounded border border-[#CCD3B5]">
                    Raster (.png)
                  </span>
                </button>

                {/* 2. Download Images (Batch 5-Panel Figures) */}
                <button
                  onClick={handleDownloadAllImagesClick}
                  disabled={downloadingImages}
                  className="w-full flex items-center justify-between py-2 px-3 rounded-lg bg-[#F2F3E2] hover:bg-[#E5E8CE] border border-[#CCD3B5] text-xs font-bold text-[#263629] transition cursor-pointer shadow-sm group"
                >
                  <div className="flex items-center gap-2">
                    <ImageIcon size={14} className="text-[#789575] group-hover:scale-110 transition" />
                    <span>Download Images</span>
                  </div>
                  <span className="text-[10px] font-mono text-[#647B62] bg-[#EEF0DC] px-1.5 py-0.5 rounded border border-[#CCD3B5]">
                    {downloadingImages ? 'Saving...' : 'All 5 Panels'}
                  </span>
                </button>

                {/* 3. Share Chart */}
                <button
                  onClick={handleShareChart}
                  className="w-full flex items-center justify-between py-2 px-3 rounded-lg bg-[#F2F3E2] hover:bg-[#E5E8CE] border border-[#CCD3B5] text-xs font-bold text-[#263629] transition cursor-pointer shadow-sm group"
                >
                  <div className="flex items-center gap-2">
                    {copiedShare ? (
                      <Check size={14} className="text-emerald-600" />
                    ) : (
                      <Share2 size={14} className="text-[#789575] group-hover:scale-110 transition" />
                    )}
                    <span>Share Chart</span>
                  </div>
                  <span className="text-[10px] font-mono text-[#647B62] bg-[#EEF0DC] px-1.5 py-0.5 rounded border border-[#CCD3B5]">
                    {copiedShare ? 'Copied!' : 'Copy Summary'}
                  </span>
                </button>
              </div>
            </div>
          )}

          {/* TAB 2: DATA EXPORTS (.json, .csv, .keras, .zip) */}
          {activeTab === 'data_exports' && (
            <div className="space-y-3">
              <div className="bg-[#EEF0DC] border border-[#CCD3B5] rounded-xl p-3.5 space-y-2.5">
                <span className="text-[11px] font-bold uppercase tracking-wider text-[#263629] block">
                  Scientific Data & Models
                </span>

                {/* Channel Matrix JSON */}
                <button
                  onClick={handleExportCurrentMatrix}
                  disabled={!currentMatrixData}
                  className="w-full flex items-center justify-between py-2 px-3 rounded-lg bg-[#F2F3E2] hover:bg-[#E5E8CE] disabled:opacity-50 border border-[#CCD3B5] text-xs font-bold text-[#263629] transition cursor-pointer shadow-sm group"
                >
                  <div className="flex items-center gap-2">
                    <FileCode size={14} className="text-[#789575]" />
                    <span>Channel Matrix (.json)</span>
                  </div>
                  <span className="text-[10px] font-mono text-[#647B62] bg-[#EEF0DC] px-1.5 py-0.5 rounded border border-[#CCD3B5]">
                    {exportedJson ? 'Downloaded' : '64×64 Grid'}
                  </span>
                </button>

                {/* Simulation Config JSON */}
                <button
                  onClick={handleExportConfig}
                  className="w-full flex items-center justify-between py-2 px-3 rounded-lg bg-[#F2F3E2] hover:bg-[#E5E8CE] border border-[#CCD3B5] text-xs font-bold text-[#263629] transition cursor-pointer shadow-sm group"
                >
                  <div className="flex items-center gap-2">
                    <FileText size={14} className="text-[#789575]" />
                    <span>Simulation Config (.json)</span>
                  </div>
                  <span className="text-[10px] font-mono text-[#647B62] bg-[#EEF0DC] px-1.5 py-0.5 rounded border border-[#CCD3B5]">
                    {exportedConfig ? 'Downloaded' : 'Parameters'}
                  </span>
                </button>

                {/* Benchmark CSV (MATLAB 6G / 5G) */}
                <a
                  href={`http://localhost:8000/api/download/csv/${activeStandard === '5g_nr' ? '5g' : '6g'}`}
                  target="_blank"
                  rel="noreferrer"
                  className="w-full flex items-center justify-between py-2 px-3 rounded-lg bg-[#F2F3E2] hover:bg-[#E5E8CE] border border-[#CCD3B5] text-xs font-bold text-[#263629] transition cursor-pointer shadow-sm group"
                >
                  <div className="flex items-center gap-2">
                    <FileSpreadsheet size={14} className="text-[#789575]" />
                    <span>Benchmark Results (.csv)</span>
                  </div>
                  <span className="text-[10px] font-mono text-[#647B62] bg-[#EEF0DC] px-1.5 py-0.5 rounded border border-[#CCD3B5]">
                    500 Samples
                  </span>
                </a>

                {/* Trained Model .keras */}
                <a
                  href="http://localhost:8000/api/download/model"
                  target="_blank"
                  rel="noreferrer"
                  className="w-full flex items-center justify-between py-2 px-3 rounded-lg bg-[#F2F3E2] hover:bg-[#E5E8CE] border border-[#CCD3B5] text-xs font-bold text-[#263629] transition cursor-pointer shadow-sm group"
                >
                  <div className="flex items-center gap-2">
                    <Database size={14} className="text-[#789575]" />
                    <span>Trained Model (.keras)</span>
                  </div>
                  <span className="text-[10px] font-mono text-[#647B62] bg-[#EEF0DC] px-1.5 py-0.5 rounded border border-[#CCD3B5]">
                    888K Weights
                  </span>
                </a>

                {/* Full Release ZIP */}
                <a
                  href="http://localhost:8000/api/download/release-zip"
                  target="_blank"
                  rel="noreferrer"
                  className="w-full flex items-center justify-between py-2 px-3 rounded-lg bg-[#789575] hover:bg-[#668263] border border-[#789575] text-xs font-bold text-[#F2F3E2] transition cursor-pointer shadow-sm group"
                >
                  <div className="flex items-center gap-2">
                    <Package size={14} className="text-[#F2F3E2]" />
                    <span>Complete Release (.zip)</span>
                  </div>
                  <span className="text-[10px] font-mono text-[#F2F3E2] bg-[#5F795C] px-1.5 py-0.5 rounded">
                    Distribution
                  </span>
                </a>
              </div>
            </div>
          )}

          {/* TAB 3: CITATION & METADATA */}
          {activeTab === 'citation' && (
            <div className="space-y-3">
              <div className="bg-[#EEF0DC] border border-[#CCD3B5] rounded-xl p-3.5 space-y-2">
                <span className="text-[11px] font-bold uppercase tracking-wider text-[#263629] block">
                  Manuscript Citation
                </span>
                <p className="text-xs text-[#263629] font-medium leading-relaxed">
                  Apurva Kumar, Vibha Patel, and Shweta Kukade. "Attention-Based Residual Learning Channel Estimation for Massive MIMO." <em>Vehicular Communications</em>, Elsevier, 2026.
                </p>
                <div className="relative mt-2">
                  <pre className="p-2.5 rounded-lg bg-[#E5E8CE] border border-[#CCD3B5] text-[10px] font-mono text-[#263629] overflow-x-auto">
                    {bibtexCode}
                  </pre>
                  <button
                    onClick={handleCopyBibtex}
                    className="absolute top-2 right-2 flex items-center gap-1 text-[10px] font-bold bg-[#789575] text-[#F2F3E2] px-2 py-0.5 rounded shadow-sm hover:bg-[#668263] cursor-pointer"
                  >
                    {copiedBibtex ? <Check size={11} /> : <Copy size={11} />}
                    <span>{copiedBibtex ? 'Copied' : 'Copy BibTeX'}</span>
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Sidebar Footer */}
        <div className="pt-3 border-t border-[#CCD3B5]/60 text-[10px] text-[#647B62] font-mono flex items-center justify-between">
          <span>Massive MIMO Studio V2.1</span>
          <span>Port 8000 Active</span>
        </div>
      </aside>
    </>
  );
}

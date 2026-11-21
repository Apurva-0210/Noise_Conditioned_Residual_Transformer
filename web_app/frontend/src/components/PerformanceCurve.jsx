import React, { useRef } from 'react';
import Plot from 'react-plotly.js';
import { Download, FileSpreadsheet, Info } from 'lucide-react';

export default function PerformanceCurve({ sweepData, is6G = false }) {
  if (!sweepData || !sweepData.curves) {
    return (
      <div className="bg-[#F2F3E2] border border-[#CCD3B5] rounded-xl p-8 flex flex-col items-center justify-center min-h-[350px] shadow-subtle">
        <p className="text-[#647B62] text-xs font-medium">Run an SNR sweep to generate numerical benchmark curves.</p>
      </div>
    );
  }

  const { snr_range, curves, scenario } = sweepData;

  const traces = [];

  if (curves.ls) {
    traces.push({
      x: snr_range,
      y: curves.ls,
      mode: 'lines+markers',
      name: 'Least Squares (LS)',
      line: { color: '#7B8B79', dash: 'dash', width: 1.8 },
      marker: { symbol: 'triangle-up', size: 6 },
      hovertemplate: 'LS @ %{x} dB: %{y:.2f} dB<extra></extra>',
    });
  }

  if (curves.mmse) {
    traces.push({
      x: snr_range,
      y: curves.mmse,
      mode: 'lines+markers',
      name: 'Wiener MMSE',
      line: { color: '#446E65', dash: 'dashdot', width: 2 },
      marker: { symbol: 'diamond', size: 6 },
      hovertemplate: 'MMSE @ %{x} dB: %{y:.2f} dB<extra></extra>',
    });
  }

  if (curves.ncrn) {
    traces.push({
      x: snr_range,
      y: curves.ncrn,
      mode: 'lines+markers',
      name: 'NCRN (ConvLSTM Baseline)',
      line: { color: '#A05344', width: 2.2 },
      marker: { symbol: 'circle', size: 6 },
      hovertemplate: 'NCRN @ %{x} dB: %{y:.2f} dB<extra></extra>',
    });
  }

  if (curves.ncrt_base) {
    traces.push({
      x: snr_range,
      y: curves.ncrt_base,
      mode: 'lines+markers',
      name: 'NCRT (Synthetic Base)',
      line: { color: '#788E52', dash: 'dot', width: 2 },
      marker: { symbol: 'cross', size: 7 },
      hovertemplate: 'NCRT Base @ %{x} dB: %{y:.2f} dB<extra></extra>',
    });
  }

  if (curves.ncrt || curves.ncrt_matlab) {
    const yData = curves.ncrt_matlab || curves.ncrt;
    traces.push({
      x: snr_range,
      y: yData,
      mode: 'lines+markers',
      name: is6G ? 'NCRT (6G Sub-THz Trained)' : 'NCRT (Proposed Transformer)',
      line: { color: '#263629', width: 3.2 },
      marker: { symbol: 'square', size: 8, color: '#789575', line: { color: '#263629', width: 1.5 } },
      hovertemplate: '<b>NCRT @ %{x} dB: %{y:.2f} dB</b><extra></extra>',
    });
  }

  const handleDownloadCsv = () => {
    let csvContent = 'data:text/csv;charset=utf-8,snr_db';
    const methods = Object.keys(curves);
    methods.forEach((m) => {
      csvContent += `,${m}_nmse_db`;
    });
    csvContent += '\n';

    snr_range.forEach((s, idx) => {
      let row = `${s}`;
      methods.forEach((m) => {
        row += `,${curves[m][idx]}`;
      });
      csvContent += row + '\n';
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `nmse_benchmark_${scenario || 'mimo'}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="bg-[#F2F3E2] border border-[#CCD3B5] rounded-xl p-4 shadow-subtle flex flex-col space-y-3">
      {/* Chart Title and Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#CCD3B5]/70 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-black text-[#263629] tracking-wide">
              Monte Carlo Channel Estimation Benchmark
            </h3>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-[#789575] text-[#F2F3E2] font-mono font-bold">
              NMSE in dB
            </span>
          </div>
          <p className="text-[11px] text-[#647B62] font-medium mt-0.5">
            Scenario: <strong className="text-[#263629] font-mono">{scenario}</strong> • Massive MIMO 64×64 Array • Lower is better
          </p>
        </div>

        {/* Export CSV button */}
        <button
          onClick={handleDownloadCsv}
          className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-[#EEF0DC] hover:bg-[#F2F3E2] border border-[#CCD3B5] text-[#263629] text-xs font-bold transition shadow-sm shrink-0 self-start sm:self-center"
          title="Download benchmark data as CSV"
        >
          <FileSpreadsheet size={13} className="text-[#789575]" />
          <span>Export CSV</span>
        </button>
      </div>

      {/* Plotly Canvas */}
      <div className="w-full h-[320px] overflow-hidden bg-[#F5F5E9] rounded-lg border border-[#CCD3B5]/60">
        <Plot
          data={traces}
          layout={{
            autosize: true,
            margin: { t: 15, b: 40, l: 45, r: 15 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: '#F5F5E9',
            xaxis: {
              title: { text: 'Signal-to-Noise Ratio (SNR in dB)', font: { color: '#647B62', size: 10 } },
              tickfont: { color: '#647B62', size: 9 },
              gridcolor: '#DCE2C8',
              dtick: 5,
            },
            yaxis: {
              title: { text: 'NMSE (dB)', font: { color: '#647B62', size: 10 } },
              tickfont: { color: '#647B62', size: 9 },
              gridcolor: '#DCE2C8',
            },
            legend: {
              x: 0.02,
              y: 0.05,
              font: { color: '#263629', size: 9 },
              bgcolor: 'rgba(242, 243, 226, 0.92)',
              bordercolor: '#CCD3B5',
              borderwidth: 1,
            },
          }}
          useResizeHandler={true}
          style={{ width: '100%', height: '100%' }}
          config={{ 
            displayModeBar: false, 
            responsive: true,
          }}
        />
      </div>

      {/* Scientific Benchmark Information Panel (Section 6) */}
      <div className="bg-[#EEF0DC] p-2.5 rounded-lg border border-[#CCD3B5] text-[10px] text-[#647B62] font-mono grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
        <div>
          <span className="block text-[9px] uppercase font-bold text-[#647B62]/80">Dataset:</span>
          <span className="text-[#263629] font-bold">{is6G ? 'MATLAB 142 GHz' : '3GPP TR 38.901'}</span>
        </div>
        <div>
          <span className="block text-[9px] uppercase font-bold text-[#647B62]/80">Scenario:</span>
          <span className="text-[#263629] font-bold">{scenario}</span>
        </div>
        <div>
          <span className="block text-[9px] uppercase font-bold text-[#647B62]/80">SNR Range:</span>
          <span className="text-[#263629] font-bold">−15 to +20 dB</span>
        </div>
        <div>
          <span className="block text-[9px] uppercase font-bold text-[#647B62]/80">Realizations:</span>
          <span className="text-[#263629] font-bold">500 Samples</span>
        </div>
        <div>
          <span className="block text-[9px] uppercase font-bold text-[#647B62]/80">Model Weights:</span>
          <span className="text-[#263629] font-bold">{is6G ? 'ncrt_6g.keras' : 'ncrt_tdl_64.keras'}</span>
        </div>
        <div>
          <span className="block text-[9px] uppercase font-bold text-[#647B62]/80">Evaluation:</span>
          <span className="text-[#263629] font-bold">VehCom Baseline</span>
        </div>
      </div>
    </div>
  );
}

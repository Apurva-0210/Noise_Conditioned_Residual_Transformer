import React, { useRef } from 'react';
import Plot from 'react-plotly.js';
import { Download, FileSpreadsheet, Info } from 'lucide-react';

export default function PerformanceCurve({ sweepData, is6G = false }) {
  if (!sweepData || !sweepData.curves) {
    return (
      <div className="gw-panel border border-[#1E293B] rounded-xl p-8 flex flex-col items-center justify-center min-h-[350px]">
        <p className="text-[#94A3B8] text-xs font-medium">Run an SNR sweep to generate numerical benchmark curves.</p>
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
      line: { color: '#64748B', dash: 'dash', width: 1.8 },
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
      line: { color: '#38BDF8', dash: 'dashdot', width: 2 },
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
      line: { color: '#F43F5E', width: 2.2 },
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
      line: { color: '#3B82F6', dash: 'dot', width: 2 },
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
      line: { color: '#D4FF00', width: 3.2 },
      marker: { symbol: 'square', size: 8, color: '#D4FF00', line: { color: '#060911', width: 1.5 } },
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
    <div className="gw-panel border border-[#1E293B]/80 rounded-lg p-4 flex flex-col space-y-3">
      {/* Chart Title and Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#1E293B]/70 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-[#F8FAFC] tracking-tight">
              Monte Carlo channel estimation benchmark
            </h3>
            <span className="text-[10px] px-2 py-0.5 rounded bg-[#2563EB] text-white font-mono">
              NMSE in dB
            </span>
          </div>
          <p className="text-[11px] text-[#94A3B8] mt-0.5">
            Scenario: <strong className="text-[#D4FF00] font-mono">{scenario}</strong> • 64×64 UPA • Lower is better
          </p>
        </div>

        {/* Export CSV button */}
        <button
          onClick={handleDownloadCsv}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[#0B1120] hover:bg-[#1E293B] border border-[#1E293B] text-[#F8FAFC] text-xs font-medium transition shrink-0 self-start sm:self-center cursor-pointer"
          title="Download benchmark data as CSV"
        >
          <FileSpreadsheet size={13} className="text-[#D4FF00]" />
          <span>Export CSV</span>
        </button>
      </div>

      {/* Plotly Canvas */}
      <div className="w-full h-[320px] overflow-hidden bg-[#060911] rounded-md border border-[#1E293B]/70">
        <Plot
          data={traces}
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
          config={{ 
            displayModeBar: false, 
            responsive: true,
          }}
        />
      </div>

      {/* Scientific Benchmark Information Panel */}
      <div className="bg-[#0B1120] p-2.5 rounded-md border border-[#1E293B] text-[10px] text-[#94A3B8] font-mono grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
        <div>
          <span className="block text-[9px] text-[#64748B]">Dataset:</span>
          <span className="text-[#F8FAFC] font-medium">{is6G ? 'MATLAB 142 GHz' : '3GPP TR 38.901'}</span>
        </div>
        <div>
          <span className="block text-[9px] text-[#64748B]">Scenario:</span>
          <span className="text-[#D4FF00] font-medium">{scenario}</span>
        </div>
        <div>
          <span className="block text-[9px] text-[#64748B]">SNR Range:</span>
          <span className="text-[#F8FAFC] font-medium">−15 to +20 dB</span>
        </div>
        <div>
          <span className="block text-[9px] text-[#64748B]">Realizations:</span>
          <span className="text-[#F8FAFC] font-medium">500 samples</span>
        </div>
        <div>
          <span className="block text-[9px] text-[#64748B]">Model Weights:</span>
          <span className="text-[#F8FAFC] font-medium">{is6G ? 'ncrt_6g.keras' : 'ncrt_tdl_64.keras'}</span>
        </div>
        <div>
          <span className="block text-[9px] text-[#64748B]">Evaluation:</span>
          <span className="text-[#F8FAFC] font-medium">VehCom Baseline</span>
        </div>
      </div>
    </div>
  );
}

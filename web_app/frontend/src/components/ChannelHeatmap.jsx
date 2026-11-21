import React, { useState, useRef } from 'react';
import Plot from 'react-plotly.js';
import Plotly from 'plotly.js-dist-min';
import { Palette, Info, Camera, Download } from 'lucide-react';

export default function ChannelHeatmap({ 
  title, 
  zData, 
  defaultColorscale = 'Viridis', 
  subtitle,
  highlight = false,
  badgeText = null,
  sharedZRange = null, // [zmin, zmax] if shared scale is active
  isErrorMatrix = false,
  matrixDim = '64×64 Massive MIMO',
  dragmode = 'zoom', // 'zoom' | 'pan'
}) {
  const [colorscale, setColorscale] = useState(
    isErrorMatrix ? 'Plasma' : defaultColorscale
  );
  const plotContainerRef = useRef(null);

  if (!zData || zData.length === 0) {
    return (
      <div className="gw-panel border border-[#1E293B] rounded-xl p-5 flex flex-col items-center justify-center min-h-[280px]">
        <div className="w-7 h-7 rounded-full border-2 border-[#1E293B] border-t-[#2563EB] animate-spin mb-2" />
        <p className="text-[#94A3B8] text-xs font-medium">Computing channel response matrix...</p>
      </div>
    );
  }

  const dim = zData.length;

  // Compute exact min/max/mean
  let maxVal = -Infinity;
  let minVal = Infinity;
  let sum = 0;
  let count = 0;
  for (let r = 0; r < dim; r++) {
    for (let c = 0; c < zData[r].length; c++) {
      const v = zData[r][c];
      if (v > maxVal) maxVal = v;
      if (v < minVal) minVal = v;
      sum += v;
      count++;
    }
  }
  const meanVal = count > 0 ? (sum / count).toFixed(3) : '--';
  maxVal = maxVal !== -Infinity ? maxVal.toFixed(3) : '--';
  minVal = minVal !== Infinity ? minVal.toFixed(3) : '--';

  // Heatmap trace
  const trace = {
    z: zData,
    type: 'heatmap',
    colorscale: colorscale,
    showscale: true,
    colorbar: {
      thickness: 8,
      len: 0.9,
      tickfont: { color: '#94A3B8', size: 8 },
      tickcolor: '#334155',
      outlinecolor: 'transparent',
    },
    hoverongaps: false,
    hovertemplate: 'Tx: %{x}<br>Rx: %{y}<br>Mag: %{z:.3f}<extra></extra>',
  };

  if (sharedZRange && !isErrorMatrix) {
    trace.zmin = sharedZRange[0];
    trace.zmax = sharedZRange[1];
  }

  const handleDownloadPng = () => {
    if (plotContainerRef.current) {
      const gd = plotContainerRef.current.querySelector('.js-plotly-plot');
      if (gd && window.Plotly) {
        window.Plotly.downloadImage(gd, {
          format: 'png',
          width: 800,
          height: 700,
          filename: title.toLowerCase().replace(/[^a-z0-9]/g, '_'),
        });
      }
    }
  };

  return (
    <div className={`gw-panel rounded-lg p-3.5 flex flex-col justify-between border transition-all duration-200 ${
      highlight 
        ? 'border-[#D4FF00]/60 ring-1 ring-[#D4FF00]/40 shadow-lg shadow-[#D4FF00]/5' 
        : 'border-[#1E293B]/80 hover:border-[#2563EB]/50'
    }`}>
      {/* Header */}
      <div className="flex items-start justify-between gap-1.5 mb-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5 flex-wrap">
            <h3 className="text-xs font-semibold text-[#F8FAFC]">
              {title}
            </h3>
            {badgeText && (
              <span className={`text-[10px] px-1.5 py-0.2 rounded font-mono font-medium border shrink-0 ${
                isErrorMatrix 
                  ? 'bg-rose-500/15 text-rose-400 border-rose-500/30'
                  : highlight
                    ? 'bg-[#D4FF00] text-[#060911] font-semibold border border-[#D4FF00]'
                    : 'bg-[#2563EB] text-white border border-[#3B82F6]'
              }`}>
                {badgeText}
              </span>
            )}
          </div>
          {subtitle && (
            <p className="text-[10px] text-[#94A3B8] mt-0.5 leading-snug">
              {subtitle}
            </p>
          )}
        </div>

        {/* Colormap, Download Icon & Dim */}
        <div className="flex items-center gap-1 shrink-0 ml-1">
          <button
            onClick={handleDownloadPng}
            className="p-1 rounded bg-[#0B1120] hover:bg-[#1E293B] border border-[#1E293B] text-[#94A3B8] hover:text-[#F8FAFC] transition cursor-pointer"
            title="Download this heatmap as PNG"
          >
            <Camera size={11} />
          </button>

          <div className="flex items-center bg-[#0B1120] border border-[#1E293B] rounded px-1.5 py-0.5 text-[9px]">
            <Palette size={10} className="text-[#60A5FA] mr-1" />
            <select
              value={colorscale}
              onChange={(e) => setColorscale(e.target.value)}
              className="bg-transparent text-[#F8FAFC] font-medium focus:outline-none cursor-pointer text-[9px]"
              title="Select colormap"
            >
              <option value="Viridis" className="bg-[#0B1120] text-[#F8FAFC]">Viridis</option>
              <option value="Plasma" className="bg-[#0B1120] text-[#F8FAFC]">Plasma</option>
              <option value="Turbo" className="bg-[#0B1120] text-[#F8FAFC]">Turbo</option>
              <option value="YlGnBu" className="bg-[#0B1120] text-[#F8FAFC]">YlGnBu</option>
              <option value="Greens" className="bg-[#0B1120] text-[#F8FAFC]">Greens</option>
              <option value="Hot" className="bg-[#0B1120] text-[#F8FAFC]">Hot</option>
            </select>
          </div>

          <span 
            className="text-[9px] px-1.5 py-0.5 rounded bg-[#0B1120] text-[#60A5FA] font-mono border border-[#1E293B]"
            title={`Sampled grid: ${dim}×${dim} (Full physical array: 64×64)`}
          >
            {dim}×{dim}
          </span>
        </div>
      </div>

      {/* Heatmap Plot */}
      <div 
        ref={plotContainerRef}
        className="w-full h-[210px] flex items-center justify-center overflow-hidden bg-[#060911] rounded-lg border border-[#1E293B]/70"
      >
        <Plot
          data={[trace]}
          layout={{
            autosize: true,
            dragmode: dragmode,
            margin: { t: 4, b: 24, l: 28, r: 4 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            xaxis: {
              title: { text: 'Tx Antenna', font: { color: '#94A3B8', size: 9 } },
              tickfont: { color: '#64748B', size: 8 },
              showgrid: false,
            },
            yaxis: {
              title: { text: 'Rx Antenna', font: { color: '#94A3B8', size: 9 } },
              tickfont: { color: '#64748B', size: 8 },
              showgrid: false,
            },
          }}
          useResizeHandler={true}
          style={{ width: '100%', height: '100%' }}
          config={{ displayModeBar: false, responsive: true }}
        />
      </div>

      {/* Telemetry Footer */}
      <div className="mt-2 pt-1.5 border-t border-[#1E293B]/60 flex items-center justify-between text-[10px] text-[#64748B] font-mono tabular-nums">
        <span>Min: <strong className="text-[#F8FAFC]">{minVal}</strong></span>
        <span>Avg: <strong className="text-[#F8FAFC]">{meanVal}</strong></span>
        <span>Peak: <strong className={highlight ? "text-[#D4FF00]" : "text-[#F8FAFC]"}>{maxVal}</strong></span>
      </div>
    </div>
  );
}

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
      <div className="bg-[#F2F3E2] border border-[#CCD3B5] rounded-xl p-5 flex flex-col items-center justify-center min-h-[280px] shadow-subtle">
        <div className="w-7 h-7 rounded-full border-2 border-[#CCD3B5] border-t-[#789575] animate-spin mb-2" />
        <p className="text-[#647B62] text-xs font-medium">Computing channel response matrix...</p>
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
      tickfont: { color: '#647B62', size: 8 },
      tickcolor: '#CCD3B5',
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
    <div className={`bg-[#F2F3E2] rounded-lg p-3.5 flex flex-col justify-between border transition-colors ${
      highlight 
        ? 'border-[#789575]' 
        : 'border-[#CCD3B5]'
    }`}>
      {/* Header */}
      <div className="flex items-start justify-between gap-1.5 mb-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5 flex-wrap">
            <h3 className="text-xs font-semibold text-[#263629]">
              {title}
            </h3>
            {badgeText && (
              <span className={`text-[10px] px-1.5 py-0.2 rounded font-mono font-medium border shrink-0 ${
                isErrorMatrix 
                  ? 'bg-[#F9E6E6] text-[#9C4141] border-[#9C4141]/30'
                  : 'bg-[#789575] text-[#F2F3E2] border-[#789575]'
              }`}>
                {badgeText}
              </span>
            )}
          </div>
          {subtitle && (
            <p className="text-[10px] text-[#647B62] mt-0.5 leading-snug">
              {subtitle}
            </p>
          )}
        </div>

        {/* Colormap, Download Icon & Dim */}
        <div className="flex items-center gap-1 shrink-0 ml-1">
          <button
            onClick={handleDownloadPng}
            className="p-1 rounded bg-[#EEF0DC] hover:bg-[#E5E8CE] border border-[#CCD3B5] text-[#647B62] hover:text-[#263629] transition cursor-pointer"
            title="Download this heatmap as PNG"
          >
            <Camera size={11} />
          </button>

          <div className="flex items-center bg-[#EEF0DC] border border-[#CCD3B5] rounded px-1.5 py-0.5 text-[9px]">
            <Palette size={10} className="text-[#647B62] mr-1" />
            <select
              value={colorscale}
              onChange={(e) => setColorscale(e.target.value)}
              className="bg-transparent text-[#263629] font-medium focus:outline-none cursor-pointer text-[9px]"
              title="Select colormap"
            >
              <option value="Viridis">Viridis</option>
              <option value="Plasma">Plasma</option>
              <option value="Turbo">Turbo</option>
              <option value="YlGnBu">YlGnBu</option>
              <option value="Greens">Greens</option>
              <option value="Hot">Hot</option>
            </select>
          </div>

          <span 
            className="text-[9px] px-1.5 py-0.5 rounded bg-[#EEF0DC] text-[#263629] font-mono border border-[#CCD3B5]"
            title={`Sampled grid: ${dim}×${dim} (Full physical array: 64×64)`}
          >
            {dim}×{dim}
          </span>
        </div>
      </div>

      {/* Heatmap Plot */}
      <div 
        ref={plotContainerRef}
        className="w-full h-[210px] flex items-center justify-center overflow-hidden bg-[#F5F5E9] rounded-lg border border-[#CCD3B5]/60"
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
              title: { text: 'Tx Antenna', font: { color: '#647B62', size: 9 } },
              tickfont: { color: '#647B62', size: 8 },
              showgrid: false,
            },
            yaxis: {
              title: { text: 'Rx Antenna', font: { color: '#647B62', size: 9 } },
              tickfont: { color: '#647B62', size: 8 },
              showgrid: false,
            },
          }}
          useResizeHandler={true}
          style={{ width: '100%', height: '100%' }}
          config={{ displayModeBar: false, responsive: true }}
        />
      </div>

      {/* Telemetry Footer */}
      <div className="mt-2 pt-1.5 border-t border-[#CCD3B5]/60 flex items-center justify-between text-[10px] text-[#647B62] font-mono tabular-nums">
        <span>Min: <strong className="text-[#263629]">{minVal}</strong></span>
        <span>Avg: <strong className="text-[#263629]">{meanVal}</strong></span>
        <span>Peak: <strong className="text-[#263629]">{maxVal}</strong></span>
      </div>
    </div>
  );
}

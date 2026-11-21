import React from 'react';
import { HelpCircle } from 'lucide-react';

export default function MetricCard({ 
  title, 
  value, 
  unit, 
  subtitle, 
  badge, 
  isHighlighted = false,
  tooltip = null,
  icon: Icon 
}) {
  return (
    <div className={`gw-panel rounded-xl p-4 border flex flex-col justify-between transition-all duration-200 ${
      isHighlighted 
        ? 'border-[#789575] ring-1 ring-[#789575]/30' 
        : 'border-[#CCD3B5]/70 hover:border-[#CCD3B5]'
    }`}>
      {/* Header Row */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="text-xs font-medium text-[#556B52] truncate">
            {title}
          </span>
          {tooltip && (
            <div className="group relative inline-block shrink-0">
              <HelpCircle size={13} className="text-[#7A8C78] hover:text-[#263629] cursor-help transition-colors" />
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block w-48 p-2 bg-[#263629] text-[#F2F3E2] text-[11px] rounded-lg shadow-dropdown z-50 pointer-events-none leading-relaxed">
                {tooltip}
              </div>
            </div>
          )}
        </div>

        {Icon && (
          <div className="w-7 h-7 rounded-lg shrink-0 flex items-center justify-center bg-[#EEF0DC]/70 backdrop-blur-md border border-[#CCD3B5]/70 text-[#556B52]">
            <Icon size={14} strokeWidth={1.5} />
          </div>
        )}
      </div>

      {/* Metric Value Row */}
      <div className="mt-3 flex items-baseline gap-1.5">
        <span className="text-2xl font-semibold tracking-tight font-mono tabular-nums text-[#263629]">
          {value}
        </span>
        {unit && (
          <span className="text-xs font-mono text-[#556B52]">
            {unit}
          </span>
        )}
      </div>

      {/* Footer Details */}
      {(subtitle || badge) && (
        <div className="mt-3 pt-2.5 border-t border-[#CCD3B5]/50 flex items-center justify-between gap-1.5 text-xs">
          {subtitle && (
            <span className="text-[11px] text-[#7A8C78] leading-tight truncate">
              {subtitle}
            </span>
          )}
          {badge && (
            <span className="px-1.5 py-0.5 rounded text-[10px] font-mono text-[#556B52] bg-[#EEF0DC]/70 backdrop-blur-md border border-[#CCD3B5]/70 shrink-0">
              {badge}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

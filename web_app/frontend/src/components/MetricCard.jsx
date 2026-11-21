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
        ? 'border-[#D4FF00]/60 ring-1 ring-[#D4FF00]/40 shadow-lg shadow-[#D4FF00]/5' 
        : 'border-[#1E293B]/80 hover:border-[#2563EB]/60'
    }`}>
      {/* Header Row */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="text-xs font-medium text-[#94A3B8] truncate">
            {title}
          </span>
          {tooltip && (
            <div className="group relative inline-block shrink-0">
              <HelpCircle size={13} className="text-[#64748B] hover:text-[#F8FAFC] cursor-help transition-colors" />
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block w-48 p-2 bg-[#0B1120] text-[#F8FAFC] border border-[#1E293B] text-[11px] rounded-lg shadow-2xl z-50 pointer-events-none leading-relaxed">
                {tooltip}
              </div>
            </div>
          )}
        </div>

        {Icon && (
          <div className={`w-7 h-7 rounded-lg shrink-0 flex items-center justify-center backdrop-blur-md border ${
            isHighlighted 
              ? 'bg-[#D4FF00]/10 border-[#D4FF00]/30 text-[#D4FF00]' 
              : 'bg-[#1E293B]/60 border-[#334155]/50 text-[#60A5FA]'
          }`}>
            <Icon size={14} strokeWidth={1.5} />
          </div>
        )}
      </div>

      {/* Metric Value Row */}
      <div className="mt-3 flex items-baseline gap-1.5">
        <span className={`text-2xl font-semibold tracking-tight font-mono tabular-nums ${
          isHighlighted ? 'text-[#D4FF00]' : 'text-[#F8FAFC]'
        }`}>
          {value}
        </span>
        {unit && (
          <span className={`text-xs font-mono ${isHighlighted ? 'text-[#D4FF00]/80' : 'text-[#64748B]'}`}>
            {unit}
          </span>
        )}
      </div>

      {/* Footer Details */}
      {(subtitle || badge) && (
        <div className="mt-3 pt-2.5 border-t border-[#1E293B]/60 flex items-center justify-between gap-1.5 text-xs">
          {subtitle && (
            <span className="text-[11px] text-[#94A3B8] leading-tight truncate">
              {subtitle}
            </span>
          )}
          {badge && (
            <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono shrink-0 ${
              isHighlighted
                ? 'text-[#060911] bg-[#D4FF00] font-semibold border border-[#D4FF00]'
                : 'text-[#94A3B8] bg-[#1E293B]/70 backdrop-blur-md border border-[#334155]/60'
            }`}>
              {badge}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

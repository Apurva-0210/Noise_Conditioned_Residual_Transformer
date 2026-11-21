import React from 'react';
import { HelpCircle } from 'lucide-react';

export default function MetricCard({ 
  title, 
  value, 
  unit, 
  subtitle, 
  badge, 
  badgeVariant = 'default', // 'default' | 'accent' | 'success' | 'warning'
  isHighlighted = false,
  tooltip = null,
  icon: Icon 
}) {
  const getBadgeClass = () => {
    switch (badgeVariant) {
      case 'accent':
        return 'bg-[#789575] text-[#F2F3E2] border-[#789575]';
      case 'success':
        return 'bg-[#E4F0E6] text-[#36663E] border-[#A7B99A]';
      case 'warning':
        return 'bg-[#F7EED9] text-[#A67824] border-[#DCE2C8]';
      default:
        return 'bg-[#EEF0DC] text-[#263629] border-[#CCD3B5]';
    }
  };

  return (
    <div className={`relative bg-[#F2F3E2] rounded-xl p-4 transition-all duration-200 border flex flex-col justify-between shadow-subtle ${
      isHighlighted 
        ? 'border-[#789575] ring-1 ring-[#789575]/25' 
        : 'border-[#CCD3B5] hover:border-[#A7B99A]'
    }`}>
      {/* Top Accent line for proposed estimator */}
      {isHighlighted && (
        <div className="absolute top-0 left-0 right-0 h-1 bg-[#789575] rounded-t-xl" />
      )}

      {/* Header Row */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="text-[11px] font-bold uppercase tracking-wider text-[#647B62] truncate">
            {title}
          </span>
          {tooltip && (
            <div className="group relative inline-block shrink-0">
              <HelpCircle size={12} className="text-[#647B62] hover:text-[#263629] cursor-help transition-colors" />
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 hidden group-hover:block w-48 p-2 bg-[#263629] text-[#F2F3E2] text-[10px] rounded-lg shadow-dropdown z-50 pointer-events-none leading-relaxed">
                {tooltip}
              </div>
            </div>
          )}
        </div>

        {Icon && (
          <div className={`p-1.5 rounded-lg shrink-0 border ${
            isHighlighted 
              ? 'bg-[#789575] text-[#F2F3E2] border-[#789575]' 
              : 'bg-[#EEF0DC] text-[#263629] border-[#CCD3B5]'
          }`}>
            <Icon size={16} />
          </div>
        )}
      </div>

      {/* Metric Value Row */}
      <div className="mt-2.5 flex items-baseline gap-1.5">
        <span className={`text-2xl font-black tracking-tight font-mono tabular-nums ${
          isHighlighted ? 'text-[#263629]' : 'text-[#263629]'
        }`}>
          {value}
        </span>
        {unit && (
          <span className="text-xs font-mono font-bold text-[#647B62]">
            {unit}
          </span>
        )}
      </div>

      {/* Footer Details */}
      {(subtitle || badge) && (
        <div className="mt-3 pt-2 border-t border-[#CCD3B5]/60 flex items-center justify-between gap-1.5 text-xs">
          {subtitle && (
            <span className="text-[10.5px] text-[#647B62] font-medium leading-tight">
              {subtitle}
            </span>
          )}
          {badge && (
            <span className={`px-1.5 py-0.5 rounded text-[9.5px] font-bold font-mono border shrink-0 ${getBadgeClass()}`}>
              {badge}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

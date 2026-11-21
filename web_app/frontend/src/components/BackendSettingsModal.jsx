import React, { useState, useEffect } from 'react';
import { 
  Server, 
  CheckCircle2, 
  AlertCircle, 
  RefreshCw, 
  ExternalLink, 
  X, 
  Globe, 
  HelpCircle,
  Zap
} from 'lucide-react';
import { getBackendUrl, setBackendUrl, resetBackendUrl, testConnection } from '../api';

export default function BackendSettingsModal({ isOpen, onClose, onConnected }) {
  const [url, setUrl] = useState('');
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState(null); // { success: boolean, message: string }

  useEffect(() => {
    if (isOpen) {
      setUrl(getBackendUrl());
      setTestResult(null);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleTest = async (testUrl) => {
    const targetUrl = testUrl || url;
    if (!targetUrl.trim()) {
      setTestResult({ success: false, message: 'Please enter a valid URL' });
      return false;
    }

    setTesting(true);
    setTestResult(null);
    try {
      const data = await testConnection(targetUrl);
      setTestResult({ 
        success: true, 
        message: `Successfully connected to FastAPI (Status: ${data.status || 'online'})` 
      });
      setTesting(false);
      return true;
    } catch (err) {
      setTestResult({ 
        success: false, 
        message: err.message || 'Failed to reach backend endpoint.' 
      });
      setTesting(false);
      return false;
    }
  };

  const handleSave = async () => {
    if (!url.trim()) return;
    setBackendUrl(url);
    const success = await handleTest(url);
    if (success && onConnected) {
      setTimeout(() => {
        onConnected();
        onClose();
      }, 700);
    }
  };

  const handleReset = () => {
    resetBackendUrl();
    const defaultUrl = getBackendUrl();
    setUrl(defaultUrl);
    setTestResult(null);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm animate-fadeIn">
      <div className="bg-[#F2F3E2] border border-[#CCD3B5] rounded-2xl shadow-2xl max-w-lg w-full overflow-hidden text-[#263629]">
        {/* Header */}
        <div className="px-6 py-4 border-b border-[#CCD3B5] flex items-center justify-between bg-[#EEF0DC]">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-[#789575]/20 border border-[#789575]/40 flex items-center justify-center text-[#263629]">
              <Server size={18} />
            </div>
            <div>
              <h3 className="text-sm font-black text-[#263629] tracking-tight">
                Backend Connection Settings
              </h3>
              <p className="text-[11px] text-[#647B62] font-medium">
                Connect your frontend to Render Cloud API or local server
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-[#647B62] hover:text-[#263629] hover:bg-[#CCD3B5]/40 transition"
          >
            <X size={18} />
          </button>
        </div>

        {/* Body */}
        <div className="p-6 space-y-4 text-xs">
          {/* Why is it offline banner if on HTTPS and url is localhost */}
          {typeof window !== 'undefined' && window.location.protocol === 'https:' && url.includes('localhost') && (
            <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-xl flex items-start gap-2.5 text-amber-900">
              <AlertCircle size={16} className="text-amber-700 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <span className="font-bold">Why is FastAPI Offline on Vercel?</span>
                <p className="text-[11px] leading-relaxed text-amber-800">
                  Your frontend is live on HTTPS, but currently configured to query <code className="bg-amber-200/50 px-1 py-0.5 rounded font-mono">http://localhost:8000</code>.
                  Paste your live <strong>Render backend URL</strong> below to link them!
                </p>
              </div>
            </div>
          )}

          {/* URL Input */}
          <div className="space-y-1.5">
            <label className="font-bold uppercase tracking-wider text-[11px] text-[#647B62] flex items-center justify-between">
              <span>Backend API URL (Render or Local)</span>
              <span className="font-mono text-[10px] text-[#647B62]">No trailing slash</span>
            </label>
            <div className="relative">
              <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://your-service-name.onrender.com"
                className="w-full bg-[#E5E8CE] border border-[#CCD3B5] rounded-xl pl-3.5 pr-24 py-2.5 font-mono text-xs text-[#263629] placeholder-[#8C9E89] focus:outline-none focus:ring-2 focus:ring-[#789575] focus:border-transparent"
              />
              <button
                type="button"
                onClick={() => handleTest()}
                disabled={testing}
                className="absolute right-1.5 top-1.5 bottom-1.5 px-3 rounded-lg bg-[#789575] text-[#F2F3E2] font-semibold flex items-center gap-1.5 hover:bg-[#668263] transition disabled:opacity-50"
              >
                {testing ? (
                  <>
                    <RefreshCw size={12} className="animate-spin" />
                    <span>Testing...</span>
                  </>
                ) : (
                  <>
                    <Zap size={12} />
                    <span>Ping</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Test Status Feedback */}
          {testing && (
            <div className="p-3 bg-[#EEF0DC] border border-[#CCD3B5] rounded-xl flex items-center gap-2.5 text-[#647B62]">
              <RefreshCw size={14} className="animate-spin text-[#789575]" />
              <div>
                <p className="font-bold text-[#263629]">Pinging backend server...</p>
                <p className="text-[10px]">
                  If your Render service was sleeping (free tier spins down after 15m), it takes ~30–45s to wake up.
                </p>
              </div>
            </div>
          )}

          {testResult && !testing && (
            <div
              className={`p-3 rounded-xl border flex items-start gap-2.5 ${
                testResult.success
                  ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-900'
                  : 'bg-rose-500/10 border-rose-500/30 text-rose-900'
              }`}
            >
              {testResult.success ? (
                <CheckCircle2 size={16} className="text-emerald-700 shrink-0 mt-0.5" />
              ) : (
                <AlertCircle size={16} className="text-rose-700 shrink-0 mt-0.5" />
              )}
              <div className="space-y-0.5">
                <span className="font-bold">
                  {testResult.success ? 'Connection Verified!' : 'Connection Failed'}
                </span>
                <p className="text-[11px] leading-relaxed opacity-90">{testResult.message}</p>
              </div>
            </div>
          )}

          {/* Tips / Instructions */}
          <div className="p-3 bg-[#EEF0DC] border border-[#CCD3B5] rounded-xl space-y-1.5 text-[11px] text-[#647B62]">
            <div className="flex items-center gap-1.5 font-bold text-[#263629]">
              <HelpCircle size={13} className="text-[#789575]" />
              <span>How to find your Render URL</span>
            </div>
            <ol className="list-decimal list-inside space-y-1 pl-1">
              <li>Open your <strong>Render Dashboard</strong> (<a href="https://dashboard.render.com" target="_blank" rel="noreferrer" className="underline hover:text-[#263629]">dashboard.render.com</a>)</li>
              <li>Click your Web Service (e.g. <code>massive-mimo-ai-api</code>)</li>
              <li>Copy the URL located under the service title (e.g. <code>https://xxx.onrender.com</code>)</li>
              <li>Paste it here and click <strong>Save & Connect</strong></li>
            </ol>
            <div className="pt-1 text-[10px] text-[#647B62]/80 italic">
              Tip: You can also add <code>VITE_API_BASE</code> in Vercel Project Settings &gt; Environment Variables.
            </div>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="px-6 py-3.5 bg-[#EEF0DC] border-t border-[#CCD3B5] flex items-center justify-between">
          <button
            onClick={handleReset}
            className="text-[11px] font-bold text-[#647B62] hover:text-[#263629] transition"
          >
            Reset to Default
          </button>
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="px-3.5 py-1.5 rounded-lg border border-[#CCD3B5] text-[#263629] font-semibold hover:bg-[#CCD3B5]/40 transition text-xs"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={testing || !url.trim()}
              className="px-4 py-1.5 rounded-lg bg-[#789575] hover:bg-[#668263] text-[#F2F3E2] font-bold shadow-sm transition text-xs flex items-center gap-1.5 disabled:opacity-50"
            >
              <CheckCircle2 size={13} />
              <span>Save & Connect</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

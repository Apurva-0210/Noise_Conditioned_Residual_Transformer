export function sanitizeBackendUrl(raw) {
  if (!raw || typeof raw !== "string") return "";
  let url = raw.trim().replace(/\/+$/, "");
  // Strip accidental trailing /api/health or /api
  url = url.replace(/\/api\/health$/i, "");
  url = url.replace(/\/api$/i, "");
  return url.replace(/\/+$/, "");
}

export function getBackendUrl() {
  if (typeof window !== "undefined") {
    const custom = localStorage.getItem("massive_mimo_api_url");
    if (custom && custom.trim()) {
      return sanitizeBackendUrl(custom);
    }
  }
  const envUrl = import.meta.env.VITE_API_BASE;
  if (envUrl && envUrl.trim()) {
    return sanitizeBackendUrl(envUrl);
  }
  return "http://localhost:8000";
}

export function setBackendUrl(url) {
  if (typeof window !== "undefined") {
    if (url && url.trim()) {
      localStorage.setItem("massive_mimo_api_url", sanitizeBackendUrl(url));
    } else {
      localStorage.removeItem("massive_mimo_api_url");
    }
  }
}

export function resetBackendUrl() {
  if (typeof window !== "undefined") {
    localStorage.removeItem("massive_mimo_api_url");
  }
}

export function getApiBase() {
  return `${getBackendUrl()}/api`;
}

export async function testConnection(customUrl) {
  const base = (customUrl ? customUrl.trim().replace(/\/+$/, "") : getBackendUrl());
  const urlToTest = `${base}/api/health`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 25000); // 25s timeout for Render wake-up
  try {
    const res = await fetch(urlToTest, { signal: controller.signal });
    clearTimeout(timeoutId);
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    return await res.json();
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === "AbortError") {
      throw new Error("Connection timed out (25s). If on Render free tier, the container may be spinning up. Please wait 15 seconds and retry.");
    }
    throw err;
  }
}

export async function checkHealth() {
  const res = await fetch(`${getApiBase()}/health`);
  if (!res.ok) throw new Error("Backend offline");
  return await res.json();
}

export async function fetchScenarios() {
  const res = await fetch(`${getApiBase()}/scenarios`);
  if (!res.ok) throw new Error("Failed to fetch scenarios");
  return await res.json();
}

export async function runEstimation(params) {
  const res = await fetch(`${getApiBase()}/estimate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error("Estimation failed");
  return await res.json();
}

export async function runSweep(params) {
  const res = await fetch(`${getApiBase()}/sweep`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error("Sweep failed");
  return await res.json();
}

// --- 6G Sub-THz & MATLAB Studio APIs ---
export async function fetchMatlabDatasets() {
  const res = await fetch(`${getApiBase()}/matlab/datasets`);
  if (!res.ok) throw new Error("Failed to fetch MATLAB datasets");
  return await res.json();
}

export async function runMatlabEstimate(params) {
  const res = await fetch(`${getApiBase()}/matlab/estimate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error("MATLAB estimation failed");
  return await res.json();
}

export async function fetchMatlabBenchmarks() {
  const res = await fetch(`${getApiBase()}/matlab/benchmarks`);
  if (!res.ok) throw new Error("Failed to fetch MATLAB benchmarks");
  return await res.json();
}

export async function fetchMatlabScript() {
  const res = await fetch(`${getApiBase()}/matlab/script`);
  if (!res.ok) throw new Error("Failed to fetch MATLAB script");
  return await res.json();
}

// Official Production Backend endpoint
const OFFICIAL_PRODUCTION_API = "https://noise-conditioned-residual-transformer.onrender.com";

export function getBackendUrl() {
  const envUrl = import.meta.env.VITE_API_BASE;
  if (envUrl && envUrl.trim()) {
    return envUrl.trim().replace(/\/+$/, "");
  }
  // If running in production (deployed on Vercel or any live host), strictly use official Render backend
  if (typeof window !== "undefined" && window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1") {
    return OFFICIAL_PRODUCTION_API;
  }
  // Local development fallback
  return "http://localhost:8000";
}

export function getApiBase() {
  return `${getBackendUrl()}/api`;
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

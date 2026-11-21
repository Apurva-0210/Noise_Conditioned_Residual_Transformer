const BACKEND_URL = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const API_BASE = `${BACKEND_URL}/api`;

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error("Backend offline");
  return await res.json();
}

export async function fetchScenarios() {
  const res = await fetch(`${API_BASE}/scenarios`);
  if (!res.ok) throw new Error("Failed to fetch scenarios");
  return await res.json();
}

export async function runEstimation(params) {
  const res = await fetch(`${API_BASE}/estimate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error("Estimation failed");
  return await res.json();
}

export async function runSweep(params) {
  const res = await fetch(`${API_BASE}/sweep`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error("Sweep failed");
  return await res.json();
}

// --- 6G Sub-THz & MATLAB Studio APIs ---
export async function fetchMatlabDatasets() {
  const res = await fetch(`${API_BASE}/matlab/datasets`);
  if (!res.ok) throw new Error("Failed to fetch MATLAB datasets");
  return await res.json();
}

export async function runMatlabEstimate(params) {
  const res = await fetch(`${API_BASE}/matlab/estimate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error("MATLAB estimation failed");
  return await res.json();
}

export async function fetchMatlabBenchmarks() {
  const res = await fetch(`${API_BASE}/matlab/benchmarks`);
  if (!res.ok) throw new Error("Failed to fetch MATLAB benchmarks");
  return await res.json();
}

export async function fetchMatlabScript() {
  const res = await fetch(`${API_BASE}/matlab/script`);
  if (!res.ok) throw new Error("Failed to fetch MATLAB script");
  return await res.json();
}

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project directories are in path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_APP_DIR = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
VEHCOM_DIR = os.path.abspath(os.path.join(WEB_APP_DIR, ".."))

sys.path.insert(0, VEHCOM_DIR)
sys.path.insert(0, os.path.join(VEHCOM_DIR, "stage2_tdl", "ncrt"))
sys.path.insert(0, os.path.join(VEHCOM_DIR, "stage2_tdl", "ncrn"))
sys.path.insert(0, os.path.join(VEHCOM_DIR, "stage2_tdl", "trust_net"))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from shared.channels import gen_tdl_random, gen_tdl_channel, SNAMES
from shared.baselines import ls_estimate, mmse_estimate, add_snr_channel, snr_aware_blend
from shared.matlab_channels import load_matlab_6g_channels
from train_ncrn_tdl import LastFrame
import ncrt_layers  # Registers custom layers

app = FastAPI(
    title="Massive MIMO AI Channel Studio API",
    description="Backend API serving NCRT (Transformer), NCRN (ConvLSTM), 6G Sub-THz, and MATLAB Datasets",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = {}
MATLAB_DATA_CACHE = {}

def load_models():
    """Load pretrained Keras models into memory."""
    models_dir = os.path.join(VEHCOM_DIR, "models")
    custom_objs = {"LastFrame": LastFrame}

    # 1. Base synthetic models
    ncrt_path = os.path.join(models_dir, "ncrt_tdl_64.keras")
    if os.path.exists(ncrt_path):
        MODELS["ncrt"] = tf.keras.models.load_model(ncrt_path, custom_objects=custom_objs, compile=False, safe_mode=False)
        print("✓ NCRT (Base Synthetic) loaded.")

    ncrn_path = os.path.join(models_dir, "ncrn_tdl_64.keras")
    if os.path.exists(ncrn_path):
        MODELS["ncrn"] = tf.keras.models.load_model(ncrn_path, custom_objects=custom_objs, compile=False, safe_mode=False)
        print("✓ NCRN (Base Synthetic) loaded.")

    # 2. MATLAB 5G + 6G fine-tuned models
    ncrt_matlab_path = os.path.join(models_dir, "ncrt_tdl_64_matlab_5g_6g.keras")
    if os.path.exists(ncrt_matlab_path):
        MODELS["ncrt_matlab"] = tf.keras.models.load_model(ncrt_matlab_path, custom_objects=custom_objs, compile=False, safe_mode=False)
        print("✓ NCRT (MATLAB 5G+6G Sub-THz Trained) loaded.")

    ncrn_matlab_path = os.path.join(models_dir, "ncrn_tdl_64_matlab_5g_6g.keras")
    if os.path.exists(ncrn_matlab_path):
        MODELS["ncrn_matlab"] = tf.keras.models.load_model(ncrn_matlab_path, custom_objects=custom_objs, compile=False, safe_mode=False)
        print("✓ NCRN (MATLAB 5G+6G Trained) loaded.")

def preload_matlab_datasets():
    """Cache MATLAB test datasets for fast UI interaction."""
    data_dir = os.path.join(VEHCOM_DIR, "data")
    
    # 6G sample
    sample_6g = os.path.join(data_dir, "test_6g_sample.mat")
    if os.path.exists(sample_6g):
        try:
            MATLAB_DATA_CACHE["6g_subthz"] = load_matlab_6g_channels(sample_6g)
            print(f"✓ Cached 6G Sub-THz dataset: {MATLAB_DATA_CACHE['6g_subthz'].shape}")
        except Exception as e:
            print(f"Failed to load 6G sample: {e}")

    # Full 6G if available
    full_6g = os.path.join(data_dir, "matlab_6g_channels.mat")
    if os.path.exists(full_6g):
        try:
            MATLAB_DATA_CACHE["6g_full"] = load_matlab_6g_channels(full_6g)
            print(f"✓ Cached full 6G dataset: {MATLAB_DATA_CACHE['6g_full'].shape}")
        except Exception as e:
            print(f"Failed to load full 6G: {e}")

    # 5G NR if available
    full_5g = os.path.join(data_dir, "matlab_5g_channels.mat")
    if os.path.exists(full_5g):
        try:
            MATLAB_DATA_CACHE["5g_full"] = load_matlab_6g_channels(full_5g)
            print(f"✓ Cached full 5G dataset: {MATLAB_DATA_CACHE['5g_full'].shape}")
        except Exception as e:
            print(f"Failed to load full 5G: {e}")

@app.on_event("startup")
def startup_event():
    load_models()
    preload_matlab_datasets()

# --- Request Schemas ---
class EstimationRequest(BaseModel):
    scenario: str = Field(default="TDL-A")
    snr_db: float = Field(default=10.0, ge=-25.0, le=30.0)
    speed_kmh: float = Field(default=30.0, ge=0.0, le=500.0)
    selected_model: str = Field(default="ncrt")

class SweepRequest(BaseModel):
    scenario: str = Field(default="TDL-A")
    snr_range: List[float] = Field(default=[-15, -10, -5, 0, 5, 10, 15, 20])
    num_samples: int = Field(default=8, ge=1, le=50)

class MatlabEstimateRequest(BaseModel):
    dataset_id: str = Field(default="6g_subthz", description="Dataset identifier: 6g_subthz, 6g_full, 5g_full")
    sample_index: int = Field(default=0, ge=0, le=500)
    snr_db: float = Field(default=5.0, ge=-25.0, le=30.0)

# --- Routes ---
@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "models_loaded": list(MODELS.keys()),
        "matlab_datasets_cached": list(MATLAB_DATA_CACHE.keys()),
        "tensorflow_version": tf.__version__,
    }

@app.get("/api/scenarios")
def get_scenarios():
    return [
        {"id": "TDL-A", "name": "3GPP TDL-A", "type": "NLOS", "delay_spread": "High (300 ns)", "description": "Urban Macro NLOS with severe multipath"},
        {"id": "TDL-B", "name": "3GPP TDL-B", "type": "NLOS", "delay_spread": "Medium (100 ns)", "description": "Urban Micro NLOS"},
        {"id": "TDL-C", "name": "3GPP TDL-C", "type": "NLOS", "delay_spread": "Long (1000 ns)", "description": "Rural / Wide-area NLOS"},
        {"id": "TDL-D", "name": "3GPP TDL-D", "type": "LOS", "delay_spread": "Low (30 ns)", "description": "Urban Micro LOS with strong direct path"},
        {"id": "TDL-E", "name": "3GPP TDL-E", "type": "LOS", "delay_spread": "Extreme", "description": "Open area LOS"},
        {"id": "random", "name": "Random Mix", "type": "Mixed", "delay_spread": "Variable", "description": "Uniform random across all profiles"},
    ]

@app.post("/api/estimate")
def run_estimation(req: EstimationRequest):
    Nr = 64
    Nt = 64
    snr = float(req.snr_db)

    h = gen_tdl_channel(req.scenario, Nr, Nt) if req.scenario in SNAMES else gen_tdl_random(Nr, Nt)

    sp = float(np.mean(h ** 2)) + 1e-10
    noise_sigma = np.sqrt(sp * 10 ** (-snr / 10))
    noise = np.random.normal(0, noise_sigma, h.shape).astype(np.float32)
    hn = (h + noise).astype(np.float32)

    h_true_last = h[-1]
    h_noisy_last = hn[-1]
    h_true_norm_sq = float(np.sum(h_true_last ** 2))

    X = np.expand_dims(add_snr_channel(hn, snr), axis=0)

    results = {}

    # Classical LS
    t0 = time.perf_counter()
    h_ls = ls_estimate(h_noisy_last)
    ls_time = (time.perf_counter() - t0) * 1000.0
    ls_nmse_lin = float(np.sum((h_true_last - h_ls) ** 2)) / h_true_norm_sq
    results["ls"] = {
        "name": "Least Squares (LS)",
        "nmse_db": round(10.0 * np.log10(max(ls_nmse_lin, 1e-12)), 2),
        "latency_ms": round(ls_time, 2),
    }

    # Classical MMSE
    t0 = time.perf_counter()
    h_mmse = mmse_estimate(h_noisy_last, noise_var=noise_sigma**2, Nt=Nt, sp=sp)
    mmse_time = (time.perf_counter() - t0) * 1000.0
    mmse_nmse_lin = float(np.sum((h_true_last - h_mmse) ** 2)) / h_true_norm_sq
    results["mmse"] = {
        "name": "Wiener MMSE",
        "nmse_db": round(10.0 * np.log10(max(mmse_nmse_lin, 1e-12)), 2),
        "latency_ms": round(mmse_time, 2),
    }

    # NCRT
    h_ncrt_est = None
    if "ncrt" in MODELS:
        t0 = time.perf_counter()
        raw_res = MODELS["ncrt"].predict(X, verbose=0)[0]
        ncrt_time = (time.perf_counter() - t0) * 1000.0
        h_ncrt_est = h_noisy_last + raw_res
        ncrt_nmse_lin = float(np.sum((h_true_last - h_ncrt_est) ** 2)) / h_true_norm_sq
        results["ncrt"] = {
            "name": "NCRT (Attention Residual)",
            "nmse_db": round(10.0 * np.log10(max(ncrt_nmse_lin, 1e-12)), 2),
            "latency_ms": round(ncrt_time, 2),
            "param_count": 888202,
        }

    # NCRN
    h_ncrn_est = None
    if "ncrn" in MODELS:
        t0 = time.perf_counter()
        raw_res_ncrn = MODELS["ncrn"].predict(X, verbose=0)[0]
        ncrn_time = (time.perf_counter() - t0) * 1000.0
        h_ncrn_est = h_noisy_last + raw_res_ncrn
        ncrn_nmse_lin = float(np.sum((h_true_last - h_ncrn_est) ** 2)) / h_true_norm_sq
        results["ncrn"] = {
            "name": "NCRN (ConvLSTM)",
            "nmse_db": round(10.0 * np.log10(max(ncrn_nmse_lin, 1e-12)), 2),
            "latency_ms": round(ncrn_time, 2),
            "param_count": 1204610,
        }

    active_est = h_ncrt_est if h_ncrt_est is not None else (h_ncrn_est if h_ncrn_est is not None else h_ls)

    step = 1  # Full 64x64 Massive MIMO array resolution
    true_mag = np.sqrt(h_true_last[::step, ::step, 0]**2 + h_true_last[::step, ::step, 1]**2)
    noisy_mag = np.sqrt(h_noisy_last[::step, ::step, 0]**2 + h_noisy_last[::step, ::step, 1]**2)
    est_mag = np.sqrt(active_est[::step, ::step, 0]**2 + active_est[::step, ::step, 1]**2)
    error_mag = np.abs(true_mag - est_mag)

    # 2D Beamspace Angular Domain (2D FFT of complex channel)
    H_complex = h_true_last[:, :, 0] + 1j * h_true_last[:, :, 1]
    beamspace = np.abs(np.fft.fftshift(np.fft.fft2(H_complex)))[::step, ::step]

    return {
        "status": "success",
        "scenario": req.scenario,
        "snr_db": snr,
        "results": results,
        "matrices": {
            "true_channel": true_mag.tolist(),
            "noisy_ls": noisy_mag.tolist(),
            "reconstructed": est_mag.tolist(),
            "error_residual": error_mag.tolist(),
            "beamspace_sparsity": beamspace.tolist(),
            "dim": 64,
            "full_dim": 64
        }
    }

@app.post("/api/sweep")
def run_snr_sweep(req: SweepRequest):
    Nr = 64
    Nt = 64
    snr_list = req.snr_range
    n_samples = req.num_samples

    methods = ["ls", "mmse"]
    if "ncrt" in MODELS:
        methods.append("ncrt")
    if "ncrn" in MODELS:
        methods.append("ncrn")

    curves = {m: [] for m in methods}

    for snr in snr_list:
        sq_errs = {m: 0.0 for m in methods}
        den = 0.0

        for _ in range(n_samples):
            h = gen_tdl_channel(req.scenario, Nr, Nt) if req.scenario in SNAMES else gen_tdl_random(Nr, Nt)
            sp = float(np.mean(h ** 2)) + 1e-10
            noise_sigma = np.sqrt(sp * 10 ** (-snr / 10))
            hn = (h + np.random.normal(0, noise_sigma, h.shape)).astype(np.float32)

            hl = hn[-1]
            hr = h[-1]
            den += float(np.sum(hr ** 2))

            sq_errs["ls"] += float(np.sum((hr - ls_estimate(hl)) ** 2))
            sq_errs["mmse"] += float(np.sum((hr - mmse_estimate(hl, noise_var=noise_sigma**2, Nt=Nt, sp=sp)) ** 2))

            X = np.expand_dims(add_snr_channel(hn, snr), axis=0)
            if "ncrt" in MODELS:
                r_ncrt = MODELS["ncrt"].predict(X, verbose=0)[0]
                sq_errs["ncrt"] += float(np.sum((hr - (hl + r_ncrt)) ** 2))
            if "ncrn" in MODELS:
                r_ncrn = MODELS["ncrn"].predict(X, verbose=0)[0]
                sq_errs["ncrn"] += float(np.sum((hr - (hl + r_ncrn)) ** 2))

        for m in methods:
            lin = sq_errs[m] / max(den, 1e-10)
            curves[m].append(round(10.0 * np.log10(max(lin, 1e-12)), 2))

    return {
        "snr_range": snr_list,
        "scenario": req.scenario,
        "curves": curves
    }

# ==============================================================================
# 6G SUB-THZ & MATLAB STUDIO ENDPOINTS
# ==============================================================================

@app.get("/api/matlab/datasets")
def list_matlab_datasets():
    return [
        {
            "id": "6g_subthz",
            "name": "6G Sub-THz (142 GHz)",
            "carrier_fc": "142.0 GHz",
            "velocity": "120 km/h (Doppler fd = 15.8 kHz)",
            "array": "64×64 UPA Sub-THz Array",
            "sparsity": "Extreme Angular Sparsity (Few dominant clusters)",
            "samples_available": len(MATLAB_DATA_CACHE.get("6g_subthz", [])) or 20,
            "description": "High-fidelity 6G Sub-THz ray-traced channel generated from MATLAB 5G/6G Toolbox."
        },
        {
            "id": "5g_full",
            "name": "5G NR mmWave (28 GHz)",
            "carrier_fc": "28.0 GHz",
            "velocity": "30 km/h (Doppler fd = 777 Hz)",
            "array": "64×64 Massive MIMO",
            "sparsity": "Medium Sparsity (TDL-C Delay Profile)",
            "samples_available": len(MATLAB_DATA_CACHE.get("5g_full", [])) or 500,
            "description": "3GPP TR 38.901 standard 5G NR TDL-C channel realizations exported from MATLAB."
        }
    ]

@app.post("/api/matlab/estimate")
def run_matlab_6g_estimation(req: MatlabEstimateRequest):
    """Evaluates a 6G sub-THz or 5G MATLAB channel sample and computes beamspace sparsity."""
    dataset_key = req.dataset_id if req.dataset_id in MATLAB_DATA_CACHE else "6g_subthz"
    dataset = MATLAB_DATA_CACHE.get(dataset_key)

    if dataset is None or len(dataset) == 0:
        raise HTTPException(status_code=404, detail="MATLAB dataset not loaded.")

    idx = req.sample_index % len(dataset)
    h_clean = dataset[idx] # (10, 64, 64, 2)
    snr = float(req.snr_db)

    sp = float(np.mean(h_clean ** 2)) + 1e-10
    noise_sigma = np.sqrt(sp * 10 ** (-snr / 10))
    noise = np.random.normal(0, noise_sigma, h_clean.shape).astype(np.float32)
    hn = (h_clean + noise).astype(np.float32)

    h_true_last = h_clean[-1]   # (64, 64, 2)
    h_noisy_last = hn[-1]       # (64, 64, 2)
    h_true_norm_sq = float(np.sum(h_true_last ** 2))

    X = np.expand_dims(add_snr_channel(hn, snr), axis=0)

    results = {}

    # 1. LS Baseline
    h_ls = ls_estimate(h_noisy_last)
    ls_nmse = 10.0 * np.log10(max(float(np.sum((h_true_last - h_ls) ** 2)) / h_true_norm_sq, 1e-12))
    results["ls"] = {"name": "LS Baseline", "nmse_db": round(ls_nmse, 2)}

    # 2. Wiener MMSE
    h_mmse = mmse_estimate(h_noisy_last, noise_var=noise_sigma**2, Nt=64, sp=sp)
    mmse_nmse = 10.0 * np.log10(max(float(np.sum((h_true_last - h_mmse) ** 2)) / h_true_norm_sq, 1e-12))
    results["mmse"] = {"name": "Wiener MMSE", "nmse_db": round(mmse_nmse, 2)}

    # 3. NCRT MATLAB 6G-Trained (Specialized Model)
    h_ncrt_6g_est = None
    if "ncrt_matlab" in MODELS:
        res_6g = MODELS["ncrt_matlab"].predict(X, verbose=0)[0]
        h_ncrt_6g_est = h_noisy_last + res_6g
        ncrt_6g_nmse = 10.0 * np.log10(max(float(np.sum((h_true_last - h_ncrt_6g_est) ** 2)) / h_true_norm_sq, 1e-12))
        results["ncrt_6g"] = {"name": "NCRT (6G Sub-THz Trained)", "nmse_db": round(ncrt_6g_nmse, 2)}

    # 4. NCRT Synthetic Base (Generalist Model)
    h_ncrt_base_est = None
    if "ncrt" in MODELS:
        res_base = MODELS["ncrt"].predict(X, verbose=0)[0]
        h_ncrt_base_est = h_noisy_last + res_base
        ncrt_base_nmse = 10.0 * np.log10(max(float(np.sum((h_true_last - h_ncrt_base_est) ** 2)) / h_true_norm_sq, 1e-12))
        results["ncrt_base"] = {"name": "NCRT (Synthetic Base)", "nmse_db": round(ncrt_base_nmse, 2)}

    active_est = h_ncrt_6g_est if h_ncrt_6g_est is not None else (h_ncrt_base_est if h_ncrt_base_est is not None else h_ls)

    # Compute 2D Spatial Responses (Full 64x64 Massive MIMO Array)
    step = 1
    true_mag = np.sqrt(h_true_last[::step, ::step, 0]**2 + h_true_last[::step, ::step, 1]**2)
    noisy_mag = np.sqrt(h_noisy_last[::step, ::step, 0]**2 + h_noisy_last[::step, ::step, 1]**2)
    est_mag = np.sqrt(active_est[::step, ::step, 0]**2 + active_est[::step, ::step, 1]**2)
    error_mag = np.abs(true_mag - est_mag)

    # 6G Beamspace (Virtual Angular Domain via 2D FFT)
    H_complex = h_true_last[:, :, 0] + 1j * h_true_last[:, :, 1]
    beamspace = np.abs(np.fft.fftshift(np.fft.fft2(H_complex)))[::step, ::step]

    return {
        "status": "success",
        "dataset_id": req.dataset_id,
        "sample_index": idx,
        "snr_db": snr,
        "results": results,
        "matrices": {
            "true_channel": true_mag.tolist(),
            "noisy_ls": noisy_mag.tolist(),
            "reconstructed": est_mag.tolist(),
            "error_residual": error_mag.tolist(),
            "beamspace_sparsity": beamspace.tolist(),
            "dim": 64,
        }
    }

@app.get("/api/matlab/benchmarks")
def get_matlab_benchmarks():
    """Returns the precomputed 500-sample 6G Sub-THz and 5G MATLAB curves from the paper."""
    csv_6g = os.path.join(VEHCOM_DIR, "outputs", "csv", "matlab_6g_eval.csv")
    csv_5g = os.path.join(VEHCOM_DIR, "outputs", "csv", "matlab_5g_eval.csv")

    data = {}
    if os.path.exists(csv_6g):
        df6g = pd.read_csv(csv_6g)
        data["6g_subthz"] = {
            "snr_db": df6g["snr_db"].tolist(),
            "ls": df6g["ls_nmse_db"].round(2).tolist(),
            "mmse": df6g["mmse_nmse_db"].round(2).tolist(),
            "ncrn": df6g["ncrn_nmse_db"].round(2).tolist(),
            "ncrt_base": df6g["ncrt_base_nmse_db"].round(2).tolist(),
            "ncrt_matlab": df6g["ncrt_matlab_nmse_db"].round(2).tolist(),
        }

    if os.path.exists(csv_5g):
        df5g = pd.read_csv(csv_5g)
        data["5g_nr"] = {
            "snr_db": df5g["snr_db"].tolist(),
            "ls": df5g["ls_nmse_db"].round(2).tolist(),
            "mmse": df5g["mmse_nmse_db"].round(2).tolist(),
            "ncrn": df5g["ncrn_nmse_db"].round(2).tolist(),
            "ncrt_base": df5g["ncrt_base_nmse_db"].round(2).tolist(),
            "ncrt_matlab": df5g["ncrt_matlab_nmse_db"].round(2).tolist(),
        }

    return data

@app.get("/api/matlab/script")
def get_matlab_export_script():
    """Returns the MATLAB script for exporting 6G/5G Massive MIMO channel tensors."""
    script_path = os.path.join(VEHCOM_DIR, "matlab", "export_6g_channels.m")
    if not os.path.exists(script_path):
        script_path = os.path.join(VEHCOM_DIR, "..", "matlab", "export_6g_channels.m")
    
    if os.path.exists(script_path):
        with open(script_path, "r") as f:
            code = f.read()
    else:
        code = "% MATLAB 6G Export Script\nfc = 142e9; % 142 GHz sub-THz\nNr = 64; Nt = 64;\n"

    return {"filename": "export_6g_channels.m", "code": code}

# --- File Download Endpoints ---
from fastapi.responses import FileResponse

@app.get("/api/download/script")
def download_script():
    script_path = os.path.join(VEHCOM_DIR, "matlab", "export_6g_channels.m")
    if not os.path.exists(script_path):
        script_path = os.path.join(VEHCOM_DIR, "..", "matlab", "export_6g_channels.m")
    if os.path.exists(script_path):
        return FileResponse(script_path, filename="export_6g_channels.m", media_type="text/x-matlab")
    raise HTTPException(status_code=404, detail="MATLAB script not found")

@app.get("/api/download/csv/{dataset}")
def download_csv(dataset: str):
    fname = f"matlab_{dataset}_eval.csv" if not dataset.endswith(".csv") else dataset
    csv_path = os.path.join(VEHCOM_DIR, "outputs", "csv", fname)
    if os.path.exists(csv_path):
        return FileResponse(csv_path, filename=fname, media_type="text/csv")
    raise HTTPException(status_code=404, detail=f"CSV {fname} not found")

@app.get("/api/download/model")
def download_model():
    model_path = os.path.join(VEHCOM_DIR, "models", "ncrt_tdl_64.keras")
    if os.path.exists(model_path):
        return FileResponse(model_path, filename="ncrt_tdl_64.keras", media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="Model file not found")

@app.get("/api/download/release-zip")
def download_release_zip():
    zip_path = os.path.join(VEHCOM_DIR, "..", "VehCom_Release.zip")
    if os.path.exists(zip_path):
        return FileResponse(zip_path, filename="VehCom_Release.zip", media_type="application/zip")
    raise HTTPException(status_code=404, detail="Release ZIP not found")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

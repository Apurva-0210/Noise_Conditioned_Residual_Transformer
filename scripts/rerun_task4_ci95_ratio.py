#!/usr/bin/env python3
"""
Task 4 Fix: Headline Combined Test Set Table with N=500 and 95% Confidence Intervals
Uses the mathematically exact ratio-of-sums estimator (Paper Eq. 7):
R = sum(Y_i) / sum(X_i), NMSE_dB = 10*log10(R)
with Delta-method ratio standard error:
Var(R) = (s_Y^2 - 2*R*s_XY + R^2*s_X^2) / (N * mean(X)^2)
SE_dB = (10 / ln(10)) * sqrt(Var(R)) / R
CI_95 = 1.96 * SE_dB
"""
import os, sys, time
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'stage2_tdl', 'ncrn'))
sys.path.insert(0, os.path.join(ROOT, 'stage2_tdl', 'ncrt'))

from shared.channels import gen_tdl_random
from shared.baselines import (mmse_estimate, cnnsparse_mask, sadlcs_postprocess,
                              add_snr_channel, snr_aware_blend)
from train_ncrn_tdl import LastFrame
import ncrt_layers

SNR_RANGE = [-15, -10, -5, 0, 5, 10, 15, 20]
DATA_DIR = os.path.join(ROOT, 'paper', 'data')
OUT_CSV_DIR = os.path.join(ROOT, 'outputs', 'csv')

def main():
    print("Loading models for Task 4...")
    customs = {'LastFrame': LastFrame}
    m_sparse = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'cnn_sparse_tdl_64.keras'), compile=False)
    m_lstm   = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'cnn_lstm_tdl_64.keras'), compile=False)
    m_sadlcs = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'sa_dlcs_tdl_64.keras'), compile=False)
    m_ncrn   = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'ncrn_tdl_64.keras'),
                                          custom_objects=customs, compile=False)
    m_ncrt   = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'ncrt_tdl_64.keras'), compile=False)

    N = 500
    batch_size = 50
    np.random.seed(42)
    rows = []

    print("\nRunning N=500 ratio-of-sums evaluation with 95% CIs across SNRs...")
    for snr in SNR_RANGE:
        t0 = time.time()
        ns = np.sqrt(10 ** (-snr / 10.0))
        H, HN, HL, HR, SP = [], [], [], [], []
        for _ in range(N):
            h = gen_tdl_random(64, 64, dt=1e-6)
            noise = np.random.normal(0, ns, h.shape).astype(np.float32)
            hn = (h + noise).astype(np.float32)
            H.append(h); HN.append(hn); HL.append(hn[-1]); HR.append(h[-1])
            SP.append(float(np.mean(h**2)))
            
        HL_arr = np.array(HL, dtype=np.float32)
        HN_arr = np.array(HN, dtype=np.float32)
        HR_arr = np.array(HR, dtype=np.float32)
        
        SPARSE_IN = np.array([HL_arr[i] * cnnsparse_mask(HL_arr[i], k=500) for i in range(N)], dtype=np.float32)
        X_SNR = np.array([add_snr_channel(hn, snr) for hn in HN], dtype=np.float32)
        
        p_sparse = m_sparse.predict(SPARSE_IN, batch_size=batch_size, verbose=0)
        p_lstm   = m_lstm.predict(HN_arr, batch_size=batch_size, verbose=0)
        p_sadlcs_raw = m_sadlcs.predict(HN_arr, batch_size=batch_size, verbose=0)
        p_ncrn_raw   = m_ncrn.predict(X_SNR, batch_size=batch_size, verbose=0)
        p_ncrt_raw   = m_ncrt.predict(X_SNR, batch_size=batch_size, verbose=0)
        
        # Collect per-sample errors (Y) and channel powers (X)
        X_pwr = np.array([np.sum(HR_arr[i] ** 2) for i in range(N)])
        
        Y_errs = {
            'ls': np.array([np.sum((HR_arr[i] - HL_arr[i]) ** 2) for i in range(N)]),
            'mmse': np.array([np.sum((HR_arr[i] - mmse_estimate(HL_arr[i], noise_var=ns**2, Nt=64, sp=SP[i])) ** 2) for i in range(N)]),
            'cnn_sparse': np.array([np.sum((HR_arr[i] - p_sparse[i]) ** 2) for i in range(N)]),
            'cnn_lstm': np.array([np.sum((HR_arr[i] - p_lstm[i]) ** 2) for i in range(N)]),
            'sa_dlcs': np.array([np.sum((HR_arr[i] - sadlcs_postprocess(p_sadlcs_raw[i], HL_arr[i])) ** 2) for i in range(N)]),
            'ncrn_b': np.array([np.sum((HR_arr[i] - snr_aware_blend(p_ncrn_raw[i], HL_arr[i], snr)) ** 2) for i in range(N)]),
            'ncrt_b': np.array([np.sum((HR_arr[i] - snr_aware_blend(p_ncrt_raw[i], HL_arr[i], snr)) ** 2) for i in range(N)]),
            'ncrn_raw': np.array([np.sum((HR_arr[i] - (HL_arr[i] + p_ncrn_raw[i])) ** 2) for i in range(N)]),
            'ncrt_raw': np.array([np.sum((HR_arr[i] - (HL_arr[i] + p_ncrt_raw[i])) ** 2) for i in range(N)]),
        }
        
        mean_X = np.mean(X_pwr)
        var_X = np.var(X_pwr, ddof=1)
        
        res = {}
        for k, Y in Y_errs.items():
            mean_Y = np.mean(Y)
            var_Y = np.var(Y, ddof=1)
            cov_XY = np.cov(Y, X_pwr, ddof=1)[0, 1]
            
            R = mean_Y / mean_X
            nmse_db = 10 * np.log10(R)
            
            # Ratio estimator asymptotic variance
            var_R = (var_Y - 2 * R * cov_XY + (R ** 2) * var_X) / (N * (mean_X ** 2))
            se_db = (10.0 / np.log(10.0)) * (np.sqrt(max(var_R, 1e-15)) / R)
            ci95 = 1.96 * se_db
            res[k] = f"{nmse_db:+.2f} +/- {ci95:.2f}"
            
        row = {
            'SNR (dB)': snr,
            'LS': res['ls'],
            'MMSE': res['mmse'],
            'CNN-Sparse': res['cnn_sparse'],
            'CNN-LSTM': res['cnn_lstm'],
            'Sa-DLCS': res['sa_dlcs'],
            'NCRN (blended)': res['ncrn_b'],
            'NCRT (blended)': res['ncrt_b'],
            'NCRN (raw)': res['ncrn_raw'],
            'NCRT (raw)': res['ncrt_raw'],
        }
        rows.append(row)
        print(f"SNR {snr:>3} dB | LS: {res['ls']} | MMSE: {res['mmse']} | NCRN-b: {res['ncrn_b']} | NCRT-b: {res['ncrt_b']} | NCRT-raw: {res['ncrt_raw']}")
        
    df_ci = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_CSV_DIR, 'headline_combined_tdl_64_ci95.csv')
    paper_csv = os.path.join(DATA_DIR, 'headline_combined_tdl_64_ci95.csv')
    df_ci.to_csv(out_csv, index=False)
    df_ci.to_csv(paper_csv, index=False)
    print(f"\nSaved updated Headline N=500 CI95 Table to {out_csv} and {paper_csv}")
    print(df_ci.to_string(index=False))

if __name__ == '__main__':
    main()

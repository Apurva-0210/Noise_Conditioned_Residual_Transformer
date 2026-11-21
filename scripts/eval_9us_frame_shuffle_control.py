#!/usr/bin/env python3
"""
Reviewer Item 5: Frame-Shuffling Control at 9 µs Spacing
Tests whether NCRT actively tracks the continuous Doppler phase rotation trajectory,
or merely performs permutation-invariant temporal denoising/smoothing.

Conditions evaluated at inference on lockstep test set (N=100, seed 42):
1. Ordered frames: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (Natural Doppler trajectory)
2. Permuted past frames: e.g. [3, 0, 8, 6, 4, 1, 7, 5, 2, 9] (Scrambled phase trajectory)
3. Reversed past frames: [8, 7, 6, 5, 4, 3, 2, 1, 0, 9] (Inverted Doppler trajectory)
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
from shared.baselines import add_snr_channel
from train_ncrn_tdl import LastFrame
import ncrt_layers

SNR_RANGE = [-15, -10, -5, 0, 5, 10, 15, 20]
DATA_DIR = os.path.join(ROOT, 'paper', 'data')
OUT_CSV_DIR = os.path.join(ROOT, 'outputs', 'csv')

def main():
    print("=" * 80, flush=True)
    print("Item 5 Control: Temporal Frame-Shuffling Test at 9 µs Spacing", flush=True)
    print("=" * 80, flush=True)

    p_ncrt = os.path.join(ROOT, 'models', 'ablations', 'ncrt_9us.keras')
    p_ncrn = os.path.join(ROOT, 'models', 'ablations', 'ncrn_9us.keras')
    
    customs = {'LastFrame': LastFrame}
    m_ncrt = tf.keras.models.load_model(p_ncrt, compile=False)
    m_ncrn = tf.keras.models.load_model(p_ncrn, custom_objects=customs, compile=False)
    
    N = 100
    np.random.seed(42)
    dt = 9e-6
    
    # Define permutation of past frames 0..8
    rng = np.random.RandomState(12345)
    perm_past = rng.permutation(9) # e.g. [3, 0, 8, 6, 4, 1, 7, 5, 2]
    perm_order = list(perm_past) + [9]
    rev_order = list(range(8, -1, -1)) + [9]
    
    print(f"Permuted frame order: {perm_order}", flush=True)
    print(f"Reversed frame order: {rev_order}", flush=True)
    
    rows = []
    
    for snr in SNR_RANGE:
        t0 = time.time()
        ns = np.sqrt(10 ** (-snr / 10.0))
        H, HN, HL, HR = [], [], [], []
        for _ in range(N):
            h = gen_tdl_random(64, 64, dt=dt)
            noise = np.random.normal(0, ns, h.shape).astype(np.float32)
            hn = (h + noise).astype(np.float32)
            H.append(h); HN.append(hn); HL.append(hn[-1]); HR.append(h[-1])
            
        HN_arr = np.array(HN, dtype=np.float32)
        HL_arr = np.array(HL, dtype=np.float32)
        HR_arr = np.array(HR, dtype=np.float32)
        
        # 1. Natural Ordered sequence
        X_ord = np.array([add_snr_channel(hn, snr) for hn in HN_arr], dtype=np.float32)
        
        # 2. Permuted sequence
        HN_perm = HN_arr[:, perm_order, :, :, :]
        X_perm = np.array([add_snr_channel(hn, snr) for hn in HN_perm], dtype=np.float32)
        
        # 3. Reversed sequence
        HN_rev = HN_arr[:, rev_order, :, :, :]
        X_rev = np.array([add_snr_channel(hn, snr) for hn in HN_rev], dtype=np.float32)
        
        # Inferences
        p_ncrt_ord = m_ncrt.predict(X_ord, batch_size=25, verbose=0)
        p_ncrt_perm = m_ncrt.predict(X_perm, batch_size=25, verbose=0)
        p_ncrt_rev = m_ncrt.predict(X_rev, batch_size=25, verbose=0)
        
        p_ncrn_ord = m_ncrn.predict(X_ord, batch_size=25, verbose=0)
        p_ncrn_perm = m_ncrn.predict(X_perm, batch_size=25, verbose=0)
        p_ncrn_rev = m_ncrn.predict(X_rev, batch_size=25, verbose=0)
        
        # Reconstruct raw channel estimates (HL_arr + r_hat)
        h_ls = HL_arr
        h_ncrt_ord  = HL_arr + p_ncrt_ord
        h_ncrt_perm = HL_arr + p_ncrt_perm
        h_ncrt_rev  = HL_arr + p_ncrt_rev
        
        h_ncrn_ord  = HL_arr + p_ncrn_ord
        h_ncrn_perm = HL_arr + p_ncrn_perm
        h_ncrn_rev  = HL_arr + p_ncrn_rev
        
        # Ratio of sums NMSE
        denom = np.sum(HR_arr ** 2)
        nmse_ls        = 10 * np.log10(np.sum((HR_arr - h_ls) ** 2) / denom)
        nmse_ncrt_ord  = 10 * np.log10(np.sum((HR_arr - h_ncrt_ord) ** 2) / denom)
        nmse_ncrt_perm = 10 * np.log10(np.sum((HR_arr - h_ncrt_perm) ** 2) / denom)
        nmse_ncrt_rev  = 10 * np.log10(np.sum((HR_arr - h_ncrt_rev) ** 2) / denom)
        
        nmse_ncrn_ord  = 10 * np.log10(np.sum((HR_arr - h_ncrn_ord) ** 2) / denom)
        nmse_ncrn_perm = 10 * np.log10(np.sum((HR_arr - h_ncrn_perm) ** 2) / denom)
        nmse_ncrn_rev  = 10 * np.log10(np.sum((HR_arr - h_ncrn_rev) ** 2) / denom)
        
        ncrt_collapse_perm = nmse_ncrt_perm - nmse_ncrt_ord
        ncrt_collapse_rev  = nmse_ncrt_rev - nmse_ncrt_ord
        
        row = {
            'SNR (dB)': snr,
            'LS (dB)': nmse_ls,
            'NCRT Ordered (dB)': nmse_ncrt_ord,
            'NCRT Permuted (dB)': nmse_ncrt_perm,
            'NCRT Reversed (dB)': nmse_ncrt_rev,
            'NCRT Permutation Penalty (dB)': ncrt_collapse_perm,
            'NCRT Reversal Penalty (dB)': ncrt_collapse_rev,
            'NCRN Ordered (dB)': nmse_ncrn_ord,
            'NCRN Permuted (dB)': nmse_ncrn_perm,
            'NCRN Reversed (dB)': nmse_ncrn_rev,
        }
        rows.append(row)
        print(f"SNR {snr:>3} dB [{time.time()-t0:.1f}s] | LS: {nmse_ls:+.2f} | "
              f"NCRT Ord: {nmse_ncrt_ord:.2f} -> Perm: {nmse_ncrt_perm:.2f} (+{ncrt_collapse_perm:+.2f} dB) | "
              f"NCRN Ord: {nmse_ncrn_ord:.2f} -> Perm: {nmse_ncrn_perm:.2f}", flush=True)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_CSV_DIR, 'mobility_9us_frame_shuffle_control.csv')
    paper_csv = os.path.join(DATA_DIR, 'mobility_9us_frame_shuffle_control.csv')
    df.to_csv(out_csv, index=False)
    df.to_csv(paper_csv, index=False)
    print(f"\nSaved frame-shuffling control results to {out_csv} and {paper_csv}", flush=True)
    print("\n" + df.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()

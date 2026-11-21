#!/usr/bin/env python3
"""
Evaluate Phase-Aligned-Average Baseline against NCRN and NCRT at 9 us pilot spacing:
- Lockstep test channels: N=100, seed 42, dt = 9e-6 s across SNR = [-15, 20] dB
- Baselines:
  1. LS (Last frame noisy observation)
  2. Naive Temporal Average (Unaligned)
  3. Phase-Aligned Average (Correlation-based Doppler estimation)
  4. Phase-Aligned Average (Oracle dominant phase alignment)
  5. NCRN zero-shot (1 us model)
  6. NCRT zero-shot (1 us model)
  7. NCRN-9us (retrained at 9 us)
  8. NCRT-9us (retrained at 9 us)
- Outputs saved to outputs/csv/mobility_9us_phase_aligned_average.csv and paper/data/
"""
import os, sys, time
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'stage2_tdl', 'ncrn'))
sys.path.insert(0, os.path.join(ROOT, 'stage2_tdl', 'ncrt'))

from shared.channels import gen_tdl_random, TIME_STEPS
from shared.baselines import add_snr_channel, snr_aware_blend
from train_ncrn_tdl import LastFrame
import ncrt_layers

SNR_RANGE = [-15, -10, -5, 0, 5, 10, 15, 20]
DT_9US = 9e-6
N_SAMPLES = 100
MODELS_DIR = os.path.join(ROOT, 'models', 'ablations')

def main():
    print("=" * 80)
    print("Evaluating Phase-Aligned-Average Baseline at 9 us Symbol Spacing")
    print("=" * 80)
    
    # 1. Load models
    customs = {'LastFrame': LastFrame}
    base_ncrn = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'ncrn_tdl_64.keras'),
                                           custom_objects=customs, compile=False)
    base_ncrt = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'ncrt_tdl_64.keras'),
                                           compile=False)
    
    p_ncrt_9us = os.path.join(MODELS_DIR, 'ncrt_9us.keras')
    p_ncrn_9us = os.path.join(MODELS_DIR, 'ncrn_9us.keras')
    m_ncrt_9us = tf.keras.models.load_model(p_ncrt_9us, compile=False)
    m_ncrn_9us = tf.keras.models.load_model(p_ncrn_9us, custom_objects=customs, compile=False)
    
    # 2. Generate lockstep evaluation channels
    np.random.seed(42)
    test_data = {}
    for snr in SNR_RANGE:
        ns = np.sqrt(10 ** (-snr / 10.0))
        H, HN, HL, HR = [], [], [], []
        for _ in range(N_SAMPLES):
            h = gen_tdl_random(64, 64, dt=DT_9US)
            noise = np.random.normal(0, ns, h.shape).astype(np.float32)
            hn = (h + noise).astype(np.float32)
            H.append(h); HN.append(hn); HL.append(hn[-1]); HR.append(h[-1])
        test_data[snr] = (H, HN, HL, HR)
        
    rows = []
    
    for snr in SNR_RANGE:
        t0 = time.time()
        H_list, HN_list, HL_list, HR_list = test_data[snr]
        den = sum(np.sum(hr**2) for hr in HR_list)
        
        # Ground truth complex target at last frame
        H_target = [hr[:,:,0] + 1j*hr[:,:,1] for hr in HR_list]
        
        # 1. LS (Last frame)
        H_ls = [hl[:,:,0] + 1j*hl[:,:,1] for hl in HL_list]
        nmse_ls = 10 * np.log10(sum(np.sum(np.abs(ht - hl)**2) for ht, hl in zip(H_target, H_ls)) / den)
        
        # 2. Naive Temporal Average (Unaligned)
        H_unaligned = []
        for hn in HN_list:
            # hn shape (10, 64, 64, 2)
            hn_c = hn[:,:,:,0] + 1j*hn[:,:,:,1]
            H_unaligned.append(np.mean(hn_c, axis=0))
        nmse_unaligned = 10 * np.log10(sum(np.sum(np.abs(ht - hu)**2) for ht, hu in zip(H_target, H_unaligned)) / den)
        
        # 3. Phase-Aligned Average (Correlation-Based Doppler Estimation)
        H_paa_corr = []
        for hn in HN_list:
            hn_c = hn[:,:,:,0] + 1j*hn[:,:,:,1] # (10, 64, 64)
            # Estimate dominant inter-frame phase shift
            R = 0.0 + 0.0j
            for t in range(TIME_STEPS - 1):
                R += np.sum(np.conj(hn_c[t]) * hn_c[t+1])
            delta_theta = np.angle(R)
            
            # De-rotate frames to align with last frame t=9
            hn_aligned = np.zeros_like(hn_c)
            for t in range(TIME_STEPS):
                rot = np.exp(1j * delta_theta * (TIME_STEPS - 1 - t))
                hn_aligned[t] = hn_c[t] * rot
            H_paa_corr.append(np.mean(hn_aligned, axis=0))
        nmse_paa_corr = 10 * np.log10(sum(np.sum(np.abs(ht - hp)**2) for ht, hp in zip(H_target, H_paa_corr)) / den)
        
        # 4. Phase-Aligned Average (Oracle Frame-by-Frame Dominant Phase Alignment)
        H_paa_oracle = []
        for h, hn in zip(H_list, HN_list):
            h_c = h[:,:,:,0] + 1j*h[:,:,:,1]
            hn_c = hn[:,:,:,0] + 1j*hn[:,:,:,1]
            ht_target = h_c[-1]
            
            hn_aligned_oracle = np.zeros_like(hn_c)
            for t in range(TIME_STEPS):
                # Oracle phase difference between frame t and target frame
                R_opt = np.sum(np.conj(h_c[t]) * ht_target)
                theta_t = np.angle(R_opt)
                hn_aligned_oracle[t] = hn_c[t] * np.exp(1j * theta_t)
            H_paa_oracle.append(np.mean(hn_aligned_oracle, axis=0))
        nmse_paa_oracle = 10 * np.log10(sum(np.sum(np.abs(ht - hpo)**2) for ht, hpo in zip(H_target, H_paa_oracle)) / den)
        
        # 5 & 6. Learned Models (NCRN & NCRT)
        X_SNR = np.array([add_snr_channel(hn, snr) for hn in HN_list], dtype=np.float32)
        p_base_ncrn = base_ncrn.predict(X_SNR, batch_size=25, verbose=0)
        p_base_ncrt = base_ncrt.predict(X_SNR, batch_size=25, verbose=0)
        p_9us_ncrn  = m_ncrn_9us.predict(X_SNR, batch_size=25, verbose=0)
        p_9us_ncrt  = m_ncrt_9us.predict(X_SNR, batch_size=25, verbose=0)
        
        def calc_nmse(pred_res, is_blend=False):
            if is_blend:
                ests = [snr_aware_blend(pr, hl, snr) for pr, hl in zip(pred_res, HL_list)]
            else:
                ests = [hl + pr for pr, hl in zip(pred_res, HL_list)]
            sq = sum(np.sum((hr - e)**2) for hr, e in zip(HR_list, ests))
            return 10 * np.log10(sq / den)
            
        nmse_base_ncrn_raw = calc_nmse(p_base_ncrn, False)
        nmse_base_ncrt_raw = calc_nmse(p_base_ncrt, False)
        nmse_9us_ncrn_raw  = calc_nmse(p_9us_ncrn, False)
        nmse_9us_ncrt_raw  = calc_nmse(p_9us_ncrt, False)
        
        row = {
            'SNR (dB)': snr,
            'LS (dB)': nmse_ls,
            'Unaligned Avg (dB)': nmse_unaligned,
            'Phase-Aligned Avg Corr (dB)': nmse_paa_corr,
            'Phase-Aligned Avg Oracle (dB)': nmse_paa_oracle,
            'NCRN_1us_raw (zero-shot)': nmse_base_ncrn_raw,
            'NCRT_1us_raw (zero-shot)': nmse_base_ncrt_raw,
            'NCRN_9us_raw (retrained)': nmse_9us_ncrn_raw,
            'NCRT_9us_raw (retrained)': nmse_9us_ncrt_raw,
            'NCRT Advantage vs PAA Oracle (dB)': nmse_paa_oracle - nmse_9us_ncrt_raw,
            'NCRT Advantage vs NCRN-9us (dB)': nmse_9us_ncrn_raw - nmse_9us_ncrt_raw,
        }
        rows.append(row)
        print(f"SNR {snr:3d} dB [{time.time()-t0:.1f}s]: "
              f"LS={nmse_ls:+.2f} | Unaligned={nmse_unaligned:+.2f} | PAA_Corr={nmse_paa_corr:+.2f} | "
              f"PAA_Oracle={nmse_paa_oracle:+.2f} | NCRN-9us={nmse_9us_ncrn_raw:+.2f} | NCRT-9us={nmse_9us_ncrt_raw:+.2f}")
              
    df = pd.DataFrame(rows)
    out_csv = os.path.join(ROOT, 'outputs', 'csv', 'mobility_9us_phase_aligned_average.csv')
    paper_csv = os.path.join(ROOT, 'paper', 'data', 'mobility_9us_phase_aligned_average.csv')
    df.to_csv(out_csv, index=False)
    df.to_csv(paper_csv, index=False)
    print(f"\nSaved results to:\n  {out_csv}\n  {paper_csv}")
    return df

if __name__ == '__main__':
    main()

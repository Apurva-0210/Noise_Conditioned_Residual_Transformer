#!/usr/bin/env python3
"""
Evaluate SNR-Error Robustness Sweep, Pilot-Window Noise Estimator,
and Multi-Seed Spread on the Lockstep Combined 64x64 TDL Test Set (N=100, seed 42).
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
from shared.baselines import add_snr_channel, snr_aware_blend, cnnsparse_mask, sadlcs_postprocess
from train_ncrn_tdl import LastFrame
import ncrt_layers

SNR_RANGE = [-15, -10, -5, 0, 5, 10, 15, 20]
DELTAS = [-6, -3, 0, 3, 6]
N_SAMPLES = 100
BATCH_SIZE = 25
MODELS_DIR = os.path.join(ROOT, 'models', 'ablations')

def generate_combined_test_channels():
    np.random.seed(42)
    test_data = {}
    for snr in SNR_RANGE:
        ns = np.sqrt(10 ** (-snr / 10.0))
        H, HN, HL, HR = [], [], [], []
        for _ in range(N_SAMPLES):
            h = gen_tdl_random(64, 64)
            noise = np.random.normal(0, ns, h.shape).astype(np.float32)
            hn = (h + noise).astype(np.float32)
            H.append(h)
            HN.append(hn)
            HL.append(hn[-1])
            HR.append(h[-1])
        test_data[snr] = {
            'H': np.array(H, dtype=np.float32),
            'HN': np.array(HN, dtype=np.float32),
            'HL': np.array(HL, dtype=np.float32),
            'HR': np.array(HR, dtype=np.float32),
            'sigma2_n': 10 ** (-snr / 10.0)
        }
    return test_data

def run_offset_sweep(test_data, m_ncrn, m_ncrt):
    print("\n" + "="*75)
    print("Running SNR Offset Sweep on Lockstep Combined Test Set (N=100, seed 42)")
    print("="*75)
    
    rows = []
    for snr in SNR_RANGE:
        data = test_data[snr]
        HN, HL, HR = data['HN'], data['HL'], data['HR']
        den = np.sum(HR ** 2)
        sq_ls = np.sum((HR - HL) ** 2)
        ls_nmse = 10 * np.log10(sq_ls / den)
        
        row = {'SNR (dB)': snr, 'LS': f"{ls_nmse:+.2f}"}
        
        for d in DELTAS:
            s_hat = snr + d
            X_hat = np.array([add_snr_channel(hn, s_hat) for hn in HN], dtype=np.float32)
            
            # Predict NCRT
            p_ncrt = m_ncrt.predict(X_hat, batch_size=BATCH_SIZE, verbose=0)
            p_ncrt_raw = HL + p_ncrt
            p_ncrt_b = np.array([snr_aware_blend(pr, hl, s_hat) for pr, hl in zip(p_ncrt, HL)])
            
            sq_ncrt_raw = np.sum((HR - p_ncrt_raw) ** 2)
            sq_ncrt_b = np.sum((HR - p_ncrt_b) ** 2)
            
            # Predict NCRN
            p_ncrn = m_ncrn.predict(X_hat, batch_size=BATCH_SIZE, verbose=0)
            p_ncrn_raw = HL + p_ncrn
            p_ncrn_b = np.array([snr_aware_blend(pr, hl, s_hat) for pr, hl in zip(p_ncrn, HL)])
            
            sq_ncrn_raw = np.sum((HR - p_ncrn_raw) ** 2)
            sq_ncrn_b = np.sum((HR - p_ncrn_b) ** 2)
            
            d_str = f"{d:+d}"
            row[f"NCRT_blend (d={d_str})"] = f"{10 * np.log10(sq_ncrt_b / den):+.2f}"
            row[f"NCRT_raw (d={d_str})"] = f"{10 * np.log10(sq_ncrt_raw / den):+.2f}"
            row[f"NCRN_blend (d={d_str})"] = f"{10 * np.log10(sq_ncrn_b / den):+.2f}"
            row[f"NCRN_raw (d={d_str})"] = f"{10 * np.log10(sq_ncrn_raw / den):+.2f}"
            
        rows.append(row)
        print(f"  SNR {snr:>3} dB | LS {ls_nmse:+6.2f} dB | "
              f"NCRT b(d=0) {row['NCRT_blend (d=+0)']} | NCRT b(d=-6) {row['NCRT_blend (d=-6)']} | NCRT b(d=+6) {row['NCRT_blend (d=+6)']}")
        
    df = pd.DataFrame(rows)
    out_csv = os.path.join(ROOT, 'outputs', 'csv', 'snr_offset_sweep_combined_tdl_64.csv')
    df.to_csv(out_csv, index=False)
    paper_csv = os.path.join(ROOT, 'paper', 'data', 'snr_offset_sweep_combined_tdl_64.csv')
    df.to_csv(paper_csv, index=False)
    print(f"Saved offset sweep to {out_csv} and {paper_csv}")
    return df

def run_noise_estimator(test_data, m_ncrn, m_ncrt):
    print("\n" + "="*75)
    print("Running Pilot-Window Noise Estimator Evaluation on Combined Set (N=100, seed 42)")
    print("="*75)
    
    rows = []
    for snr in SNR_RANGE:
        data = test_data[snr]
        HN, HL, HR = data['HN'], data['HL'], data['HR']
        den = np.sum(HR ** 2)
        sq_ls = np.sum((HR - HL) ** 2)
        ls_nmse = 10 * np.log10(sq_ls / den)
        
        # 1. Pilot-window 2nd-order difference noise estimation for each sample
        # HN shape: (N, T=10, 64, 64, 2)
        d2 = HN[:, 2:] - 2 * HN[:, 1:-1] + HN[:, :-2] # shape (N, 8, 64, 64, 2)
        sigma2_hat = np.mean(d2 ** 2, axis=(1, 2, 3, 4)) / 6.0 # variance per component
        snr_hat_arr = -10.0 * np.log10(np.maximum(1e-12, sigma2_hat))
        
        snr_mean = float(np.mean(snr_hat_arr))
        snr_std = float(np.std(snr_hat_arr))
        snr_bias = float(np.mean(snr_hat_arr - snr))
        snr_rmse = float(np.sqrt(np.mean((snr_hat_arr - snr) ** 2)))
        
        # 2. Model predictions with Oracle SNR
        X_oracle = np.array([add_snr_channel(hn, snr) for hn in HN], dtype=np.float32)
        p_ncrt_oracle = m_ncrt.predict(X_oracle, batch_size=BATCH_SIZE, verbose=0)
        p_ncrn_oracle = m_ncrn.predict(X_oracle, batch_size=BATCH_SIZE, verbose=0)
        
        # 3. Model predictions with Estimated SNR
        X_est = np.array([add_snr_channel(hn, float(s_hat)) for hn, s_hat in zip(HN, snr_hat_arr)], dtype=np.float32)
        p_ncrt_est = m_ncrt.predict(X_est, batch_size=BATCH_SIZE, verbose=0)
        p_ncrn_est = m_ncrn.predict(X_est, batch_size=BATCH_SIZE, verbose=0)
        
        # Blended estimates
        ncrt_b_oracle = np.array([snr_aware_blend(pr, hl, snr) for pr, hl in zip(p_ncrt_oracle, HL)])
        ncrt_b_est = np.array([snr_aware_blend(pr, hl, float(s_hat)) for pr, hl, s_hat in zip(p_ncrt_est, HL, snr_hat_arr)])
        
        ncrn_b_oracle = np.array([snr_aware_blend(pr, hl, snr) for pr, hl in zip(p_ncrn_oracle, HL)])
        ncrn_b_est = np.array([snr_aware_blend(pr, hl, float(s_hat)) for pr, hl, s_hat in zip(p_ncrn_est, HL, snr_hat_arr)])
        
        # Raw estimates
        ncrt_raw_oracle = HL + p_ncrt_oracle
        ncrt_raw_est = HL + p_ncrt_est
        ncrn_raw_oracle = HL + p_ncrn_oracle
        ncrn_raw_est = HL + p_ncrn_est
        
        nmse_ncrt_b_oracle = 10 * np.log10(np.sum((HR - ncrt_b_oracle) ** 2) / den)
        nmse_ncrt_b_est = 10 * np.log10(np.sum((HR - ncrt_b_est) ** 2) / den)
        nmse_ncrn_b_oracle = 10 * np.log10(np.sum((HR - ncrn_b_oracle) ** 2) / den)
        nmse_ncrn_b_est = 10 * np.log10(np.sum((HR - ncrn_b_est) ** 2) / den)
        
        nmse_ncrt_raw_oracle = 10 * np.log10(np.sum((HR - ncrt_raw_oracle) ** 2) / den)
        nmse_ncrt_raw_est = 10 * np.log10(np.sum((HR - ncrt_raw_est) ** 2) / den)
        nmse_ncrn_raw_oracle = 10 * np.log10(np.sum((HR - ncrn_raw_oracle) ** 2) / den)
        nmse_ncrn_raw_est = 10 * np.log10(np.sum((HR - ncrn_raw_est) ** 2) / den)
        
        row = {
            'SNR (dB)': snr,
            'LS NMSE (dB)': f"{ls_nmse:+.2f}",
            'SNR_est_mean': f"{snr_mean:+.2f}",
            'SNR_est_bias': f"{snr_bias:+.3f}",
            'SNR_est_std': f"{snr_std:.3f}",
            'SNR_est_rmse': f"{snr_rmse:.3f}",
            'NCRT_blend_oracle': f"{nmse_ncrt_b_oracle:+.2f}",
            'NCRT_blend_est': f"{nmse_ncrt_b_est:+.2f}",
            'Delta_NCRT_blend': f"{(nmse_ncrt_b_est - nmse_ncrt_b_oracle):+.3f}",
            'NCRT_raw_oracle': f"{nmse_ncrt_raw_oracle:+.2f}",
            'NCRT_raw_est': f"{nmse_ncrt_raw_est:+.2f}",
            'NCRN_blend_oracle': f"{nmse_ncrn_b_oracle:+.2f}",
            'NCRN_blend_est': f"{nmse_ncrn_b_est:+.2f}",
            'NCRN_raw_oracle': f"{nmse_ncrn_raw_oracle:+.2f}",
            'NCRN_raw_est': f"{nmse_ncrn_raw_est:+.2f}",
        }
        rows.append(row)
        print(f"  SNR {snr:>3} dB | Est SNR {snr_mean:+6.2f} dB (bias {snr_bias:+.3f}, std {snr_std:.3f}) | "
              f"NCRT b: oracle {nmse_ncrt_b_oracle:+6.2f} vs est {nmse_ncrt_b_est:+6.2f} (diff {nmse_ncrt_b_est - nmse_ncrt_b_oracle:+.3f} dB)")
        
    df = pd.DataFrame(rows)
    out_csv = os.path.join(ROOT, 'outputs', 'csv', 'noise_estimator_combined_tdl_64.csv')
    df.to_csv(out_csv, index=False)
    paper_csv = os.path.join(ROOT, 'paper', 'data', 'noise_estimator_combined_tdl_64.csv')
    df.to_csv(paper_csv, index=False)
    print(f"Saved noise estimator evaluation to {out_csv} and {paper_csv}")
    return df

def run_ablation_spread(test_data):
    print("\n" + "="*75)
    print("Running Multi-Seed Spread Evaluation on Lockstep Combined Test Set")
    print("="*75)
    
    seeds = [42, 100, 2024]
    customs = {'LastFrame': LastFrame}
    all_models = [
        'cnn_sparse_matched', 'cnn_lstm_matched', 'sa_dlcs_matched',
        'cnn_lstm_resid_small', 'cnn_lstm_resid_large',
        'ncrt_no_rms', 'ncrt_no_adaln', 'ncrt_no_temp_attn',
        'ncrt_no_axial_attn', 'ncrt_no_toeplitz'
    ]
    
    # Load all trained models per seed
    loaded_models = {}
    for mid in all_models:
        loaded_models[mid] = {}
        for s in seeds:
            p = os.path.join(MODELS_DIR, f"{mid}_seed{s}.keras")
            if os.path.exists(p):
                loaded_models[mid][s] = tf.keras.models.load_model(p, custom_objects=customs, compile=False)
                
    # Model evaluation categories
    rows = []
    for snr in SNR_RANGE:
        data = test_data[snr]
        HN, HL, HR = data['HN'], data['HL'], data['HR']
        den = np.sum(HR ** 2)
        
        row = {'SNR (dB)': snr}
        
        for mid in all_models:
            seed_nmse = []
            for s in seeds:
                m = loaded_models[mid][s]
                if mid == 'cnn_sparse_matched':
                    X_sp = np.array([np.concatenate([cnnsparse_mask(hl, 500),
                                                    np.full((64, 64, 1), snr/30.0, np.float32)], axis=-1)
                                     for hl in HL], dtype=np.float32)
                    p = m.predict(X_sp, batch_size=BATCH_SIZE, verbose=0)
                    sq = np.sum((HR - p) ** 2)
                elif mid == 'cnn_lstm_matched':
                    X_10 = np.array([add_snr_channel(hn, snr) for hn in HN], dtype=np.float32)
                    p = m.predict(X_10, batch_size=BATCH_SIZE, verbose=0)
                    sq = np.sum((HR - p) ** 2)
                elif mid == 'sa_dlcs_matched':
                    X_10 = np.array([add_snr_channel(hn, snr) for hn in HN], dtype=np.float32)
                    p_raw = m.predict(X_10, batch_size=BATCH_SIZE, verbose=0)
                    p_post = np.array([sadlcs_postprocess(pr, hl) for pr, hl in zip(p_raw, HL)])
                    sq = np.sum((HR - p_post) ** 2)
                else: # residual raw
                    X_10 = np.array([add_snr_channel(hn, snr) for hn in HN], dtype=np.float32)
                    p_res = m.predict(X_10, batch_size=BATCH_SIZE, verbose=0)
                    sq = np.sum((HR - (HL + p_res)) ** 2)
                    
                val_db = 10 * np.log10(sq / den)
                seed_nmse.append(val_db)
                row[f"{mid}_s{s}"] = f"{val_db:+.2f}"
                
            mean_val = np.mean(seed_nmse)
            std_val = np.std(seed_nmse)
            row[f"{mid}_mean"] = f"{mean_val:+.2f}"
            row[f"{mid}_std"] = f"{std_val:.3f}"
            
        rows.append(row)
        print(f"  SNR {snr:>3} dB computed across all seeds.")
        
    df = pd.DataFrame(rows)
    out_csv = os.path.join(ROOT, 'outputs', 'csv', 'ablation_ladder_combined_tdl_64_spread.csv')
    df.to_csv(out_csv, index=False)
    paper_csv = os.path.join(ROOT, 'paper', 'data', 'ablation_ladder_combined_tdl_64_spread.csv')
    df.to_csv(paper_csv, index=False)
    print(f"Saved ablation ladder spread to {out_csv} and {paper_csv}")
    return df

def main():
    print("Loading base NCRN and NCRT models...")
    customs = {'LastFrame': LastFrame}
    m_ncrn = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'ncrn_tdl_64.keras'),
                                        custom_objects=customs, compile=False)
    m_ncrt = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'ncrt_tdl_64.keras'),
                                        compile=False)
    
    print("Generating lockstep combined test channels (N=100, seed 42)...")
    test_data = generate_combined_test_channels()
    
    # 1. Run SNR offset sweep
    run_offset_sweep(test_data, m_ncrn, m_ncrt)
    
    # 2. Run Pilot-Window Noise Estimator
    run_noise_estimator(test_data, m_ncrn, m_ncrt)
    
    # 3. Run Multi-Seed Spread for Ablations
    run_ablation_spread(test_data)
    
    print("\nAll evaluations complete and saved!")

if __name__ == '__main__':
    main()

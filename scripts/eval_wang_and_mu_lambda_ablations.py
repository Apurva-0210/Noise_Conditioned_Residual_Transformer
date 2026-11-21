#!/usr/bin/env python3
"""
Evaluate:
1. Wang et al. (IEEE TCCN 2025) Sa-DLCS models on the exact test arrays behind combined_test_tdl_64.csv
   (where LS reads -28.13 / -28.14 dB at 20 dB), reporting all 4 variants with per-seed values.
2. mu-only and lambda-only runs on both checkpoints (cnn_lstm_tdl_64.keras and sa_dlcs_tdl_64.keras).
3. Diagnostic measurements on why ConvLSTM-ISTA alone is so much worse than CNN-LSTM alone.
"""
import os, sys, time
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from shared.channels import gen_tdl_random, TIME_STEPS
from shared.baselines import sadlcs_postprocess

SNR_RANGE = [-15, -10, -5, 0, 5, 10, 15, 20]
NR, NT = 64, 64

# DFT Matrix for Wang baseline
n_idx = np.arange(NR)
k_idx = np.arange(NR)
DFT_MAT = np.exp(-2j * np.pi * np.outer(k_idx, n_idx) / NR) / np.sqrt(NR)

def vectorized_preprocess(R_cols, mode='arith', eps=1e-8):
    mags = np.abs(R_cols)
    max_idx = np.argmax(mags, axis=-1, keepdims=True)
    mask_excl = np.ones_like(mags, dtype=bool)
    np.put_along_axis(mask_excl, max_idx, False, axis=-1)
    
    if mode == 'arith':
        taus = np.sum(mags * mask_excl, axis=-1, keepdims=True) / (NR - 1)
    elif mode == 'harm':
        inv_mags = 1.0 / (mags + eps)
        taus = (NR - 1) / (np.sum(inv_mags * mask_excl, axis=-1, keepdims=True) + eps)
    else:
        raise ValueError(f"Unknown mode {mode}")
    
    gamma = (mags >= taus)
    R_masked = R_cols * gamma
    X = np.concatenate([R_masked.real, R_masked.imag], axis=-1).astype(np.float32)
    return X, gamma

def main():
    print("="*75)
    print("1. Generating Exact Test Arrays Behind combined_test_tdl_64.csv")
    print("="*75)
    
    # Exactly matching eval_ncrt_on_legacy_tdl.py: np.random.seed(42) set ONCE before the SNR loop
    n_samples = 100
    np.random.seed(42)
    
    test_data = {}
    for snr in SNR_RANGE:
        ns = np.sqrt(10 ** (-snr / 10.0))
        X_list, HL_list, HR_list = [], [], []
        for _ in range(n_samples):
            h = gen_tdl_random(NR, NT)  # (TIME_STEPS, NR, NT, 2)
            noise = np.random.normal(0, ns, h.shape).astype(np.float32)
            hn = (h + noise).astype(np.float32)
            X_list.append(hn)
            HL_list.append(hn[-1])
            HR_list.append(h[-1])
        test_data[snr] = (
            np.array(X_list, dtype=np.float32),
            np.array(HL_list, dtype=np.float32),
            np.array(HR_list, dtype=np.float32)
        )
        
    print("Verifying LS Baseline NMSE:")
    for snr in SNR_RANGE:
        X, HL, HR = test_data[snr]
        den = np.sum(HR ** 2)
        sq_ls = np.sum((HR - HL) ** 2)
        ls_nmse = 10 * np.log10(sq_ls / den)
        print(f"  SNR {snr:3d} dB: LS = {ls_nmse:6.2f} dB")
        
    print("\n" + "="*75)
    print("2. Evaluating mu-only and lambda-only Ablations on Both Checkpoints")
    print("="*75)
    
    m_cnn = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'cnn_lstm_tdl_64.keras'), compile=False)
    m_sa  = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'sa_dlcs_tdl_64.keras'), compile=False)
    
    # We will test on each checkpoint:
    # - raw
    # - full ISTA (mu=0.5, lam=0.05, K=3)
    # - mu-only (mu=0.5, lam=0.0, K=3)
    # - lambda-only (mu=0.0, lam=0.05, K=3)
    
    abl_results = {
        'snr': SNR_RANGE,
        'ls': [],
        'cnn_raw': [],
        'cnn_full_ista': [],
        'cnn_mu_only': [],
        'cnn_lam_only': [],
        'sa_raw': [],
        'sa_full_ista': [],
        'sa_mu_only': [],
        'sa_lam_only': []
    }
    
    # Also collect diagnostics at 20 dB
    diag_info = {}
    
    for snr in SNR_RANGE:
        X, HL, HR = test_data[snr]
        den = np.sum(HR ** 2)
        sq_ls = np.sum((HR - HL) ** 2)
        abl_results['ls'].append(10 * np.log10(sq_ls / den))
        
        # Predict
        p_cnn = m_cnn.predict(X, batch_size=20, verbose=0)
        p_sa  = m_sa.predict(X, batch_size=20, verbose=0)
        
        # CNN-LSTM variants
        sq_cnn_raw = np.sum((HR - p_cnn) ** 2)
        sq_cnn_full = sum(np.sum((hr - sadlcs_postprocess(pr, hl, lam=0.05, K=3, mu=0.5))**2) for hr, pr, hl in zip(HR, p_cnn, HL))
        sq_cnn_mu   = sum(np.sum((hr - sadlcs_postprocess(pr, hl, lam=0.0,  K=3, mu=0.5))**2) for hr, pr, hl in zip(HR, p_cnn, HL))
        sq_cnn_lam  = sum(np.sum((hr - sadlcs_postprocess(pr, hl, lam=0.05, K=3, mu=0.0))**2) for hr, pr, hl in zip(HR, p_cnn, HL))
        
        abl_results['cnn_raw'].append(10 * np.log10(sq_cnn_raw / den))
        abl_results['cnn_full_ista'].append(10 * np.log10(sq_cnn_full / den))
        abl_results['cnn_mu_only'].append(10 * np.log10(sq_cnn_mu / den))
        abl_results['cnn_lam_only'].append(10 * np.log10(sq_cnn_lam / den))
        
        # ConvLSTM-ISTA variants
        sq_sa_raw  = np.sum((HR - p_sa) ** 2)
        sq_sa_full = sum(np.sum((hr - sadlcs_postprocess(pr, hl, lam=0.05, K=3, mu=0.5))**2) for hr, pr, hl in zip(HR, p_sa, HL))
        sq_sa_mu   = sum(np.sum((hr - sadlcs_postprocess(pr, hl, lam=0.0,  K=3, mu=0.5))**2) for hr, pr, hl in zip(HR, p_sa, HL))
        sq_sa_lam  = sum(np.sum((hr - sadlcs_postprocess(pr, hl, lam=0.05, K=3, mu=0.0))**2) for hr, pr, hl in zip(HR, p_sa, HL))
        
        abl_results['sa_raw'].append(10 * np.log10(sq_sa_raw / den))
        abl_results['sa_full_ista'].append(10 * np.log10(sq_sa_full / den))
        abl_results['sa_mu_only'].append(10 * np.log10(sq_sa_mu / den))
        abl_results['sa_lam_only'].append(10 * np.log10(sq_sa_lam / den))
        
        if snr == 20:
            # Compute detailed diagnostics at 20 dB
            # 1. Amplitude ratio ||H_hat|| / ||H||
            amp_ratio_cnn = np.sqrt(np.sum(p_cnn ** 2) / den)
            amp_ratio_sa  = np.sqrt(np.sum(p_sa ** 2) / den)
            
            # 2. Bias power ||E[H_hat - H]||^2 / ||H||^2
            mean_err_cnn = np.mean(p_cnn - HR, axis=0)
            mean_err_sa  = np.mean(p_sa - HR, axis=0)
            bias_pwr_cnn = np.sum(mean_err_cnn ** 2) / (den / n_samples)
            bias_pwr_sa  = np.sum(mean_err_sa ** 2) / (den / n_samples)
            
            # 3. Variance of prediction
            var_cnn = np.var(p_cnn)
            var_sa  = np.var(p_sa)
            var_hr  = np.var(HR)
            
            # 4. Correlation with noise N = HL - HR
            N_noise = HL - HR
            corr_noise_cnn = np.sum((p_cnn - HR) * N_noise) / (np.linalg.norm(p_cnn - HR) * np.linalg.norm(N_noise) + 1e-12)
            corr_noise_sa  = np.sum((p_sa - HR) * N_noise) / (np.linalg.norm(p_sa - HR) * np.linalg.norm(N_noise) + 1e-12)
            
            diag_info = {
                'amp_ratio_cnn': amp_ratio_cnn,
                'amp_ratio_sa': amp_ratio_sa,
                'bias_pwr_cnn': bias_pwr_cnn,
                'bias_pwr_sa': bias_pwr_sa,
                'var_cnn': var_cnn,
                'var_sa': var_sa,
                'var_hr': var_hr,
                'corr_noise_cnn': corr_noise_cnn,
                'corr_noise_sa': corr_noise_sa,
            }
            
    df_abl = pd.DataFrame(abl_results)
    out_abl_csv = os.path.join(ROOT, 'outputs', 'csv', 'ablation_mu_lambda_checkpoints.csv')
    df_abl.to_csv(out_abl_csv, index=False)
    df_abl.to_csv(os.path.join(ROOT, 'paper', 'data', 'ablation_mu_lambda_checkpoints.csv'), index=False)
    print("Saved mu/lambda ablation table:")
    print(df_abl.to_string(index=False))
    
    print("\n" + "="*75)
    print("3. Evaluating Saved Wang et al. Models on the Exact Test Arrays")
    print("="*75)
    
    # 4 variants x 3 seeds = 12 models
    wang_variants = [
        ('arith', 'wangproto'),
        ('harm',  'wangproto'),
        ('arith', 'curriculum'),
        ('harm',  'curriculum'),
    ]
    SEEDS = [42, 100, 2024]
    
    wang_raw_results = {
        'snr': SNR_RANGE,
        'ls': abl_results['ls']
    }
    
    # Prepare data for each SNR
    for mode, proto in wang_variants:
        for seed in SEEDS:
            col_name = f"{mode}_{proto}_s{seed}"
            wang_raw_results[col_name] = []
            
    for mode, proto in wang_variants:
        for seed in SEEDS:
            m_path = os.path.join(ROOT, 'models', 'sa_dlcs_wang', f"cenn_wang_{mode}_{proto}_seed{seed}.keras")
            model = tf.keras.models.load_model(m_path, compile=False)
            col_name = f"{mode}_{proto}_s{seed}"
            
            for snr in SNR_RANGE:
                X, HL, HR = test_data[snr]
                den = np.sum(HR ** 2)
                snr_norm = snr / 30.0
                
                # Convert HL and HR to complex
                # HL shape: (100, 64, 64, 2)
                sq_err = 0.0
                for i in range(n_samples):
                    hl_c = (HL[i, :, :, 0] + 1j * HL[i, :, :, 1]).astype(np.complex64)
                    hr_c = (HR[i, :, :, 0] + 1j * HR[i, :, :, 1]).astype(np.complex64)
                    
                    Y_cols = hl_c.T  # (64, 64)
                    R_cols = Y_cols @ DFT_MAT.conj()  # (64, 64)
                    X_cols, gammas = vectorized_preprocess(R_cols, mode=mode)
                    
                    if proto == 'curriculum':
                        snr_col = np.full((NT, 1), snr_norm, dtype=np.float32)
                        X_cols = np.concatenate([X_cols, snr_col], axis=-1)
                        
                    c_hat = model(X_cols, training=False).numpy()  # (64, 128)
                    c_hat_c = (c_hat[:, :NR] + 1j * c_hat[:, NR:]) * gammas
                    H_recon = (c_hat_c @ DFT_MAT.T).T  # back to (64, 64)
                    
                    sq_err += np.sum(np.abs(hr_c - H_recon) ** 2)
                    
                wang_raw_results[col_name].append(10 * np.log10(sq_err / den))
                
    df_wang = pd.DataFrame(wang_raw_results)
    out_wang_csv = os.path.join(ROOT, 'outputs', 'csv', 'sa_dlcs_wang_on_combined_test_tdl_64.csv')
    df_wang.to_csv(out_wang_csv, index=False)
    df_wang.to_csv(os.path.join(ROOT, 'paper', 'data', 'sa_dlcs_wang_on_combined_test_tdl_64.csv'), index=False)
    print("Saved Wang evaluation table:")
    print(df_wang.to_string(index=False))
    
    print("\n" + "="*75)
    print("4. Diagnostic Analysis at 20 dB (ConvLSTM-ISTA vs CNN-LSTM)")
    print("="*75)
    for k, v in diag_info.items():
        print(f"  {k:20s}: {v}")
        
if __name__ == '__main__':
    main()

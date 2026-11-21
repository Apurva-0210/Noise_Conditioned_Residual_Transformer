#!/usr/bin/env python3
"""
Evaluate Spectral Efficiency Gap in the exact Table 7 Configuration:
- Nominal noise scaling: ns = sqrt(10^(-SNR/10)), sigma2_n = 10^(-SNR/10) (LS achieves ~48.82 bps/Hz at 20 dB)
- 4-stream SU-MIMO spatial multiplexing with SVD precoding and LMMSE equalization (16-QAM)
- Estimators:
  1. Genie (Perfect CSI)
  2. LS
  3. NCRN Raw & Blended
  4. NCRT Raw & Blended
  5. White-noise error control (calibrated to exact sample NMSE of NCRN and NCRT)
  6. De-biased NCRN & NCRT (empirical mean error \bar{E} removed: H_hat - \bar{E})
  7. De-biased + Rescaled NCRN & NCRT (mean error removed and amplitude rescaled to 1.0)
- Outputs saved to outputs/csv/link_level_se_table7_whitenoise_debiased.csv and paper/data/
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
from shared.baselines import add_snr_channel, snr_aware_blend
from train_ncrn_tdl import LastFrame
import ncrt_layers

SNR_RANGE = [-15, -10, -5, 0, 5, 10, 15, 20]
N_STREAMS = 4
N_SAMPLES = 50
SYMS_PER_CH = 500

def qam16_mod(n_syms):
    bits = np.random.randint(0, 2, n_syms * 4)
    b = bits.reshape(-1, 4)
    mapping = {(0,0): -3, (0,1): -1, (1,1): 1, (1,0): 3}
    I = np.array([mapping[(x[0], x[1])] for x in b])
    Q = np.array([mapping[(x[2], x[3])] for x in b])
    s = (I + 1j*Q) / np.sqrt(10.0)
    return bits, s

def run_simulation():
    print("=" * 80)
    print("Evaluating SE Gap in Table 7 Configuration with Mean-Error Removal")
    print("=" * 80)
    
    m_ncrn = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'ncrn_tdl_64.keras'),
                                        custom_objects={'LastFrame': LastFrame}, compile=False)
    m_ncrt = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'ncrt_tdl_64.keras'), compile=False)
    
    np.random.seed(42)
    
    rows = []
    
    for snr in SNR_RANGE:
        t0 = time.time()
        # Exact Table 7 nominal noise scaling
        ns = np.sqrt(10 ** (-snr / 10.0))
        sigma2_n = 10 ** (-snr / 10.0)
        
        # 1. Generate channels and observations
        H_true_list, H_noisy_list, HL_list, HR_list = [], [], [], []
        for _ in range(N_SAMPLES):
            h = gen_tdl_random(64, 64)
            noise = np.random.normal(0, ns, h.shape).astype(np.float32)
            hn = (h + noise).astype(np.float32)
            H_true_list.append(h)
            H_noisy_list.append(hn)
            HL_list.append(hn[-1])
            HR_list.append(h[-1])
            
        HL_arr = np.array(HL_list, dtype=np.float32)
        HN_arr = np.array(H_noisy_list, dtype=np.float32)
        HR_arr = np.array(HR_list, dtype=np.float32)
        
        X_SNR = np.array([add_snr_channel(hn, snr) for hn in HN_arr], dtype=np.float32)
        p_ncrn_raw = m_ncrn.predict(X_SNR, batch_size=25, verbose=0)
        p_ncrt_raw = m_ncrt.predict(X_SNR, batch_size=25, verbose=0)
        
        H_true = [hr[:, :, 0] + 1j*hr[:, :, 1] for hr in HR_arr]
        H_ls   = [hl[:, :, 0] + 1j*hl[:, :, 1] for hl in HL_arr]
        
        H_ncrn_raw = [(hl[:,:,0] + pn[:,:,0]) + 1j*(hl[:,:,1] + pn[:,:,1]) for hl, pn in zip(HL_arr, p_ncrn_raw)]
        H_ncrt_raw = [(hl[:,:,0] + pt[:,:,0]) + 1j*(hl[:,:,1] + pt[:,:,1]) for hl, pt in zip(HL_arr, p_ncrt_raw)]
        
        H_ncrn_blend = [snr_aware_blend(pn, hl, snr)[:, :, 0] + 1j*snr_aware_blend(pn, hl, snr)[:, :, 1] for hl, pn in zip(HL_arr, p_ncrn_raw)]
        H_ncrt_blend = [snr_aware_blend(pt, hl, snr)[:, :, 0] + 1j*snr_aware_blend(pt, hl, snr)[:, :, 1] for hl, pt in zip(HL_arr, p_ncrt_raw)]
        
        # 2. Compute empirical mean estimation error (bias) across the N_SAMPLES
        err_ncrn_raw = [hest - ht for hest, ht in zip(H_ncrn_raw, H_true)]
        err_ncrt_raw = [hest - ht for hest, ht in zip(H_ncrt_raw, H_true)]
        
        mean_bias_ncrn = np.mean(err_ncrn_raw, axis=0) # (64, 64)
        mean_bias_ncrt = np.mean(err_ncrt_raw, axis=0) # (64, 64)
        
        bias_pwr_ncrn = float(np.sum(np.abs(mean_bias_ncrn) ** 2))
        bias_pwr_ncrt = float(np.sum(np.abs(mean_bias_ncrt) ** 2))
        
        # 3. Construct de-biased estimates
        H_ncrn_debiased = [hest - mean_bias_ncrn for hest in H_ncrn_raw]
        H_ncrt_debiased = [hest - mean_bias_ncrt for hest in H_ncrt_raw]
        
        # 4. Construct amplitude-rescaled estimates
        amp_r_ncrn = np.mean([np.linalg.norm(hest, 'fro') / np.linalg.norm(ht, 'fro') for hest, ht in zip(H_ncrn_raw, H_true)])
        amp_r_ncrt = np.mean([np.linalg.norm(hest, 'fro') / np.linalg.norm(ht, 'fro') for hest, ht in zip(H_ncrt_raw, H_true)])
        
        H_ncrn_rescaled = [hest / amp_r_ncrn for hest in H_ncrn_raw]
        H_ncrt_rescaled = [hest / amp_r_ncrt for hest in H_ncrt_raw]
        
        H_ncrn_debiased_rescaled = [(hest - mean_bias_ncrn) / (np.linalg.norm(hest - mean_bias_ncrn, 'fro') / np.linalg.norm(ht, 'fro')) for hest, ht in zip(H_ncrn_raw, H_true)]
        H_ncrt_debiased_rescaled = [(hest - mean_bias_ncrt) / (np.linalg.norm(hest - mean_bias_ncrt, 'fro') / np.linalg.norm(ht, 'fro')) for hest, ht in zip(H_ncrt_raw, H_true)]
        
        # 5. Construct synthetic isotropic white-noise controls matching exact sample NMSE
        H_white_ncrn, H_white_ncrt = [], []
        for i in range(N_SAMPLES):
            ht = H_true[i]
            # NCRN error power
            err_pwr_n = np.sum(np.abs(H_ncrn_raw[i] - ht) ** 2)
            w_raw_n = np.random.normal(0, 1, ht.shape) + 1j*np.random.normal(0, 1, ht.shape)
            w_scaled_n = w_raw_n * np.sqrt(err_pwr_n / np.sum(np.abs(w_raw_n) ** 2))
            H_white_ncrn.append(ht + w_scaled_n)
            
            # NCRT error power
            err_pwr_t = np.sum(np.abs(H_ncrt_raw[i] - ht) ** 2)
            w_raw_t = np.random.normal(0, 1, ht.shape) + 1j*np.random.normal(0, 1, ht.shape)
            w_scaled_t = w_raw_t * np.sqrt(err_pwr_t / np.sum(np.abs(w_raw_t) ** 2))
            H_white_ncrt.append(ht + w_scaled_t)
            
        H_hat_dict = {
            'Genie': H_true,
            'LS': H_ls,
            'NCRN (raw)': H_ncrn_raw,
            'NCRN (blended)': H_ncrn_blend,
            'NCRN (white control)': H_white_ncrn,
            'NCRN (debiased)': H_ncrn_debiased,
            'NCRN (rescaled)': H_ncrn_rescaled,
            'NCRN (debiased+rescaled)': H_ncrn_debiased_rescaled,
            'NCRT (raw)': H_ncrt_raw,
            'NCRT (blended)': H_ncrt_blend,
            'NCRT (white control)': H_white_ncrt,
            'NCRT (debiased)': H_ncrt_debiased,
            'NCRT (rescaled)': H_ncrt_rescaled,
            'NCRT (debiased+rescaled)': H_ncrt_debiased_rescaled,
        }
        
        se_lmmse = {m: [] for m in H_hat_dict}
        sinr_streams = {m: [[] for _ in range(N_STREAMS)] for m in H_hat_dict}
        
        for i in range(N_SAMPLES):
            Ht = H_true[i]
            _, qam16_syms_i = qam16_mod(SYMS_PER_CH * N_STREAMS)
            S_qam16 = qam16_syms_i.reshape(N_STREAMS, SYMS_PER_CH)
            Noise = (np.random.normal(0, np.sqrt(sigma2_n/2), (64, SYMS_PER_CH)) +
                     1j*np.random.normal(0, np.sqrt(sigma2_n/2), (64, SYMS_PER_CH)))
            
            for m in H_hat_dict:
                H_est = H_hat_dict[m][i]
                U_est, S_est, Vh_est = np.linalg.svd(H_est, full_matrices=False)
                F_prec = Vh_est[:N_STREAMS, :].conj().T
                
                H_eff_true = Ht @ F_prec
                H_eff_est = H_est @ F_prec
                Y = H_eff_true @ S_qam16 + Noise
                
                G_lmmse = np.linalg.inv(H_eff_est.conj().T @ H_eff_est + sigma2_n * np.eye(N_STREAMS)) @ H_eff_est.conj().T
                
                C_lmmse = G_lmmse @ H_eff_true
                
                se_i = 0.0
                for s_idx in range(N_STREAMS):
                    sig_pwr = np.abs(C_lmmse[s_idx, s_idx]) ** 2
                    int_pwr = np.sum(np.abs(C_lmmse[s_idx, :]) ** 2) - sig_pwr
                    noise_pwr = sigma2_n * np.sum(np.abs(G_lmmse[s_idx, :]) ** 2)
                    sinr_val = sig_pwr / (int_pwr + noise_pwr + 1e-12)
                    sinr_streams[m][s_idx].append(sinr_val)
                    se_i += np.log2(1.0 + sinr_val)
                se_lmmse[m].append(se_i)
                
        # Format row
        row = {
            'SNR (dB)': snr,
            'SE_Genie': np.mean(se_lmmse['Genie']),
            'SE_LS': np.mean(se_lmmse['LS']),
            'SE_NCRN_raw': np.mean(se_lmmse['NCRN (raw)']),
            'SE_NCRN_blend': np.mean(se_lmmse['NCRN (blended)']),
            'SE_NCRN_white_control': np.mean(se_lmmse['NCRN (white control)']),
            'SE_NCRN_debiased': np.mean(se_lmmse['NCRN (debiased)']),
            'SE_NCRN_rescaled': np.mean(se_lmmse['NCRN (rescaled)']),
            'SE_NCRN_debiased_rescaled': np.mean(se_lmmse['NCRN (debiased+rescaled)']),
            'SE_NCRT_raw': np.mean(se_lmmse['NCRT (raw)']),
            'SE_NCRT_blend': np.mean(se_lmmse['NCRT (blended)']),
            'SE_NCRT_white_control': np.mean(se_lmmse['NCRT (white control)']),
            'SE_NCRT_debiased': np.mean(se_lmmse['NCRT (debiased)']),
            'SE_NCRT_rescaled': np.mean(se_lmmse['NCRT (rescaled)']),
            'SE_NCRT_debiased_rescaled': np.mean(se_lmmse['NCRT (debiased+rescaled)']),
            'Amp_Ratio_NCRN': amp_r_ncrn,
            'Amp_Ratio_NCRT': amp_r_ncrt,
            'Bias_Pwr_NCRN': bias_pwr_ncrn,
            'Bias_Pwr_NCRT': bias_pwr_ncrt,
            'SINR_Stream4_NCRN_raw_dB': 10*np.log10(np.mean(sinr_streams['NCRN (raw)'][3])),
            'SINR_Stream4_NCRN_white_dB': 10*np.log10(np.mean(sinr_streams['NCRN (white control)'][3])),
            'SINR_Stream4_NCRN_debiased_dB': 10*np.log10(np.mean(sinr_streams['NCRN (debiased)'][3])),
            'SINR_Stream4_NCRT_raw_dB': 10*np.log10(np.mean(sinr_streams['NCRT (raw)'][3])),
            'SINR_Stream4_NCRT_white_dB': 10*np.log10(np.mean(sinr_streams['NCRT (white control)'][3])),
            'SINR_Stream4_NCRT_debiased_dB': 10*np.log10(np.mean(sinr_streams['NCRT (debiased)'][3])),
        }
        rows.append(row)
        print(f"SNR {snr:3d} dB [{time.time()-t0:.1f}s]: "
              f"LS={row['SE_LS']:.2f} | "
              f"NCRT Raw={row['SE_NCRT_raw']:.2f} (White={row['SE_NCRT_white_control']:.2f}, Debias={row['SE_NCRT_debiased']:.2f}) | "
              f"NCRN Raw={row['SE_NCRN_raw']:.2f} (White={row['SE_NCRN_white_control']:.2f}, Debias={row['SE_NCRN_debiased']:.2f}) | "
              f"BiasPwr: NCRN={bias_pwr_ncrn:.2f}, NCRT={bias_pwr_ncrt:.2f}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(ROOT, 'outputs', 'csv', 'link_level_se_table7_whitenoise_debiased.csv')
    paper_csv = os.path.join(ROOT, 'paper', 'data', 'link_level_se_table7_whitenoise_debiased.csv')
    df.to_csv(out_csv, index=False)
    df.to_csv(paper_csv, index=False)
    print(f"\nSaved results to:\n  {out_csv}\n  {paper_csv}")
    return df

if __name__ == '__main__':
    run_simulation()

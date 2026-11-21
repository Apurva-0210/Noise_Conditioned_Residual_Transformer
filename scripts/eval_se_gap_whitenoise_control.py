#!/usr/bin/env python3
"""
Reviewer Item 2: Link-Level Spectral Efficiency Gap & White-Noise Error Control
Repeats the Table 8 MIMO transmission run with:
1. Exact noise scaling matching the NMSE tables: sigma2_n = sp * 10^(-SNR/10) where sp = mean(h^2)
2. LS estimator included with matched noise scaling
3. White-Noise Error Control: Synthetic isotropic Gaussian error H + E_white matching the exact NMSE of NCRT/NCRN
4. 4-stream SU-MIMO spatial multiplexing with SVD precoding and LMMSE equalization
5. Per-stream SINRs, spectral efficiency, amplitude ratios, and systematic bias power decomposition.
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
N_SAMPLES = 50   # 50 independent channels
SYMS_PER_CH = 500 # 500 symbols per stream = 2,000 symbols/channel -> 100,000 symbols per SNR point

OUT_CSV_DIR = os.path.join(ROOT, 'outputs', 'csv')
DATA_DIR = os.path.join(ROOT, 'paper', 'data')

def qpsk_mod(n_syms):
    bits = np.random.randint(0, 2, n_syms * 2)
    b = bits.reshape(-1, 2)
    s = ((1.0 - 2.0*b[:,0]) + 1j*(1.0 - 2.0*b[:,1])) / np.sqrt(2.0)
    return bits, s

def qpsk_demod(s_hat):
    b0 = (s_hat.real < 0).astype(int)
    b1 = (s_hat.imag < 0).astype(int)
    return np.column_stack([b0, b1]).reshape(-1)

def qam16_mod(n_syms):
    bits = np.random.randint(0, 2, n_syms * 4)
    b = bits.reshape(-1, 4)
    mapping = {(0,0): -3, (0,1): -1, (1,1): 1, (1,0): 3}
    I = np.array([mapping[(x[0], x[1])] for x in b])
    Q = np.array([mapping[(x[2], x[3])] for x in b])
    s = (I + 1j*Q) / np.sqrt(10.0)
    return bits, s

def qam16_demod(s_hat):
    s = s_hat * np.sqrt(10.0)
    def pam4_demod(val):
        b0 = (val > 0).astype(int)
        b1 = (np.abs(val) < 2).astype(int)
        return b0, b1
    b0, b1 = pam4_demod(s.real)
    b2, b3 = pam4_demod(s.imag)
    return np.column_stack([b0, b1, b2, b3]).reshape(-1)

def main():
    print("=" * 80)
    print("Item 2: Link-Level 20 dB SE Gap Evaluation with White-Noise-Error Control")
    print("=" * 80)
    
    m_ncrn = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'ncrn_tdl_64.keras'),
                                        custom_objects={'LastFrame': LastFrame}, compile=False)
    m_ncrt = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'ncrt_tdl_64.keras'), compile=False)
    
    np.random.seed(42)
    results = []
    
    for snr in SNR_RANGE:
        t0 = time.time()
        
        # 1. Generate channels with exact NMSE-table noise scaling
        H, HN, HL, HR, SIGMA2_N, SP = [], [], [], [], [], []
        for _ in range(N_SAMPLES):
            h = gen_tdl_random(64, 64)
            sp = float(np.mean(h ** 2))
            sigma2_n = sp * 10 ** (-snr / 10.0)
            ns = np.sqrt(sigma2_n)
            noise = np.random.normal(0, ns, h.shape).astype(np.float32)
            hn = (h + noise).astype(np.float32)
            H.append(h); HN.append(hn); HL.append(hn[-1]); HR.append(h[-1])
            SIGMA2_N.append(sigma2_n); SP.append(sp)
            
        HL_arr = np.array(HL, dtype=np.float32)
        HN_arr = np.array(HN, dtype=np.float32)
        HR_arr = np.array(HR, dtype=np.float32)
        
        X_SNR = np.array([add_snr_channel(hn, snr) for hn in HN_arr], dtype=np.float32)
        p_ncrn_raw = m_ncrn.predict(X_SNR, batch_size=25, verbose=0)
        p_ncrt_raw = m_ncrt.predict(X_SNR, batch_size=25, verbose=0)
        
        H_true_list = [hr[:, :, 0] + 1j*hr[:, :, 1] for hr in HR_arr]
        H_ls_list   = [hl[:, :, 0] + 1j*hl[:, :, 1] for hl in HL_arr]
        
        H_ncrt_raw_list = [(hl[:,:,0] + pt[:,:,0]) + 1j*(hl[:,:,1] + pt[:,:,1]) for hl, pt in zip(HL_arr, p_ncrt_raw)]
        H_ncrn_raw_list = [(hl[:,:,0] + pn[:,:,0]) + 1j*(hl[:,:,1] + pn[:,:,1]) for hl, pn in zip(HL_arr, p_ncrn_raw)]
        
        H_ncrt_blend_list = [snr_aware_blend(pt, hl, snr)[:, :, 0] + 1j*snr_aware_blend(pt, hl, snr)[:, :, 1] for hl, pt in zip(HL_arr, p_ncrt_raw)]
        H_ncrn_blend_list = [snr_aware_blend(pn, hl, snr)[:, :, 0] + 1j*snr_aware_blend(pn, hl, snr)[:, :, 1] for hl, pn in zip(HL_arr, p_ncrn_raw)]
        
        # Construct Synthetic White-Noise Controls matching exact sample NMSE
        H_white_ncrt_list = []
        H_white_ncrn_list = []
        for i in range(N_SAMPLES):
            Ht = H_true_list[i]
            # NCRT error power
            err_ncrt_pwr = np.sum(np.abs(H_ncrt_raw_list[i] - Ht) ** 2)
            # White complex noise with exact same error power
            w_raw = (np.random.normal(0, 1, Ht.shape) + 1j*np.random.normal(0, 1, Ht.shape))
            w_scaled = w_raw * np.sqrt(err_ncrt_pwr / np.sum(np.abs(w_raw) ** 2))
            H_white_ncrt_list.append(Ht + w_scaled)
            
            # NCRN error power
            err_ncrn_pwr = np.sum(np.abs(H_ncrn_raw_list[i] - Ht) ** 2)
            w_raw2 = (np.random.normal(0, 1, Ht.shape) + 1j*np.random.normal(0, 1, Ht.shape))
            w_scaled2 = w_raw2 * np.sqrt(err_ncrn_pwr / np.sum(np.abs(w_raw2) ** 2))
            H_white_ncrn_list.append(Ht + w_scaled2)
            
        H_hat_dict = {
            'Genie': H_true_list,
            'LS': H_ls_list,
            'NCRT (raw)': H_ncrt_raw_list,
            'NCRT (blended)': H_ncrt_blend_list,
            'WhiteControl (NCRT)': H_white_ncrt_list,
            'NCRN (raw)': H_ncrn_raw_list,
            'NCRN (blended)': H_ncrn_blend_list,
            'WhiteControl (NCRN)': H_white_ncrn_list,
        }
        
        # Link-level simulation
        se_lmmse = {m: [] for m in H_hat_dict}
        sinr_streams = {m: [[] for _ in range(N_STREAMS)] for m in H_hat_dict}
        amp_ratios = {m: [] for m in H_hat_dict}
        errors = {m: [] for m in H_hat_dict}
        
        for i in range(N_SAMPLES):
            H_true = H_true_list[i]
            sigma2_n = SIGMA2_N[i]
            
            # QAM-16 transmit symbols
            _, qam16_syms_i = qam16_mod(SYMS_PER_CH * N_STREAMS)
            S_qam16 = qam16_syms_i.reshape(N_STREAMS, SYMS_PER_CH)
            
            Noise = (np.random.normal(0, np.sqrt(sigma2_n/2), (64, SYMS_PER_CH)) +
                     1j*np.random.normal(0, np.sqrt(sigma2_n/2), (64, SYMS_PER_CH)))
            
            for m in H_hat_dict:
                H_est = H_hat_dict[m][i]
                
                # Metrics
                amp_r = np.linalg.norm(H_est, 'fro') / np.linalg.norm(H_true, 'fro')
                amp_ratios[m].append(amp_r)
                errors[m].append(H_est - H_true)
                
                # SVD Precoding
                _, _, Vh = np.linalg.svd(H_est, full_matrices=False)
                V = Vh.conj().T
                V_prec = V[:, :N_STREAMS]
                
                H_eff_true = H_true @ V_prec
                H_eff_est  = H_est  @ V_prec
                
                # LMMSE Equalization
                G = H_eff_est.conj().T @ H_eff_est
                W_lmmse = np.linalg.solve(G + sigma2_n * np.eye(N_STREAMS), H_eff_est.conj().T)
                
                WH = W_lmmse @ H_eff_true
                se_sample = 0.0
                for k in range(N_STREAMS):
                    sig_pwr = np.abs(WH[k, k]) ** 2
                    isi_pwr = np.sum(np.abs(WH[k, :]) ** 2) - sig_pwr
                    noise_pwr = sigma2_n * np.sum(np.abs(W_lmmse[k, :]) ** 2)
                    sinr_k = sig_pwr / max(isi_pwr + noise_pwr, 1e-12)
                    se_sample += np.log2(1.0 + sinr_k)
                    sinr_streams[m][k].append(10 * np.log10(sinr_k))
                se_lmmse[m].append(se_sample)
                
        # Systematic bias power ||E[H_hat - H]||_F^2
        bias_power = {m: float(np.sum(np.abs(np.mean(errors[m], axis=0)) ** 2)) for m in H_hat_dict}
        
        row = {
            'SNR (dB)': snr,
            'SE Genie': np.mean(se_lmmse['Genie']),
            'SE LS': np.mean(se_lmmse['LS']),
            'SE NCRT Raw': np.mean(se_lmmse['NCRT (raw)']),
            'SE WhiteControl NCRT': np.mean(se_lmmse['WhiteControl (NCRT)']),
            'SE NCRN Raw': np.mean(se_lmmse['NCRN (raw)']),
            'SE WhiteControl NCRN': np.mean(se_lmmse['WhiteControl (NCRN)']),
            'AmpRatio LS': np.mean(amp_ratios['LS']),
            'AmpRatio NCRT': np.mean(amp_ratios['NCRT (raw)']),
            'AmpRatio WhiteControl NCRT': np.mean(amp_ratios['WhiteControl (NCRT)']),
            'AmpRatio NCRN': np.mean(amp_ratios['NCRN (raw)']),
            'BiasPwr LS': bias_power['LS'],
            'BiasPwr NCRT': bias_power['NCRT (raw)'],
            'BiasPwr WhiteControl NCRT': bias_power['WhiteControl (NCRT)'],
            'BiasPwr NCRN': bias_power['NCRN (raw)'],
            'Stream 1 SINR (NCRT)': np.mean(sinr_streams['NCRT (raw)'][0]),
            'Stream 4 SINR (NCRT)': np.mean(sinr_streams['NCRT (raw)'][3]),
            'Stream 4 SINR (WhiteCtrl NCRT)': np.mean(sinr_streams['WhiteControl (NCRT)'][3]),
            'Stream 1 SINR (NCRN)': np.mean(sinr_streams['NCRN (raw)'][0]),
            'Stream 4 SINR (NCRN)': np.mean(sinr_streams['NCRN (raw)'][3]),
            'Stream 4 SINR (WhiteCtrl NCRN)': np.mean(sinr_streams['WhiteControl (NCRN)'][3]),
        }
        results.append(row)
        print(f"SNR {snr:>3} dB [{time.time()-t0:.1f}s] | "
              f"SE: LS={row['SE LS']:.2f}, NCRT={row['SE NCRT Raw']:.2f}, WhiteCtrl(NCRT)={row['SE WhiteControl NCRT']:.2f} | "
              f"NCRN={row['SE NCRN Raw']:.2f}, WhiteCtrl(NCRN)={row['SE WhiteControl NCRN']:.2f} | "
              f"Str4 SINR: NCRT={row['Stream 4 SINR (NCRT)']:.1f} vs White={row['Stream 4 SINR (WhiteCtrl NCRT)']:.1f} dB")
        
    df = pd.DataFrame(results)
    out_csv = os.path.join(OUT_CSV_DIR, 'link_level_se_whitenoise_control.csv')
    paper_csv = os.path.join(DATA_DIR, 'link_level_se_whitenoise_control.csv')
    df.to_csv(out_csv, index=False)
    df.to_csv(paper_csv, index=False)
    print(f"\nSaved Link-Level SE White-Noise Control results to {out_csv} and {paper_csv}")
    print("\n" + df.to_string(index=False))

if __name__ == '__main__':
    main()

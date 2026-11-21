#!/usr/bin/env python3
"""
Link-Level BER and Spectral Efficiency Simulation with explicit Raw and Blended models.
N_s = 4 streams, 64x64 Massive MIMO, QPSK and 16-QAM, LMMSE headline equalizer.
Evaluates:
- LS
- NCRN (blended)
- NCRN (raw)
- NCRT (blended)
- NCRT (raw)
- Genie (oracle CSI)
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
N_SAMPLES = 100
N_STREAMS = 4
SYMS_PER_CH = 1000  # 4 streams * 1000 syms = 4000 syms/channel -> 400,000 syms/SNR point

def qpsk_mod(n_syms):
    bits = np.random.randint(0, 2, n_syms * 2)
    b0 = bits[0::2] * 2 - 1
    b1 = bits[1::2] * 2 - 1
    syms = (b0 + 1j * b1) / np.sqrt(2.0)
    return bits, syms

def qpsk_demod(s_hat):
    b0 = (s_hat.real > 0).astype(int)
    b1 = (s_hat.imag > 0).astype(int)
    bits = np.empty(len(s_hat) * 2, dtype=int)
    bits[0::2] = b0
    bits[1::2] = b1
    return bits

def qam16_mod(n_syms):
    bits = np.random.randint(0, 2, n_syms * 4)
    b = bits.reshape(n_syms, 4)
    pam_i = (2 * b[:, 0] - 1) * (2 - (2 * b[:, 1] - 1))
    pam_q = (2 * b[:, 2] - 1) * (2 - (2 * b[:, 3] - 1))
    syms = (pam_i + 1j * pam_q) / np.sqrt(10.0)
    return bits, syms

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
    print("="*75)
    print("Link-Level BER and Spectral Efficiency (Raw & Blended, N_s=4 streams)")
    print("="*75)
    
    m_ncrn = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'ncrn_tdl_64.keras'),
                                        custom_objects={'LastFrame': LastFrame}, compile=False)
    m_ncrt = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'ncrt_tdl_64.keras'), compile=False)
    
    np.random.seed(42)
    results = []
    
    for snr in SNR_RANGE:
        t0 = time.time()
        ns = np.sqrt(10 ** (-snr / 10.0))
        sigma2_n = 10 ** (-snr / 10.0)
        
        # 1. Generate channels (lockstep combined set)
        H, HN, HL, HR = [], [], [], []
        for _ in range(N_SAMPLES):
            h = gen_tdl_random(64, 64)
            noise = np.random.normal(0, ns, h.shape).astype(np.float32)
            hn = (h + noise).astype(np.float32)
            H.append(h); HN.append(hn); HL.append(hn[-1]); HR.append(h[-1])
            
        X_SNR = np.array([add_snr_channel(hn, snr) for hn in HN], dtype=np.float32)
        p_ncrn = m_ncrn.predict(X_SNR, batch_size=25, verbose=0)
        p_ncrt = m_ncrt.predict(X_SNR, batch_size=25, verbose=0)
        
        # Method estimates
        H_hat_dict = {
            'LS': [hl[:, :, 0] + 1j*hl[:, :, 1] for hl in HL],
            'NCRN_blend': [snr_aware_blend(pn, hl, snr)[:, :, 0] + 1j*snr_aware_blend(pn, hl, snr)[:, :, 1] for hl, pn in zip(HL, p_ncrn)],
            'NCRN_raw': [(hl + pn)[:, :, 0] + 1j*(hl + pn)[:, :, 1] for hl, pn in zip(HL, p_ncrn)],
            'NCRT_blend': [snr_aware_blend(pt, hl, snr)[:, :, 0] + 1j*snr_aware_blend(pt, hl, snr)[:, :, 1] for hl, pt in zip(HL, p_ncrt)],
            'NCRT_raw': [(hl + pt)[:, :, 0] + 1j*(hl + pt)[:, :, 1] for hl, pt in zip(HL, p_ncrt)],
            'Genie': [hr[:, :, 0] + 1j*hr[:, :, 1] for hr in HR],
        }
        
        bit_errors_qpsk = {m: 0 for m in H_hat_dict}
        bit_errors_qam16 = {m: 0 for m in H_hat_dict}
        total_bits_qpsk = 0
        total_bits_qam16 = 0
        se_lmmse = {m: [] for m in H_hat_dict}
        
        for i in range(N_SAMPLES):
            H_true = HR[i][:, :, 0] + 1j*HR[i][:, :, 1] # (64, 64)
            
            qpsk_bits_i, qpsk_syms_i = qpsk_mod(SYMS_PER_CH * N_STREAMS)
            qam16_bits_i, qam16_syms_i = qam16_mod(SYMS_PER_CH * N_STREAMS)
            
            S_qpsk = qpsk_syms_i.reshape(N_STREAMS, SYMS_PER_CH)
            S_qam16 = qam16_syms_i.reshape(N_STREAMS, SYMS_PER_CH)
            
            total_bits_qpsk += len(qpsk_bits_i)
            total_bits_qam16 += len(qam16_bits_i)
            
            Noise_qpsk = (np.random.normal(0, np.sqrt(sigma2_n/2), (64, SYMS_PER_CH)) +
                          1j*np.random.normal(0, np.sqrt(sigma2_n/2), (64, SYMS_PER_CH)))
            Noise_qam16 = (np.random.normal(0, np.sqrt(sigma2_n/2), (64, SYMS_PER_CH)) +
                           1j*np.random.normal(0, np.sqrt(sigma2_n/2), (64, SYMS_PER_CH)))
            
            for m in H_hat_dict:
                H_est = H_hat_dict[m][i]
                
                _, _, Vh = np.linalg.svd(H_est, full_matrices=False)
                V_precoder = Vh.conj().T[:, :N_STREAMS]
                
                H_eff_true = H_true @ V_precoder
                H_eff_est  = H_est  @ V_precoder
                
                Y_qpsk = H_eff_true @ S_qpsk + Noise_qpsk
                Y_qam16 = H_eff_true @ S_qam16 + Noise_qam16
                
                # LMMSE
                G = H_eff_est.conj().T @ H_eff_est
                W_lmmse = np.linalg.solve(G + sigma2_n * np.eye(N_STREAMS), H_eff_est.conj().T)
                
                # Equalize QPSK
                S_hat_raw = W_lmmse @ Y_qpsk
                gain = np.diag(W_lmmse @ H_eff_est).copy().reshape(-1, 1)
                gain[np.abs(gain) < 1e-6] = 1.0
                S_hat = S_hat_raw / gain
                bits_dec = qpsk_demod(S_hat.reshape(-1))
                bit_errors_qpsk[m] += np.sum(qpsk_bits_i != bits_dec)
                
                # Equalize 16-QAM
                S_hat_raw_qam = W_lmmse @ Y_qam16
                S_hat_qam = S_hat_raw_qam / gain
                bits_dec_qam = qam16_demod(S_hat_qam.reshape(-1))
                bit_errors_qam16[m] += np.sum(qam16_bits_i != bits_dec_qam)
                
                # Spectral efficiency under imperfect CSI
                H_comp = W_lmmse @ H_eff_true
                sig_pwr = np.abs(np.diag(H_comp)) ** 2
                tot_pwr = np.sum(np.abs(H_comp) ** 2, axis=1)
                isi_pwr = tot_pwr - sig_pwr
                noise_pwr = sigma2_n * np.sum(np.abs(W_lmmse) ** 2, axis=1)
                sinr = sig_pwr / np.maximum(1e-12, isi_pwr + noise_pwr)
                se_lmmse[m].append(np.sum(np.log2(1.0 + sinr)))
                
        row = {'SNR_dB': snr}
        for m in H_hat_dict:
            row[f'{m}_BER_QPSK'] = bit_errors_qpsk[m] / total_bits_qpsk
            row[f'{m}_BER_16QAM'] = bit_errors_qam16[m] / total_bits_qam16
            row[f'{m}_SE_bps_Hz'] = float(np.mean(se_lmmse[m]))
            
        results.append(row)
        print(f"SNR {snr:>3} dB ({time.time()-t0:.1f}s) | "
              f"SE: LS {row['LS_SE_bps_Hz']:.2f} | "
              f"NCRN_b {row['NCRN_blend_SE_bps_Hz']:.2f} | NCRN_r {row['NCRN_raw_SE_bps_Hz']:.2f} | "
              f"NCRT_b {row['NCRT_blend_SE_bps_Hz']:.2f} | NCRT_r {row['NCRT_raw_SE_bps_Hz']:.2f} | "
              f"Genie {row['Genie_SE_bps_Hz']:.2f}")
        
    df = pd.DataFrame(results)
    out_csv = os.path.join(ROOT, 'outputs', 'csv', 'link_level_ber_se_comprehensive.csv')
    df.to_csv(out_csv, index=False)
    paper_csv = os.path.join(ROOT, 'paper', 'data', 'link_level_ber_se_comprehensive.csv')
    df.to_csv(paper_csv, index=False)
    print(f"\nSaved comprehensive link-level results to {out_csv} and {paper_csv}")
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()

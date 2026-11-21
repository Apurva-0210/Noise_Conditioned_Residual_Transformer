#!/usr/bin/env python3
"""
Link-level MIMO transmission evaluation for LS, NCRN, and NCRT:
- Massive MIMO 64x64 array
- SU-MIMO spatial multiplexing with N_s = 4 spatial streams (SVD precoding on H_hat)
- QPSK and 16-QAM modulations (uncoded)
- Equalizers: LMMSE (headline) and Zero-Forcing (ZF)
- Metrics: Bit Error Rate (BER) and Spectral Efficiency (bps/Hz) under imperfect CSI
- Output saved to outputs/csv/link_level_ber_se.csv
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
N_SAMPLES = 50   # 50 channel realizations per SNR
SYMS_PER_CH = 500 # 500 symbols * 4 streams = 2000 syms/ch -> 100,000 syms = 200,000 bits QPSK, 400,000 bits 16-QAM

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
    print("="*75)
    print("Link-Level BER and Spectral Efficiency Simulation (N_s=4 streams)")
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
        
        # 1. Generate channels
        H, HN, HL, HR = [], [], [], []
        for _ in range(N_SAMPLES):
            h = gen_tdl_random(64, 64)
            noise = np.random.normal(0, ns, h.shape).astype(np.float32)
            hn = (h + noise).astype(np.float32)
            H.append(h)
            HN.append(hn)
            HL.append(hn[-1])
            HR.append(h[-1])
            
        X_SNR = np.array([add_snr_channel(hn, snr) for hn in HN], dtype=np.float32)
        p_ncrn = m_ncrn.predict(X_SNR, batch_size=25, verbose=0)
        p_ncrt = m_ncrt.predict(X_SNR, batch_size=25, verbose=0)
        
        # Method estimates
        H_hat_dict = {
            'LS': [hl[:, :, 0] + 1j*hl[:, :, 1] for hl in HL],
            'NCRN': [snr_aware_blend(pn, hl, snr)[:, :, 0] + 1j*snr_aware_blend(pn, hl, snr)[:, :, 1] for hl, pn in zip(HL, p_ncrn)],
            'NCRT': [snr_aware_blend(pt, hl, snr)[:, :, 0] + 1j*snr_aware_blend(pt, hl, snr)[:, :, 1] for hl, pt in zip(HL, p_ncrt)],
            'Genie': [hr[:, :, 0] + 1j*hr[:, :, 1] for hr in HR],
        }
        
        bit_errors_qpsk = {m: {'LMMSE': 0, 'ZF': 0} for m in H_hat_dict}
        bit_errors_qam16 = {m: {'LMMSE': 0, 'ZF': 0} for m in H_hat_dict}
        total_bits_qpsk = 0
        total_bits_qam16 = 0
        se_lmmse = {m: [] for m in H_hat_dict}
        
        for i in range(N_SAMPLES):
            H_true = HR[i][:, :, 0] + 1j*HR[i][:, :, 1] # (64, 64)
            
            # Generate transmit symbols for N_s streams
            qpsk_bits_i, qpsk_syms_i = qpsk_mod(SYMS_PER_CH * N_STREAMS)
            qam16_bits_i, qam16_syms_i = qam16_mod(SYMS_PER_CH * N_STREAMS)
            
            S_qpsk = qpsk_syms_i.reshape(N_STREAMS, SYMS_PER_CH)   # (4, SYMS_PER_CH)
            S_qam16 = qam16_syms_i.reshape(N_STREAMS, SYMS_PER_CH)
            
            total_bits_qpsk += len(qpsk_bits_i)
            total_bits_qam16 += len(qam16_bits_i)
            
            # Complex Gaussian noise for reception
            Noise_qpsk = (np.random.normal(0, np.sqrt(sigma2_n/2), (64, SYMS_PER_CH)) +
                          1j*np.random.normal(0, np.sqrt(sigma2_n/2), (64, SYMS_PER_CH)))
            Noise_qam16 = (np.random.normal(0, np.sqrt(sigma2_n/2), (64, SYMS_PER_CH)) +
                           1j*np.random.normal(0, np.sqrt(sigma2_n/2), (64, SYMS_PER_CH)))
            
            for m in H_hat_dict:
                H_est = H_hat_dict[m][i]
                
                # SVD beamforming / precoding based on estimated channel H_est
                # H_est = U S V^H
                _, _, Vh = np.linalg.svd(H_est, full_matrices=False)
                V = Vh.conj().T
                V_precoder = V[:, :N_STREAMS] # (64, 4)
                
                # Transmit over true channel
                H_eff_true = H_true @ V_precoder # (64, 4)
                H_eff_est  = H_est  @ V_precoder # (64, 4)
                
                # Received signals
                Y_qpsk = H_eff_true @ S_qpsk + Noise_qpsk
                Y_qam16 = H_eff_true @ S_qam16 + Noise_qam16
                
                # Receivers:
                # 1. LMMSE: W = (H_eff_est^H H_eff_est + sigma2_n * I)^-1 H_eff_est^H
                G = H_eff_est.conj().T @ H_eff_est
                W_lmmse = np.linalg.solve(G + sigma2_n * np.eye(N_STREAMS), H_eff_est.conj().T)
                
                # 2. ZF: W = (H_eff_est^H H_eff_est)^-1 H_eff_est^H
                W_zf = np.linalg.pinv(H_eff_est)
                
                # Equalize QPSK
                for eq_name, W in [('LMMSE', W_lmmse), ('ZF', W_zf)]:
                    S_hat_raw = W @ Y_qpsk
                    # Normalize by diagonal gain
                    gain = np.diag(W @ H_eff_est).copy().reshape(-1, 1)
                    gain[np.abs(gain) < 1e-6] = 1.0
                    S_hat = S_hat_raw / gain
                    bits_dec = qpsk_demod(S_hat.reshape(-1))
                    bit_errors_qpsk[m][eq_name] += np.sum(qpsk_bits_i != bits_dec)
                    
                    # Equalize 16-QAM
                    S_hat_raw16 = W @ Y_qam16
                    S_hat16 = S_hat_raw16 / gain
                    bits_dec16 = qam16_demod(S_hat16.reshape(-1))
                    bit_errors_qam16[m][eq_name] += np.sum(qam16_bits_i != bits_dec16)
                
                # Compute Spectral Efficiency (LMMSE)
                # SINR_k = |(W H_eff)_kk|^2 / (sum_{j!=k} |(W H_eff)_kj|^2 + sigma2_n * ||w_k||^2)
                WH = W_lmmse @ H_eff_true # (4, 4)
                se_sample = 0.0
                for k in range(N_STREAMS):
                    sig_pwr = np.abs(WH[k, k]) ** 2
                    isi_pwr = np.sum(np.abs(WH[k, :]) ** 2) - sig_pwr
                    noise_pwr = sigma2_n * np.sum(np.abs(W_lmmse[k, :]) ** 2)
                    sinr_k = sig_pwr / max(isi_pwr + noise_pwr, 1e-12)
                    se_sample += np.log2(1.0 + sinr_k)
                se_lmmse[m].append(se_sample)
                
        row = {
            'SNR_dB': snr,
            'LS_BER_QPSK': bit_errors_qpsk['LS']['LMMSE'] / total_bits_qpsk,
            'NCRN_BER_QPSK': bit_errors_qpsk['NCRN']['LMMSE'] / total_bits_qpsk,
            'NCRT_BER_QPSK': bit_errors_qpsk['NCRT']['LMMSE'] / total_bits_qpsk,
            'Genie_BER_QPSK': bit_errors_qpsk['Genie']['LMMSE'] / total_bits_qpsk,
            
            'LS_BER_16QAM': bit_errors_qam16['LS']['LMMSE'] / total_bits_qam16,
            'NCRN_BER_16QAM': bit_errors_qam16['NCRN']['LMMSE'] / total_bits_qam16,
            'NCRT_BER_16QAM': bit_errors_qam16['NCRT']['LMMSE'] / total_bits_qam16,
            'Genie_BER_16QAM': bit_errors_qam16['Genie']['LMMSE'] / total_bits_qam16,
            
            'LS_SE_bps_Hz': np.mean(se_lmmse['LS']),
            'NCRN_SE_bps_Hz': np.mean(se_lmmse['NCRN']),
            'NCRT_SE_bps_Hz': np.mean(se_lmmse['NCRT']),
            'Genie_SE_bps_Hz': np.mean(se_lmmse['Genie']),
        }
        results.append(row)
        print(f"SNR {snr:>3d} dB [{time.time()-t0:.1f}s] | "
              f"QPSK BER: LS={row['LS_BER_QPSK']:.4f}, NCRN={row['NCRN_BER_QPSK']:.4f}, NCRT={row['NCRT_BER_QPSK']:.4f} | "
              f"SE (bps/Hz): LS={row['LS_SE_bps_Hz']:.2f}, NCRN={row['NCRN_SE_bps_Hz']:.2f}, NCRT={row['NCRT_SE_bps_Hz']:.2f}, Genie={row['Genie_SE_bps_Hz']:.2f}")

    df_out = pd.DataFrame(results)
    out_csv = os.path.join(ROOT, 'outputs', 'csv', 'link_level_ber_se.csv')
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved link-level evaluation to: {out_csv}")
    print("\n" + df_out.to_string(index=False))

if __name__ == '__main__':
    main()

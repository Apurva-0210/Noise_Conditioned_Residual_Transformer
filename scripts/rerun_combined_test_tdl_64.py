#!/usr/bin/env python3
"""
Re-run all 7 estimators on one fresh, unified draw of 64x64 TDL channels:
1. LS
2. Wiener/MMSE (oracle)
3. CNN-Sparse
4. CNN-LSTM
5. Sa-DLCS
6. NCRN (final checkpoint, with snr_aware_blend)
7. NCRT (final checkpoint, with snr_aware_blend)

Exact lockstep: all estimators evaluate on the identical channel and noise realizations.
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
from shared.baselines import (ls_estimate, mmse_estimate, cnnsparse_mask,
                              sadlcs_postprocess, add_snr_channel, snr_aware_blend)
from train_ncrn_tdl import LastFrame
import ncrt_layers

SNR_RANGE = [-15, -10, -5, 0, 5, 10, 15, 20]
METHODS = ['cnn_lstm', 'cnn_sparse', 'ls', 'mmse', 'sa_dlcs', 'ncrn', 'ncrt', 'ncrn_raw', 'ncrt_raw']

def main():
    print("="*75)
    print("Re-running All 7 Estimators on a Unified Lockstep TDL-64 Test Draw")
    print("="*75)

    print("Loading models...")
    m_sparse = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'cnn_sparse_tdl_64.keras'), compile=False)
    m_lstm   = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'cnn_lstm_tdl_64.keras'), compile=False)
    m_sadlcs = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'sa_dlcs_tdl_64.keras'), compile=False)
    m_ncrn   = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'ncrn_tdl_64.keras'),
                                          custom_objects={'LastFrame': LastFrame}, compile=False)
    m_ncrt   = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'ncrt_tdl_64.keras'), compile=False)
    print("All 5 neural models loaded successfully.")

    n_samples = 100
    batch_size = 20
    np.random.seed(42)

    rows = []
    raw_blend_rows = []

    for snr in SNR_RANGE:
        t0 = time.time()
        ns = np.sqrt(10 ** (-snr / 10.0))

        # 1. Generate data
        H, HN, HL, HR, SP = [], [], [], [], []
        for _ in range(n_samples):
            h = gen_tdl_random(64, 64)
            noise = np.random.normal(0, ns, h.shape).astype(np.float32)
            hn = (h + noise).astype(np.float32)
            H.append(h)
            HN.append(hn)
            HL.append(hn[-1])
            HR.append(h[-1])
            SP.append(float(np.mean(h**2)))

        HN_arr = np.array(HN, dtype=np.float32)
        HL_arr = np.array(HL, dtype=np.float32)
        
        # Prepare inputs for neural networks
        # CNN-Sparse: hl * mask
        SPARSE_IN = []
        for i in range(n_samples):
            mask = cnnsparse_mask(HL_arr[i], k=500)
            SPARSE_IN.append(HL_arr[i] * mask)
        SPARSE_IN = np.array(SPARSE_IN, dtype=np.float32)

        # SNR conditioned input for NCRN and NCRT
        X_SNR = np.array([add_snr_channel(hn, snr) for hn in HN], dtype=np.float32)

        # Batch predictions
        p_sparse = m_sparse.predict(SPARSE_IN, batch_size=batch_size, verbose=0)
        p_lstm   = m_lstm.predict(HN_arr, batch_size=batch_size, verbose=0)
        p_sadlcs_raw = m_sadlcs.predict(HN_arr, batch_size=batch_size, verbose=0)
        p_ncrn_raw   = m_ncrn.predict(X_SNR, batch_size=batch_size, verbose=0)
        p_ncrt_raw   = m_ncrt.predict(X_SNR, batch_size=batch_size, verbose=0)

        # Accumulate squared errors
        den = 0.0
        sq = {m: 0.0 for m in METHODS}

        for i in range(n_samples):
            hr = HR[i]
            hl = HL_arr[i]
            sp = SP[i]
            den += np.sum(hr ** 2)

            # LS
            sq['ls'] += np.sum((hr - hl) ** 2)

            # MMSE oracle
            pred_mmse = mmse_estimate(hl, noise_var=ns**2, Nt=64, sp=sp)
            sq['mmse'] += np.sum((hr - pred_mmse) ** 2)

            # CNN-Sparse
            sq['cnn_sparse'] += np.sum((hr - p_sparse[i]) ** 2)

            # CNN-LSTM
            sq['cnn_lstm'] += np.sum((hr - p_lstm[i]) ** 2)

            # Sa-DLCS
            pred_sa = sadlcs_postprocess(p_sadlcs_raw[i], hl)
            sq['sa_dlcs'] += np.sum((hr - pred_sa) ** 2)

            # NCRN (blended)
            pred_ncrn = snr_aware_blend(p_ncrn_raw[i], hl, snr_db_known=snr)
            sq['ncrn'] += np.sum((hr - pred_ncrn) ** 2)

            # NCRN (raw: alpha=1.0)
            sq['ncrn_raw'] += np.sum((hr - (hl + p_ncrn_raw[i])) ** 2)

            # NCRT (blended)
            pred_ncrt = snr_aware_blend(p_ncrt_raw[i], hl, snr_db_known=snr)
            sq['ncrt'] += np.sum((hr - pred_ncrt) ** 2)

            # NCRT (raw: alpha=1.0)
            sq['ncrt_raw'] += np.sum((hr - (hl + p_ncrt_raw[i])) ** 2)

        row = {
            'SNR (dB)': snr,
            'CNN-LSTM NMSE (dB)': f"{10 * np.log10(sq['cnn_lstm'] / den):+.2f}",
            'CNN-Sparse NMSE (dB)': f"{10 * np.log10(sq['cnn_sparse'] / den):+.2f}",
            'LS NMSE (dB)': f"{10 * np.log10(sq['ls'] / den):+.2f}",
            'Wiener/MMSE (oracle) NMSE (dB)': f"{10 * np.log10(sq['mmse'] / den):+.2f}",
            'Sa-DLCS NMSE (dB)': f"{10 * np.log10(sq['sa_dlcs'] / den):+.2f}",
            'NCRN (proposed) NMSE (dB)': f"{10 * np.log10(sq['ncrn'] / den):+.2f}",
            'NCRT (proposed) NMSE (dB)': f"{10 * np.log10(sq['ncrt'] / den):+.2f}",
        }
        rows.append(row)

        raw_blend_rows.append({
            'snr': snr,
            'ls_db': float(10 * np.log10(sq['ls'] / den)),
            'ncrt_raw_db': float(10 * np.log10(sq['ncrt_raw'] / den)),
            'ncrt_blended_db': float(10 * np.log10(sq['ncrt'] / den)),
            'ncrn_raw_db': float(10 * np.log10(sq['ncrn_raw'] / den)),
            'ncrn_blended_db': float(10 * np.log10(sq['ncrn'] / den)),
        })

        print(f"SNR {snr:>3d} dB [{time.time()-t0:.1f}s] | "
              f"LS: {row['LS NMSE (dB)']} | "
              f"MMSE: {row['Wiener/MMSE (oracle) NMSE (dB)']} | "
              f"LSTM: {row['CNN-LSTM NMSE (dB)']} | "
              f"Sparse: {row['CNN-Sparse NMSE (dB)']} | "
              f"Sa-DLCS: {row['Sa-DLCS NMSE (dB)']} | "
              f"NCRN: {row['NCRN (proposed) NMSE (dB)']} | "
              f"NCRT: {row['NCRT (proposed) NMSE (dB)']} | "
              f"NCRT-Raw: {10 * np.log10(sq['ncrt_raw'] / den):+.2f} dB | "
              f"NCRN-Raw: {10 * np.log10(sq['ncrn_raw'] / den):+.2f} dB")

    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(ROOT, 'outputs', 'csv', 'combined_test_tdl_64_rerun.csv')
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved lockstep unified evaluation to: {out_csv}")

    df_rb = pd.DataFrame(raw_blend_rows)
    out_rb = os.path.join(ROOT, 'outputs', 'csv', 'eval_ncrt_combined_test_raw_blended.csv')
    df_rb.to_csv(out_rb, index=False)
    print(f"Saved raw vs blended evaluation to: {out_rb}")
    print("\n" + df_out.to_string(index=False))
    print("\nRaw vs Blended:")
    print(df_rb.to_string(index=False))
    return df_out, df_rb

if __name__ == '__main__':
    main()

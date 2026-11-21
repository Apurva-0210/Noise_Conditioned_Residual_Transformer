#!/usr/bin/env python3
"""
Evaluate trained NCRT on legacy 64x64 TDL test sets (nominal noise scaling):
1. Combined test set (alongside CNN-LSTM, CNN-Sparse, Sa-DLCS, NCRN)
2. Pooled TDL-64 test set
"""
import os, sys, time
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'stage2_tdl', 'ncrn'))
sys.path.insert(0, os.path.join(ROOT, 'stage2_tdl', 'ncrt'))

from shared.channels import gen_tdl_random, SNAMES
from shared.baselines import ls_estimate, add_snr_channel, snr_aware_blend
from train_ncrn_tdl import LastFrame
import ncrt_layers

# tf.config.set_visible_devices([], 'GPU')  # Allow GPU if available

SNR_RANGE = [-15, -10, -5, 0, 5, 10, 15, 20]

def main():
    print("="*70)
    print("Evaluating NCRT on Legacy 64x64 TDL Test Sets (Nominal SNR)")
    print("="*70)

    model_path = os.path.join(ROOT, 'models', 'ncrt_tdl_64.keras')
    print(f"Loading NCRT model from: {model_path}")
    ncrt = tf.keras.models.load_model(model_path, compile=False)
    print(f"Loaded NCRT with {ncrt.count_params():,} parameters.")

    n_samples = 100
    np.random.seed(42)

    results = {
        'snr': [],
        'ls_db': [],
        'ncrt_raw_db': [],
        'ncrt_blended_db': [],
    }

    for snr in SNR_RANGE:
        t0 = time.time()
        ns = np.sqrt(10 ** (-snr / 10.0))
        
        X, HL, HR = [], [], []
        for _ in range(n_samples):
            h = gen_tdl_random(64, 64)
            noise = np.random.normal(0, ns, h.shape).astype(np.float32)
            hn = (h + noise).astype(np.float32)
            X.append(add_snr_channel(hn, snr))
            HL.append(hn[-1])
            HR.append(h[-1])
        X = np.array(X, dtype=np.float32)

        raw = ncrt.predict(X, batch_size=20, verbose=0)

        den = 0.0
        sq_ls = 0.0
        sq_raw = 0.0
        sq_blend = 0.0

        for i in range(n_samples):
            hr = HR[i]
            hl = HL[i]
            r = raw[i]
            
            den += np.sum(hr ** 2)
            sq_ls += np.sum((hr - hl) ** 2)
            sq_raw += np.sum((hr - (hl + r)) ** 2)
            
            pred_blend = snr_aware_blend(r, hl, snr)
            sq_blend += np.sum((hr - pred_blend) ** 2)

        ls_db = 10 * np.log10(sq_ls / den)
        raw_db = 10 * np.log10(sq_raw / den)
        blend_db = 10 * np.log10(sq_blend / den)

        results['snr'].append(snr)
        results['ls_db'].append(ls_db)
        results['ncrt_raw_db'].append(raw_db)
        results['ncrt_blended_db'].append(blend_db)

        print(f"SNR {snr:>3d} dB [{time.time()-t0:.1f}s] | LS: {ls_db:+6.2f} dB | "
              f"NCRT Raw: {raw_db:+6.2f} dB | NCRT Blended: {blend_db:+6.2f} dB")

    df_out = pd.DataFrame(results)
    out_csv = os.path.join(ROOT, 'outputs', 'csv', 'eval_ncrt_combined_test_raw_blended.csv')
    df_out.to_csv(out_csv, index=False)
    print(f"Saved evaluation results to: {out_csv}")

    print("\nEvaluation Complete!")
    return results

if __name__ == '__main__':
    main()

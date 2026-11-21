#!/usr/bin/env python3
"""
Evaluate Protocol-Matched Baselines and NCRT Ablations across 10 3GPP TR 38.901 Scenarios.
Outputs per-scenario summaries and a macro-averaged ten-scenario CSV.
"""
import os, sys, time
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'stage2_tdl', 'ncrn'))
sys.path.insert(0, os.path.join(ROOT, 'stage2_tdl', 'ncrt'))

from shared.channels import gen_tdl_channel, SCENARIOS, SNAMES
from shared.baselines import add_snr_channel, snr_aware_blend, sadlcs_postprocess, cnnsparse_mask
from train_ncrn_tdl import LastFrame
import ncrt_layers

MODELS_DIR = os.path.join(ROOT, 'models', 'ablations')
SNR_RANGE = [-15, -10, -5, 0, 5, 10, 15, 20]
N_SAMPLES = 25  # per scenario per SNR

def main():
    customs = {'LastFrame': LastFrame}
    model_ids = [
        'cnn_sparse_matched', 'cnn_lstm_matched', 'sa_dlcs_matched',
        'cnn_lstm_resid_small', 'cnn_lstm_resid_large',
        'ncrt_no_rms', 'ncrt_no_adaln', 'ncrt_no_temp_attn',
        'ncrt_no_axial_attn', 'ncrt_no_toeplitz'
    ]
    
    # Load all models (seed 42)
    models = {}
    for mid in model_ids:
        p = os.path.join(MODELS_DIR, f"{mid}_seed42.keras")
        models[mid] = tf.keras.models.load_model(p, custom_objects=customs, compile=False)
        
    # Also load full NCRN and NCRT for reference
    ncrn = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'ncrn_tdl_64.keras'),
                                      custom_objects=customs, compile=False)
    ncrt = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'ncrt_tdl_64.keras'),
                                      custom_objects=customs, compile=False)
    
    # Dictionary to accumulate NMSE per scenario and overall
    # Keys will be column names
    col_names = [
        'LS', 'CNN-Sparse (matched)', 'CNN-LSTM (matched)', 'Sa-DLCS (matched)',
        'CNN-LSTM-resid (18k, raw)', 'CNN-LSTM-resid (18k, blend)',
        'CNN-LSTM-resid (1.2M, raw)', 'CNN-LSTM-resid (1.2M, blend)',
        'NCRN (raw)', 'NCRN (blend)',
        'NCRT-full (raw)', 'NCRT-full (blend)',
        'NCRT-no_rms (raw)', 'NCRT-no_rms (blend)',
        'NCRT-no_adaln (raw)', 'NCRT-no_adaln (blend)',
        'NCRT-no_temp_attn (raw)', 'NCRT-no_temp_attn (blend)',
        'NCRT-no_axial_attn (raw)', 'NCRT-no_axial_attn (blend)',
        'NCRT-no_toeplitz (raw)', 'NCRT-no_toeplitz (blend)'
    ]
    
    scenario_results = {sc: {col: {s: 0.0 for s in SNR_RANGE} for col in col_names} for sc in SNAMES}
    macro_results = {col: {s: 0.0 for s in SNR_RANGE} for col in col_names}
    
    print(f"Starting evaluation across 10 scenarios ({N_SAMPLES} samples/scenario/SNR)...")
    np.random.seed(42)
    
    for sc in SNAMES:
        t0 = time.time()
        for snr in SNR_RANGE:
            ns = np.sqrt(10 ** (-snr / 10.0))
            H, HN, HL, HR = [], [], [], []
            for _ in range(N_SAMPLES):
                h = gen_tdl_channel(sc, 64, 64)
                noise = np.random.normal(0, ns, h.shape).astype(np.float32)
                hn = (h + noise).astype(np.float32)
                H.append(h); HN.append(hn); HL.append(hn[-1]); HR.append(h[-1])
                
            den = sum(np.sum(hr**2) for hr in HR)
            sq_ls = sum(np.sum((hr - hl)**2) for hr, hl in zip(HR, HL))
            scenario_results[sc]['LS'][snr] = 10 * np.log10(sq_ls / den)
            
            # 1. CNN-Sparse
            X_sp = np.array([np.concatenate([cnnsparse_mask(hl, 500),
                                            np.full((64, 64, 1), snr/30.0, np.float32)], axis=-1)
                             for hl in HL], dtype=np.float32)
            p_sp = models['cnn_sparse_matched'].predict(X_sp, batch_size=25, verbose=0)
            sq = sum(np.sum((hr - pred)**2) for hr, pred in zip(HR, p_sp))
            scenario_results[sc]['CNN-Sparse (matched)'][snr] = 10 * np.log10(sq / den)
            
            # Common 10-step inputs with SNR channel
            X_10 = np.array([add_snr_channel(hn, snr) for hn in HN], dtype=np.float32)
            
            # 2. CNN-LSTM matched
            p_cl = models['cnn_lstm_matched'].predict(X_10, batch_size=25, verbose=0)
            sq = sum(np.sum((hr - pred)**2) for hr, pred in zip(HR, p_cl))
            scenario_results[sc]['CNN-LSTM (matched)'][snr] = 10 * np.log10(sq / den)
            
            # 3. Sa-DLCS matched
            p_sa_raw = models['sa_dlcs_matched'].predict(X_10, batch_size=25, verbose=0)
            sq = sum(np.sum((hr - sadlcs_postprocess(pr, hl))**2) for hr, pr, hl in zip(HR, p_sa_raw, HL))
            scenario_results[sc]['Sa-DLCS (matched)'][snr] = 10 * np.log10(sq / den)
            
            # 4. CNN-LSTM-resid small (18k)
            pr_18k = models['cnn_lstm_resid_small'].predict(X_10, batch_size=25, verbose=0)
            sq_raw = sum(np.sum((hr - (hl + pr))**2) for hr, pr, hl in zip(HR, pr_18k, HL))
            sq_b = sum(np.sum((hr - snr_aware_blend(pr, hl, snr))**2) for hr, pr, hl in zip(HR, pr_18k, HL))
            scenario_results[sc]['CNN-LSTM-resid (18k, raw)'][snr] = 10 * np.log10(sq_raw / den)
            scenario_results[sc]['CNN-LSTM-resid (18k, blend)'][snr] = 10 * np.log10(sq_b / den)
            
            # 5. CNN-LSTM-resid large (1.2M)
            pr_12m = models['cnn_lstm_resid_large'].predict(X_10, batch_size=25, verbose=0)
            sq_raw = sum(np.sum((hr - (hl + pr))**2) for hr, pr, hl in zip(HR, pr_12m, HL))
            sq_b = sum(np.sum((hr - snr_aware_blend(pr, hl, snr))**2) for hr, pr, hl in zip(HR, pr_12m, HL))
            scenario_results[sc]['CNN-LSTM-resid (1.2M, raw)'][snr] = 10 * np.log10(sq_raw / den)
            scenario_results[sc]['CNN-LSTM-resid (1.2M, blend)'][snr] = 10 * np.log10(sq_b / den)
            
            # 6. NCRN reference
            pr_ncrn = ncrn.predict(X_10, batch_size=25, verbose=0)
            sq_raw = sum(np.sum((hr - (hl + pr))**2) for hr, pr, hl in zip(HR, pr_ncrn, HL))
            sq_b = sum(np.sum((hr - snr_aware_blend(pr, hl, snr))**2) for hr, pr, hl in zip(HR, pr_ncrn, HL))
            scenario_results[sc]['NCRN (raw)'][snr] = 10 * np.log10(sq_raw / den)
            scenario_results[sc]['NCRN (blend)'][snr] = 10 * np.log10(sq_b / den)
            
            # 7. NCRT full reference
            pr_ncrt = ncrt.predict(X_10, batch_size=25, verbose=0)
            sq_raw = sum(np.sum((hr - (hl + pr))**2) for hr, pr, hl in zip(HR, pr_ncrt, HL))
            sq_b = sum(np.sum((hr - snr_aware_blend(pr, hl, snr))**2) for hr, pr, hl in zip(HR, pr_ncrt, HL))
            scenario_results[sc]['NCRT-full (raw)'][snr] = 10 * np.log10(sq_raw / den)
            scenario_results[sc]['NCRT-full (blend)'][snr] = 10 * np.log10(sq_b / den)
            
            # 8. NCRT ablations
            for ab in ['no_rms', 'no_adaln', 'no_temp_attn', 'no_axial_attn', 'no_toeplitz']:
                pr_ab = models[f'ncrt_{ab}'].predict(X_10, batch_size=25, verbose=0)
                sq_raw = sum(np.sum((hr - (hl + pr))**2) for hr, pr, hl in zip(HR, pr_ab, HL))
                sq_b = sum(np.sum((hr - snr_aware_blend(pr, hl, snr))**2) for hr, pr, hl in zip(HR, pr_ab, HL))
                scenario_results[sc][f'NCRT-{ab} (raw)'][snr] = 10 * np.log10(sq_raw / den)
                scenario_results[sc][f'NCRT-{ab} (blend)'][snr] = 10 * np.log10(sq_b / den)
                
        print(f"  Finished scenario {sc:10s} in {time.time()-t0:.1f}s")
        
        # Save individual scenario CSV
        sc_rows = []
        for snr in SNR_RANGE:
            row = {'SNR (dB)': snr}
            for col in col_names:
                row[col] = f"{scenario_results[sc][col][snr]:+.2f}"
            sc_rows.append(row)
        sc_df = pd.DataFrame(sc_rows)
        sc_df.to_csv(os.path.join(ROOT, 'outputs', 'csv', f'ablation_ladder_{sc}.csv'), index=False)
        
    # Compute macro-average across 10 scenarios
    macro_rows = []
    for snr in SNR_RANGE:
        row = {'SNR (dB)': snr}
        for col in col_names:
            avg_val = np.mean([scenario_results[sc][col][snr] for sc in SNAMES])
            row[col] = f"{avg_val:+.2f}"
        macro_rows.append(row)
    macro_df = pd.DataFrame(macro_rows)
    macro_out = os.path.join(ROOT, 'outputs', 'csv', 'ablation_ladder_ten_scenarios.csv')
    macro_df.to_csv(macro_out, index=False)
    
    paper_macro = os.path.join(ROOT, 'paper', 'data', 'ablation_ladder_ten_scenarios.csv')
    macro_df.to_csv(paper_macro, index=False)
    print(f"\nSaved ten-scenario macro-average to {macro_out} and {paper_macro}")
    print(macro_df.to_string(index=False))

if __name__ == '__main__':
    main()

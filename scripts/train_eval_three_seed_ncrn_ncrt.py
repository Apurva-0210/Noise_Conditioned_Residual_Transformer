#!/usr/bin/env python3
"""
Train and Evaluate Headline NCRN and NCRT Across Three Training Seeds (42, 100, 2024)
on the Lockstep Combined TDL-64 Test Set (N=100, seed 42, SNR -15 to 20 dB).

Reports:
- Genuine 3-seed spread for both NCRN and NCRT
- Raw NMSE, Heuristic Blended NMSE, and Measured Plug-in Blended NMSE (alpha_hat = L/(L+R))
- NCRT lead over NCRN (Raw and Plug-in)
- Complete single-value CSVs with means, standard deviations, and individual seed columns
"""
import os, sys, time, pickle, argparse
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'stage2_tdl', 'ncrn'))
sys.path.insert(0, os.path.join(ROOT, 'stage2_tdl', 'ncrt'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from shared.channels import gen_tdl_random, TIME_STEPS
from shared.baselines import add_snr_channel, snr_aware_blend
from train_ncrn_tdl import LastFrame
from run_ablation_and_baselines_suite import (
    build_ncrt_variant, build_cnn_lstm_resid_large,
    make_matched_dataset, SNR_RANGE, SEEDS, BATCH_SIZE, EPOCHS, STEPS_PER_EPOCH
)

MODELS_DIR = os.path.join(ROOT, 'models', 'ablations')
os.makedirs(MODELS_DIR, exist_ok=True)
OUT_CSV_DIR = os.path.join(ROOT, 'outputs', 'csv')
DATA_DIR = os.path.join(ROOT, 'paper', 'data')
os.makedirs(OUT_CSV_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def train_ncrt_seed(seed, epochs=EPOCHS, steps=STEPS_PER_EPOCH):
    save_path = os.path.join(MODELS_DIR, f"ncrt_full_seed{seed}.keras")
    if os.path.exists(save_path):
        print(f"[NCRT Seed {seed}] Already exists at {save_path}, skipping training.")
        return save_path

    print(f"\n{'='*70}\nTraining Full NCRT (Seed {seed}) | Epochs: {epochs}, Steps: {steps}\n{'='*70}")
    np.random.seed(seed); tf.random.set_seed(seed)
    model = build_ncrt_variant(ablation=None)
    opt = tf.keras.optimizers.Adam(3e-4, clipnorm=1.0)
    model.compile(optimizer=opt, loss='mse')
    print(f"Model {model.name} has {model.count_params():,} parameters.")

    ds = make_matched_dataset(64, 64, BATCH_SIZE, epochs, steps,
                              target_type='residual', is_sparse=False)

    cbs = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=12, restore_best_weights=True, verbose=1),
    ]

    t0 = time.time()
    model.fit(ds, steps_per_epoch=steps, epochs=epochs, callbacks=cbs, verbose=2)
    model.save(save_path)
    print(f"Saved Full NCRT (Seed {seed}) to {save_path} in {time.time()-t0:.1f}s.")
    return save_path


def evaluate_three_seed_headline(ncrn_models, ncrt_models, n_samples=100):
    print(f"\n{'='*75}\nEvaluating 3-Seed Headline NCRN vs NCRT on Lockstep TDL-64 Test Set\n{'='*75}")
    np.random.seed(42)

    # 1. Generate Lockstep Test Channels & Noise Draws
    test_channels = {}
    for snr in SNR_RANGE:
        ns = np.sqrt(10 ** (-snr / 10.0))
        H, HN, HL, HR = [], [], [], []
        for _ in range(n_samples):
            h = gen_tdl_random(64, 64)
            noise = np.random.normal(0, ns, h.shape).astype(np.float32)
            hn = (h + noise).astype(np.float32)
            H.append(h); HN.append(hn); HL.append(hn[-1]); HR.append(h[-1])
        test_channels[snr] = (np.array(H), np.array(HN), np.array(HL), np.array(HR))

    detailed_rows = []
    summary_rows = []

    for snr in SNR_RANGE:
        H_arr, HN_arr, HL_arr, HR_arr = test_channels[snr]
        den = np.sum(HR_arr ** 2)
        sq_ls = np.sum((HR_arr - HL_arr) ** 2)
        ls_nmse = 10 * np.log10(sq_ls / den)
        L_lin = sq_ls / den

        X_SNR = np.array([add_snr_channel(hn, snr) for hn in HN_arr], dtype=np.float32)

        # Evaluate NCRN across 3 seeds
        ncrn_raw_seeds = []
        ncrn_blend_seeds = []
        ncrn_plugin_seeds = []
        ncrn_alpha_seeds = []

        for s, m in zip(SEEDS, ncrn_models):
            p = m.predict(X_SNR, batch_size=25, verbose=0)
            h_raw = HL_arr + p
            sq_raw = np.sum((HR_arr - h_raw) ** 2)
            nmse_raw = 10 * np.log10(sq_raw / den)
            R_lin = sq_raw / den

            h_blend = np.array([snr_aware_blend(pr, hl, snr) for pr, hl in zip(p, HL_arr)])
            nmse_blend = 10 * np.log10(np.sum((HR_arr - h_blend) ** 2) / den)

            alpha_hat = L_lin / (L_lin + R_lin)
            h_plugin = HL_arr + alpha_hat * p
            nmse_plugin = 10 * np.log10(np.sum((HR_arr - h_plugin) ** 2) / den)

            ncrn_raw_seeds.append(nmse_raw)
            ncrn_blend_seeds.append(nmse_blend)
            ncrn_plugin_seeds.append(nmse_plugin)
            ncrn_alpha_seeds.append(alpha_hat)

        # Evaluate NCRT across 3 seeds
        ncrt_raw_seeds = []
        ncrt_blend_seeds = []
        ncrt_plugin_seeds = []
        ncrt_alpha_seeds = []

        for s, m in zip(SEEDS, ncrt_models):
            p = m.predict(X_SNR, batch_size=25, verbose=0)
            h_raw = HL_arr + p
            sq_raw = np.sum((HR_arr - h_raw) ** 2)
            nmse_raw = 10 * np.log10(sq_raw / den)
            R_lin = sq_raw / den

            h_blend = np.array([snr_aware_blend(pr, hl, snr) for pr, hl in zip(p, HL_arr)])
            nmse_blend = 10 * np.log10(np.sum((HR_arr - h_blend) ** 2) / den)

            alpha_hat = L_lin / (L_lin + R_lin)
            h_plugin = HL_arr + alpha_hat * p
            nmse_plugin = 10 * np.log10(np.sum((HR_arr - h_plugin) ** 2) / den)

            ncrt_raw_seeds.append(nmse_raw)
            ncrt_blend_seeds.append(nmse_blend)
            ncrt_plugin_seeds.append(nmse_plugin)
            ncrt_alpha_seeds.append(alpha_hat)

        # Calculate statistics
        ncrn_raw_mean = np.mean(ncrn_raw_seeds)
        ncrn_raw_std = np.std(ncrn_raw_seeds)
        ncrn_plugin_mean = np.mean(ncrn_plugin_seeds)
        ncrn_plugin_std = np.std(ncrn_plugin_seeds)

        ncrt_raw_mean = np.mean(ncrt_raw_seeds)
        ncrt_raw_std = np.std(ncrt_raw_seeds)
        ncrt_plugin_mean = np.mean(ncrt_plugin_seeds)
        ncrt_plugin_std = np.std(ncrt_plugin_seeds)

        raw_lead = ncrn_raw_mean - ncrt_raw_mean  # positive means NCRT outperforms NCRN
        plugin_lead = ncrn_plugin_mean - ncrt_plugin_mean

        det_row = {
            'SNR (dB)': snr,
            'LS': f"{ls_nmse:+.2f}",
            'NCRN_raw_s42': f"{ncrn_raw_seeds[0]:+.2f}",
            'NCRN_raw_s100': f"{ncrn_raw_seeds[1]:+.2f}",
            'NCRN_raw_s2024': f"{ncrn_raw_seeds[2]:+.2f}",
            'NCRN_raw_mean': f"{ncrn_raw_mean:+.2f}",
            'NCRN_raw_std': f"{ncrn_raw_std:.3f}",
            'NCRN_plugin_mean': f"{ncrn_plugin_mean:+.2f}",
            'NCRN_plugin_std': f"{ncrn_plugin_std:.3f}",
            'NCRT_raw_s42': f"{ncrt_raw_seeds[0]:+.2f}",
            'NCRT_raw_s100': f"{ncrt_raw_seeds[1]:+.2f}",
            'NCRT_raw_s2024': f"{ncrt_raw_seeds[2]:+.2f}",
            'NCRT_raw_mean': f"{ncrt_raw_mean:+.2f}",
            'NCRT_raw_std': f"{ncrt_raw_std:.3f}",
            'NCRT_plugin_mean': f"{ncrt_plugin_mean:+.2f}",
            'NCRT_plugin_std': f"{ncrt_plugin_std:.3f}",
            'NCRT_Raw_Lead_dB': f"{raw_lead:+.2f}",
            'NCRT_Plugin_Lead_dB': f"{plugin_lead:+.2f}",
        }
        detailed_rows.append(det_row)

        print(f"SNR {snr:>3} dB: LS={ls_nmse:+6.2f} | "
              f"NCRN Raw={ncrn_raw_mean:+6.2f}±{ncrn_raw_std:.2f} (Plug-in={ncrn_plugin_mean:+6.2f}) | "
              f"NCRT Raw={ncrt_raw_mean:+6.2f}±{ncrt_raw_std:.2f} (Plug-in={ncrt_plugin_mean:+6.2f}) | "
              f"Lead: Raw={raw_lead:+5.2f} dB, Plug-in={plugin_lead:+5.2f} dB")

    df_det = pd.DataFrame(detailed_rows)
    df_det.to_csv(os.path.join(OUT_CSV_DIR, 'headline_three_seed_ncrn_ncrt_eval.csv'), index=False)
    df_det.to_csv(os.path.join(DATA_DIR, 'headline_three_seed_ncrn_ncrt_eval.csv'), index=False)
    print(f"\nSaved headline three-seed evaluation to {os.path.join(OUT_CSV_DIR, 'headline_three_seed_ncrn_ncrt_eval.csv')}")
    return df_det


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    if not args.eval_only:
        for s in SEEDS:
            train_ncrt_seed(s)

    # Load 3 seeds of NCRN (cnn_lstm_resid_large)
    customs = {'LastFrame': LastFrame}
    ncrn_models = []
    for s in SEEDS:
        p = os.path.join(MODELS_DIR, f"cnn_lstm_resid_large_seed{s}.keras")
        print(f"Loading NCRN seed {s} from {p}")
        ncrn_models.append(tf.keras.models.load_model(p, custom_objects=customs, compile=False))

    # Load 3 seeds of NCRT (ncrt_full)
    ncrt_models = []
    for s in SEEDS:
        p = os.path.join(MODELS_DIR, f"ncrt_full_seed{s}.keras")
        print(f"Loading NCRT seed {s} from {p}")
        ncrt_models.append(tf.keras.models.load_model(p, compile=False))

    evaluate_three_seed_headline(ncrn_models, ncrt_models)


if __name__ == '__main__':
    main()

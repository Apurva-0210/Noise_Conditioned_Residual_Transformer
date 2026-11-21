#!/usr/bin/env python3
"""
Evaluate and Train NCRN and NCRT at 9 us Symbol Spacing (Full-Slot 5G NR Numerology).
Demonstrates whether NCRT tracks rapid Doppler phase evolution (up to 90 deg across window)
or merely performs temporal smoothing.
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
from shared.baselines import add_snr_channel, snr_aware_blend
from train_ncrn_tdl import LastFrame
import ncrt_layers

from scripts.run_ablation_and_baselines_suite import (
    build_ncrt_variant, build_cnn_lstm_resid_large
)

SNR_RANGE = [-15, -10, -5, 0, 5, 10, 15, 20]
DT_9US = 9e-6
N_SAMPLES = 100
BATCH_SIZE = 4
EPOCHS = 30
STEPS_PER_EPOCH = 50
MODELS_DIR = os.path.join(ROOT, 'models', 'ablations')
os.makedirs(MODELS_DIR, exist_ok=True)

def make_dataset_9us(Nr=64, Nt=64, batch_size=4, total_epochs=30, steps_per_epoch=50):
    total = total_epochs * steps_per_epoch
    step = [0]
    def _gen():
        while True:
            prog = step[0] / total
            snr_min = (5.0 if prog < 0.33 else 0.0 if prog < 0.67 else -5.0)
            snr_max = 15.0
            X, Y = [], []
            for _ in range(batch_size):
                h = gen_tdl_random(Nr, Nt, dt=DT_9US)
                snr_db = float(np.random.uniform(snr_min, snr_max))
                sp = float(np.mean(h**2)) + 1e-10
                ns = np.sqrt(sp * 10**(-snr_db/10))
                hn = (h + np.random.normal(0, ns, h.shape)).astype(np.float32)
                x_in = add_snr_channel(hn, snr_db)
                y_out = (h[-1] - hn[-1]).astype(np.float32)
                X.append(x_in)
                Y.append(y_out)
            step[0] += 1
            yield np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
            
    return tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec((batch_size, TIME_STEPS, Nr, Nt, 3), tf.float32),
            tf.TensorSpec((batch_size, Nr, Nt, 2), tf.float32),
        )
    ).prefetch(tf.data.AUTOTUNE)

def train_models_9us():
    np.random.seed(42); tf.random.set_seed(42)
    
    # 1. Train NCRT-9us
    p_ncrt = os.path.join(MODELS_DIR, 'ncrt_9us.keras')
    if not os.path.exists(p_ncrt):
        print("\n" + "="*70)
        print("Training NCRT at 9 us Pilot Spacing (3-Phase Curriculum, 30 Epochs)")
        print("="*70)
        m_ncrt = build_ncrt_variant()
        opt = tf.keras.optimizers.Adam(3e-4, clipnorm=1.0)
        m_ncrt.compile(optimizer=opt, loss='mse')
        ds = make_dataset_9us(batch_size=BATCH_SIZE, total_epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
        t0 = time.time()
        m_ncrt.fit(ds, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, verbose=2)
        m_ncrt.save(p_ncrt)
        print(f"Saved NCRT-9us in {time.time()-t0:.1f}s.")
    else:
        print(f"NCRT-9us already exists at {p_ncrt}")
        m_ncrt = tf.keras.models.load_model(p_ncrt, compile=False)
        
    # 2. Train NCRN-9us
    p_ncrn = os.path.join(MODELS_DIR, 'ncrn_9us.keras')
    if not os.path.exists(p_ncrn):
        print("\n" + "="*70)
        print("Training NCRN at 9 us Pilot Spacing (3-Phase Curriculum, 30 Epochs)")
        print("="*70)
        m_ncrn = build_cnn_lstm_resid_large()
        opt = tf.keras.optimizers.Adam(3e-4)
        m_ncrn.compile(optimizer=opt, loss='mse')
        ds = make_dataset_9us(batch_size=BATCH_SIZE, total_epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
        t0 = time.time()
        m_ncrn.fit(ds, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, verbose=2)
        m_ncrn.save(p_ncrn)
        print(f"Saved NCRN-9us in {time.time()-t0:.1f}s.")
    else:
        print(f"NCRN-9us already exists at {p_ncrn}")
        m_ncrn = tf.keras.models.load_model(p_ncrn, custom_objects={'LastFrame': LastFrame}, compile=False)
        
    return m_ncrt, m_ncrn

def main():
    customs = {'LastFrame': LastFrame}
    
    # 1. Load base 1us models
    print("Loading base 1us NCRN and NCRT models...")
    base_ncrn = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'ncrn_tdl_64.keras'),
                                           custom_objects=customs, compile=False)
    base_ncrt = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'ncrt_tdl_64.keras'),
                                           compile=False)
    
    # 2. Train or load 9us models
    m_ncrt_9us, m_ncrn_9us = train_models_9us()
    
    # 3. Generate lockstep 9us evaluation data
    print("\n" + "="*75)
    print("Generating Lockstep 9 us Evaluation Channels (N=100, seed 42)")
    print("="*75)
    np.random.seed(42)
    test_data_9us = {}
    for snr in SNR_RANGE:
        ns = np.sqrt(10 ** (-snr / 10.0))
        H, HN, HL, HR = [], [], [], []
        for _ in range(N_SAMPLES):
            h = gen_tdl_random(64, 64, dt=DT_9US)
            noise = np.random.normal(0, ns, h.shape).astype(np.float32)
            hn = (h + noise).astype(np.float32)
            H.append(h); HN.append(hn); HL.append(hn[-1]); HR.append(h[-1])
        test_data_9us[snr] = (H, HN, HL, HR)
        
    # 4. Evaluate models
    rows = []
    print("\n" + "="*75)
    print("Evaluating Estimators at 9 us Pilot Spacing (Full-Slot 5G NR Grid)")
    print("="*75)
    
    for snr in SNR_RANGE:
        H, HN, HL, HR = test_data_9us[snr]
        den = sum(np.sum(hr**2) for hr in HR)
        sq_ls = sum(np.sum((hr - hl)**2) for hr, hl in zip(HR, HL))
        ls_nmse = 10 * np.log10(sq_ls / den)
        
        X_10 = np.array([add_snr_channel(hn, snr) for hn in HN], dtype=np.float32)
        
        # Predictions
        # A. Base (1us trained) models tested zero-shot on 9us
        p_base_ncrn = base_ncrn.predict(X_10, batch_size=25, verbose=0)
        p_base_ncrt = base_ncrt.predict(X_10, batch_size=25, verbose=0)
        
        # B. Retrained (9us trained) models
        p_retrained_ncrn = m_ncrn_9us.predict(X_10, batch_size=25, verbose=0)
        p_retrained_ncrt = m_ncrt_9us.predict(X_10, batch_size=25, verbose=0)
        
        def calc_nmse(pred_res, is_blend=False):
            if is_blend:
                ests = [snr_aware_blend(pr, hl, snr) for pr, hl in zip(pred_res, HL)]
            else:
                ests = [hl + pr for pr, hl in zip(pred_res, HL)]
            sq = sum(np.sum((hr - e)**2) for hr, e in zip(HR, ests))
            return 10 * np.log10(sq / den)
            
        nmse_base_ncrn_raw = calc_nmse(p_base_ncrn, False)
        nmse_base_ncrn_b   = calc_nmse(p_base_ncrn, True)
        nmse_base_ncrt_raw = calc_nmse(p_base_ncrt, False)
        nmse_base_ncrt_b   = calc_nmse(p_base_ncrt, True)
        
        nmse_9us_ncrn_raw = calc_nmse(p_retrained_ncrn, False)
        nmse_9us_ncrn_b   = calc_nmse(p_retrained_ncrn, True)
        nmse_9us_ncrt_raw = calc_nmse(p_retrained_ncrt, False)
        nmse_9us_ncrt_b   = calc_nmse(p_retrained_ncrt, True)
        
        row = {
            'SNR (dB)': snr,
            'LS NMSE (dB)': f"{ls_nmse:+.2f}",
            'NCRN_1us_raw (zero-shot)': f"{nmse_base_ncrn_raw:+.2f}",
            'NCRN_1us_blend (zero-shot)': f"{nmse_base_ncrn_b:+.2f}",
            'NCRT_1us_raw (zero-shot)': f"{nmse_base_ncrt_raw:+.2f}",
            'NCRT_1us_blend (zero-shot)': f"{nmse_base_ncrt_b:+.2f}",
            'NCRN_9us_raw (retrained)': f"{nmse_9us_ncrn_raw:+.2f}",
            'NCRN_9us_blend (retrained)': f"{nmse_9us_ncrn_b:+.2f}",
            'NCRT_9us_raw (retrained)': f"{nmse_9us_ncrt_raw:+.2f}",
            'NCRT_9us_blend (retrained)': f"{nmse_9us_ncrt_b:+.2f}",
        }
        rows.append(row)
        print(f"SNR {snr:>3} dB | LS {ls_nmse:+6.2f} | "
              f"NCRT-9us raw {nmse_9us_ncrt_raw:+6.2f} | NCRN-9us raw {nmse_9us_ncrn_raw:+6.2f} | "
              f"NCRT-1us zero-shot {nmse_base_ncrt_raw:+6.2f}")
        
    df = pd.DataFrame(rows)
    out_csv = os.path.join(ROOT, 'outputs', 'csv', 'mobility_9us_spacing_eval.csv')
    df.to_csv(out_csv, index=False)
    paper_csv = os.path.join(ROOT, 'paper', 'data', 'mobility_9us_spacing_eval.csv')
    df.to_csv(paper_csv, index=False)
    print(f"\nSaved 9 us mobility evaluation to {out_csv} and {paper_csv}")
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()

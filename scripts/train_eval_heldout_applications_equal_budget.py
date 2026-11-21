#!/usr/bin/env python3
"""
Held-Out Partition Fine-Tuning & Evaluation with Strictly EQUAL Budgets:
- 500 training links (indices 0..499)
- 100 strictly held-out evaluation links (indices 500..599)
- Tests two controlled budget configurations:
  1. Equal Budget A: Exactly 6 epochs for both NCRN and NCRT (matching earlier in-sample NCRN)
  2. Equal Budget B: Exactly 15 epochs for both NCRN and NCRT (matching earlier in-sample NCRT)
- Evaluates across SNR -15 to 20 dB on:
  * Combined 100 held-out links
  * UAV A2G 50 held-out links
  * V2V mobility 50 held-out links
- Outputs saved to outputs/csv/application_heldout_equal_budget_eval.csv and paper/data/
"""
import os, sys, time, pickle, argparse
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'stage2_tdl', 'ncrn'))
sys.path.insert(0, os.path.join(ROOT, 'stage2_tdl', 'ncrt'))

from shared.matlab_channels import load_matlab_6g_channels, make_matlab_dataset
from shared.baselines import ls_estimate, mmse_estimate, add_snr_channel, snr_aware_blend
from train_ncrn_tdl import LastFrame, build_ncrn
from train_ncrt_tdl import build_ncrt, SkipNonFinite
import ncrt_layers

SNR_RANGE = [-15, -10, -5, 0, 5, 10, 15, 20]
TRAIN_LINKS = 500
TEST_LINKS = 100
STEPS_PER_EPOCH = 50
BS = 4

def train_ncrn(train_channels, base_weights_path, out_path, epochs):
    print(f"\n--- Fine-tuning NCRN (Budget: {epochs} epochs) ---")
    model = build_ncrn(64, 64)
    if os.path.exists(base_weights_path):
        pre = tf.keras.models.load_model(base_weights_path, custom_objects={'LastFrame': LastFrame}, compile=False)
        model.set_weights(pre.get_weights())
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse', metrics=['mae'])
    ds = make_matlab_dataset(train_channels, batch_size=BS, total_epochs=epochs, steps_per_epoch=STEPS_PER_EPOCH)
    t0 = time.time()
    model.fit(ds, steps_per_epoch=STEPS_PER_EPOCH, epochs=epochs, verbose=1)
    print(f"NCRN ({epochs} ep) complete in {time.time()-t0:.1f}s -> {out_path}")
    model.save(out_path)
    return model

def train_ncrt(train_channels, base_weights_path, out_path, epochs):
    print(f"\n--- Fine-tuning NCRT (Budget: {epochs} epochs) ---")
    net = build_ncrt(64, 64)
    model = SkipNonFinite(net.inputs, net.outputs)
    if os.path.exists(base_weights_path):
        pre = tf.keras.models.load_model(base_weights_path, compile=False)
        net.set_weights(pre.get_weights())
    # batch 2 with accum 2 gives effective batch 4
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4, clipnorm=1.0, gradient_accumulation_steps=2), loss='mse', metrics=['mae'])
    ds = make_matlab_dataset(train_channels, batch_size=2, total_epochs=epochs, steps_per_epoch=STEPS_PER_EPOCH * 2)
    t0 = time.time()
    model.fit(ds, steps_per_epoch=STEPS_PER_EPOCH * 2, epochs=epochs, verbose=1)
    print(f"NCRT ({epochs} ep) complete in {time.time()-t0:.1f}s -> {out_path}")
    net.save(out_path)
    return net

def evaluate_models(models_dict, test_channels, tag=""):
    print(f"\nEvaluating on {tag} ({len(test_channels)} links)...")
    res = []
    for snr in SNR_RANGE:
        t0 = time.time()
        H, HN, HL = [], [], []
        sigma2_n = 10 ** (-snr / 10.0)
        ns = np.sqrt(sigma2_n)
        for h in test_channels:
            noise = np.random.normal(0, ns, h.shape).astype(np.float32)
            hn = (h + noise).astype(np.float32)
            H.append(h[-1]); HN.append(hn); HL.append(hn[-1])
            
        H_arr = np.array(H, dtype=np.float32)
        HL_arr = np.array(HL, dtype=np.float32)
        HN_arr = np.array(HN, dtype=np.float32)
        
        den = np.sum(H_arr ** 2)
        nmse_ls = 10 * np.log10(np.sum((H_arr - HL_arr) ** 2) / den)
        
        X_SNR = np.array([add_snr_channel(hn, snr) for hn in HN_arr], dtype=np.float32)
        
        row = {'SNR': snr, 'Dataset': tag, 'LS': nmse_ls}
        
        for name, m in models_dict.items():
            pred = m.predict(X_SNR, batch_size=25, verbose=0)
            H_raw = HL_arr + pred
            H_blend = np.array([snr_aware_blend(pr, hl, snr) for pr, hl in zip(pred, HL_arr)])
            
            nmse_raw = 10 * np.log10(np.sum((H_arr - H_raw) ** 2) / den)
            nmse_blend = 10 * np.log10(np.sum((H_arr - H_blend) ** 2) / den)
            
            row[f'{name}_raw'] = nmse_raw
            row[f'{name}_blend'] = nmse_blend
            
        res.append(row)
        print(f"[{tag}] SNR {snr:3d} dB [{time.time()-t0:.1f}s]: LS={nmse_ls:.2f} | " + 
              " | ".join(f"{name}_raw={row[f'{name}_raw']:.2f}" for name in models_dict))
    return pd.DataFrame(res)

def main():
    uav_v2x_mat = os.path.join(ROOT, 'data', 'matlab_uav_v2x_combined.mat')
    all_channels = load_matlab_6g_channels(uav_v2x_mat, target_Nr=64, target_Nt=64)
    train_channels = all_channels[:TRAIN_LINKS]
    test_combined = all_channels[TRAIN_LINKS:]
    
    uav_mat = os.path.join(ROOT, 'data', 'matlab_uav_channels.mat')
    v2v_mat = os.path.join(ROOT, 'data', 'matlab_v2v_channels.mat')
    uav_channels = load_matlab_6g_channels(uav_mat, target_Nr=64, target_Nt=64)
    v2v_channels = load_matlab_6g_channels(v2v_mat, target_Nr=64, target_Nt=64)
    test_uav = uav_channels[250:] # 50 links
    test_v2v = v2v_channels[250:] # 50 links
    
    base_ncrn = os.path.join(ROOT, 'models', 'ncrn_tdl_64.keras')
    base_ncrt = os.path.join(ROOT, 'models', 'ncrt_tdl_64.keras')
    
    out_dir = os.path.join(ROOT, 'models', 'heldout_app')
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Budget A: 6 Epochs
    p_ncrn_ep6 = os.path.join(out_dir, 'ncrn_heldout_ep6.keras')
    p_ncrt_ep6 = os.path.join(out_dir, 'ncrt_heldout_ep6.keras')
    if not os.path.exists(p_ncrn_ep6):
        m_ncrn_ep6 = train_ncrn(train_channels, base_ncrn, p_ncrn_ep6, epochs=6)
    else:
        m_ncrn_ep6 = tf.keras.models.load_model(p_ncrn_ep6, custom_objects={'LastFrame': LastFrame}, compile=False)
        
    if not os.path.exists(p_ncrt_ep6):
        m_ncrt_ep6 = train_ncrt(train_channels, base_ncrt, p_ncrt_ep6, epochs=6)
    else:
        m_ncrt_ep6 = tf.keras.models.load_model(p_ncrt_ep6, compile=False)
        
    # 2. Budget B: 15 Epochs
    p_ncrn_ep15 = os.path.join(out_dir, 'ncrn_heldout_ep15.keras')
    p_ncrt_ep15 = os.path.join(out_dir, 'ncrt_heldout_ep15.keras')
    if not os.path.exists(p_ncrn_ep15):
        m_ncrn_ep15 = train_ncrn(train_channels, base_ncrn, p_ncrn_ep15, epochs=15)
    else:
        m_ncrn_ep15 = tf.keras.models.load_model(p_ncrn_ep15, custom_objects={'LastFrame': LastFrame}, compile=False)
        
    if not os.path.exists(p_ncrt_ep15):
        m_ncrt_ep15 = train_ncrt(train_channels, base_ncrt, p_ncrt_ep15, epochs=15)
    else:
        m_ncrt_ep15 = tf.keras.models.load_model(p_ncrt_ep15, compile=False)
        
    models_dict = {
        'NCRN_ep6': m_ncrn_ep6,
        'NCRT_ep6': m_ncrt_ep6,
        'NCRN_ep15': m_ncrn_ep15,
        'NCRT_ep15': m_ncrt_ep15,
    }
    
    np.random.seed(42)
    df_comb = evaluate_models(models_dict, test_combined, tag='Combined')
    df_uav = evaluate_models(models_dict, test_uav, tag='UAV_A2G')
    df_v2v = evaluate_models(models_dict, test_v2v, tag='V2V_Mobility')
    
    df_all = pd.concat([df_comb, df_uav, df_v2v], ignore_index=True)
    out_csv = os.path.join(ROOT, 'outputs', 'csv', 'application_heldout_equal_budget_eval.csv')
    paper_csv = os.path.join(ROOT, 'paper', 'data', 'application_heldout_equal_budget_eval.csv')
    df_all.to_csv(out_csv, index=False)
    df_all.to_csv(paper_csv, index=False)
    print(f"\nSaved equal budget evaluation to:\n  {out_csv}\n  {paper_csv}")

if __name__ == '__main__':
    main()

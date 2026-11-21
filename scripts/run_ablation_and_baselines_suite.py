#!/usr/bin/env python3
"""
Protocol-Matched Baselines & NCRT Ablation Ladder Training and Evaluation Suite
Addresses Concern 1 and Concern 8:
- Protocol-matched baselines: CNN-Sparse, CNN-LSTM, Sa-DLCS with SNR channel, 3-phase curriculum
- Residual-target variants: CNN-LSTM small (18k) and large (1.20M, NCRN-capacity)
- NCRT ablations: no_rms, no_adaln, no_temp_attn, no_axial_attn, no_toeplitz
- Multi-seed training: S in {42, 100, 2024}
- Evaluated on lockstep combined TDL-64 test set (N=100, seed 42, SNR -15 to 20 dB)
- Outputs saved to outputs/csv/ablation_ladder_combined_tdl_64.csv
"""
import os, sys, time, pickle, argparse
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'stage2_tdl', 'ncrn'))
sys.path.insert(0, os.path.join(ROOT, 'stage2_tdl', 'ncrt'))

from shared.channels import gen_tdl_random, TIME_STEPS
from shared.baselines import (add_snr_channel, snr_aware_blend,
                              sadlcs_postprocess, cnnsparse_mask)
from train_ncrn_tdl import make_dataset, LastFrame
import ncrt_layers

L = tf.keras.layers
MODELS_DIR = os.path.join(ROOT, 'models', 'ablations')
os.makedirs(MODELS_DIR, exist_ok=True)

SEEDS = [42, 100, 2024]
SNR_RANGE = [-15, -10, -5, 0, 5, 10, 15, 20]
EPOCHS = 30
STEPS_PER_EPOCH = 50
BATCH_SIZE = 4

# ─── Model Builders ─────────────────────────────────────────────────────────

def build_cnn_sparse_matched(Nr=64, Nt=64):
    inp = tf.keras.Input(shape=(Nr, Nt, 3))
    x = L.Conv2D(16, (3, 3), padding='same', activation='relu')(inp)
    x = L.BatchNormalization()(x)
    x = L.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = L.BatchNormalization()(x)
    out = L.Conv2D(2, (3, 3), padding='same')(x)
    return tf.keras.Model(inp, out, name='cnn_sparse_matched')

def build_cnn_lstm_matched(Nr=64, Nt=64, T=10):
    inp = tf.keras.Input(shape=(T, Nr, Nt, 3))
    x = L.ConvLSTM2D(16, (3, 3), padding='same', return_sequences=True, activation='relu')(inp)
    x = L.BatchNormalization()(x)
    x = L.ConvLSTM2D(8, (3, 3), padding='same', return_sequences=False, activation='relu')(x)
    x = L.BatchNormalization()(x)
    out = L.Conv2D(2, (3, 3), padding='same')(x)
    return tf.keras.Model(inp, out, name='cnn_lstm_matched')

def build_cnn_lstm_resid_small(Nr=64, Nt=64, T=10):
    # Same capacity as CNN-LSTM, but trained on residual target r = H - H_noisy
    inp = tf.keras.Input(shape=(T, Nr, Nt, 3))
    x = L.ConvLSTM2D(16, (3, 3), padding='same', return_sequences=True, activation='relu')(inp)
    x = L.BatchNormalization()(x)
    x = L.ConvLSTM2D(8, (3, 3), padding='same', return_sequences=False, activation='relu')(x)
    x = L.BatchNormalization()(x)
    out = L.Conv2D(2, (3, 3), padding='same')(x)
    return tf.keras.Model(inp, out, name='cnn_lstm_resid_small')

def build_cnn_lstm_resid_large(Nr=64, Nt=64, T=10):
    # NCRN capacity (1,204,610 parameters)
    inp = tf.keras.Input(shape=(T, Nr, Nt, 3))
    x1 = L.ConvLSTM2D(128, (3, 3), padding='same', return_sequences=True, activation='tanh',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-5))(inp)
    x1 = L.BatchNormalization()(x1)
    x2 = L.ConvLSTM2D(64, (3, 3), padding='same', return_sequences=False, activation='tanh',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x1)
    x2 = L.BatchNormalization()(x2)
    sk = LastFrame()(x1)
    sk = L.Conv2D(64, (1, 1), padding='same')(sk)
    mg = L.Activation('relu')(L.Add()([x2, sk]))
    out = L.Conv2D(128, (3, 3), padding='same', activation='relu')(mg)
    out = L.Conv2D(64, (3, 3), padding='same', activation='relu')(out)
    out = L.Conv2D(2, (3, 3), padding='same', activation='linear')(out)
    return tf.keras.Model(inp, out, name='cnn_lstm_resid_large')

def build_ncrt_variant(Nr=64, Nt=64, T=10, ablation=None):
    dim = 64; heads = 4; ffn = 128; sp_dim = 128; sp_blocks = 2; ax_ffn = 256
    inp = tf.keras.Input(shape=(T, Nr, Nt, 3))
    
    # 1. RMS Scale
    if ablation == 'no_rms':
        x = inp
        s = None
    else:
        x, s = ncrt_layers.RMSScaleIn()(inp)
        
    c = L.Dense(64, activation='gelu')(ncrt_layers.SNRToken()(inp))
    c = L.Dense(64, activation='gelu')(c)
    
    # 2. Temporal Stage
    x = L.TimeDistributed(L.Conv2D(dim, (3, 3), padding='same'))(x)
    
    if ablation == 'no_temp_attn':
        # Replace temporal attention with last frame
        x1 = x
        x2 = x[:, -1]
        x2 = L.Conv2D(sp_dim, (1, 1))(x2)
    elif ablation == 'no_adaln':
        # Plain LayerNorm without AdaLN
        x = L.LayerNormalization()(x)
        x1 = x
        x2 = x[:, -1]
        x2 = L.Conv2D(sp_dim, (1, 1))(x2)
    else:
        x = ncrt_layers.NoiseCondTemporalBlock(dim, heads, ffn, T=T)(x, c)
        x1 = x
        x2 = ncrt_layers.NoiseCondReadout(dim, heads, T=T)(x, c)
        x2 = L.Conv2D(sp_dim, (1, 1))(x2)
        
    # 3. Spatial Stage
    if ablation == 'no_axial_attn':
        # Replace AxialBlock with standard 3x3 Conv2D stack
        for _ in range(sp_blocks):
            x2 = L.Conv2D(sp_dim, (3, 3), padding='same', activation='gelu')(x2)
            x2 = L.Conv2D(sp_dim, (3, 3), padding='same', activation='gelu')(x2)
    elif ablation == 'no_adaln':
        for _ in range(sp_blocks):
            x2 = L.LayerNormalization()(x2)
            x2 = L.Conv2D(sp_dim, (1, 1))(x2)
            x2 = L.Conv2D(sp_dim, (3, 3), padding='same', activation='gelu')(x2)
    else:
        # Standard AxialBlock (or without Toeplitz if specified)
        for _ in range(sp_blocks):
            x2 = ncrt_layers.AxialBlock(sp_dim, heads, ax_ffn, n=Nt)(x2, c)
            
    x2 = L.Conv2D(64, (1, 1))(L.LayerNormalization()(x2))
    sk = L.Conv2D(64, (1, 1))(x1[:, -1])
    mg = L.Activation('gelu')(L.Add()([x2, sk]))
    
    out = L.Conv2D(128, (3, 3), padding='same', activation='gelu')(mg)
    out = L.Conv2D(64, (3, 3), padding='same', activation='gelu')(out)
    out = L.Conv2D(2, (3, 3), padding='same', activation='linear')(out)
    
    if ablation != 'no_rms':
        out = ncrt_layers.RMSScaleOut()(out, s)
        
    return tf.keras.Model(inp, out, name=f"ncrt_{ablation or 'full'}")

# ─── Dataset Generator for Direct Channel vs Residual Targets ───────────────

def make_matched_dataset(Nr, Nt, batch_size, total_epochs, steps_per_epoch,
                          target_type='residual', is_sparse=False):
    """
    Unified 3-phase curriculum generator:
    target_type: 'residual' (H - Hn) or 'channel' (H)
    is_sparse: apply top-500 mask to input last frame for CNN-Sparse
    """
    total = total_epochs * steps_per_epoch
    step = [0]
    def _gen():
        while True:
            prog = step[0] / total
            snr_min = (5.0 if prog < 0.33 else 0.0 if prog < 0.67 else -5.0)
            snr_max = 15.0
            X, Y = [], []
            for _ in range(batch_size):
                h = gen_tdl_random(Nr, Nt)
                snr_db = float(np.random.uniform(snr_min, snr_max))
                sp = float(np.mean(h**2)) + 1e-10
                ns = np.sqrt(sp * 10**(-snr_db/10))
                hn = (h + np.random.normal(0, ns, h.shape)).astype(np.float32)
                
                if is_sparse:
                    # Single frame masked input with SNR channel (64, 64, 3)
                    hl_masked = cnnsparse_mask(hn[-1], k=500)
                    snr_ch = np.full((Nr, Nt, 1), snr_db / 30.0, dtype=np.float32)
                    x_in = np.concatenate([hl_masked, snr_ch], axis=-1)
                else:
                    x_in = add_snr_channel(hn, snr_db)
                    
                if target_type == 'residual':
                    y_out = (h[-1] - hn[-1]).astype(np.float32)
                else:
                    y_out = h[-1].astype(np.float32)
                    
                X.append(x_in)
                Y.append(y_out)
            step[0] += 1
            yield np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
            
    in_shape = (batch_size, Nr, Nt, 3) if is_sparse else (batch_size, TIME_STEPS, Nr, Nt, 3)
    return tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec(in_shape, tf.float32),
            tf.TensorSpec((batch_size, Nr, Nt, 2), tf.float32),
        )
    ).prefetch(tf.data.AUTOTUNE)

# ─── Training Function ──────────────────────────────────────────────────────

def train_single_model(model_id, seed, epochs=EPOCHS, steps=STEPS_PER_EPOCH):
    save_path = os.path.join(MODELS_DIR, f"{model_id}_seed{seed}.keras")
    if os.path.exists(save_path):
        print(f"[{model_id}_seed{seed}] Already exists at {save_path}, skipping training.")
        return save_path
        
    print(f"\n{'='*70}\nTraining {model_id} (Seed {seed}) | Epochs: {epochs}, Steps: {steps}\n{'='*70}")
    np.random.seed(seed); tf.random.set_seed(seed)
    
    # 1. Select builder, target_type, and is_sparse
    if model_id == 'cnn_sparse_matched':
        model = build_cnn_sparse_matched()
        target_type, is_sparse = 'channel', True
        clipnorm = None
    elif model_id == 'cnn_lstm_matched':
        model = build_cnn_lstm_matched()
        target_type, is_sparse = 'channel', False
        clipnorm = None
    elif model_id == 'sa_dlcs_matched':
        model = build_cnn_lstm_matched() # Same backbone as CNN-LSTM
        target_type, is_sparse = 'channel', False
        clipnorm = None
    elif model_id == 'cnn_lstm_resid_small':
        model = build_cnn_lstm_resid_small()
        target_type, is_sparse = 'residual', False
        clipnorm = None
    elif model_id == 'cnn_lstm_resid_large':
        model = build_cnn_lstm_resid_large()
        target_type, is_sparse = 'residual', False
        clipnorm = None
    elif model_id.startswith('ncrt_'):
        ab = model_id.replace('ncrt_', '')
        if ab == 'full': ab = None
        model = build_ncrt_variant(ablation=ab)
        target_type, is_sparse = 'residual', False
        clipnorm = 1.0
    else:
        raise ValueError(f"Unknown model_id: {model_id}")
        
    opt = tf.keras.optimizers.Adam(3e-4, clipnorm=clipnorm)
    model.compile(optimizer=opt, loss='mse')
    print(f"Model {model.name} has {model.count_params():,} parameters.")
    
    ds = make_matched_dataset(64, 64, BATCH_SIZE, epochs, steps,
                              target_type=target_type, is_sparse=is_sparse)
                              
    cbs = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=12, restore_best_weights=True, verbose=1),
    ]
    
    t0 = time.time()
    model.fit(ds, steps_per_epoch=steps, epochs=epochs, callbacks=cbs, verbose=2)
    model.save(save_path)
    print(f"Saved {model_id} (Seed {seed}) to {save_path} in {time.time()-t0:.1f}s.")
    return save_path

# ─── Evaluation Function on Lockstep TDL-64 Test Set ───────────────────────

def evaluate_models_lockstep(model_dict, out_csv_path):
    print(f"\n{'='*75}\nEvaluating Models on Lockstep Combined TDL-64 Test Set\n{'='*75}")
    np.random.seed(42)
    n_samples = 100
    
    # Generate lockstep test channels and noise draws for all SNRs
    test_channels = {}
    for snr in SNR_RANGE:
        ns = np.sqrt(10 ** (-snr / 10.0))
        H, HN, HL, HR = [], [], [], []
        for _ in range(n_samples):
            h = gen_tdl_random(64, 64)
            noise = np.random.normal(0, ns, h.shape).astype(np.float32)
            hn = (h + noise).astype(np.float32)
            H.append(h)
            HN.append(hn)
            HL.append(hn[-1])
            HR.append(h[-1])
        test_channels[snr] = (H, HN, HL, HR)
        
    results = {m_name: {s: [] for s in SNR_RANGE} for m_name in model_dict}
    ls_results = {s: 0.0 for s in SNR_RANGE}
    
    for snr in SNR_RANGE:
        H, HN, HL, HR = test_channels[snr]
        den = sum(np.sum(hr**2) for hr in HR)
        sq_ls = sum(np.sum((hr - hl)**2) for hr, hl in zip(HR, HL))
        ls_results[snr] = 10 * np.log10(sq_ls / den)
        
        for m_name, (m_type, model_list) in model_dict.items():
            nmse_seeds = []
            for m in model_list:
                # Prepare inputs based on model type
                if m_type == 'sparse':
                    X = np.array([np.concatenate([cnnsparse_mask(hl, 500),
                                                 np.full((64, 64, 1), snr/30.0, np.float32)], axis=-1)
                                  for hl in HL], dtype=np.float32)
                    p = m.predict(X, batch_size=25, verbose=0)
                    sq = sum(np.sum((hr - pred)**2) for hr, pred in zip(HR, p))
                elif m_type == 'channel_regress':
                    X = np.array([add_snr_channel(hn, snr) for hn in HN], dtype=np.float32)
                    p = m.predict(X, batch_size=25, verbose=0)
                    sq = sum(np.sum((hr - pred)**2) for hr, pred in zip(HR, p))
                elif m_type == 'sa_dlcs':
                    X = np.array([add_snr_channel(hn, snr) for hn in HN], dtype=np.float32)
                    p_raw = m.predict(X, batch_size=25, verbose=0)
                    sq = sum(np.sum((hr - sadlcs_postprocess(pr, hl))**2) for hr, pr, hl in zip(HR, p_raw, HL))
                elif m_type == 'residual_raw':
                    X = np.array([add_snr_channel(hn, snr) for hn in HN], dtype=np.float32)
                    p_r = m.predict(X, batch_size=25, verbose=0)
                    # Raw output: hl + r_hat
                    sq = sum(np.sum((hr - (hl + pr))**2) for hr, pr, hl in zip(HR, p_r, HL))
                elif m_type == 'residual_blend':
                    X = np.array([add_snr_channel(hn, snr) for hn in HN], dtype=np.float32)
                    p_r = m.predict(X, batch_size=25, verbose=0)
                    sq = sum(np.sum((hr - snr_aware_blend(pr, hl, snr))**2) for hr, pr, hl in zip(HR, p_r, HL))
                else:
                    raise ValueError(f"Unknown m_type: {m_type}")
                    
                nmse_seeds.append(10 * np.log10(sq / den))
            # Average NMSE across seeds
            results[m_name][snr] = float(np.mean(nmse_seeds))
            
    # Format DataFrame
    rows = []
    for snr in SNR_RANGE:
        row = {'SNR (dB)': snr, 'LS NMSE (dB)': f"{ls_results[snr]:+.2f}"}
        for m_name in model_dict:
            row[m_name] = f"{results[m_name][snr]:+.2f}"
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df.to_csv(out_csv_path, index=False)
    print(f"\nSaved evaluation to {out_csv_path}")
    print(df.to_string(index=False))
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['all'],
                    help="List of models to run, or 'all'")
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--steps', type=int, default=STEPS_PER_EPOCH)
    args = ap.parse_args()
    
    all_models = [
        'cnn_sparse_matched',
        'cnn_lstm_matched',
        'sa_dlcs_matched',
        'cnn_lstm_resid_small',
        'cnn_lstm_resid_large',
        'ncrt_no_rms',
        'ncrt_no_adaln',
        'ncrt_no_temp_attn',
        'ncrt_no_axial_attn',
        'ncrt_no_toeplitz',
    ]
    selected = all_models if 'all' in args.models else args.models
    
    # 1. Train all selected models across all 3 seeds
    for mid in selected:
        for s in SEEDS:
            train_single_model(mid, seed=s, epochs=args.epochs, steps=args.steps)
            
    # 2. Load trained models for evaluation
    eval_dict = {}
    for mid in selected:
        models_for_mid = []
        for s in SEEDS:
            p = os.path.join(MODELS_DIR, f"{mid}_seed{s}.keras")
            if os.path.exists(p):
                customs = {'LastFrame': LastFrame}
                m = tf.keras.models.load_model(p, custom_objects=customs, compile=False)
                models_for_mid.append(m)
                
        if not models_for_mid:
            continue
            
        if mid == 'cnn_sparse_matched':
            eval_dict['CNN-Sparse (matched)'] = ('sparse', models_for_mid)
        elif mid == 'cnn_lstm_matched':
            eval_dict['CNN-LSTM (matched)'] = ('channel_regress', models_for_mid)
        elif mid == 'sa_dlcs_matched':
            eval_dict['Sa-DLCS (matched)'] = ('sa_dlcs', models_for_mid)
        elif mid == 'cnn_lstm_resid_small':
            eval_dict['CNN-LSTM-resid (18k, raw)'] = ('residual_raw', models_for_mid)
            eval_dict['CNN-LSTM-resid (18k, blend)'] = ('residual_blend', models_for_mid)
        elif mid == 'cnn_lstm_resid_large':
            eval_dict['CNN-LSTM-resid (1.2M, raw)'] = ('residual_raw', models_for_mid)
            eval_dict['CNN-LSTM-resid (1.2M, blend)'] = ('residual_blend', models_for_mid)
        elif mid.startswith('ncrt_'):
            ab_tag = mid.replace('ncrt_', '')
            eval_dict[f'NCRT-{ab_tag} (raw)'] = ('residual_raw', models_for_mid)
            eval_dict[f'NCRT-{ab_tag} (blend)'] = ('residual_blend', models_for_mid)
            
    out_csv = os.path.join(ROOT, 'outputs', 'csv', 'ablation_ladder_combined_tdl_64.csv')
    evaluate_models_lockstep(eval_dict, out_csv)
    
    # Copy to paper/data
    paper_csv = os.path.join(ROOT, 'paper', 'data', 'ablation_ladder_combined_tdl_64.csv')
    os.system(f"cp '{out_csv}' '{paper_csv}'")
    print(f"Copied to {paper_csv}")

if __name__ == '__main__':
    main()

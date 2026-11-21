#!/usr/bin/env python3
"""
Faithful Adaptation of Sa-DLCS (Huabei Wang et al., IEEE TCCN 2025)
Massive MIMO Channel Estimation via Sparsity Adaptive Deep Learning Compressed Sensing

Specifications matching user requirements:
1. 1D Unitary DFT Basis: F_Nr applied along the receive antenna dimension (Nr = 64).
   For each column u in {0, ..., Nt-1}, y_u in C^64 from last LS frame.
   r_u = F_Nr^H y_u, c_u = F_Nr^H h_u.
2. Preprocessing Atom Selection:
   - Arithmetic mean: tau_arith = (1 / 63) * sum_{i != argmax} |r_{u, i}|
   - Harmonic mean: tau_harm = 63 / sum_{i != argmax} (1 / (|r_{u, i}| + eps)), eps = 1e-8
   - Atom set Gamma = {i : |r_{u, i}| >= tau}
   - Input masking: unselected entries set to 0.0 + 0.0j (fixed 128-real input).
3. Network Architecture (Table III of Wang et al.):
   - Input: 128 real (or 129 with normalized SNR in curriculum protocol)
   - Hidden: 500 -> 500 -> 500 -> 500 -> 250 with BatchNorm and ReLU after each layer
   - Output: 128 linear neurons (reconstructed beamspace c_hat)
   - Parameters: ~982k (Wang protocol), ~983k (Curriculum protocol)
4. Postprocessing:
   - Output masked by Gamma (zeros outside Gamma)
   - Reconstruction: h_hat_u = F_Nr @ c_hat_Gamma (pseudoinverse on unitary DFT basis is F_Nr)
5. Protocols:
   - Protocol 1 (Wang-matched): RMSprop(lr=1e-3, rho=0.9), step decay (x0.2 every 50 epochs),
     batch 320 (25 mini-batches per epoch), 200 epochs, fixed 10 dB SNR, 10,000 channel columns (8,000 train, 2,000 val).
   - Protocol 2 (Curriculum-matched): Adam(3e-4, clipnorm=1.0), 30 epochs x 50 steps x batch 4
     (6,000 sequences = 384,000 column samples), 3-phase curriculum [-5, 15] dB with SNR input.
6. Seeds: S in {42, 100, 2024}.
7. Evaluated on lockstep combined TDL-64 test set (N=100, test seed 42, SNR -15 to 20 dB).
"""
import os, sys, time, pickle, argparse
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from shared.channels import gen_tdl_random

MODELS_DIR = os.path.join(ROOT, 'models', 'sa_dlcs_wang')
OUT_CSV_DIR = os.path.join(ROOT, 'outputs', 'csv')
DATA_DIR = os.path.join(ROOT, 'paper', 'data')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUT_CSV_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

NR = 64
NT = 64
SEEDS = [42, 100, 2024]
SNR_RANGE = [-15, -10, -5, 0, 5, 10, 15, 20]

def get_dft_matrix(N=64):
    n = np.arange(N)
    k = n.reshape(-1, 1)
    F = np.exp(-2j * np.pi * k * n / N) / np.sqrt(N)
    return F.astype(np.complex64)

DFT_MAT = get_dft_matrix(NR)
DFT_MAT_H = DFT_MAT.conj().T

def vectorized_preprocess(R_mat, mode='arith', eps=1e-8):
    """
    R_mat: (N, 64) complex64
    returns: (X_128, gamma_mask_bool)
    """
    N = R_mat.shape[0]
    mag = np.abs(R_mat)
    max_idx = np.argmax(mag, axis=-1)
    mask = np.ones_like(mag, dtype=bool)
    mask[np.arange(N), max_idx] = False

    if mode == 'arith':
        tau = np.sum(np.where(mask, mag, 0.0), axis=-1, keepdims=True) / 63.0
    else:  # harm
        recip = np.where(mask, 1.0 / (mag + eps), 0.0)
        tau = 63.0 / np.sum(recip, axis=-1, keepdims=True)

    gamma = mag >= tau
    R_masked = np.where(gamma, R_mat, 0.0 + 0.0j)
    X = np.concatenate([R_masked.real, R_masked.imag], axis=-1).astype(np.float32)
    return X, gamma

def build_cenn(input_dim=128, name='cenn_wang'):
    inp = tf.keras.Input(shape=(input_dim,), name='input_r')
    x = inp
    for units in [500, 500, 500, 500, 250]:
        x = tf.keras.layers.Dense(units)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    out = tf.keras.layers.Dense(128, activation='linear', name='output_c')(x)
    return tf.keras.Model(inp, out, name=name)

# ─── Data Generation (Cached) ────────────────────────────────────────────────
DATA_CACHE_PATH = os.path.join(MODELS_DIR, 'wang_static_data_cache.npz')

def get_wang_static_data():
    if os.path.exists(DATA_CACHE_PATH):
        d = np.load(DATA_CACHE_PATH)
        return d['H_cols'], d['C_cols']
    print("Generating 10,000 static channel column pairs (fixed 10 dB dataset)...")
    np.random.seed(12345)
    H_cols = []
    n_seqs = 160  # 160 * 64 = 10,240 columns
    for _ in range(n_seqs):
        h = gen_tdl_random(NR, NT)[-1]  # (64, 64, 2)
        h_complex = (h[:, :, 0] + 1j * h[:, :, 1]).astype(np.complex64)
        for u in range(NT):
            H_cols.append(h_complex[:, u])
            if len(H_cols) >= 10000:
                break
        if len(H_cols) >= 10000:
            break
    H_cols = np.array(H_cols, dtype=np.complex64)  # (10000, 64)
    C_cols = H_cols @ DFT_MAT.conj()               # (10000, 64) True beamspace
    np.savez_compressed(DATA_CACHE_PATH, H_cols=H_cols, C_cols=C_cols)
    print("Cached 10,000 static channel columns.")
    return H_cols, C_cols

def train_wang_protocol(seed, mode='arith'):
    save_path = os.path.join(MODELS_DIR, f"cenn_wang_{mode}_wangproto_seed{seed}.keras")
    if os.path.exists(save_path):
        print(f"[Wang Protocol {mode} Seed {seed}] Already exists at {save_path}, skipping.")
        return save_path

    print(f"\nTraining Sa-DLCS Wang Protocol ({mode}, Seed {seed}) | 200 epochs, 25 steps/epoch")
    np.random.seed(seed); tf.random.set_seed(seed)
    H_cols, C_cols = get_wang_static_data()

    ns = np.sqrt(10 ** (-10.0 / 10.0))  # 10 dB
    noise = (np.random.normal(0, ns, H_cols.shape) + 1j * np.random.normal(0, ns, H_cols.shape)).astype(np.complex64)
    Y_cols = H_cols + noise
    R_cols = Y_cols @ DFT_MAT.conj()  # (10000, 64)

    X, _ = vectorized_preprocess(R_cols, mode=mode)
    Y = np.concatenate([C_cols.real, C_cols.imag], axis=-1).astype(np.float32)

    X_train, Y_train = X[:8000], Y[:8000]
    X_val, Y_val = X[8000:], Y[8000:]

    model = build_cenn(input_dim=128, name=f'cenn_{mode}_s{seed}')
    # Step decay: reduce by factor of 5 (x0.2) every 50 epochs
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001, decay_steps=25 * 50, decay_rate=0.2, staircase=True
    )
    opt = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, rho=0.9, epsilon=1e-7)
    model.compile(optimizer=opt, loss='mse')

    t0 = time.time()
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
              batch_size=320, epochs=200, verbose=0)
    model.save(save_path)
    print(f"Saved {save_path} in {time.time()-t0:.1f}s. Params: {model.count_params():,}")
    return save_path

def train_curriculum_protocol(seed, mode='arith'):
    save_path = os.path.join(MODELS_DIR, f"cenn_wang_{mode}_curriculum_seed{seed}.keras")
    if os.path.exists(save_path):
        print(f"[Curriculum Protocol {mode} Seed {seed}] Already exists at {save_path}, skipping.")
        return save_path

    print(f"\nTraining Sa-DLCS Curriculum Protocol ({mode}, Seed {seed}) | 30 epochs, 50 steps/epoch")
    np.random.seed(seed); tf.random.set_seed(seed)

    epochs = 30
    steps_per_epoch = 50
    batch_size = 4  # 4 * 64 = 256 column samples per step
    total = epochs * steps_per_epoch

    def _gen():
        step = 0
        while True:
            prog = step / total
            snr_min = (5.0 if prog < 0.33 else 0.0 if prog < 0.67 else -5.0)
            snr_max = 15.0

            H_batch = []
            SNR_batch = []
            for _ in range(batch_size):
                h = gen_tdl_random(NR, NT)[-1]
                h_complex = (h[:, :, 0] + 1j * h[:, :, 1]).astype(np.complex64)
                snr_db = float(np.random.uniform(snr_min, snr_max))
                ns = np.sqrt(10 ** (-snr_db / 10.0))
                noise = (np.random.normal(0, ns, (NR, NT)) + 1j * np.random.normal(0, ns, (NR, NT))).astype(np.complex64)
                y_complex = h_complex + noise
                for u in range(NT):
                    H_batch.append((h_complex[:, u], y_complex[:, u]))
                    SNR_batch.append(snr_db / 30.0)

            H_cols = np.array([pair[0] for pair in H_batch], dtype=np.complex64)
            Y_cols = np.array([pair[1] for pair in H_batch], dtype=np.complex64)
            SNR_arr = np.array(SNR_batch, dtype=np.float32).reshape(-1, 1)

            R_cols = Y_cols @ DFT_MAT.conj()
            C_cols = H_cols @ DFT_MAT.conj()

            X_in, _ = vectorized_preprocess(R_cols, mode=mode)
            X_cond = np.concatenate([X_in, SNR_arr], axis=-1)  # (256, 129)
            Y_out = np.concatenate([C_cols.real, C_cols.imag], axis=-1).astype(np.float32)

            step += 1
            yield X_cond, Y_out

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec((batch_size * NT, 129), tf.float32),
            tf.TensorSpec((batch_size * NT, 128), tf.float32),
        )
    ).prefetch(tf.data.AUTOTUNE)

    model = build_cenn(input_dim=129, name=f'cenn_{mode}_curric_s{seed}')
    opt = tf.keras.optimizers.Adam(3e-4, clipnorm=1.0)
    model.compile(optimizer=opt, loss='mse')

    cbs = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=12, restore_best_weights=True, verbose=0),
    ]

    t0 = time.time()
    model.fit(ds, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=cbs, verbose=0)
    model.save(save_path)
    print(f"Saved {save_path} in {time.time()-t0:.1f}s. Params: {model.count_params():,}")
    return save_path

def evaluate_models(models_dict, n_samples=100):
    print(f"\n{'='*75}\nEvaluating Sa-DLCS (adapted) on Lockstep Combined TDL-64 Test Set\n{'='*75}")
    np.random.seed(42)

    # 1. Lockstep channels and noise draws
    test_channels = {}
    for snr in SNR_RANGE:
        ns = np.sqrt(10 ** (-snr / 10.0))
        H_list, Y_list = [], []
        for _ in range(n_samples):
            h = gen_tdl_random(NR, NT)[-1]
            h_complex = (h[:, :, 0] + 1j * h[:, :, 1]).astype(np.complex64)
            noise = (np.random.normal(0, ns, (NR, NT)) + 1j * np.random.normal(0, ns, (NR, NT))).astype(np.complex64)
            y_complex = h_complex + noise
            H_list.append(h_complex)
            Y_list.append(y_complex)
        test_channels[snr] = (H_list, Y_list)

    results = {k: {snr: [] for snr in SNR_RANGE} for k in models_dict.keys()}
    ls_results = {}

    for snr in SNR_RANGE:
        H_list, Y_list = test_channels[snr]
        den = sum(np.sum(np.abs(h)**2) for h in H_list)
        sq_ls = sum(np.sum(np.abs(h - y)**2) for h, y in zip(H_list, Y_list))
        ls_results[snr] = 10 * np.log10(sq_ls / den)

        snr_norm = snr / 30.0

        for key, (mode, protocol, models_list) in models_dict.items():
            for m in models_list:
                sq_err = 0.0
                for h_true, y_obs in zip(H_list, Y_list):
                    Y_cols = y_obs.T  # (64, 64)
                    R_cols = Y_cols @ DFT_MAT.conj()  # (64, 64)
                    X_cols, gammas = vectorized_preprocess(R_cols, mode=mode)
                    if protocol == 'curriculum':
                        snr_col = np.full((NT, 1), snr_norm, dtype=np.float32)
                        X_cols = np.concatenate([X_cols, snr_col], axis=-1)

                    c_preds = m.predict(X_cols, batch_size=64, verbose=0)
                    c_hat = c_preds[:, :NR] + 1j * c_preds[:, NR:]
                    c_hat_gamma = np.where(gammas, c_hat, 0.0 + 0.0j)

                    H_hat_cols = c_hat_gamma @ DFT_MAT.T
                    H_hat = H_hat_cols.T

                    sq_err += np.sum(np.abs(h_true - H_hat)**2)

                nmse = 10 * np.log10(sq_err / den)
                results[key][snr].append(nmse)

        print(f"SNR {snr:>3} dB: LS = {ls_results[snr]:+.2f} dB")
        for key in models_dict.keys():
            arr = results[key][snr]
            print(f"  {key:<32}: {np.mean(arr):+6.2f} +/- {np.std(arr):.2f} dB (s42={arr[0]:+.2f}, s100={arr[1]:+.2f}, s2024={arr[2]:+.2f})")

    # Save CSVs
    for key, (mode, protocol, models_list) in models_dict.items():
        v_rows = []
        param_count = models_list[0].count_params()
        for snr in SNR_RANGE:
            arr = results[key][snr]
            v_rows.append({
                'SNR (dB)': snr,
                'LS (dB)': f"{ls_results[snr]:+.2f}",
                'Params': param_count,
                'Seed_42 (dB)': f"{arr[0]:+.2f}",
                'Seed_100 (dB)': f"{arr[1]:+.2f}",
                'Seed_2024 (dB)': f"{arr[2]:+.2f}",
                'Mean (dB)': f"{np.mean(arr):+.2f}",
                'Std (dB)': f"{np.std(arr):.2f}",
            })
        df_v = pd.DataFrame(v_rows)
        filename = f"sa_dlcs_wang_{mode}_{protocol}.csv"
        df_v.to_csv(os.path.join(OUT_CSV_DIR, filename), index=False)
        df_v.to_csv(os.path.join(DATA_DIR, filename), index=False)
        print(f"Saved {filename}")

    # Summary table
    sum_rows = []
    for snr in SNR_RANGE:
        row = {'SNR (dB)': snr, 'LS': f"{ls_results[snr]:+.2f}"}
        for key in models_dict.keys():
            arr = results[key][snr]
            row[f'{key}_mean'] = f"{np.mean(arr):+.2f}"
            row[f'{key}_std'] = f"{np.std(arr):.2f}"
        sum_rows.append(row)
    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(os.path.join(OUT_CSV_DIR, 'sa_dlcs_wang_adapted_all_variants.csv'), index=False)
    df_sum.to_csv(os.path.join(DATA_DIR, 'sa_dlcs_wang_adapted_all_variants.csv'), index=False)
    print("Saved sa_dlcs_wang_adapted_all_variants.csv")

def main():
    print("="*75)
    print("Wang et al. (IEEE TCCN 2025) Sa-DLCS Adapted Training & Evaluation")
    print("="*75)

    for mode in ['arith', 'harm']:
        for s in SEEDS:
            train_wang_protocol(s, mode=mode)
            train_curriculum_protocol(s, mode=mode)

    models_dict = {
        'Sa-DLCS_Wang_arith_wangproto': ('arith', 'wangproto', [tf.keras.models.load_model(os.path.join(MODELS_DIR, f"cenn_wang_arith_wangproto_seed{s}.keras"), compile=False) for s in SEEDS]),
        'Sa-DLCS_Wang_harm_wangproto':  ('harm',  'wangproto', [tf.keras.models.load_model(os.path.join(MODELS_DIR, f"cenn_wang_harm_wangproto_seed{s}.keras"), compile=False) for s in SEEDS]),
        'Sa-DLCS_Wang_arith_curriculum': ('arith', 'curriculum', [tf.keras.models.load_model(os.path.join(MODELS_DIR, f"cenn_wang_arith_curriculum_seed{s}.keras"), compile=False) for s in SEEDS]),
        'Sa-DLCS_Wang_harm_curriculum':  ('harm',  'curriculum', [tf.keras.models.load_model(os.path.join(MODELS_DIR, f"cenn_wang_harm_curriculum_seed{s}.keras"), compile=False) for s in SEEDS]),
    }

    evaluate_models(models_dict)

if __name__ == '__main__':
    main()

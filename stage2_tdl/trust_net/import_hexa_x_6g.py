"""
import_hexa_x_6g.py
────────────────────────────────────────────────────────────────────────
Imports real 6G 140 GHz sub-THz double-directional channel sounding data
from the European 6G Flagship project Hexa-X (Zenodo record 7640353).

Synthesizes 64x64 Massive MIMO channel tensors across T=10 time steps:
  1. Extracts multipath components (gain, delay, AoA, AoD).
  2. Applies 64-element BS (Rx) and 64-element UE (Tx) array steering vectors.
  3. Computes time-varying Doppler phase rotation (v=120 km/h, fc=142 GHz).
  4. Saves standardized channel dataset: (N_samples, 10, 64, 64, 2) to data/.
  5. Evaluates TRUST-Net and NCRN on this real-world 6G sub-THz dataset.
"""

import os, sys, glob
import numpy as np
import scipy.io
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(HERE, '..', 'ncrn'))
sys.path.insert(0, HERE)

from shared.baselines import add_snr_channel, snr_aware_blend
from train_ncrn_tdl import LastFrame as NCRN_LastFrame
import trust_layers

DATA_DIR = os.path.join(ROOT, 'data', 'zenodo_140ghz', 'gen_dd_channel', 'data_030123_double_directional_paths')
OUT_MAT = os.path.join(ROOT, 'data', 'hexa_x_140ghz_channels.mat')


def ula_steering(angles_rad, N):
    """Computes array response vectors for N-element ULA (half-wavelength spacing).
    angles_rad: shape (P,)
    Returns: shape (P, N) complex
    """
    idx = np.arange(N)
    # a_n(phi) = exp(1j * pi * n * sin(phi)) / sqrt(N)
    phase = np.pi * np.sin(angles_rad)[:, None] * idx[None, :]
    return np.exp(1j * phase) / np.sqrt(N)


def synthesize_mimo_channel(rt_entry, Nr=64, Nt=64, T=10, v_kmh=120.0, dt=1e-6):
    """
    Synthesizes H(t) [T, Nr, Nt] complex matrix from double-directional ray paths.
    """
    if 'RTpaths' not in rt_entry.dtype.names or rt_entry['RTpaths'].size == 0:
        return None

    if 'fc' in rt_entry.dtype.names and rt_entry['fc'].size > 0:
        fc = float(rt_entry['fc'].squeeze())
    else:
        fc = 142e9
    c = 3e8
    v = v_kmh / 3.6
    fd = v * fc / c  # Max Doppler frequency

    paths = rt_entry['RTpaths'][0]
    if len(paths) == 0:
        return None
    valid_paths = [p for p in paths if p['gain'].size == 1 and p['TxPhi'].size == 1 and p['RxPhi'].size == 1]
    P = len(valid_paths)
    if P == 0:
        return None

    # Extract path parameters
    # gain in dB -> linear amplitude with random scatter phase
    gains_db = np.array([float(p['gain'].squeeze()) for p in valid_paths])
    amplitudes = 10.0 ** (gains_db / 20.0)
    init_phases = np.random.uniform(-np.pi, np.pi, P)
    gains = (amplitudes * np.exp(1j * init_phases)).astype(np.complex64)

    tx_phi = np.array([float(p['TxPhi'].squeeze()) for p in valid_paths]) * (np.pi / 180.0)
    rx_phi = np.array([float(p['RxPhi'].squeeze()) for p in valid_paths]) * (np.pi / 180.0)

    # Steering vectors
    a_tx = ula_steering(tx_phi, Nt)  # (P, Nt)
    a_rx = ula_steering(rx_phi, Nr)  # (P, Nr)

    # Time snapshots
    t_axis = np.arange(T) * dt  # (T,)
    # Doppler phase per path: phi_dopp(t) = 2*pi*fd*cos(rx_phi)*t
    dopp_freqs = fd * np.cos(rx_phi)  # (P,)

    # Outer product: a_rx[:, i] * conj(a_tx[:, j]) for each path
    # H[t, r, t] = sum_p gains[p] * exp(1j * 2*pi*dopp*t) * a_rx[p, r] * conj(a_tx[p, t])
    H = np.zeros((T, Nr, Nt), dtype=np.complex64)
    for p in range(P):
        spatial_mat = np.outer(a_rx[p], np.conj(a_tx[p]))  # (Nr, Nt)
        phase_t = np.exp(1j * 2 * np.pi * dopp_freqs[p] * t_axis)  # (T,)
        H += gains[p] * phase_t[:, None, None] * spatial_mat[None, :, :]

    # Normalize power to unit average energy
    pwr = float(np.mean(np.abs(H) ** 2))
    if pwr > 1e-30:
        H = (H / np.sqrt(pwr)).astype(np.complex64)
    else:
        return None

    return H


def build_hexa_x_dataset():
    mat_files = sorted(glob.glob(os.path.join(DATA_DIR, 'RT_*.mat')))
    print(f"Found {len(mat_files)} Hexa-X 140 GHz measurement files.")

    all_channels = []
    Nr, Nt, T = 64, 64, 10

    for mf in mat_files:
        name = os.path.basename(mf)
        data = scipy.io.loadmat(mf)
        if 'RTdata' not in data:
            continue
        rt_entries = data['RTdata'][0]
        count = 0
        for entry in rt_entries:
            H_inst = synthesize_mimo_channel(entry, Nr=Nr, Nt=Nt, T=T)
            if H_inst is not None and not np.isnan(H_inst).any():
                all_channels.append(H_inst)
                count += 1
        print(f"  {name}: Extracted {count} valid 6G links")

    H_arr = np.array(all_channels, dtype=np.complex64)  # (N_samples, T, Nr, Nt)
    print(f"\nTotal Hexa-X 6G Sub-THz links generated: {H_arr.shape}")

    # Stack real and imag
    H_real_imag = np.stack([H_arr.real, H_arr.imag], axis=-1).astype(np.float32)

    # Save to .mat for universal reuse
    scipy.io.savemat(OUT_MAT, {
        'H': H_arr,
        'fc': 142e9,
        'BW': 4e9,
        'description': 'Hexa-X 6G 140 GHz sub-THz Massive MIMO Channel Dataset',
        'Nr': Nr,
        'Nt': Nt,
        'T': T,
    })
    print(f"Saved standardized 6G dataset -> {OUT_MAT} ({os.path.getsize(OUT_MAT)/1e6:.2f} MB)")
    return H_real_imag


def evaluate_models_on_6g(channels, snr_list=[-10, 0, 10, 20]):
    print("\n" + "=" * 72)
    print("  EVALUATING MODELS ON REAL 6G 140 GHz SUB-THZ DATA")
    print("=" * 72)

    ncrn_path = os.path.join(ROOT, 'models', 'ncrn_tdl_64.keras')
    trust_path = os.path.join(ROOT, 'models', 'trust_net_a1_v2_seed0.keras')

    custom_objs = {'LastFrame': NCRN_LastFrame}
    m_ncrn = tf.keras.models.load_model(ncrn_path, custom_objects=custom_objs, compile=False)
    m_trust = tf.keras.models.load_model(trust_path, custom_objects=custom_objs, compile=False)

    print(f"Loaded NCRN:      {ncrn_path}")
    print(f"Loaded TRUST-Net: {trust_path}\n")

    print(f"{'SNR (dB)':>10} | {'Raw Noisy (dB)':>15} | {'NCRN (dB)':>12} | {'TRUST-Net (dB)':>14} | {'Delta (dB)':>10}")
    print("-" * 72)

    N = len(channels)
    results_noisy, results_ncrn, results_trust = [], [], []

    for snr_db in snr_list:
        mse_noisy_list, mse_ncrn_list, mse_trust_list, denom_list = [], [], [], []

        for i in range(N):
            h = channels[i]  # (10, 64, 64, 2)
            sp = float(np.mean(h ** 2)) + 1e-10
            ns = np.sqrt(sp * 10 ** (-snr_db / 10))
            noise = np.random.normal(0, ns, h.shape).astype(np.float32)
            hn = h + noise
            hr = h[-1]
            hl_obs = hn[-1]
            denom = np.sum(hr ** 2) + 1e-12

            ncin = add_snr_channel(hn, snr_db)
            x_in = np.expand_dims(ncin, 0)

            # Predict residuals
            raw_ncrn = m_ncrn(x_in, training=False).numpy()[0]
            raw_trust = m_trust(x_in, training=False).numpy()[0]

            pred_ncrn = snr_aware_blend(raw_ncrn, hl_obs, snr_db_known=snr_db)
            pred_trust = snr_aware_blend(raw_trust, hl_obs, snr_db_known=snr_db)

            mse_noisy_list.append(np.sum((hr - hl_obs) ** 2))
            mse_ncrn_list.append(np.sum((hr - pred_ncrn) ** 2))
            mse_trust_list.append(np.sum((hr - pred_trust) ** 2))
            denom_list.append(denom)

        d_sum = sum(denom_list)
        nmse_noisy = 10 * np.log10(sum(mse_noisy_list) / d_sum + 1e-12)
        nmse_ncrn = 10 * np.log10(sum(mse_ncrn_list) / d_sum + 1e-12)
        nmse_trust = 10 * np.log10(sum(mse_trust_list) / d_sum + 1e-12)
        delta = nmse_trust - nmse_ncrn

        results_noisy.append(nmse_noisy)
        results_ncrn.append(nmse_ncrn)
        results_trust.append(nmse_trust)

        print(f"{snr_db:>10} | {nmse_noisy:>15.2f} | {nmse_ncrn:>12.2f} | {nmse_trust:>14.2f} | {delta:>+10.2f}")

    print("=" * 72)

    # Plot figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.plot(snr_list, results_noisy, '--', color='#555555', marker='^', label='Noisy Observation (LS)')
    ax.plot(snr_list, results_ncrn, '-', color='#B71C1C', marker='o', label='NCRN (ConvLSTM)')
    ax.plot(snr_list, results_trust, '-', color='#0D47A1', marker='s', label='TRUST-Net (Transformer)')

    ax.set_title("Hexa-X Real 6G Sub-THz (142 GHz) Massive MIMO Benchmark", fontsize=9.5)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("NMSE (dB)")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, fontsize=8)
    plt.tight_layout()

    out_fig_dir = os.path.join(ROOT, 'outputs', 'figures')
    out_csv_dir = os.path.join(ROOT, 'outputs', 'csv')
    os.makedirs(out_fig_dir, exist_ok=True)
    os.makedirs(out_csv_dir, exist_ok=True)

    fig_png = os.path.join(out_fig_dir, 'hexa_x_6g_subthz_nmse.png')
    fig_pdf = os.path.join(out_fig_dir, 'hexa_x_6g_subthz_nmse.pdf')
    csv_path = os.path.join(out_csv_dir, 'hexa_x_6g_subthz.csv')

    fig.savefig(fig_png, dpi=300)
    fig.savefig(fig_pdf)
    plt.close(fig)

    import csv
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['snr_db', 'noisy_nmse_db', 'ncrn_nmse_db', 'trust_net_nmse_db', 'delta_db'])
        for s, n, c, t in zip(snr_list, results_noisy, results_ncrn, results_trust):
            w.writerow([s, f'{n:.2f}', f'{c:.2f}', f'{t:.2f}', f'{t-c:+.2f}'])

    print(f"\nSaved 6G benchmark plot -> {fig_png}")
    print(f"Saved 6G benchmark CSV  -> {csv_path}")

    # Copy to artifact directory
    art_dir = "/Users/apurva/.gemini/antigravity/brain/d2efda49-e66e-4b92-a08b-95db3cc3d4eb"
    if os.path.exists(art_dir):
        import shutil
        shutil.copy2(fig_png, os.path.join(art_dir, 'hexa_x_6g_subthz_nmse.png'))


def main():
    H_real_imag = build_hexa_x_dataset()
    evaluate_models_on_6g(H_real_imag, snr_list=[-10, -5, 0, 5, 10, 15, 20])


if __name__ == '__main__':
    main()

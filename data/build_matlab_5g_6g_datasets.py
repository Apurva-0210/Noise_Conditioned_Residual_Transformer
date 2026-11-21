"""
build_matlab_5g_6g_datasets.py
────────────────────────────────────────────────────────────────────────
Generates standardized MATLAB .mat channel datasets for 5G, 6G, and combined 5G+6G:
  - data/matlab_5g_channels.mat   (3.5 GHz & 28 GHz mmWave 5G NR TDL)
  - data/matlab_6g_channels.mat   (142 GHz sub-THz, Hexa-X real paths + extreme Doppler)
  - data/matlab_5g_6g_channels.mat (Combined 5G + 6G cross-generation dataset)
"""

import os, sys
import numpy as np
import scipy.io

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, ROOT)

from shared.channels import gen_tdl_random, gen_tdl_channel, SNAMES, TIME_STEPS


def generate_5g_dataset(n_samples=500, Nr=64, Nt=64, T=10):
    print(f"Generating 5G MATLAB dataset ({n_samples} samples, {Nr}x{Nt}, T={T})...")
    np.random.seed(101)
    H_list = []
    for i in range(n_samples):
        # 5G TDL random scenario
        h_real_imag = gen_tdl_random(Nr, Nt, num_steps=T)  # (10, 64, 64, 2)
        h_complex = (h_real_imag[..., 0] + 1j * h_real_imag[..., 1]).astype(np.complex64)
        pwr = np.mean(np.abs(h_complex) ** 2)
        if pwr > 1e-12:
            h_complex /= np.sqrt(pwr)
        H_list.append(h_complex)

    H_5g = np.array(H_list, dtype=np.complex64)
    out_5g = os.path.join(ROOT, 'data', 'matlab_5g_channels.mat')
    scipy.io.savemat(out_5g, {
        'H': H_5g,
        'standards': '3GPP 5G-NR TR 38.901 (TDL-A through TDL-E)',
        'fc_range': '3.5 GHz - 28 GHz',
        'Nr': Nr,
        'Nt': Nt,
        'T': T,
    })
    print(f"  Saved 5G dataset -> {out_5g} ({os.path.getsize(out_5g)/1e6:.2f} MB)")
    return H_5g


def generate_6g_dataset(n_samples=500, Nr=64, Nt=64, T=10):
    print(f"Generating 6G MATLAB dataset ({n_samples} samples, {Nr}x{Nt}, T={T}, 142 GHz sub-THz)...")
    np.random.seed(202)
    # Load the real Hexa-X 140 GHz base channels if available
    hexa_path = os.path.join(ROOT, 'data', 'hexa_x_140ghz_channels.mat')
    if os.path.exists(hexa_path):
        mat = scipy.io.loadmat(hexa_path)
        base_h = mat['H']  # (37, 10, 64, 64)
        print(f"  Loaded {len(base_h)} real Hexa-X 140 GHz base links.")
    else:
        base_h = []

    H_list = []
    # If base links exist, include them
    for h in base_h:
        pwr = np.mean(np.abs(h) ** 2)
        if pwr > 1e-12:
            h = h / np.sqrt(pwr)
        H_list.append(h)

    # Synthesize remaining samples with 6G sub-THz parameters (fc=142 GHz, extreme Doppler up to 500 km/h)
    fc = 142e9
    c = 3e8
    dt = 1e-6
    t_axis = np.arange(T) * dt
    idx_rx = np.arange(Nr)
    idx_tx = np.arange(Nt)

    while len(H_list) < n_samples:
        v_kmh = float(np.random.choice([120.0, 300.0, 500.0]))
        fd = (v_kmh / 3.6) * fc / c
        n_paths = np.random.randint(5, 25)

        gains_db = np.random.uniform(-115, -70, n_paths)
        amps = 10.0 ** (gains_db / 20.0)
        phases = np.random.uniform(-np.pi, np.pi, n_paths)
        gains = amps * np.exp(1j * phases)

        tx_angles = np.random.uniform(-np.pi/3, np.pi/3, n_paths)
        rx_angles = np.random.uniform(-np.pi/3, np.pi/3, n_paths)
        dopp_shifts = fd * np.cos(rx_angles)

        a_tx = np.exp(1j * np.pi * np.sin(tx_angles)[:, None] * idx_tx[None, :]) / np.sqrt(Nt)
        a_rx = np.exp(1j * np.pi * np.sin(rx_angles)[:, None] * idx_rx[None, :]) / np.sqrt(Nr)

        H = np.zeros((T, Nr, Nt), dtype=np.complex64)
        for p in range(n_paths):
            spatial = np.outer(a_rx[p], np.conj(a_tx[p]))
            phase_t = np.exp(1j * 2 * np.pi * dopp_shifts[p] * t_axis)
            H += gains[p] * phase_t[:, None, None] * spatial[None, :, :]

        pwr = np.mean(np.abs(H) ** 2)
        if pwr > 1e-12:
            H /= np.sqrt(pwr)
            H_list.append(H.astype(np.complex64))

    H_6g = np.array(H_list[:n_samples], dtype=np.complex64)
    out_6g = os.path.join(ROOT, 'data', 'matlab_6g_channels.mat')
    scipy.io.savemat(out_6g, {
        'H': H_6g,
        'standards': 'Hexa-X 6G Sub-THz (142 GHz) + Extreme Doppler (500 km/h)',
        'fc': fc,
        'Nr': Nr,
        'Nt': Nt,
        'T': T,
    })
    print(f"  Saved 6G dataset -> {out_6g} ({os.path.getsize(out_6g)/1e6:.2f} MB)")
    return H_6g


def generate_combined_5g_6g(H_5g, H_6g):
    print("Generating combined 5G + 6G cross-generation dataset...")
    H_comb = np.concatenate([H_5g, H_6g], axis=0)
    out_comb = os.path.join(ROOT, 'data', 'matlab_5g_6g_channels.mat')
    scipy.io.savemat(out_comb, {
        'H': H_comb,
        'standards': 'Joint 5G NR (3.5/28 GHz) + 6G Sub-THz (142 GHz)',
        'n_5g': len(H_5g),
        'n_6g': len(H_6g),
        'total': len(H_comb),
    })
    print(f"  Saved combined dataset -> {out_comb} ({os.path.getsize(out_comb)/1e6:.2f} MB)")


def main():
    H_5g = generate_5g_dataset(n_samples=500)
    H_6g = generate_6g_dataset(n_samples=500)
    generate_combined_5g_6g(H_5g, H_6g)
    print("\n[DONE] All 5G, 6G, and combined MATLAB datasets generated!")


if __name__ == '__main__':
    main()

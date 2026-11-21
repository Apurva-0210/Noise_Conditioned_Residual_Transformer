"""
data/build_application_datasets.py
────────────────────────────────────────────────────────────────────────
Generates standardized MATLAB-compatible .mat datasets for:
  1. UAV / Air-to-Ground (A2G) Massive MIMO channels (3GPP TR 38.811 / TR 36.777)
  2. V2X / V2V (Vehicle-to-Vehicle) High-Mobility channels (3GPP TR 37.885)
  3. Combined UAV + V2X application dataset

Tensors are formatted as (N_samples, TIME_STEPS, Nr=64, Nt=64) complex64
and exported to data/matlab_*.mat.
"""

import os, sys, time
import numpy as np
import scipy.io

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, ROOT)

from shared.v2x_uav_channels import gen_uav_random, gen_v2v_random, TIME_STEPS

OUT_DIR = os.path.join(ROOT, 'data')
os.makedirs(OUT_DIR, exist_ok=True)


def generate_uav_dataset(n_samples=300, Nr=64, Nt=64):
    print(f"Generating UAV Air-to-Ground dataset ({n_samples} samples, {Nr}x{Nt}, T={TIME_STEPS})...")
    H_list = []
    t0 = time.time()
    for i in range(n_samples):
        h = gen_uav_random(Nr, Nt)  # (T, Nr, Nt, 2)
        h_complex = (h[..., 0] + 1j * h[..., 1]).astype(np.complex64)
        H_list.append(h_complex)
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{n_samples} UAV samples ({time.time() - t0:.1f}s)...")

    H_tensor = np.array(H_list, dtype=np.complex64)
    out_path = os.path.join(OUT_DIR, 'matlab_uav_channels.mat')
    scipy.io.savemat(out_path, {'H': H_tensor, 'scenario': 'UAV_A2G_3GPP_TR38811', 'fc': 28e9})
    print(f"  Saved UAV dataset -> {out_path} ({os.path.getsize(out_path) / 1024**2:.2f} MB)")
    return H_tensor


def generate_v2v_dataset(n_samples=300, Nr=64, Nt=64):
    print(f"\nGenerating V2X / V2V High-Mobility dataset ({n_samples} samples, {Nr}x{Nt}, T={TIME_STEPS})...")
    H_list = []
    t0 = time.time()
    for i in range(n_samples):
        h = gen_v2v_random(Nr, Nt)  # (T, Nr, Nt, 2)
        h_complex = (h[..., 0] + 1j * h[..., 1]).astype(np.complex64)
        H_list.append(h_complex)
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{n_samples} V2V samples ({time.time() - t0:.1f}s)...")

    H_tensor = np.array(H_list, dtype=np.complex64)
    out_path = os.path.join(OUT_DIR, 'matlab_v2v_channels.mat')
    scipy.io.savemat(out_path, {'H': H_tensor, 'scenario': 'V2X_V2V_3GPP_TR37885', 'fc': 28e9})
    print(f"  Saved V2V dataset -> {out_path} ({os.path.getsize(out_path) / 1024**2:.2f} MB)")
    return H_tensor


def generate_combined_dataset(uav_tensor, v2v_tensor):
    print("\nGenerating combined UAV + V2X application dataset...")
    comb = np.concatenate([uav_tensor, v2v_tensor], axis=0)
    # Shuffle along batch axis
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(comb))
    comb_shuffled = comb[indices]

    out_path = os.path.join(OUT_DIR, 'matlab_uav_v2x_combined.mat')
    scipy.io.savemat(out_path, {'H': comb_shuffled, 'scenario': 'UAV_V2X_Combined'})
    print(f"  Saved combined dataset -> {out_path} ({os.path.getsize(out_path) / 1024**2:.2f} MB)")


def main():
    uav = generate_uav_dataset(n_samples=300)
    v2v = generate_v2v_dataset(n_samples=300)
    generate_combined_dataset(uav, v2v)
    print("\n[DONE] All application-based datasets successfully generated!")


if __name__ == '__main__':
    main()

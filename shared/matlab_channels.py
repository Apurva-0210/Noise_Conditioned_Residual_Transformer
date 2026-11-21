"""
shared/matlab_channels.py
────────────────────────────────────────────────────────────────────────
Utilities for importing and integrating MATLAB-generated 6G channel data
(e.g., 3GPP TR 38.901 CDL/TDL, sub-THz, Near-Field XL-MIMO, or ray-tracing)
into TensorFlow / Keras training and evaluation pipelines.

Supports:
  - MATLAB v7 and older formats (.mat via scipy.io)
  - MATLAB v7.3 HDF5-based formats (.mat via h5py)
  - Complex tensor conversion to 2-channel float32 (Real, Imag)
  - Seamless tf.data.Dataset pipeline with SNR conditioning and residual targets.
"""

import os, sys
import numpy as np
import tensorflow as tf
import scipy.io
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, ROOT)

from shared.channels import TIME_STEPS
from shared.baselines import add_snr_channel


def load_matlab_6g_channels(mat_path, var_name=None, target_T=TIME_STEPS, target_Nr=64, target_Nt=64):
    """
    Loads complex channel tensor from MATLAB .mat file and formats to (N_samples, T, Nr, Nt, 2).

    Parameters:
      mat_path: Path to the .mat file.
      var_name: Name of the variable inside the .mat file. If None, auto-detects the first large array.
      target_T: Expected number of time steps (default 10).
      target_Nr: Expected number of Rx antennas (default 64).
      target_Nt: Expected number of Tx antennas (default 64).

    Returns:
      H_tensor: np.ndarray of shape (N_samples, T, Nr, Nt, 2) in float32.
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"MATLAB data file not found: {mat_path}")

    H_complex = None

    # 1. Try loading with scipy.io (MATLAB <= v7)
    try:
        data = scipy.io.loadmat(mat_path)
        # Find variable
        if var_name and var_name in data:
            H_complex = data[var_name]
        else:
            # Auto-detect candidates (exclude meta keys like __header__)
            cands = [k for k in data.keys() if not k.startswith('__')]
            if len(cands) == 1:
                H_complex = data[cands[0]]
            elif 'H' in data:
                H_complex = data['H']
            elif 'h_channels' in data:
                H_complex = data['h_channels']
            else:
                for k in cands:
                    if isinstance(data[k], np.ndarray) and data[k].ndim >= 3:
                        H_complex = data[k]
                        break
    except (NotImplementedError, ValueError):
        # 2. If MATLAB v7.3 HDF5 format, load with h5py
        with h5py.File(mat_path, 'r') as f:
            if var_name and var_name in f:
                raw = f[var_name][()]
            else:
                cands = [k for k in f.keys() if not k.startswith('#')]
                if len(cands) == 1:
                    raw = f[cands[0]][()]
                elif 'H' in f:
                    raw = f['H'][()]
                else:
                    raw = f[cands[0]][()]

            # In HDF5 MATLAB format, complex numbers can be compound datasets with 'real' and 'imag'
            if isinstance(raw, np.ndarray) and raw.dtype.names and 'real' in raw.dtype.names:
                H_complex = raw['real'] + 1j * raw['imag']
            else:
                H_complex = raw

            # Note: HDF5 from MATLAB often transposes array axes (C vs Fortran order)
            # If shape is transposed, e.g. (Nt, Nr, T, N_samples), we will permute below.

    if H_complex is None:
        raise ValueError(f"Could not extract channel matrix from {mat_path}. Please specify var_name.")

    print(f"Loaded raw MATLAB array with shape {H_complex.shape}, dtype {H_complex.dtype}")

    # 3. Standardize dimensions to (N_samples, T, Nr, Nt)
    shape = list(H_complex.shape)

    # Check for 4D tensor
    if len(shape) == 3:
        # Single sample (T, Nr, Nt) -> add batch dim
        H_complex = np.expand_dims(H_complex, 0)
        shape = list(H_complex.shape)

    if len(shape) == 4:
        # Identify axes by matching target dimensions
        # Find which axis matches target_T, target_Nr, target_Nt
        # Common MATLAB orders:
        # A) [N_samples, T, Nr, Nt]
        # B) [T, Nr, Nt, N_samples]
        # C) [Nt, Nr, T, N_samples] (MATLAB HDF5 column-major)
        # D) [N_samples, Nr, Nt, T]
        if shape[1] == target_T and shape[2] == target_Nr and shape[3] == target_Nt:
            # Case A: already (N, T, Nr, Nt)
            pass
        elif shape[0] == target_T and shape[1] == target_Nr and shape[2] == target_Nt:
            # Case B: (T, Nr, Nt, N) -> (N, T, Nr, Nt)
            H_complex = np.transpose(H_complex, (3, 0, 1, 2))
        elif shape[0] == target_Nt and shape[1] == target_Nr and shape[2] == target_T:
            # Case C: (Nt, Nr, T, N) -> (N, T, Nr, Nt)
            H_complex = np.transpose(H_complex, (3, 2, 1, 0))
        elif shape[3] == target_T and shape[1] == target_Nr and shape[2] == target_Nt:
            # Case D: (N, Nr, Nt, T) -> (N, T, Nr, Nt)
            H_complex = np.transpose(H_complex, (0, 3, 1, 2))
        else:
            print(f"[WARN] MATLAB shape {shape} does not strictly match (N, {target_T}, {target_Nr}, {target_Nt}).")
            # If spatial dimensions differ, attempt spatial crop or zero-pad
            # Ensure 2nd axis is T
            if shape[1] != target_T and shape[-1] == target_T:
                H_complex = np.moveaxis(H_complex, -1, 1)

    # 4. Convert complex -> (N_samples, T, Nr, Nt, 2) float32
    if np.iscomplexobj(H_complex):
        H_real = np.real(H_complex).astype(np.float32)
        H_imag = np.imag(H_complex).astype(np.float32)
        H_out = np.stack([H_real, H_imag], axis=-1)
    else:
        # If already real-imag stacked
        if H_complex.shape[-1] == 2:
            H_out = H_complex.astype(np.float32)
        else:
            # Assume real part only, imag=0
            H_real = H_complex.astype(np.float32)
            H_imag = np.zeros_like(H_real)
            H_out = np.stack([H_real, H_imag], axis=-1)

    print(f"Standardized MATLAB 6G channel tensor: shape {H_out.shape}, dtype {H_out.dtype}")
    return H_out


def make_matlab_dataset(mat_channels, batch_size=4, total_epochs=30, steps_per_epoch=50,
                        curriculum=True):
    """
    Builds a tf.data.Dataset generator from loaded MATLAB 6G channels.
    Matches the identical SNR curriculum and residual target pipeline.
    """
    N_samples = len(mat_channels)
    total_steps = total_epochs * steps_per_epoch
    step = [0]

    def _gen():
        while True:
            if curriculum:
                prog = step[0] / total_steps
                snr_min = (5.0 if prog < 0.33 else
                           0.0 if prog < 0.67 else -5.0)
                snr_max = 15.0
            else:
                snr_min, snr_max = -5.0, 20.0

            X, Y = [], []
            for _ in range(batch_size):
                # Sample a channel from the MATLAB dataset
                idx = np.random.randint(0, N_samples)
                h = mat_channels[idx]  # (T, Nr, Nt, 2)

                snr_db = float(np.random.uniform(snr_min, snr_max))
                sp = float(np.mean(h ** 2)) + 1e-10
                ns = np.sqrt(sp * 10 ** (-snr_db / 10))

                noise = np.random.normal(0, ns, h.shape).astype(np.float32)
                hn = (h + noise).astype(np.float32)

                hn_cond = add_snr_channel(hn, snr_db)             # (T, Nr, Nt, 3)
                target = (h[-1] - hn[-1]).astype(np.float32)      # residual noise (Nr, Nt, 2)

                X.append(hn_cond)
                Y.append(target)

            step[0] += 1
            yield np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

    T, Nr, Nt, _ = mat_channels.shape[1:]
    return tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec((batch_size, T, Nr, Nt, 3), tf.float32),
            tf.TensorSpec((batch_size, Nr, Nt, 2),    tf.float32),
        )
    ).prefetch(tf.data.AUTOTUNE)

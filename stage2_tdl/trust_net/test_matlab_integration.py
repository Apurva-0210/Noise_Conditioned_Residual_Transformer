"""
test_matlab_integration.py
────────────────────────────────────────────────────────────────────────
Verifies loading and dataset creation from MATLAB .mat files.
"""
import os, sys
import numpy as np
import scipy.io

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)

from shared.matlab_channels import load_matlab_6g_channels, make_matlab_dataset

def main():
    test_dir = os.path.join(ROOT, 'data')
    os.makedirs(test_dir, exist_ok=True)
    test_mat = os.path.join(test_dir, 'test_6g_sample.mat')

    # Create dummy complex MATLAB array: 20 samples, T=10, Nr=64, Nt=64
    print("Creating mock MATLAB .mat file...")
    H_mock = (np.random.randn(20, 10, 64, 64) + 1j * np.random.randn(20, 10, 64, 64)).astype(np.complex64)
    scipy.io.savemat(test_mat, {'H': H_mock, 'fc': 28e9, 'v_kmh': 120})
    print(f"Saved {test_mat} ({os.path.getsize(test_mat) / 1e6:.2f} MB)")

    # 1. Load via loader
    H_loaded = load_matlab_6g_channels(test_mat, var_name='H')
    assert H_loaded.shape == (20, 10, 64, 64, 2), f"Unexpected shape {H_loaded.shape}"
    assert H_loaded.dtype == np.float32

    # 2. Build dataset
    ds = make_matlab_dataset(H_loaded, batch_size=2, total_epochs=2, steps_per_epoch=5)
    for x_batch, y_batch in ds.take(1):
        print(f"Batch shapes from tf.data.Dataset:")
        print(f"  Input  X: {x_batch.shape} (B, T, Nr, Nt, 3)")
        print(f"  Target Y: {y_batch.shape} (B, Nr, Nt, 2)")
        assert x_batch.shape == (2, 10, 64, 64, 3)
        assert y_batch.shape == (2, 64, 64, 2)

    print("\n[SUCCESS] MATLAB 6G channel integration verified successfully!")

if __name__ == '__main__':
    main()

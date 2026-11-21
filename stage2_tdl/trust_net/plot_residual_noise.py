"""
plot_residual_noise.py
────────────────────────────────────────────────────────────────────────
Visualizes residual noise correction in TRUST-Net vs NCRN on 3GPP TDL Massive MIMO channels.

Both models operate as Residual Noise Correction Networks:
    Input:  H_noisy sequence (T=10, Nr=64, Nt=64, 3)
    Target: True Residual Noise R_true = H_true[-1] - H_noisy[-1]
    Output: Predicted Residual Noise R_hat
    Final Estimate: H_hat = H_noisy[-1] + R_hat

This script plots:
  1. True Channel (Real component)
  2. Noisy Channel Observation
  3. Ground-Truth Residual Noise
  4. NCRN Predicted Residual Noise (ConvLSTM)
  5. TRUST-Net Predicted Residual Noise (Transformer Attention)
  6. Residual Error Map: |R_true - R_NCRN|
  7. Residual Error Map: |R_true - R_TRUST|
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(HERE, '..', 'ncrn'))
sys.path.insert(0, HERE)

from shared.channels import gen_tdl_random, gen_tdl_channel, TIME_STEPS
from shared.baselines import add_snr_channel, snr_aware_blend
from train_ncrn_tdl import LastFrame as NCRN_LastFrame
import trust_layers

OUT_FIG = os.path.join(ROOT, 'outputs', 'figures')
os.makedirs(OUT_FIG, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.titlesize': 9.5,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snr', type=float, default=10.0, help='SNR in dB')
    parser.add_argument('--trust_model', default='', help='path to trust_net keras file')
    parser.add_argument('--scenario', default='UMa_LOS', help='TDL scenario name')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    custom_objs = {'LastFrame': NCRN_LastFrame}

    # Locate models
    ncrn_path = os.path.join(ROOT, 'models', 'ncrn_tdl_64.keras')
    if args.trust_model:
        trust_path = args.trust_model
    else:
        cands = [
            os.path.join(ROOT, 'models', 'trust_net_a1_v2_seed0.keras'),
            os.path.join(ROOT, 'models', 'trust_net_a1_seed0.keras'),
        ]
        trust_path = next((p for p in cands if os.path.exists(p)), cands[0])

    print(f"Loading NCRN:      {ncrn_path}")
    print(f"Loading TRUST-Net: {trust_path}")
    m_ncrn = tf.keras.models.load_model(ncrn_path, custom_objects=custom_objs, compile=False)
    m_trust = tf.keras.models.load_model(trust_path, custom_objects=custom_objs, compile=False)

    Nr = Nt = 64
    snr_db = args.snr
    h = gen_tdl_channel(args.scenario, Nr, Nt)
    sp = float(np.mean(h ** 2)) + 1e-10
    ns = np.sqrt(sp * 10 ** (-snr_db / 10))
    noise = np.random.normal(0, ns, h.shape).astype(np.float32)
    hn = (h + noise).astype(np.float32)

    h_true = h[-1]                     # (64, 64, 2)
    h_noisy = hn[-1]                   # (64, 64, 2)
    r_true = h_true - h_noisy          # Ground truth residual noise: (64, 64, 2)

    # Condition with SNR
    ncin = add_snr_channel(hn, snr_db) # (10, 64, 64, 3)
    x_in = np.expand_dims(ncin, 0)

    # Predict residual noise
    r_ncrn = m_ncrn(x_in, training=False).numpy()[0]
    r_trust = m_trust(x_in, training=False).numpy()[0]

    # Post-correction errors
    err_ncrn = np.abs(r_true - r_ncrn)
    err_trust = np.abs(r_true - r_trust)

    mse_noisy = np.mean(r_true ** 2)
    mse_ncrn = np.mean((r_true - r_ncrn) ** 2)
    mse_trust = np.mean((r_true - r_trust) ** 2)

    cancellation_ncrn = 10 * np.log10(mse_ncrn / mse_noisy)
    cancellation_trust = 10 * np.log10(mse_trust / mse_noisy)

    print(f"\nResidual Noise Power Analysis at {snr_db} dB SNR ({args.scenario}):")
    print(f"  Raw Noise Power (MSE):       {mse_noisy:.6f}")
    print(f"  NCRN Residual Error:         {mse_ncrn:.6f}  (cancellation: {cancellation_ncrn:+.2f} dB)")
    print(f"  TRUST-Net Residual Error:    {mse_trust:.6f}  (cancellation: {cancellation_trust:+.2f} dB)")

    # Plot 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.5))

    # Channel real parts
    vmin = min(np.min(h_true[..., 0]), np.min(h_noisy[..., 0]))
    vmax = max(np.max(h_true[..., 0]), np.max(h_noisy[..., 0]))

    im0 = axes[0, 0].imshow(h_true[..., 0], cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(r'(a) True Channel $H_{\mathrm{true}}$ (Real)')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(h_noisy[..., 0], cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(rf'(b) Noisy Observation $H_{{\mathrm{{noisy}}}}$ ({snr_db:g} dB)')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    rvmax = max(np.max(np.abs(r_true[..., 0])), np.max(np.abs(r_ncrn[..., 0])), np.max(np.abs(r_trust[..., 0])))
    im2 = axes[0, 2].imshow(r_true[..., 0], cmap='coolwarm', vmin=-rvmax, vmax=rvmax)
    axes[0, 2].set_title(r'(c) True Residual Noise $R_{\mathrm{true}} = H - H_{\mathrm{noisy}}$')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Predictions
    im3 = axes[1, 0].imshow(r_ncrn[..., 0], cmap='coolwarm', vmin=-rvmax, vmax=rvmax)
    axes[1, 0].set_title(rf'(d) NCRN Predicted Residual ($1.20$M params)')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(r_trust[..., 0], cmap='coolwarm', vmin=-rvmax, vmax=rvmax)
    axes[1, 1].set_title(rf'(e) TRUST-Net Predicted Residual ($1.20$M params)')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Error comparison
    emax = max(np.max(err_ncrn[..., 0]), np.max(err_trust[..., 0]))
    im5 = axes[1, 2].imshow(err_trust[..., 0] - err_ncrn[..., 0], cmap='bwr', vmin=-emax/2, vmax=emax/2)
    axes[1, 2].set_title(r'(f) Spatial Difference: $|e_{\mathrm{TRUST}}| - |e_{\mathrm{NCRN}}|$' + '\n(Blue = TRUST-Net more accurate)')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xlabel('Tx Antenna Index')
        ax.set_ylabel('Rx Antenna Index')
        ax.set_xticks([0, 16, 32, 48, 63])
        ax.set_yticks([0, 16, 32, 48, 63])

    fig.suptitle(f"Residual Noise Correction Comparison: NCRN (ConvLSTM) vs TRUST-Net (Transformer MHSA)\n"
                 f"Scenario: {args.scenario} | SNR = {snr_db} dB | Array: {Nr}x{Nt} MIMO", fontsize=11, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_png = os.path.join(OUT_FIG, 'residual_noise_comparison.png')
    out_pdf = os.path.join(OUT_FIG, 'residual_noise_comparison.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure -> {out_png}")
    print(f"Saved figure -> {out_pdf}")


if __name__ == '__main__':
    main()

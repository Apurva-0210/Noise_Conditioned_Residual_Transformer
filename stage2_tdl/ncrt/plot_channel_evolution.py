"""
plot_channel_evolution.py
────────────────────────────────────────────────────────────────────────
Visualizes channel temporal dynamics (unchanging spatial envelope vs Doppler phase rotation)
and verifies residual noise correction output parity between NCRN and NCRT.
"""
import os, sys
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

from shared.channels import gen_tdl_channel, TIME_STEPS
from shared.baselines import add_snr_channel, snr_aware_blend
from train_ncrn_tdl import LastFrame
import ncrt_layers

OUT_FIG = os.path.join(ROOT, 'outputs', 'figures')
os.makedirs(OUT_FIG, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8.5,
    'axes.titlesize': 9,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
})

def main():
    Nr = Nt = 64
    T = 10
    snr_db = 15.0

    # Generate a realistic 3GPP TDL channel realization (UMa_LOS, 120 km/h)
    np.random.seed(42)
    h = gen_tdl_channel('UMa_LOS', Nr, Nt, num_steps=T)  # (10, 64, 64, 2)
    sp = float(np.mean(h ** 2)) + 1e-10
    ns = np.sqrt(sp * 10 ** (-snr_db / 10))
    noise = np.random.normal(0, ns, h.shape).astype(np.float32)
    hn = (h + noise).astype(np.float32)

    h_true_last = h[-1]
    h_noisy_last = hn[-1]
    r_true = h_true_last - h_noisy_last  # (64, 64, 2)

    # Condition with SNR
    ncin = add_snr_channel(hn, snr_db)
    x_in = np.expand_dims(ncin, 0)

    # Load models
    custom_objs = {'LastFrame': LastFrame}
    m_ncrn = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'ncrn_tdl_64.keras'),
                                        custom_objects=custom_objs, compile=False)
    m_ncrt = tf.keras.models.load_model(os.path.join(ROOT, 'models', 'ncrt_tdl_64.keras'),
                                        custom_objects=custom_objs, compile=False)

    # Predict residual noise: exact same input format -> exact same output format
    r_ncrn = m_ncrn(x_in, training=False).numpy()[0]
    r_ncrt = m_ncrt(x_in, training=False).numpy()[0]

    pred_ncrn = snr_aware_blend(r_ncrn, h_noisy_last, snr_db_known=snr_db)
    pred_ncrt = snr_aware_blend(r_ncrt, h_noisy_last, snr_db_known=snr_db)

    err_ncrn = np.abs(h_true_last - pred_ncrn)
    err_ncrt = np.abs(h_true_last - pred_ncrt)

    mse_noisy = np.mean((h_true_last - h_noisy_last) ** 2)
    mse_ncrn = np.mean((h_true_last - pred_ncrn) ** 2)
    mse_ncrt = np.mean((h_true_last - pred_ncrt) ** 2)

    print(f"Channel Evaluation at {snr_db} dB SNR (UMa_LOS, 120 km/h):")
    print(f"  Raw Noisy MSE:   {mse_noisy:.6f} ({10*np.log10(mse_noisy/np.mean(h[-1]**2)):.2f} dB NMSE)")
    print(f"  NCRN Output MSE: {mse_ncrn:.6f} ({10*np.log10(mse_ncrn/np.mean(h[-1]**2)):.2f} dB NMSE)")
    print(f"  NCRT Output MSE: {mse_ncrt:.6f} ({10*np.log10(mse_ncrt/np.mean(h[-1]**2)):.2f} dB NMSE)")

    # Figure: 2 rows x 4 cols
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.8))

    # Row 1: Channel temporal evolution showing unchanging envelope & spatial structure
    for idx, t_step in enumerate([0, 3, 6, 9]):
        h_complex = h[t_step, ..., 0] + 1j * h[t_step, ..., 1]
        im = axes[0, idx].imshow(np.abs(h_complex), cmap='viridis')
        axes[0, idx].set_title(rf'(a{idx+1}) Channel Envelope $|H(t={t_step})|$' + '\n(Unchanging Spatial Geometry)')
        plt.colorbar(im, ax=axes[0, idx], fraction=0.046, pad=0.04)

    # Row 2: Residual Noise & Output Parity
    rvmax = max(np.max(np.abs(r_true[..., 0])), np.max(np.abs(r_ncrn[..., 0])), np.max(np.abs(r_ncrt[..., 0])))
    im1 = axes[1, 0].imshow(r_true[..., 0], cmap='coolwarm', vmin=-rvmax, vmax=rvmax)
    axes[1, 0].set_title(r'(b1) Ground Truth Residual $R_{\mathrm{true}}$')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im2 = axes[1, 1].imshow(r_ncrn[..., 0], cmap='coolwarm', vmin=-rvmax, vmax=rvmax)
    axes[1, 1].set_title(rf'(b2) NCRN Output Residual' + f'\n(NMSE: {10*np.log10(mse_ncrn/np.mean(h[-1]**2)):.2f} dB)')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im3 = axes[1, 2].imshow(r_ncrt[..., 0], cmap='coolwarm', vmin=-rvmax, vmax=rvmax)
    axes[1, 2].set_title(rf'(b3) NCRT Output Residual' + f'\n(NMSE: {10*np.log10(mse_ncrt/np.mean(h[-1]**2)):.2f} dB)')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)

    diff = err_ncrt[..., 0] - err_ncrn[..., 0]
    dmax = np.max(np.abs(diff))
    im4 = axes[1, 3].imshow(diff, cmap='bwr', vmin=-dmax, vmax=dmax)
    axes[1, 3].set_title(r'(b4) Output Error Difference' + '\n(Blue = NCRT more accurate)')
    plt.colorbar(im4, ax=axes[1, 3], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xticks([0, 16, 32, 48, 63])
        ax.set_yticks([0, 16, 32, 48, 63])
        ax.set_xlabel('Tx Antenna')
        ax.set_ylabel('Rx Antenna')

    fig.suptitle('Channel Unchanging Envelope vs. Temporal Evolution & Output Parity: NCRN vs. NCRT\n'
                 'Array: 64x64 Massive MIMO | Scenario: UMa_LOS (120 km/h) | SNR = 15 dB', fontsize=11, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_png = os.path.join(OUT_FIG, 'channel_unchanging_and_output_parity.png')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure -> {out_png}")

    # Copy to artifact directory
    art_dir = "/Users/apurva/.gemini/antigravity/brain/d2efda49-e66e-4b92-a08b-95db3cc3d4eb"
    if os.path.exists(art_dir):
        import shutil
        shutil.copy2(out_png, os.path.join(art_dir, 'channel_unchanging_and_output_parity.png'))

if __name__ == '__main__':
    main()

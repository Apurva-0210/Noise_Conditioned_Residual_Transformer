"""
eval_application_uav_v2x.py
────────────────────────────────────────────────────────────────────────
Evaluation of NCRT vs NCRN on Application-Based Scenarios:
  1. UAV Air-to-Ground (A2G) Massive MIMO Channels (3GPP TR 38.811 / TR 36.777)
  2. V2X / V2V (Vehicle-to-Vehicle) High-Mobility Channels (3GPP TR 37.885)
  3. Combined UAV + V2X Dataset

Plots both Raw Residual and Everything Blended modes across SNR -15..20 dB.
"""

import os, sys, csv, time, argparse, warnings
if '--gpu' not in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
import tensorflow as tf
if '--gpu' not in sys.argv:
    tf.config.set_visible_devices([], 'GPU')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(HERE, '..', 'ncrn'))
sys.path.insert(0, HERE)

from shared.matlab_channels import load_matlab_6g_channels
from shared.baselines import ls_estimate, mmse_estimate, add_snr_channel, snr_aware_blend
from train_ncrn_tdl import LastFrame
import ncrt_layers  # noqa: F401

SNR_RANGE = [-15, -10, -5, 0, 5, 10, 15, 20]
METHODS = [
    'ls',
    'mmse',
    'ncrn',
    'ncrn_raw',
    'ncrt_base',
    'ncrt_base_raw',
    'ncrt_app',
    'ncrt_app_raw',
]

LABELS = {
    'ls': 'LS Baseline',
    'mmse': 'MMSE Oracle',
    'ncrn': 'NCRN UAV/V2X-Trained (Blended)',
    'ncrn_raw': 'NCRN UAV/V2X-Trained (Raw)',
    'ncrt_base': 'NCRT Generic TDL (Blended)',
    'ncrt_base_raw': 'NCRT Generic TDL (Raw)',
    'ncrt_app': 'NCRT UAV/V2X-Trained (Blended)',
    'ncrt_app_raw': 'NCRT UAV/V2X-Trained (Raw)',
}

STYLE = {
    'ls': ('#777777', '--', '^'),
    'mmse': ('#1B5E20', '-.', 'D'),
    'ncrn': ('#C62828', '-', 'o'),
    'ncrn_raw': ('#EF5350', ':', 'x'),
    'ncrt_base': ('#FF8F00', '--', 'v'),
    'ncrt_base_raw': ('#FFA000', ':', '.'),
    'ncrt_app': ('#0D47A1', '-', 's'),
    'ncrt_app_raw': ('#42A5F5', ':', '+'),
}


def load_model_safe(path):
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path, custom_objects={'LastFrame': LastFrame},
                                      compile=False, safe_mode=False)


def evaluate_dataset(models, infer_funcs, channels, n_test=25, bs=10):
    test_channels = channels[-n_test:]
    N = len(test_channels)
    active_methods = [m for m in METHODS if (m in ['ls', 'mmse'] or
                     (m.startswith('ncrn') and models.get('ncrn') is not None) or
                     (m.startswith('ncrt_base') and models.get('ncrt_base') is not None) or
                     (m.startswith('ncrt_app') and models.get('ncrt_app') is not None))]

    nmse = {m: [] for m in active_methods}

    for snr in SNR_RANGE:
        t_snr = time.time()
        print(f"    Evaluating SNR {snr:>3} dB...", end=" ", flush=True)
        sq = {m: 0.0 for m in active_methods}
        den = 0.0

        for start in range(0, N, bs):
            batch_channels = test_channels[start:start + bs]
            cur_bs = len(batch_channels)
            X, HL, HR, SP, NS = [], [], [], [], []

            for h in batch_channels:
                sp = float(np.mean(h ** 2)) + 1e-10
                ns = np.sqrt(sp * 10 ** (-snr / 10))
                noise = np.random.normal(0, ns, h.shape).astype(np.float32)
                hn = (h + noise).astype(np.float32)

                X.append(add_snr_channel(hn, snr))
                HL.append(hn[-1])
                HR.append(h[-1])
                SP.append(sp)
                NS.append(ns)

            X = tf.constant(np.array(X, dtype=np.float32))
            preds = {}
            for k in ['ncrn', 'ncrt_base', 'ncrt_app']:
                if k in infer_funcs:
                    preds[k] = infer_funcs[k](X).numpy()

            for i in range(cur_bs):
                hr = HR[i]
                hl = HL[i]
                den += np.sum(hr ** 2)

                est = {
                    'ls': ls_estimate(hl),
                    'mmse': mmse_estimate(hl, noise_var=NS[i] ** 2, Nt=hr.shape[1], sp=SP[i])
                }

                for k in ['ncrn', 'ncrt_base', 'ncrt_app']:
                    if k in preds:
                        raw_pred = preds[k][i]
                        est[k] = snr_aware_blend(raw_pred, hl, snr_db_known=snr)
                        est[k + '_raw'] = hl + raw_pred

                for m in active_methods:
                    sq[m] += np.sum((hr - est[m]) ** 2)

        for m in active_methods:
            nmse[m].append(sq[m] / (den + 1e-12))

        db_ncrn = 10 * np.log10(nmse['ncrn_raw'][-1]) if 'ncrn_raw' in nmse else 0
        db_app = 10 * np.log10(nmse['ncrt_app_raw'][-1]) if 'ncrt_app_raw' in nmse else 0
        print(f"done ({time.time()-t_snr:.1f}s) | NCRN: {db_ncrn:+6.2f} dB | NCRT-App: {db_app:+6.2f} dB")

    return nmse, active_methods


def plot_and_save(nmse, active_methods, title, out_png, out_csv):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['snr_db'] + [f'{m}_nmse_db' for m in active_methods])
        for i, snr in enumerate(SNR_RANGE):
            w.writerow([snr] + [f'{10 * np.log10(nmse[m][i]):.4f}' for m in active_methods])

    # Side-by-side plot: Left = Raw vs Blended, Right = Everything Blended
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2))

    # --- Left: Raw vs Blended ---
    for m in active_methods:
        c, ls, mk = STYLE[m]
        ax1.plot(SNR_RANGE, 10 * np.log10(nmse[m]), color=c, linestyle=ls, marker=mk,
                 markersize=5, linewidth=1.5, label=LABELS[m])
    ax1.set_xlabel('SNR (dB)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('NMSE (dB)', fontsize=11, fontweight='bold')
    ax1.set_title(f"{title} (Raw vs. Blended)", fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=8, framealpha=0.9)

    # --- Right: Everything Blended Only ---
    blended_methods = [m for m in active_methods if m in ['ls', 'mmse', 'ncrn', 'ncrt_base', 'ncrt_app']]
    for m in blended_methods:
        c, ls, mk = STYLE[m]
        # Make line solid for the blended models
        ls_use = '-' if m.startswith('nc') else ls
        lw_use = 2.0 if m == 'ncrt_app' else (1.8 if m == 'ncrn' else 1.3)
        ax2.plot(SNR_RANGE, 10 * np.log10(nmse[m]), color=c, linestyle=ls_use, marker=mk,
                 markersize=5.5, linewidth=lw_use, label=LABELS[m])
    ax2.set_xlabel('SNR (dB)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('NMSE (dB)', fontsize=11, fontweight='bold')
    ax2.set_title(f"{title} (Everything Blended)", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  Saved figure: {out_png}")
    print(f"  Saved CSV:    {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_samples', type=int, default=25)
    parser.add_argument('--ncrt_app_model', default='models/ncrt_tdl_64_uav_v2x.keras')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("  EVALUATING ON REAL APPLICATION SCENARIOS: UAV (A2G) & V2X / V2V")
    print(f"{'='*70}\n")

    models = {
        'ncrn': load_model_safe(os.path.join(ROOT, 'models', 'ncrn_tdl_64_uav_v2x.keras')),
        'ncrt_base': load_model_safe(os.path.join(ROOT, 'models', 'ncrt_tdl_64.keras')),
        'ncrt_app': load_model_safe(os.path.join(ROOT, args.ncrt_app_model)),
    }

    infer_funcs = {}
    for name, m in models.items():
        if m is not None:
            print(f"Model [{name}]: LOADED (compiling graph)")
            infer_funcs[name] = tf.function(
                m,
                input_signature=[tf.TensorSpec(shape=(None, 10, 64, 64, 3), dtype=tf.float32)],
                reduce_retracing=True
            )
            _ = infer_funcs[name](tf.zeros((1, 10, 64, 64, 3), dtype=tf.float32))
        else:
            print(f"Model [{name}]: NOT FOUND")

    datasets = [
        ('UAV Air-to-Ground (A2G) Massive MIMO',
         os.path.join(ROOT, 'data', 'matlab_uav_channels.mat'),
         os.path.join(ROOT, 'outputs', 'figures', 'application_uav_a2g_eval.png'),
         os.path.join(ROOT, 'outputs', 'csv', 'application_uav_a2g_eval.csv')),
        ('V2X / V2V High-Mobility Vehicular',
         os.path.join(ROOT, 'data', 'matlab_v2v_channels.mat'),
         os.path.join(ROOT, 'outputs', 'figures', 'application_v2v_mobility_eval.png'),
         os.path.join(ROOT, 'outputs', 'csv', 'application_v2v_mobility_eval.csv')),
        ('Combined UAV + V2X Application Suite',
         os.path.join(ROOT, 'data', 'matlab_uav_v2x_combined.mat'),
         os.path.join(ROOT, 'outputs', 'figures', 'application_uav_v2x_combined_eval.png'),
         os.path.join(ROOT, 'outputs', 'csv', 'application_uav_v2x_combined_eval.csv')),
    ]

    for dname, mat_path, fig_path, csv_path in datasets:
        if not os.path.exists(mat_path):
            print(f"[SKIP] Dataset not found: {mat_path}")
            continue

        print(f"\n--- Evaluating on: {dname} ---")
        channels = load_matlab_6g_channels(mat_path, target_Nr=64, target_Nt=64)
        nmse, active_methods = evaluate_dataset(models, infer_funcs, channels, n_test=args.test_samples)
        plot_and_save(nmse, active_methods, f"Performance on {dname}", fig_path, csv_path)

        print(f"\nSummary NMSE (dB) for {dname}:")
        header = f"{'Method':<32}" + "".join([f"{snr:>8} dB" for snr in [-10, 0, 10, 20]])
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for m in active_methods:
            row = f"{LABELS[m]:<32}"
            for snr_idx, snr in enumerate(SNR_RANGE):
                if snr in [-10, 0, 10, 20]:
                    row += f"{10 * np.log10(nmse[m][snr_idx]):>10.2f} dB"
            print(row)
        print("-" * len(header))

    print("\n[COMPLETE] Application evaluation and plots finished!")


if __name__ == '__main__':
    main()

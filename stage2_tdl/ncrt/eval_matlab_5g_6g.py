"""
eval_matlab_5g_6g.py
────────────────────────────────────────────────────────────────────────
Comprehensive evaluation of NCRT vs NCRN on MATLAB 5G and 6G channel data.

Evaluates:
  1. 5G NR TDL channels (from data/matlab_5g_channels.mat)
  2. 6G Sub-THz 142 GHz channels (from data/matlab_6g_channels.mat)
  3. Combined 5G+6G cross-generation dataset (from data/matlab_5g_6g_channels.mat)

Models compared:
  - LS (Least Squares)
  - MMSE
  - NCRN (ConvLSTM baseline)
  - NCRT (Pre-trained on synthetic TDL)
  - NCRT (Trained on MATLAB 5G+6G dataset)

Outputs:
  - outputs/csv/matlab_5g_6g_eval.csv
  - outputs/figures/matlab_5g_eval.png
  - outputs/figures/matlab_6g_eval.png
  - outputs/figures/matlab_5g_6g_combined_eval.png
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
    'ncrt_matlab',
    'ncrt_matlab_raw',
]

LABELS = {
    'ls': 'LS',
    'mmse': 'MMSE',
    'ncrn': 'NCRN MATLAB-Trained (blended)',
    'ncrn_raw': 'NCRN MATLAB-Trained (raw)',
    'ncrt_base': 'NCRT Synthetic (blended)',
    'ncrt_base_raw': 'NCRT Synthetic (raw)',
    'ncrt_matlab': 'NCRT MATLAB-Trained (blended)',
    'ncrt_matlab_raw': 'NCRT MATLAB-Trained (raw)',
}

STYLE = {
    'ls': ('#555555', '--', '^'),
    'mmse': ('#1B5E20', '-.', 'D'),
    'ncrn': ('#B71C1C', '-', 'o'),
    'ncrn_raw': ('#E57373', ':', 'x'),
    'ncrt_base': ('#FF6F00', '--', 'v'),
    'ncrt_base_raw': ('#FFA000', ':', '.'),
    'ncrt_matlab': ('#0D47A1', '-', 's'),
    'ncrt_matlab_raw': ('#1976D2', ':', '+'),
}


def load_model_safe(path):
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path, custom_objects={'LastFrame': LastFrame},
                                      compile=False, safe_mode=False)


def evaluate_dataset(models, infer_funcs, channels, n_test=20, bs=10):
    """
    Evaluates models on a set of channel tensors (N_samples, T, Nr, Nt, 2).
    Uses the last n_test samples as held-out test split.
    """
    test_channels = channels[-n_test:]
    N = len(test_channels)
    active_methods = [m for m in METHODS if (m in ['ls', 'mmse'] or 
                     (m.startswith('ncrn') and models.get('ncrn') is not None) or
                     (m.startswith('ncrt_base') and models.get('ncrt_base') is not None) or
                     (m.startswith('ncrt_matlab') and models.get('ncrt_matlab') is not None))]

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
            for k in ['ncrn', 'ncrt_base', 'ncrt_matlab']:
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

                for k in ['ncrn', 'ncrt_base', 'ncrt_matlab']:
                    if k in preds:
                        raw_pred = preds[k][i]
                        est[k] = snr_aware_blend(raw_pred, hl, snr_db_known=snr)
                        est[k + '_raw'] = hl + raw_pred

                for m in active_methods:
                    sq[m] += np.sum((hr - est[m]) ** 2)

        for m in active_methods:
            nmse[m].append(sq[m] / (den + 1e-12))
        
        db_ncrn = 10 * np.log10(nmse['ncrn_raw'][-1]) if 'ncrn_raw' in nmse else 0
        db_matlab = 10 * np.log10(nmse['ncrt_matlab_raw'][-1]) if 'ncrt_matlab_raw' in nmse else 0
        print(f"done ({time.time()-t_snr:.1f}s) | NCRN: {db_ncrn:+6.2f} dB | NCRT-Matlab: {db_matlab:+6.2f} dB")

    return nmse, active_methods


def plot_and_save(nmse, active_methods, title, out_png, out_csv):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['snr_db'] + [f'{m}_nmse_db' for m in active_methods])
        for i, snr in enumerate(SNR_RANGE):
            w.writerow([snr] + [f'{10 * np.log10(nmse[m][i]):.4f}' for m in active_methods])

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for m in active_methods:
        c, ls, mk = STYLE[m]
        ax.plot(SNR_RANGE, 10 * np.log10(nmse[m]), color=c, linestyle=ls, marker=mk,
                markersize=5, linewidth=1.5, label=LABELS[m])

    ax.set_xlabel('SNR (dB)', fontsize=11, fontweight='bold')
    ax.set_ylabel('NMSE (dB)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  Saved figure: {out_png}")
    print(f"  Saved CSV:    {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_samples', type=int, default=20)
    parser.add_argument('--ncrt_matlab_model', default='models/ncrt_tdl_64_matlab_5g_6g.keras')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("  EVALUATING MODEL PERFORMANCE ON MATLAB 5G & 6G CHANNEL DATA")
    print(f"{'='*70}\n")

    # Load models
    models = {
        'ncrn': load_model_safe(os.path.join(ROOT, 'models', 'ncrn_tdl_64_matlab_5g_6g.keras')),
        'ncrt_base': load_model_safe(os.path.join(ROOT, 'models', 'ncrt_tdl_64.keras')),
        'ncrt_matlab': load_model_safe(os.path.join(ROOT, args.ncrt_matlab_model)),
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
            # Warm up
            _ = infer_funcs[name](tf.zeros((1, 10, 64, 64, 3), dtype=tf.float32))
        else:
            print(f"Model [{name}]: NOT FOUND")

    # Datasets to evaluate
    datasets = [
        ('5G NR TDL Channels (MATLAB)',
         os.path.join(ROOT, 'data', 'matlab_5g_channels.mat'),
         os.path.join(ROOT, 'outputs', 'figures', 'matlab_5g_eval.png'),
         os.path.join(ROOT, 'outputs', 'csv', 'matlab_5g_eval.csv')),
        ('6G Sub-THz 142 GHz Channels (Hexa-X / MATLAB)',
         os.path.join(ROOT, 'data', 'matlab_6g_channels.mat'),
         os.path.join(ROOT, 'outputs', 'figures', 'matlab_6g_eval.png'),
         os.path.join(ROOT, 'outputs', 'csv', 'matlab_6g_eval.csv')),
        ('Combined 5G + 6G Cross-Generation Dataset',
         os.path.join(ROOT, 'data', 'matlab_5g_6g_channels.mat'),
         os.path.join(ROOT, 'outputs', 'figures', 'matlab_5g_6g_combined_eval.png'),
         os.path.join(ROOT, 'outputs', 'csv', 'matlab_5g_6g_combined_eval.csv')),
    ]

    summary_results = {}

    for dname, mat_path, fig_path, csv_path in datasets:
        if not os.path.exists(mat_path):
            print(f"[SKIP] Dataset not found: {mat_path}")
            continue

        print(f"\n--- Evaluating on: {dname} ---")
        channels = load_matlab_6g_channels(mat_path, target_Nr=64, target_Nt=64)
        nmse, active_methods = evaluate_dataset(models, infer_funcs, channels, n_test=args.test_samples)
        plot_and_save(nmse, active_methods, f"Performance on {dname}", fig_path, csv_path)

        # Print summary table at key SNRs (-10, 0, 10, 20 dB)
        print(f"\nSummary NMSE (dB) for {dname}:")
        header = f"{'Method':<28}" + "".join([f"{snr:>8} dB" for snr in [-10, 0, 10, 20]])
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for m in active_methods:
            row = f"{LABELS[m]:<28}"
            for snr_idx, snr in enumerate(SNR_RANGE):
                if snr in [-10, 0, 10, 20]:
                    row += f"{10 * np.log10(nmse[m][snr_idx]):>10.2f} dB"
            print(row)
        print("-" * len(header))

        summary_results[dname] = {m: [10 * np.log10(v) for v in nmse[m]] for m in active_methods}

    print("\n[COMPLETE] All MATLAB evaluations and figure generation complete!")


if __name__ == '__main__':
    main()

"""
eval_ncrt_vs_ncrn.py
────────────────────────────────────────────────────────────────────────
NCRT vs NCRN on identical TDL samples (same channel + noise realisation
fed to both models), full-pilot, 64x64, SNR -15..20 dB.

Both models are scored raw (h_noisy + residual) and with NCRN's
snr_aware_blend, so neither gets a post-processing advantage.

  python eval_ncrt_vs_ncrn.py --samples 100
Outputs:
  outputs/csv/ncrt_vs_ncrn_full_pilot.csv
  outputs/figures/ncrt_vs_ncrn_full_pilot.png (+ .pdf)
"""
import os, sys, csv, time, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(HERE, '..', 'ncrn'))
sys.path.insert(0, HERE)

from shared.channels import gen_tdl_random, gen_tdl_channel, SNAMES
from shared.baselines import ls_estimate, mmse_estimate, add_snr_channel, snr_aware_blend
from train_ncrn_tdl import LastFrame
import ncrt_layers  # noqa: F401  (registers custom layers)

SNR_RANGE = [-15, -10, -5, 0, 5, 10, 15, 20]
METHODS = ['ls', 'mmse', 'ncrn', 'ncrn_raw', 'ncrt', 'ncrt_raw']
LABELS = {'ls': 'LS', 'mmse': 'MMSE', 'ncrn': 'NCRN (ConvLSTM, blended)',
          'ncrn_raw': 'NCRN (raw residual)', 'ncrt': 'NCRT (Transformer, blended)',
          'ncrt_raw': 'NCRT (raw residual)'}
STYLE = {'ls': ('#555555', '--', '^'), 'mmse': ('#1B5E20', '-.', 'D'),
         'ncrn': ('#B71C1C', '-', 'o'), 'ncrn_raw': ('#E57373', ':', 'x'),
         'ncrt': ('#0D47A1', '-', 's'), 'ncrt_raw': ('#64B5F6', ':', '+')}


def load(path):
    return tf.keras.models.load_model(path, custom_objects={'LastFrame': LastFrame},
                                      compile=False, safe_mode=False)


def evaluate(models, n_samples, scenario=None, Nr=64, Nt=64, bs=8):
    """Returns {method: [lin NMSE per SNR]} using identical inputs for all methods."""
    nmse = {m: [] for m in METHODS}
    for snr in SNR_RANGE:
        sq = {m: 0.0 for m in METHODS}; den = 0.0
        for start in range(0, n_samples, bs):
            X, HL, HR, SP, NS = [], [], [], [], []
            for _ in range(min(bs, n_samples - start)):
                h = gen_tdl_channel(scenario, Nr, Nt) if scenario else gen_tdl_random(Nr, Nt)
                sp = float(np.mean(h ** 2)) + 1e-10
                ns = np.sqrt(sp * 10 ** (-snr / 10))
                hn = (h + np.random.normal(0, ns, h.shape)).astype(np.float32)
                X.append(add_snr_channel(hn, snr)); HL.append(hn[-1]); HR.append(h[-1])
                SP.append(sp); NS.append(ns)
            X = np.array(X)
            raw = {k: models[k].predict(X, verbose=0, batch_size=bs) for k in ('ncrn', 'ncrt')}
            for i, (hl, hr) in enumerate(zip(HL, HR)):
                den += np.sum(hr ** 2)
                est = {'ls': ls_estimate(hl),
                       'mmse': mmse_estimate(hl, noise_var=NS[i] ** 2, Nt=Nt, sp=SP[i])}
                for k in ('ncrn', 'ncrt'):
                    est[k] = snr_aware_blend(raw[k][i], hl, snr_db_known=snr)
                    est[k + '_raw'] = hl + raw[k][i]
                for m in METHODS:
                    sq[m] += np.sum((hr - est[m]) ** 2)
        for m in METHODS:
            nmse[m].append(sq[m] / den)
        db = lambda m: 10 * np.log10(nmse[m][-1])
        print(f"  SNR {snr:>3} dB | LS {db('ls'):+6.2f} | MMSE {db('mmse'):+6.2f} | "
              f"NCRN raw {db('ncrn_raw'):+6.2f} | NCRT raw {db('ncrt_raw'):+6.2f} | "
              f"Δ {db('ncrt_raw') - db('ncrn_raw'):+5.2f} dB")
    return nmse


def save(nmse, tag):
    csv_path = os.path.join(ROOT, 'outputs', 'csv', f'ncrt_vs_ncrn_{tag}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['snr_db'] + [f'{m}_nmse_db' for m in METHODS])
        for i, snr in enumerate(SNR_RANGE):
            w.writerow([snr] + [f'{10 * np.log10(nmse[m][i]):.4f}' for m in METHODS])
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    for m in METHODS:
        c, ls, mk = STYLE[m]
        ax.plot(SNR_RANGE, 10 * np.log10(nmse[m]), color=c, linestyle=ls, marker=mk,
                markersize=4.5, linewidth=1.4, label=LABELS[m])
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('NMSE (dB)')
    ax.set_title(f'NCRT vs NCRN — {tag.replace("_", " ")}', fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6); ax.legend(loc='lower left', fontsize=7)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(ROOT, 'outputs', 'figures', f'ncrt_vs_ncrn_{tag}.{ext}'), dpi=300)
    plt.close(fig)
    print(f"  Saved {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=100, help='samples per SNR')
    ap.add_argument('--ncrt', default=os.path.join(ROOT, 'models', 'ncrt_tdl_64.keras'))
    ap.add_argument('--per_scenario', action='store_true')
    ap.add_argument('--seed', type=int, default=1234)
    a = ap.parse_args()
    np.random.seed(a.seed)

    models = {'ncrn': load(os.path.join(ROOT, 'models', 'ncrn_tdl_64.keras')), 'ncrt': load(a.ncrt)}
    for k, m in models.items():
        x = np.random.randn(1, 10, 64, 64, 3).astype(np.float32)
        m.predict(x, verbose=0); t0 = time.time()
        for _ in range(10): m.predict(x, verbose=0)
        print(f"  {k.upper()}: {m.count_params():,} params, {(time.time() - t0) * 100:.1f} ms/sample")

    print(f"\n  FULL-PILOT, all 10 TDL scenarios mixed, {a.samples} samples/SNR")
    save(evaluate(models, a.samples), 'full_pilot')
    if a.per_scenario:
        for s in SNAMES:
            print(f"\n  Scenario {s}")
            save(evaluate(models, max(a.samples // 4, 8), scenario=s), f'{s}_full_pilot')


if __name__ == '__main__':
    main()

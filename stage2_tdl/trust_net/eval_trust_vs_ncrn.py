"""
eval_trust_vs_ncrn.py
────────────────────────────────────────────────────────────────────────
Head-to-head verification of TRUST-Net vs NCRN under identical
3GPP TR 38.901 TDL channel conditions.

Evaluates:
  - LS (Least Squares baseline)
  - MMSE (Linear MMSE baseline)
  - NCRN (Wide ConvLSTM2D, 1.20M params)
  - TRUST-Net (Temporal Self-Attention, matched 1.19M params)
  - DAUGT (if present)

Metrics:
  - NMSE (Normalized Mean Square Error in dB)
  - MAE (Mean Absolute Error)
  - Inference Latency (ms per sample)
  - Model Parameter Count & Efficiency

Outputs:
  - outputs/csv/trust_vs_ncrn_full_pilot.csv
  - outputs/csv/trust_vs_ncrn_sparse_pilot.csv
  - outputs/figures/trust_vs_ncrn_nmse.png (+ .pdf)
"""

import os, sys, time, csv, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(HERE, '..', 'ncrn'))
sys.path.insert(0, HERE)

from shared.channels import gen_tdl_random, TIME_STEPS
from shared.baselines import (ls_estimate, mmse_estimate, add_snr_channel,
                              snr_aware_blend)
from shared.pilot import gen_pilot_mask
from shared.evidential import ubmmse_fuse, decompose_uncertainty
from train_ncrn_tdl import LastFrame as NCRN_LastFrame
import trust_layers  # registers custom layers

MODELS_DIR = os.path.join(ROOT, 'models')
OUT_FIG = os.path.join(ROOT, 'outputs', 'figures')
OUT_CSV = os.path.join(ROOT, 'outputs', 'csv')
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_CSV, exist_ok=True)

SNR_RANGE = [-15, -10, -5, 0, 5, 10, 15, 20]
PILOT_DENSITY = 0.35

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.frameon': True,
    'legend.framealpha': 0.92,
    'axes.linewidth': 0.7,
    'grid.linewidth': 0.4,
    'grid.linestyle': ':',
    'grid.alpha': 0.45,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

COLORS = {
    'ls': '#555555',
    'mmse': '#1B5E20',
    'ncrn': '#B71C1C',
    'ncrn_raw': '#E57373',
    'trust_net': '#0D47A1',
    'trust_net_raw': '#64B5F6',
    'trust_net_a2': '#4A148C',
    'daugt': '#E65100',
}
STYLES = {
    'ls': '--',
    'mmse': '-.',
    'ncrn': '-',
    'ncrn_raw': ':',
    'trust_net': '-',
    'trust_net_raw': ':',
    'trust_net_a2': '-.',
    'daugt': '-',
}
MARKERS = {
    'ls': '^',
    'mmse': 'D',
    'ncrn': 'o',
    'ncrn_raw': 'x',
    'trust_net': 's',
    'trust_net_raw': '+',
    'trust_net_a2': 'v',
    'daugt': 'P',
}
LABELS = {
    'ls': 'LS',
    'mmse': 'MMSE',
    'ncrn': 'NCRN (ConvLSTM, blended)',
    'ncrn_raw': 'NCRN (raw residual)',
    'trust_net': 'TRUST-Net (MHSA, blended)',
    'trust_net_raw': 'TRUST-Net (raw residual)',
    'trust_net_a2': 'TRUST-Net-a2 (Evidential)',
    'daugt': 'DAUGT V2 (Transformer)',
}


def load_model_safe(path, custom=None):
    if not os.path.exists(path):
        return None
    try:
        return tf.keras.models.load_model(path, custom_objects=custom, compile=False, safe_mode=False)
    except Exception as e:
        print(f"  [WARN] Failed to load {path}: {e}")
        return None


def benchmark_latency(model, shape=(1, TIME_STEPS, 64, 64, 3), n_warmup=10, n_iter=50):
    """Measures single-sample inference latency in milliseconds."""
    dummy = np.random.randn(*shape).astype(np.float32)
    # Warmup
    for _ in range(n_warmup):
        _ = model.predict(dummy, verbose=0)
    # Timed run
    t0 = time.time()
    for _ in range(n_iter):
        _ = model.predict(dummy, verbose=0)
    avg_ms = ((time.time() - t0) / n_iter) * 1000.0
    return avg_ms


def evaluate_regime(models, Nr=64, Nt=64, n_samples=50, sparse=False):
    regime_name = "SPARSE-PILOT (35%)" if sparse else "FULL-PILOT (100%)"
    print(f"\n{'='*72}")
    print(f"  EVALUATION REGIME: {regime_name} | {Nr}x{Nt} | {n_samples} samples/SNR")
    print(f"{'='*72}")

    active_methods = ['ls', 'mmse']
    if 'ncrn' in models:
        active_methods.extend(['ncrn', 'ncrn_raw'])
    if 'trust_net' in models:
        active_methods.extend(['trust_net', 'trust_net_raw'])
    if 'trust_net_a2' in models:
        active_methods.append('trust_net_a2')
    if 'daugt' in models:
        active_methods.append('daugt')

    nmse_results = {m: [] for m in active_methods}
    mae_results = {m: [] for m in active_methods}

    for snr in SNR_RANGE:
        sq_errs = {m: [] for m in active_methods}
        abs_errs = {m: [] for m in active_methods}
        denom_sum = 0.0

        for _ in range(n_samples):
            h = gen_tdl_random(Nr, Nt)
            sp = float(np.mean(h ** 2)) + 1e-10
            ns = np.sqrt(sp * 10 ** (-snr / 10))
            hn = (h + np.random.normal(0, ns, h.shape)).astype(np.float32)
            hr = h[-1]
            denom_sum += np.sum(hr ** 2) + 1e-12

            if sparse:
                mask = gen_pilot_mask(Nr, Nt, PILOT_DENSITY)
            else:
                mask = np.ones((Nr, Nt), dtype=np.float32)

            hl_obs = hn[-1] * mask[..., None]

            # 1. LS Baseline
            h_ls = ls_estimate(hl_obs)
            sq_errs['ls'].append(np.sum((hr - h_ls) ** 2))
            abs_errs['ls'].append(np.mean(np.abs(hr - h_ls)))

            # 2. MMSE Baseline
            h_mmse = mmse_estimate(hl_obs, noise_var=ns ** 2, Nt=Nt, sp=sp)
            sq_errs['mmse'].append(np.sum((hr - h_mmse) ** 2))
            abs_errs['mmse'].append(np.mean(np.abs(hr - h_mmse)))

            # Masked input sequence for models
            hn_obs_seq = hn * mask[None, ..., None]
            ncin = add_snr_channel(hn_obs_seq, snr)

            # 3. NCRN
            if 'ncrn' in models:
                raw_ncrn = models['ncrn'].predict(ncin[None, ...], verbose=0)[0]
                pred_ncrn_blend = snr_aware_blend(raw_ncrn, hl_obs, snr_db_known=snr)
                pred_ncrn_raw = hl_obs + raw_ncrn
                sq_errs['ncrn'].append(np.sum((hr - pred_ncrn_blend) ** 2))
                abs_errs['ncrn'].append(np.mean(np.abs(hr - pred_ncrn_blend)))
                sq_errs['ncrn_raw'].append(np.sum((hr - pred_ncrn_raw) ** 2))
                abs_errs['ncrn_raw'].append(np.mean(np.abs(hr - pred_ncrn_raw)))

            # 4. TRUST-Net a1
            if 'trust_net' in models:
                raw_trust = models['trust_net'].predict(ncin[None, ...], verbose=0)[0]
                pred_trust_blend = snr_aware_blend(raw_trust, hl_obs, snr_db_known=snr)
                pred_trust_raw = hl_obs + raw_trust
                sq_errs['trust_net'].append(np.sum((hr - pred_trust_blend) ** 2))
                abs_errs['trust_net'].append(np.mean(np.abs(hr - pred_trust_blend)))
                sq_errs['trust_net_raw'].append(np.sum((hr - pred_trust_raw) ** 2))
                abs_errs['trust_net_raw'].append(np.mean(np.abs(hr - pred_trust_raw)))

            # 5. TRUST-Net a2 (Evidential)
            if 'trust_net_a2' in models:
                raw_a2 = models['trust_net_a2'].predict(ncin[None, ...], verbose=0)[0]
                # Gamma sits at channels 0 and 4
                gamma = np.stack([raw_a2[..., 0], raw_a2[..., 4]], axis=-1)
                pred_a2 = hl_obs + gamma
                sq_errs['trust_net_a2'].append(np.sum((hr - pred_a2) ** 2))
                abs_errs['trust_net_a2'].append(np.mean(np.abs(hr - pred_a2)))

        for m in active_methods:
            lin_nmse = sum(sq_errs[m]) / denom_sum
            nmse_results[m].append(lin_nmse)
            mae_results[m].append(float(np.mean(abs_errs[m])))

        print(f"  SNR {snr:>3} dB | " + " | ".join(
            f"{m.upper()}: {10 * np.log10(nmse_results[m][-1] + 1e-12):+6.2f} dB"
            for m in ['ls', 'mmse', 'ncrn', 'trust_net'] if m in active_methods
        ))

    return active_methods, nmse_results, mae_results


def plot_comparison(active_methods, nmse_results, tag="full_pilot"):
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    for m in active_methods:
        if m not in COLORS:
            continue
        vals_db = [10 * np.log10(v + 1e-12) for v in nmse_results[m]]
        ax.plot(SNR_RANGE, vals_db, color=COLORS[m], linestyle=STYLES.get(m, '-'),
                marker=MARKERS.get(m, 'o'), markersize=4.5, linewidth=1.4,
                label=LABELS.get(m, m))

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('NMSE (dB)')
    title = 'TRUST-Net vs NCRN — Full-Pilot Regime' if 'full' in tag else 'TRUST-Net vs NCRN — Sparse-Pilot Regime (35%)'
    ax.set_title(title, fontweight='bold', pad=8)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim(min(SNR_RANGE) - 1, max(SNR_RANGE) + 1)
    ax.legend(loc='lower left', framealpha=0.92)
    fig.tight_layout()

    out_png = os.path.join(OUT_FIG, f'trust_vs_ncrn_{tag}.png')
    out_pdf = os.path.join(OUT_FIG, f'trust_vs_ncrn_{tag}.pdf')
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"\n  Saved plot -> {out_png}")


def save_csv(active_methods, nmse_results, mae_results, tag="full_pilot"):
    csv_path = os.path.join(OUT_CSV, f'trust_vs_ncrn_{tag}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['snr_db']
        for m in active_methods:
            header.extend([f'{m}_nmse_db', f'{m}_nmse_lin', f'{m}_mae'])
        writer.writerow(header)

        for i, snr in enumerate(SNR_RANGE):
            row = [snr]
            for m in active_methods:
                lin = nmse_results[m][i]
                db = 10 * np.log10(lin + 1e-12)
                mae = mae_results[m][i]
                row.extend([f'{db:.4f}', f'{lin:.6e}', f'{mae:.6f}'])
            writer.writerow(row)
    print(f"  Saved CSV  -> {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=50, help='samples per SNR')
    parser.add_argument('--trust_model', default='', help='path to trust_net keras file')
    parser.add_argument('--eval_sparse', action='store_true', help='also run sparse pilot eval')
    args = parser.parse_args()

    custom_objs = {'LastFrame': NCRN_LastFrame}

    # Locate models
    ncrn_path = os.path.join(MODELS_DIR, 'ncrn_tdl_64.keras')
    if args.trust_model:
        trust_path = args.trust_model
    else:
        # Default search
        cands = [
            os.path.join(MODELS_DIR, 'trust_net_a1_seed0.keras'),
            os.path.join(MODELS_DIR, 'trust_net_a1.keras'),
        ]
        trust_path = next((p for p in cands if os.path.exists(p)), cands[0])

    trust_a2_path = os.path.join(MODELS_DIR, 'trust_net_a2_seed0.keras')

    models = {}
    print("\nLoading models for verification...")
    ncrn_m = load_model_safe(ncrn_path, custom_objs)
    if ncrn_m is not None:
        models['ncrn'] = ncrn_m
        print(f"  [OK] Loaded NCRN: {ncrn_path} ({ncrn_m.count_params():,} params)")
    else:
        print(f"  [MISSING] NCRN not found at {ncrn_path}")

    trust_m = load_model_safe(trust_path, custom_objs)
    if trust_m is not None:
        models['trust_net'] = trust_m
        print(f"  [OK] Loaded TRUST-Net: {trust_path} ({trust_m.count_params():,} params)")
    else:
        print(f"  [MISSING] TRUST-Net not found at {trust_path}")

    if os.path.exists(trust_a2_path):
        trust_a2_m = load_model_safe(trust_a2_path, custom_objs)
        if trust_a2_m is not None:
            models['trust_net_a2'] = trust_a2_m
            print(f"  [OK] Loaded TRUST-Net-a2: {trust_a2_path} ({trust_a2_m.count_params():,} params)")

    if not models:
        print("ERROR: Neither NCRN nor TRUST-Net could be loaded. Exiting.")
        return

    # Latency and parameter benchmarking
    print(f"\n{'='*72}")
    print("  MODEL ARCHITECTURE & BENCHMARK")
    print(f"{'='*72}")
    if 'ncrn' in models and 'trust_net' in models:
        ncrn_p = models['ncrn'].count_params()
        trust_p = models['trust_net'].count_params()
        diff_pct = (trust_p - ncrn_p) / ncrn_p * 100.0
        print(f"  NCRN Parameters:      {ncrn_p:,}")
        print(f"  TRUST-Net Parameters: {trust_p:,}  ({diff_pct:+.2f}% vs NCRN)")

        print("\n  Measuring inference latency (50 single-sample runs)...")
        ncrn_lat = benchmark_latency(models['ncrn'])
        trust_lat = benchmark_latency(models['trust_net'])
        print(f"  NCRN Latency:      {ncrn_lat:.2f} ms/sample")
        print(f"  TRUST-Net Latency: {trust_lat:.2f} ms/sample (speedup: {ncrn_lat/trust_lat:.2f}x)")

    # 1. Full-pilot evaluation
    active, nmse, mae = evaluate_regime(models, Nr=64, Nt=64, n_samples=args.samples, sparse=False)
    plot_comparison(active, nmse, tag="full_pilot")
    save_csv(active, nmse, mae, tag="full_pilot")

    # 2. Sparse-pilot evaluation (if requested)
    if args.eval_sparse:
        active_sp, nmse_sp, mae_sp = evaluate_regime(models, Nr=64, Nt=64, n_samples=args.samples, sparse=True)
        plot_comparison(active_sp, nmse_sp, tag="sparse_pilot")
        save_csv(active_sp, nmse_sp, mae_sp, tag="sparse_pilot")

    # Summary table
    print(f"\n{'='*72}")
    print("  VERIFICATION SUMMARY: NMSE (dB)")
    print(f"{'='*72}")
    header_str = f"{'SNR (dB)':>10} | {'LS':>8} | {'MMSE':>8}"
    if 'ncrn' in active:
        header_str += f" | {'NCRN':>8}"
    if 'trust_net' in active:
        header_str += f" | {'TRUST-Net':>10} | {'Delta (dB)':>10}"
    print(header_str)
    print("-" * len(header_str))

    for i, snr in enumerate(SNR_RANGE):
        ls_db = 10 * np.log10(nmse['ls'][i] + 1e-12)
        mmse_db = 10 * np.log10(nmse['mmse'][i] + 1e-12)
        row_str = f"{snr:>10} | {ls_db:>8.2f} | {mmse_db:>8.2f}"
        if 'ncrn' in active:
            ncrn_db = 10 * np.log10(nmse['ncrn'][i] + 1e-12)
            row_str += f" | {ncrn_db:>8.2f}"
        if 'trust_net' in active:
            trust_db = 10 * np.log10(nmse['trust_net'][i] + 1e-12)
            delta = trust_db - ncrn_db if 'ncrn' in active else 0.0
            row_str += f" | {trust_db:>10.2f} | {delta:>+10.2f}"
        print(row_str)

    print("=" * len(header_str))
    print("\nVerification complete.")


if __name__ == '__main__':
    main()

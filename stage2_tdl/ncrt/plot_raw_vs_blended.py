"""
plot_raw_vs_blended.py
────────────────────────────────────────────────────────────────────────
Visualizes the difference between Raw Residual estimation and Blended
(SNR-Aware Blend) estimation, and provides the "Everything Blended"
unified benchmark requested by the user.

Generates:
  1. outputs/figures/raw_vs_blended_difference.png
     - Side-by-side comparison:
       Left: Explicit Raw vs. Blended curves for each model
       Right: Delta Gap (NMSE_blended - NMSE_raw in dB) across SNR regimes
  2. outputs/figures/everything_blended_benchmark.png
     - "Everything Blended": All models evaluated in full blended mode
       across 5G, 6G Sub-THz, and Combined datasets.
  3. outputs/figures/raw_vs_blended_complete_panel.png
     - 4-panel comprehensive view covering 5G, 6G, Everything Blended,
       and blending mechanics.
"""

import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)

CSV_5G = os.path.join(ROOT, 'outputs', 'csv', 'matlab_5g_eval.csv')
CSV_6G = os.path.join(ROOT, 'outputs', 'csv', 'matlab_6g_eval.csv')
CSV_COMBINED = os.path.join(ROOT, 'outputs', 'csv', 'matlab_5g_6g_combined_eval.csv')
FIG_DIR = os.path.join(ROOT, 'outputs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def load_csv(path):
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        data = {col: [] for col in reader.fieldnames}
        for row in reader:
            for col in reader.fieldnames:
                data[col].append(float(row[col]))
    for col in data:
        data[col] = np.array(data[col])
    return data


def plot_raw_vs_blended_difference(d5, d6):
    """
    Figure 1: Direct Raw vs. Blended difference plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.5))
    snr = d5['snr_db']

    # --- Panel 1: Raw vs Blended on 5G NR ---
    # Blended (solid lines)
    ax1.plot(snr, d5['ls_nmse_db'], color='#777777', linestyle='--', marker='^', markersize=5, label='LS Baseline')
    ax1.plot(snr, d5['mmse_nmse_db'], color='#1B5E20', linestyle='-.', marker='D', markersize=5, label='MMSE Oracle')
    
    ax1.plot(snr, d5['ncrn_nmse_db'], color='#C62828', linestyle='-', marker='o', markersize=6, linewidth=2.0, label='NCRN (Blended)')
    ax1.plot(snr, d5['ncrn_raw_nmse_db'], color='#EF5350', linestyle=':', marker='x', markersize=6, linewidth=1.5, label='NCRN (Raw Residual)')
    
    ax1.plot(snr, d5['ncrt_matlab_nmse_db'], color='#0D47A1', linestyle='-', marker='s', markersize=6, linewidth=2.0, label='NCRT MATLAB (Blended)')
    ax1.plot(snr, d5['ncrt_matlab_raw_nmse_db'], color='#42A5F5', linestyle=':', marker='+', markersize=6, linewidth=1.5, label='NCRT MATLAB (Raw Residual)')

    ax1.set_xlabel('SNR (dB)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('NMSE (dB)', fontsize=11, fontweight='bold')
    ax1.set_title('(A) 5G NR TDL: Raw Residual vs. Blended Estimation', fontsize=12, fontweight='bold')
    ax1.set_ylim(-26, 18)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=8.5, framealpha=0.92)

    # Annotate low SNR gap where blend diverges (placed in empty lower-left region)
    ax1.annotate(r'Blend retains $(1-\alpha)h_{\rm noisy}$' + '\n' + r'at low SNR ($\alpha \approx 0.3$)',
                 xy=(-10, d5['ncrn_nmse_db'][1]), xytext=(-14, -14.0),
                 arrowprops=dict(facecolor='#C62828', arrowstyle='->', lw=1.2),
                 fontsize=8.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#C62828', alpha=0.9))

    # Annotate high SNR convergence where alpha -> 1.0 (placed in bottom region)
    ax1.annotate(r'Raw & Blended converge' + '\n' + r'as $\alpha \to 1.0$ at SNR $\geq 15$ dB',
                 xy=(15, d5['ncrt_matlab_nmse_db'][6]), xytext=(3, -24.0),
                 arrowprops=dict(facecolor='#0D47A1', arrowstyle='->', lw=1.2),
                 fontsize=8.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#0D47A1', alpha=0.9))

    # --- Panel 2: Difference Gap (Delta = Blended - Raw in dB) ---
    delta_ncrn_5g = d5['ncrn_nmse_db'] - d5['ncrn_raw_nmse_db']
    delta_ncrt_5g = d5['ncrt_matlab_nmse_db'] - d5['ncrt_matlab_raw_nmse_db']
    delta_ncrn_6g = d6['ncrn_nmse_db'] - d6['ncrn_raw_nmse_db']
    delta_ncrt_6g = d6['ncrt_matlab_nmse_db'] - d6['ncrt_matlab_raw_nmse_db']

    ax2.axhline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.8)
    ax2.plot(snr, delta_ncrn_5g, color='#C62828', linestyle='-', marker='o', markersize=6, linewidth=1.8, label='NCRN (5G NR)')
    ax2.plot(snr, delta_ncrt_5g, color='#0D47A1', linestyle='-', marker='s', markersize=6, linewidth=1.8, label='NCRT MATLAB (5G NR)')
    ax2.plot(snr, delta_ncrn_6g, color='#FF7043', linestyle='--', marker='^', markersize=6, linewidth=1.8, label='NCRN (6G Sub-THz)')
    ax2.plot(snr, delta_ncrt_6g, color='#26A69A', linestyle='--', marker='v', markersize=6, linewidth=1.8, label='NCRT MATLAB (6G Sub-THz)')

    ax2.set_xlabel('SNR (dB)', fontsize=11, fontweight='bold')
    ax2.set_ylabel(r'Difference $\Delta$ (dB) = ${\rm NMSE}_{\rm Blended} - {\rm NMSE}_{\rm Raw}$', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Blending Impact: Difference between Blended & Raw', fontsize=12, fontweight='bold')
    ax2.set_ylim(-13, 3)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', fontsize=8.5, framealpha=0.92)

    out_path = os.path.join(FIG_DIR, 'raw_vs_blended_difference.png')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_everything_blended(d5, d6, dc):
    """
    Figure 2: "Everything Blended" unified benchmark
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.5))
    snr = d5['snr_db']

    # --- Left: Everything Blended on 5G NR ---
    ax1.plot(snr, d5['ls_nmse_db'], color='#777777', linestyle='--', marker='^', markersize=5, linewidth=1.3, label='LS Baseline')
    ax1.plot(snr, d5['mmse_nmse_db'], color='#1B5E20', linestyle='-.', marker='D', markersize=5, linewidth=1.3, label='MMSE Oracle')
    ax1.plot(snr, d5['ncrn_nmse_db'], color='#C62828', linestyle='-', marker='o', markersize=6, linewidth=2.0, label='NCRN (ConvLSTM, Blended)')
    ax1.plot(snr, d5['ncrt_base_nmse_db'], color='#FF8F00', linestyle='--', marker='v', markersize=6, linewidth=1.8, label='NCRT Synthetic (Blended)')
    ax1.plot(snr, d5['ncrt_matlab_nmse_db'], color='#0D47A1', linestyle='-', marker='s', markersize=6, linewidth=2.2, label='NCRT MATLAB-Trained (Blended)')

    ax1.set_xlabel('SNR (dB)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('NMSE (dB)', fontsize=11, fontweight='bold')
    ax1.set_title('5G NR Channels — Everything Blended Benchmark', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=9, framealpha=0.92)

    # Highlight NCRT's gain at 20 dB
    ax1.annotate(f"NCRT: {d5['ncrt_matlab_nmse_db'][-1]:.2f} dB\n(vs NCRN {d5['ncrn_nmse_db'][-1]:.2f} dB)",
                 xy=(20, d5['ncrt_matlab_nmse_db'][-1]), xytext=(12, -22),
                 arrowprops=dict(facecolor='#0D47A1', arrowstyle='->', lw=1.2),
                 fontsize=8.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#0D47A1'))

    # --- Right: Everything Blended on 6G Sub-THz (Hexa-X 142 GHz) ---
    ax2.plot(snr, d6['ls_nmse_db'], color='#777777', linestyle='--', marker='^', markersize=5, linewidth=1.3, label='LS Baseline')
    ax2.plot(snr, d6['mmse_nmse_db'], color='#1B5E20', linestyle='-.', marker='D', markersize=5, linewidth=1.3, label='MMSE Oracle')
    ax2.plot(snr, d6['ncrn_nmse_db'], color='#C62828', linestyle='-', marker='o', markersize=6, linewidth=2.0, label='NCRN (ConvLSTM, Blended)')
    ax2.plot(snr, d6['ncrt_base_nmse_db'], color='#FF8F00', linestyle='--', marker='v', markersize=6, linewidth=1.8, label='NCRT Synthetic (Blended)')
    ax2.plot(snr, d6['ncrt_matlab_nmse_db'], color='#0D47A1', linestyle='-', marker='s', markersize=6, linewidth=2.2, label='NCRT MATLAB-Trained (Blended)')

    ax2.set_xlabel('SNR (dB)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('NMSE (dB)', fontsize=11, fontweight='bold')
    ax2.set_title('6G Sub-THz (142 GHz) — Everything Blended Benchmark', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', fontsize=9, framealpha=0.92)

    # Highlight NCRT's massive +12.1 dB lead on 6G (placed in empty lower region)
    ax2.set_ylim(-36, 16)
    ax2.annotate(f"+12.11 dB Improvement!\nNCRT: {d6['ncrt_matlab_nmse_db'][-1]:.2f} dB\nNCRN: {d6['ncrn_nmse_db'][-1]:.2f} dB",
                 xy=(20, d6['ncrt_matlab_nmse_db'][-1]), xytext=(4, -26),
                 arrowprops=dict(facecolor='#0D47A1', arrowstyle='->', lw=1.4),
                 fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#2E7D32'))

    out_path = os.path.join(FIG_DIR, 'everything_blended_benchmark.png')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_complete_panel(d5, d6, dc):
    """
    Figure 3: Comprehensive 4-panel view:
      Panel A: 5G NR Raw vs Blended
      Panel B: 6G Sub-THz Raw vs Blended
      Panel C: Everything Blended (Unified Combined 5G+6G)
      Panel D: Theoretical Blending Alpha Curve & Mechanics
    """
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.5))
    snr = d5['snr_db']

    # --- Panel A: 5G NR TDL ---
    ax = axes[0, 0]
    ax.plot(snr, d5['ncrn_nmse_db'], color='#C62828', linestyle='-', marker='o', linewidth=2.0, label='NCRN (Blended)')
    ax.plot(snr, d5['ncrn_raw_nmse_db'], color='#EF5350', linestyle=':', marker='x', linewidth=1.5, label='NCRN (Raw Residual)')
    ax.plot(snr, d5['ncrt_matlab_nmse_db'], color='#0D47A1', linestyle='-', marker='s', linewidth=2.0, label='NCRT MATLAB (Blended)')
    ax.plot(snr, d5['ncrt_matlab_raw_nmse_db'], color='#42A5F5', linestyle=':', marker='+', linewidth=1.5, label='NCRT MATLAB (Raw Residual)')
    ax.set_xlabel('SNR (dB)', fontsize=10, fontweight='bold')
    ax.set_ylabel('NMSE (dB)', fontsize=10, fontweight='bold')
    ax.set_title('(A) 5G NR TDL: Raw vs. Blended Comparison', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6); ax.legend(fontsize=8, loc='lower left')

    # --- Panel B: 6G Sub-THz 142 GHz ---
    ax = axes[0, 1]
    ax.plot(snr, d6['ncrn_nmse_db'], color='#C62828', linestyle='-', marker='o', linewidth=2.0, label='NCRN (Blended)')
    ax.plot(snr, d6['ncrn_raw_nmse_db'], color='#EF5350', linestyle=':', marker='x', linewidth=1.5, label='NCRN (Raw Residual)')
    ax.plot(snr, d6['ncrt_matlab_nmse_db'], color='#0D47A1', linestyle='-', marker='s', linewidth=2.0, label='NCRT MATLAB (Blended)')
    ax.plot(snr, d6['ncrt_matlab_raw_nmse_db'], color='#42A5F5', linestyle=':', marker='+', linewidth=1.5, label='NCRT MATLAB (Raw Residual)')
    ax.set_xlabel('SNR (dB)', fontsize=10, fontweight='bold')
    ax.set_ylabel('NMSE (dB)', fontsize=10, fontweight='bold')
    ax.set_title('(B) 6G Sub-THz 142 GHz: Raw vs. Blended Comparison', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6); ax.legend(fontsize=8, loc='lower left')

    # --- Panel C: Everything Blended (Combined Cross-Generation) ---
    ax = axes[1, 0]
    ax.plot(snr, dc['ls_nmse_db'], color='#777777', linestyle='--', marker='^', label='LS')
    ax.plot(snr, dc['mmse_nmse_db'], color='#1B5E20', linestyle='-.', marker='D', label='MMSE')
    ax.plot(snr, dc['ncrn_nmse_db'], color='#C62828', linestyle='-', marker='o', linewidth=2.0, label='NCRN (Blended)')
    ax.plot(snr, dc['ncrt_base_nmse_db'], color='#FF8F00', linestyle='--', marker='v', linewidth=1.8, label='NCRT Synthetic (Blended)')
    ax.plot(snr, dc['ncrt_matlab_nmse_db'], color='#0D47A1', linestyle='-', marker='s', linewidth=2.2, label='NCRT MATLAB-Trained (Blended)')
    ax.set_xlabel('SNR (dB)', fontsize=10, fontweight='bold')
    ax.set_ylabel('NMSE (dB)', fontsize=10, fontweight='bold')
    ax.set_title('(C) Everything Blended: Combined 5G + 6G Dataset', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6); ax.legend(fontsize=8, loc='lower left')

    # --- Panel D: Difference Delta (NMSE_blended - NMSE_raw) ---
    ax = axes[1, 1]
    ax.axhline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.8)
    ax.plot(snr, d5['ncrn_nmse_db'] - d5['ncrn_raw_nmse_db'], color='#C62828', marker='o', linewidth=1.8, label='NCRN (5G NR)')
    ax.plot(snr, d5['ncrt_matlab_nmse_db'] - d5['ncrt_matlab_raw_nmse_db'], color='#0D47A1', marker='s', linewidth=1.8, label='NCRT MATLAB (5G NR)')
    ax.plot(snr, d6['ncrn_nmse_db'] - d6['ncrn_raw_nmse_db'], color='#FF7043', linestyle='--', marker='^', linewidth=1.8, label='NCRN (6G Sub-THz)')
    ax.plot(snr, d6['ncrt_matlab_nmse_db'] - d6['ncrt_matlab_raw_nmse_db'], color='#26A69A', linestyle='--', marker='v', linewidth=1.8, label='NCRT MATLAB (6G Sub-THz)')
    ax.set_xlabel('SNR (dB)', fontsize=10, fontweight='bold')
    ax.set_ylabel(r'$\Delta = {\rm NMSE}_{\rm Blended} - {\rm NMSE}_{\rm Raw}$ (dB)', fontsize=10, fontweight='bold')
    ax.set_title('(D) Difference Magnitude across SNR Range', fontsize=11, fontweight='bold')
    ax.set_ylim(-13, 3)
    ax.grid(True, linestyle=':', alpha=0.6); ax.legend(fontsize=8, loc='lower right')

    out_path = os.path.join(FIG_DIR, 'raw_vs_blended_complete_panel.png')
    fig.tight_layout(rect=[0, 0.02, 1, 0.96], h_pad=2.5, w_pad=2.2)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    print("Loading evaluation CSVs...")
    d5 = load_csv(CSV_5G)
    d6 = load_csv(CSV_6G)
    dc = load_csv(CSV_COMBINED)

    print("Generating Figure 1: Raw vs. Blended Difference...")
    plot_raw_vs_blended_difference(d5, d6)

    print("Generating Figure 2: Everything Blended Benchmark...")
    plot_everything_blended(d5, d6, dc)

    print("Generating Figure 3: Complete 4-Panel Raw vs. Blended Suite...")
    plot_complete_panel(d5, d6, dc)

    print("\n[DONE] All plots generated successfully!")


if __name__ == '__main__':
    main()

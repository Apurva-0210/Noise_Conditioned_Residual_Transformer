"""
plot_individual_10_tdl.py
────────────────────────────────────────────────────────────────────────
Generates individual, dedicated, high-resolution plots for each of the
10 3GPP TR 38.901 TDL channel scenarios, one by one.

Also generates:
1. scenario_all_10_ncrt_curves.png: All 10 scenarios plotted together for NCRT
2. scenario_all_10_ncrn_curves.png: All 10 scenarios plotted together for NCRN
3. scenario_all_10_delta_curves.png: The NCRT improvement delta vs SNR across all 10 scenarios
"""

import os, sys, shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
ARTIFACT_DIR = '/Users/apurva/.gemini/antigravity/brain/d2efda49-e66e-4b92-a08b-95db3cc3d4eb'
FIG_DIR = os.path.join(ROOT, 'outputs', 'figures')
CSV_DIR = os.path.join(ROOT, 'outputs', 'csv')

from shared.channels import SCENARIOS

ORDERED_SCENARIOS = [
    ('InH_LOS', 'Indoor Hotspot LOS', '01'),
    ('UMi_LOS', 'Urban Micro LOS', '02'),
    ('UMa_LOS', 'Urban Macro LOS (120 km/h)', '03'),
    ('RMa_LOS', 'Rural Macro LOS (300 km/h)', '04'),
    ('RMa_HST', 'Rural Macro High-Speed Train (500 km/h)', '05'),
    ('InH_NLOS', 'Indoor Hotspot NLOS', '06'),
    ('UMi_NLOS', 'Urban Micro NLOS', '07'),
    ('UMa_NLOS', 'Urban Macro NLOS (120 km/h)', '08'),
    ('RMa_NLOS', 'Rural Macro NLOS (300 km/h)', '09'),
    ('UMi_O2I', 'Urban Micro Outdoor-to-Indoor', '10'),
]

STYLE = {
    'ls': ('#757575', '--', '^', 1.4, 'LS Baseline'),
    'mmse': ('#2E7D32', '-.', 'D', 1.6, 'MMSE Oracle'),
    'ncrn_raw': ('#EF9A9A', ':', 'x', 1.5, 'NCRN (Raw Residual)'),
    'ncrn': ('#C62828', '-', 'o', 2.2, 'NCRN (ConvLSTM, Blended)'),
    'ncrt_raw': ('#90CAF9', ':', '+', 1.6, 'NCRT (Raw Residual)'),
    'ncrt': ('#1565C0', '-', 's', 2.5, 'NCRT (Transformer, Blended)')
}

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'lines.markersize': 5.5,
    'grid.alpha': 0.45,
    'grid.linestyle': ':'
})


def plot_one_scenario(sname, desc, idx):
    csv_path = os.path.join(CSV_DIR, f'ncrt_vs_ncrn_{sname}_full_pilot.csv')
    df = pd.read_csv(csv_path)
    sinfo = SCENARIOS[sname]
    
    speed = sinfo['v']
    fc_ghz = sinfo['fc'] / 1e9
    taps_count = len(sinfo['taps'])
    k_val = sinfo['K']
    k_str = f"Rician K = {k_val} dB" if k_val is not None else "Rayleigh Fading (NLOS)"
    group_str = sinfo['group']
    
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    
    # Plot baseline curves
    ax.plot(df['snr_db'], df['ls_nmse_db'], color=STYLE['ls'][0], linestyle=STYLE['ls'][1],
            marker=STYLE['ls'][2], linewidth=STYLE['ls'][3], label=STYLE['ls'][4], alpha=0.8)
    ax.plot(df['snr_db'], df['mmse_nmse_db'], color=STYLE['mmse'][0], linestyle=STYLE['mmse'][1],
            marker=STYLE['mmse'][2], linewidth=STYLE['mmse'][3], label=STYLE['mmse'][4], alpha=0.9)
    
    # Plot raw curves
    ax.plot(df['snr_db'], df['ncrn_raw_nmse_db'], color=STYLE['ncrn_raw'][0], linestyle=STYLE['ncrn_raw'][1],
            marker=STYLE['ncrn_raw'][2], linewidth=STYLE['ncrn_raw'][3], label=STYLE['ncrn_raw'][4], alpha=0.75)
    ax.plot(df['snr_db'], df['ncrt_raw_nmse_db'], color=STYLE['ncrt_raw'][0], linestyle=STYLE['ncrt_raw'][1],
            marker=STYLE['ncrt_raw'][2], linewidth=STYLE['ncrt_raw'][3], label=STYLE['ncrt_raw'][4], alpha=0.75)
    
    # Plot blended curves (highlighted)
    ax.plot(df['snr_db'], df['ncrn_nmse_db'], color=STYLE['ncrn'][0], linestyle=STYLE['ncrn'][1],
            marker=STYLE['ncrn'][2], linewidth=STYLE['ncrn'][3], label=STYLE['ncrn'][4], alpha=0.95)
    ax.plot(df['snr_db'], df['ncrt_nmse_db'], color=STYLE['ncrt'][0], linestyle=STYLE['ncrt'][1],
            marker=STYLE['ncrt'][2], linewidth=STYLE['ncrt'][3], label=STYLE['ncrt'][4], alpha=0.95)
    
    # Fill between blended NCRN and NCRT to highlight the gain
    ax.fill_between(df['snr_db'], df['ncrn_nmse_db'], df['ncrt_nmse_db'], 
                    where=(df['ncrt_nmse_db'] <= df['ncrn_nmse_db']),
                    interpolate=True, color='#BBDEFB', alpha=0.45, label='NCRT Advantage Margin')
    
    # Metric values at 20 dB SNR
    r20_ncrn = df.loc[df.snr_db == 20, 'ncrn_nmse_db'].values[0]
    r20_ncrt = df.loc[df.snr_db == 20, 'ncrt_nmse_db'].values[0]
    delta20 = abs(r20_ncrt - r20_ncrn)
    
    # Compact, sleek badge for scenario parameters and key gain (minimal size, unobtrusive)
    k_short = f"K={k_val}dB" if k_val is not None else "Rayleigh"
    textstr = (
        f"{sname} ({speed} km/h, {fc_ghz:.1f} GHz, {taps_count} taps, {k_short})\n"
        f"NCRT Gain @ 20 dB: +{delta20:.2f} dB  (NCRT: {r20_ncrt:.2f} dB vs NCRN: {r20_ncrn:.2f} dB)"
    )
    
    props = dict(boxstyle='round,pad=0.3', facecolor='#F8F9FA', edgecolor='#B0BEC5', alpha=0.9, linewidth=0.8)
    ax.text(0.03, 0.04, textstr, transform=ax.transAxes, fontsize=7.5,
            verticalalignment='bottom', bbox=props, family='sans-serif')
    
    ax.set_xlabel('SNR (dB)', fontweight='bold')
    ax.set_ylabel('NMSE (dB)', fontweight='bold')
    ax.set_title(f"Scenario {idx}: {sname} — {desc}\nMassive MIMO (64×64, T=10): NCRT vs. NCRN", fontweight='bold')
    ax.grid(True)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    fig.tight_layout()
    
    fname = f"scenario_{idx}_{sname}.png"
    out_path = os.path.join(FIG_DIR, fname)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Generated {out_path}")
    
    # Copy to artifact directory
    art_path = os.path.join(ARTIFACT_DIR, fname)
    shutil.copyfile(out_path, art_path)
    print(f"Copied to {art_path}")


def plot_all_scenarios_comparison():
    """Generates comparison figures across all 10 scenarios."""
    # Plot 1: All 10 scenarios for NCRT (Blended)
    fig1, ax1 = plt.subplots(figsize=(10.0, 6.0))
    # Plot 2: All 10 scenarios for NCRN (Blended)
    fig2, ax2 = plt.subplots(figsize=(10.0, 6.0))
    # Plot 3: NCRT Improvement Delta vs SNR across all 10 scenarios
    fig3, ax3 = plt.subplots(figsize=(10.0, 6.0))
    
    colors = [
        '#1E88E5', '#0D47A1', '#00ACC1', '#00897B', '#43A047',
        '#E53935', '#D81B60', '#8E24AA', '#5E35B1', '#FB8C00'
    ]
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'x', '+']
    
    for i, (sname, desc, idx) in enumerate(ORDERED_SCENARIOS):
        csv_path = os.path.join(CSV_DIR, f'ncrt_vs_ncrn_{sname}_full_pilot.csv')
        df = pd.read_csv(csv_path)
        c = colors[i]
        m = markers[i]
        lbl = f"{idx}. {sname} ({desc})"
        
        # NCRT curve
        ax1.plot(df['snr_db'], df['ncrt_nmse_db'], color=c, marker=m, linewidth=1.8, label=lbl, alpha=0.9)
        # NCRN curve
        ax2.plot(df['snr_db'], df['ncrn_nmse_db'], color=c, marker=m, linewidth=1.8, label=lbl, alpha=0.9)
        # Delta curve (NCRT Gain = NCRN - NCRT in dB)
        delta = df['ncrn_nmse_db'] - df['ncrt_nmse_db']
        ax3.plot(df['snr_db'], delta, color=c, marker=m, linewidth=1.8, label=lbl, alpha=0.9)
        
    ax1.set_xlabel('SNR (dB)', fontweight='bold')
    ax1.set_ylabel('NMSE (dB)', fontweight='bold')
    ax1.set_title('NCRT (Transformer, Blended) — Performance Across All 10 3GPP TDL Scenarios', fontweight='bold')
    ax1.grid(True)
    ax1.legend(loc='lower left', fontsize=8, ncol=2)
    fig1.tight_layout()
    f1 = os.path.join(FIG_DIR, 'scenario_all_10_ncrt_curves.png')
    fig1.savefig(f1, dpi=300)
    plt.close(fig1)
    shutil.copyfile(f1, os.path.join(ARTIFACT_DIR, 'scenario_all_10_ncrt_curves.png'))
    
    ax2.set_xlabel('SNR (dB)', fontweight='bold')
    ax2.set_ylabel('NMSE (dB)', fontweight='bold')
    ax2.set_title('NCRN (ConvLSTM, Blended) — Performance Across All 10 3GPP TDL Scenarios', fontweight='bold')
    ax2.grid(True)
    ax2.legend(loc='lower left', fontsize=8, ncol=2)
    fig2.tight_layout()
    f2 = os.path.join(FIG_DIR, 'scenario_all_10_ncrn_curves.png')
    fig2.savefig(f2, dpi=300)
    plt.close(fig2)
    shutil.copyfile(f2, os.path.join(ARTIFACT_DIR, 'scenario_all_10_ncrn_curves.png'))
    
    ax3.set_xlabel('SNR (dB)', fontweight='bold')
    ax3.set_ylabel('NCRT Improvement Margin Δ (dB)', fontweight='bold')
    ax3.set_title('NCRT Estimation Margin over NCRN (dB) vs SNR Across All 10 TDL Scenarios', fontweight='bold')
    ax3.set_ylim(-0.5, 9.5)
    ax3.grid(True)
    ax3.legend(loc='upper left', fontsize=8, ncol=2, frameon=True, framealpha=0.9)
    fig3.tight_layout()
    f3 = os.path.join(FIG_DIR, 'scenario_all_10_delta_curves.png')
    fig3.savefig(f3, dpi=300)
    plt.close(fig3)
    shutil.copyfile(f3, os.path.join(ARTIFACT_DIR, 'scenario_all_10_delta_curves.png'))
    
    print("Generated and copied all 10 scenario progression figures.")


def main():
    print("Generating individual scenario plots one by one...")
    for sname, desc, idx in ORDERED_SCENARIOS:
        plot_one_scenario(sname, desc, idx)
    plot_all_scenarios_comparison()
    print("All individual plots successfully generated!")


if __name__ == '__main__':
    main()

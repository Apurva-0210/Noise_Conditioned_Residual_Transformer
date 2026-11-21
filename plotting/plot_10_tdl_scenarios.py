"""
plot_10_tdl_scenarios.py
────────────────────────────────────────────────────────────────────────
Generates publication-quality figures for the 10 3GPP TR 38.901 TDL channel scenarios:
1. ncrt_vs_ncrn_10_tdl_scenarios_grid.png: 2x5 multi-panel NMSE vs SNR curves
2. ncrt_vs_ncrn_10_tdl_delta_summary.png: Bar chart & scenario breakdown of NCRT gains
3. ncrt_vs_ncrn_10_tdl_raw_vs_blended.png: Raw Residual vs Blended comparison across scenarios
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

LOS_HST_SCENARIOS = ['InH_LOS', 'UMi_LOS', 'UMa_LOS', 'RMa_LOS', 'RMa_HST']
NLOS_O2I_SCENARIOS = ['InH_NLOS', 'UMi_NLOS', 'UMa_NLOS', 'RMa_NLOS', 'UMi_O2I']
ALL_10 = LOS_HST_SCENARIOS + NLOS_O2I_SCENARIOS

STYLE = {
    'ls': ('#777777', '--', '^', 1.2, 'LS Baseline'),
    'mmse': ('#2E7D32', '-.', 'D', 1.3, 'MMSE Oracle'),
    'ncrn': ('#C62828', '-', 'o', 1.8, 'NCRN (ConvLSTM, Blended)'),
    'ncrn_raw': ('#EF9A9A', ':', 'x', 1.3, 'NCRN (Raw Residual)'),
    'ncrt': ('#1565C0', '-', 's', 2.0, 'NCRT (Transformer, Blended)'),
    'ncrt_raw': ('#90CAF9', ':', '+', 1.5, 'NCRT (Raw Residual)')
}

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 7.5,
    'figure.titlesize': 13,
    'lines.markersize': 4.5,
    'grid.alpha': 0.45,
    'grid.linestyle': ':'
})


def plot_grid():
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharex=True, sharey=True)
    
    for row_idx, scen_group in enumerate([LOS_HST_SCENARIOS, NLOS_O2I_SCENARIOS]):
        for col_idx, sname in enumerate(scen_group):
            ax = axes[row_idx, col_idx]
            csv_path = os.path.join(CSV_DIR, f'ncrt_vs_ncrn_{sname}_full_pilot.csv')
            df = pd.read_csv(csv_path)
            
            sinfo = SCENARIOS[sname]
            speed = sinfo['v']
            fc_ghz = sinfo['fc'] / 1e9
            k_str = f", K={sinfo['K']}dB" if sinfo['K'] is not None else ""
            
            for m in ['ls', 'mmse', 'ncrn_raw', 'ncrn', 'ncrt_raw', 'ncrt']:
                col = f'{m}_nmse_db' if m != 'ncrn' and m != 'ncrt' else f'{m}_nmse_db'
                c, ls, mk, lw, lbl = STYLE[m]
                ax.plot(df['snr_db'], df[col], color=c, linestyle=ls, marker=mk, 
                        linewidth=lw, label=lbl, alpha=0.9)
            
            # Compute delta at 20 dB
            r20_ncrn = df.loc[df.snr_db == 20, 'ncrn_nmse_db'].values[0]
            r20_ncrt = df.loc[df.snr_db == 20, 'ncrt_nmse_db'].values[0]
            delta20 = r20_ncrt - r20_ncrn
            
            ax.set_title(f"{sname}\n({speed} km/h, {fc_ghz:.1f} GHz{k_str})", fontweight='bold', fontsize=9.5)
            ax.grid(True)
            
            # Callout box for 20 dB gain
            box_color = '#E8F5E9' if delta20 < -3.0 else '#E3F2FD'
            ax.text(0.04, 0.06, f"NCRT Gain @ 20dB:\n{abs(delta20):.2f} dB better", 
                    transform=ax.transAxes, fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc=box_color, ec='#388E3C' if delta20 < -3.0 else '#1976D2', lw=1.2))

            if col_idx == 0:
                ax.set_ylabel("NMSE (dB)", fontweight='bold')
            if row_idx == 1:
                ax.set_xlabel("SNR (dB)", fontweight='bold')

    # Shared legend placed cleanly above subplots without touching titles
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.91), ncol=6, frameon=True, fontsize=9.5)
    
    fig.suptitle("Massive MIMO (64×64) Channel Estimation Across All 10 3GPP TR 38.901 TDL Scenarios\nNCRT (Temporal Transformer) vs. NCRN (ConvLSTM2D)", 
                 fontweight='bold', y=0.98, fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    
    out_path = os.path.join(FIG_DIR, 'ncrt_vs_ncrn_10_tdl_scenarios_grid.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")
    
    # Copy to artifact directory
    art_path = os.path.join(ARTIFACT_DIR, 'ncrt_vs_ncrn_10_tdl_scenarios_grid.png')
    shutil.copyfile(out_path, art_path)
    print(f"Copied to {art_path}")


def plot_delta_summary():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    scen_labels = []
    ncrn_20 = []
    ncrt_20 = []
    mmse_20 = []
    deltas_20 = []
    deltas_10 = []
    
    for s in ALL_10:
        csv_path = os.path.join(CSV_DIR, f'ncrt_vs_ncrn_{s}_full_pilot.csv')
        df = pd.read_csv(csv_path)
        scen_labels.append(s)
        
        n_20 = df.loc[df.snr_db == 20, 'ncrn_nmse_db'].values[0]
        t_20 = df.loc[df.snr_db == 20, 'ncrt_nmse_db'].values[0]
        m_20 = df.loc[df.snr_db == 20, 'mmse_nmse_db'].values[0]
        
        n_10 = df.loc[df.snr_db == 10, 'ncrn_nmse_db'].values[0]
        t_10 = df.loc[df.snr_db == 10, 'ncrt_nmse_db'].values[0]
        
        ncrn_20.append(n_20)
        ncrt_20.append(t_20)
        mmse_20.append(m_20)
        deltas_20.append(abs(t_20 - n_20))
        deltas_10.append(abs(t_10 - n_10))
    
    x = np.arange(len(ALL_10))
    width = 0.26
    
    # Left subplot: Absolute NMSE @ 20 dB SNR
    ax1.bar(x - width, mmse_20, width, label='MMSE Oracle', color='#2E7D32', alpha=0.85)
    ax1.bar(x, ncrn_20, width, label='NCRN (ConvLSTM)', color='#C62828', alpha=0.85)
    ax1.bar(x + width, ncrt_20, width, label='NCRT (Transformer)', color='#1565C0', alpha=0.9)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(scen_labels, rotation=35, ha='right', fontweight='bold', fontsize=8.5)
    ax1.set_ylabel("NMSE (dB) @ 20 dB SNR (Lower is Better)", fontweight='bold')
    ax1.set_title("Absolute Channel Estimation Accuracy at 20 dB SNR", fontweight='bold')
    ax1.grid(True, axis='y')
    ax1.legend(loc='upper right')
    ax1.axvline(4.5, color='gray', linestyle='--', alpha=0.6)
    ax1.text(2.0, -32.5, "LOS & HST Scenarios", ha='center', fontsize=9, fontweight='bold', color='#1B5E20')
    ax1.text(7.0, -32.5, "NLOS & O2I Scenarios", ha='center', fontsize=9, fontweight='bold', color='#B71C1C')

    # Right subplot: NCRT Net Accuracy Gain over NCRN (in dB)
    bars20 = ax2.bar(x - width/2, deltas_20, width, label='NCRT Gain @ 20 dB SNR', color='#0288D1', alpha=0.9)
    bars10 = ax2.bar(x + width/2, deltas_10, width, label='NCRT Gain @ 10 dB SNR', color='#FFA000', alpha=0.85)
    
    for bar in bars20:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"+{yval:.2f}", ha='center', va='bottom', fontsize=7.5, fontweight='bold')
        
    ax2.set_xticks(x)
    ax2.set_xticklabels(scen_labels, rotation=35, ha='right', fontweight='bold', fontsize=8.5)
    ax2.set_ylabel("NCRT Margin over NCRN (dB)", fontweight='bold')
    ax2.set_title("NCRT Channel Estimation Improvement over NCRN (dB)", fontweight='bold')
    ax2.grid(True, axis='y')
    ax2.legend(loc='upper left')
    ax2.axvline(4.5, color='gray', linestyle='--', alpha=0.6)
    ax2.set_ylim(0, 7.2)
    
    fig.suptitle("Performance Breakdown Across All 10 3GPP TDL Scenarios: NCRT vs. NCRN", 
                 fontweight='bold', fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    
    out_path = os.path.join(FIG_DIR, 'ncrt_vs_ncrn_10_tdl_delta_summary.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")
    
    art_path = os.path.join(ARTIFACT_DIR, 'ncrt_vs_ncrn_10_tdl_delta_summary.png')
    shutil.copyfile(out_path, art_path)
    print(f"Copied to {art_path}")


def plot_raw_vs_blended():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    scen_labels = []
    # Compare at SNR = -15 dB (where blending dampens noise) and SNR = 20 dB (where raw and blend converge)
    ncrn_raw_m15, ncrn_bl_m15 = [], []
    ncrt_raw_m15, ncrt_bl_m15 = [], []
    
    ncrn_raw_20, ncrn_bl_20 = [], []
    ncrt_raw_20, ncrt_bl_20 = [], []
    
    for s in ALL_10:
        csv_path = os.path.join(CSV_DIR, f'ncrt_vs_ncrn_{s}_full_pilot.csv')
        df = pd.read_csv(csv_path)
        scen_labels.append(s)
        
        ncrn_raw_m15.append(df.loc[df.snr_db == -15, 'ncrn_raw_nmse_db'].values[0])
        ncrn_bl_m15.append(df.loc[df.snr_db == -15, 'ncrn_nmse_db'].values[0])
        ncrt_raw_m15.append(df.loc[df.snr_db == -15, 'ncrt_raw_nmse_db'].values[0])
        ncrt_bl_m15.append(df.loc[df.snr_db == -15, 'ncrt_nmse_db'].values[0])
        
        ncrn_raw_20.append(df.loc[df.snr_db == 20, 'ncrn_raw_nmse_db'].values[0])
        ncrn_bl_20.append(df.loc[df.snr_db == 20, 'ncrn_nmse_db'].values[0])
        ncrt_raw_20.append(df.loc[df.snr_db == 20, 'ncrt_raw_nmse_db'].values[0])
        ncrt_bl_20.append(df.loc[df.snr_db == 20, 'ncrt_nmse_db'].values[0])
        
    x = np.arange(len(ALL_10))
    width = 0.2
    
    # Left: Low SNR (-15 dB) - Regime where noise dampening is critical
    ax1.bar(x - 1.5*width, ncrn_raw_m15, width, label='NCRN (Raw Residual)', color='#EF9A9A')
    ax1.bar(x - 0.5*width, ncrn_bl_m15, width, label='NCRN (Blended)', color='#C62828')
    ax1.bar(x + 0.5*width, ncrt_raw_m15, width, label='NCRT (Raw Residual)', color='#90CAF9')
    ax1.bar(x + 1.5*width, ncrt_bl_m15, width, label='NCRT (Blended)', color='#1565C0')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(scen_labels, rotation=35, ha='right', fontweight='bold', fontsize=8.5)
    ax1.set_ylabel("NMSE (dB) @ -15 dB SNR", fontweight='bold')
    ax1.set_title("Low SNR Regime (-15 dB): Raw Residual vs. SNR-Aware Blending", fontweight='bold')
    ax1.set_ylim(-3, 19)
    ax1.grid(True, axis='y')
    ax1.legend(loc='upper left', fontsize=8.5, frameon=True)
    ax1.axvline(4.5, color='gray', linestyle='--', alpha=0.6)
    
    # Right: High SNR (+20 dB) - Regime where residual provides full correction
    ax2.bar(x - 1.5*width, ncrn_raw_20, width, label='NCRN (Raw Residual)', color='#EF9A9A')
    ax2.bar(x - 0.5*width, ncrn_bl_20, width, label='NCRN (Blended)', color='#C62828')
    ax2.bar(x + 0.5*width, ncrt_raw_20, width, label='NCRT (Raw Residual)', color='#90CAF9')
    ax2.bar(x + 1.5*width, ncrt_bl_20, width, label='NCRT (Blended)', color='#1565C0')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(scen_labels, rotation=35, ha='right', fontweight='bold', fontsize=8.5)
    ax2.set_ylabel("NMSE (dB) @ +20 dB SNR", fontweight='bold')
    ax2.set_title("High SNR Regime (+20 dB): Convergence of Raw and Blended", fontweight='bold')
    ax2.set_ylim(-36, -14)
    ax2.grid(True, axis='y')
    ax2.legend(loc='lower left', fontsize=8.5, frameon=True)
    ax2.axvline(4.5, color='gray', linestyle='--', alpha=0.6)
    
    fig.suptitle("Raw Residual vs. SNR-Aware Blending Across All 10 TDL Scenarios", fontweight='bold', fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    
    out_path = os.path.join(FIG_DIR, 'ncrt_vs_ncrn_10_tdl_raw_vs_blended.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")
    
    art_path = os.path.join(ARTIFACT_DIR, 'ncrt_vs_ncrn_10_tdl_raw_vs_blended.png')
    shutil.copyfile(out_path, art_path)
    print(f"Copied to {art_path}")


def copy_individual_plots():
    for s in ALL_10:
        src = os.path.join(FIG_DIR, f'ncrt_vs_ncrn_{s}_full_pilot.png')
        if os.path.exists(src):
            dst = os.path.join(ARTIFACT_DIR, f'ncrt_vs_ncrn_{s}_full_pilot.png')
            shutil.copyfile(src, dst)
    print("Copied all individual scenario figures to artifact directory.")


if __name__ == '__main__':
    plot_grid()
    plot_delta_summary()
    plot_raw_vs_blended()
    copy_individual_plots()

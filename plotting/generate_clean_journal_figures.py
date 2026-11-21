#!/usr/bin/env python3
"""
generate_clean_journal_figures.py
────────────────────────────────────────────────────────────────────────
Generates 100% publication-compliant figures for IEEE / Elsevier submissions:
  1. NO titles inside figures (titles belong strictly in LaTeX \\caption{}).
  2. Exactly ONE plot per file (all multi-panel figures broken into individual files).
  3. Exported in both vector PDF and 600 DPI PNG.
  4. Inward-pointing tick marks, serif typography (Times New Roman), publication palette.
"""
import os, sys, csv, zipfile, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CSV_DIR = os.path.join(ROOT, 'outputs', 'csv')
OUT_DIR = os.path.join(ROOT, 'outputs', 'figures_clean')
ARTIFACT_DIR = '/Users/apurva/.gemini/antigravity/brain/d2efda49-e66e-4b92-a08b-95db3cc3d4eb'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Strict Publication Matplotlib Style ──────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'Times', 'DejaVu Serif', 'Liberation Serif'],
    'mathtext.fontset':  'stix',
    'font.size':         9.5,
    'axes.labelsize':    10.0,
    'xtick.labelsize':   8.5,
    'ytick.labelsize':   8.5,
    'legend.fontsize':   8.0,
    'legend.frameon':    True,
    'legend.framealpha': 0.95,
    'legend.edgecolor':  '0.6',
    'legend.fancybox':   False,
    'axes.linewidth':    0.8,
    'axes.edgecolor':    '0.2',
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.major.size':  3.5,
    'ytick.major.size':  3.5,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'xtick.top':         True,
    'ytick.right':       True,
    'lines.linewidth':   1.3,
    'lines.markersize':  4.5,
    'grid.linewidth':    0.45,
    'grid.linestyle':    ':',
    'grid.alpha':        0.55,
    'grid.color':        '0.65',
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'savefig.facecolor': 'white',
    'savefig.dpi':       600,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

# Color palette & styling
STYLE = {
    'ls':        ('#000000', '--', '^', 'LS'),
    'mmse':      ('#1B5E20', '-.', 'D', 'MMSE'),
    'sadlcs':    ('#0D3B66', ':',  'v', 'Sa-DLCS'),
    'cnn_lstm':  ('#E65100', '-.', 'p', 'CNN-LSTM'),
    'cnn_sparse':('#4A148C', ':',  '*', 'CNN-Sparse'),
    'ncrn':      ('#B71C1C', '-',  'o', 'NCRN (blended)'),
    'ncrn_raw':  ('#E57373', ':',  'x', 'NCRN (raw)'),
    'ncrt':      ('#0D47A1', '-',  's', 'NCRT (blended)'),
    'ncrt_raw':  ('#64B5F6', ':',  '+', 'NCRT (raw)'),
}

SCENARIO_NAMES = [
    'InH_LOS', 'UMi_LOS', 'UMa_LOS', 'RMa_LOS',
    'InH_NLOS', 'UMi_NLOS', 'UMa_NLOS', 'RMa_NLOS',
    'UMi_O2I', 'RMa_HST'
]

SCENARIO_DISPLAY = {
    'InH_LOS':  'InH LOS (3 km/h)',
    'UMi_LOS':  'UMi LOS (30 km/h)',
    'UMa_LOS':  'UMa LOS (120 km/h)',
    'RMa_LOS':  'RMa LOS (300 km/h)',
    'InH_NLOS': 'InH NLOS (3 km/h)',
    'UMi_NLOS': 'UMi NLOS (30 km/h)',
    'UMa_NLOS': 'UMa NLOS (120 km/h)',
    'RMa_NLOS': 'RMa NLOS (300 km/h)',
    'UMi_O2I':  'UMi O2I (3 km/h)',
    'RMa_HST':  'RMa HST (500 km/h)',
}

def save_clean_figure(fig, filename_base, figsize=(5.0, 3.8)):
    """Saves single-plot figure with zero internal title in both PNG and PDF with ample margins."""
    fig.tight_layout(pad=0.6)
    png_path = os.path.join(OUT_DIR, f"{filename_base}.png")
    pdf_path = os.path.join(OUT_DIR, f"{filename_base}.pdf")
    fig.savefig(png_path, dpi=600, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Saved {filename_base}.png and .pdf")

# ─── 1. Figure 2: Pooled Full-Pilot TDL Benchmark ─────────────────────
def plot_figure_2_pooled():
    csv_p = os.path.join(CSV_DIR, 'ncrt_vs_ncrn_full_pilot.csv')
    if not os.path.exists(csv_p): return
    df = pd.read_csv(csv_p)
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    snr = df['snr_db'].values
    
    for m in ['ls', 'mmse', 'ncrn', 'ncrn_raw', 'ncrt', 'ncrt_raw']:
        col = f'{m}_nmse_db'
        if col in df:
            c, ls, mk, lbl = STYLE[m]
            ax.plot(snr, df[col], color=c, linestyle=ls, marker=mk, label=lbl)
            
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('NMSE (dB)')
    ax.set_ylim(-35, 15)
    ax.grid(True)
    ax.legend(loc='lower left', fontsize=8.0)
    save_clean_figure(fig, 'fig2_pooled_tdl_benchmark')

# ─── 2. Figure 4: Individual Bar Charts (20 dB NMSE & NCRT Margin) ────
def plot_figure_4_single_panels():
    # Load 20 dB values across 10 scenarios
    vals_20 = {'mmse': [], 'ncrn': [], 'ncrt': []}
    for scen in SCENARIO_NAMES:
        csv_p = os.path.join(CSV_DIR, f'ncrt_vs_ncrn_{scen}_full_pilot.csv')
        df = pd.read_csv(csv_p)
        r20 = df[df['snr_db'] == 20]
        vals_20['mmse'].append(float(r20['mmse_nmse_db'].values[0]))
        vals_20['ncrn'].append(float(r20['ncrn_nmse_db'].values[0]))
        vals_20['ncrt'].append(float(r20['ncrt_nmse_db'].values[0]))
        
    labels = [s.replace('_', ' ') for s in SCENARIO_NAMES]
    x = np.arange(len(labels))
    w = 0.26
    
    # Panel 4A: Absolute NMSE at 20 dB
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.bar(x - w, vals_20['mmse'], width=w, color='#2E7D32', label='MMSE')
    ax.bar(x,     vals_20['ncrn'], width=w, color='#C62828', label='NCRN')
    ax.bar(x + w, vals_20['ncrt'], width=w, color='#1565C0', label='NCRT')
    ax.set_ylabel('NMSE at 20 dB (dB)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha='right', fontsize=8.5)
    ax.set_ylim(-36, -14)
    ax.grid(True, axis='y')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=8.5, frameon=True)
    save_clean_figure(fig, 'fig4a_absolute_nmse_20db')
    
    # Panel 4B: NCRT Margin over NCRN (dB)
    margins = np.array(vals_20['ncrn']) - np.array(vals_20['ncrt'])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bar_colors = ['#0D47A1' if m >= 0 else '#D32F2F' for m in margins]
    ax.bar(x, margins, width=0.55, color=bar_colors, edgecolor='black', linewidth=0.6)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('NCRT Margin over NCRN (dB)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha='right', fontsize=8.5)
    ax.set_ylim(-0.5, 7.2)
    ax.grid(True, axis='y')
    save_clean_figure(fig, 'fig4b_ncrt_margin_20db')

# ─── 3. Figure 5: NCRT Margin over NCRN vs SNR (All 10 Scenarios) ─────
def plot_figure_5_margin_vs_snr():
    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for idx, scen in enumerate(SCENARIO_NAMES):
        df = pd.read_csv(os.path.join(CSV_DIR, f'ncrt_vs_ncrn_{scen}_full_pilot.csv'))
        snr = df['snr_db'].values
        margin = df['ncrn_nmse_db'].values - df['ncrt_nmse_db'].values
        ax.plot(snr, margin, marker='o', linewidth=1.3, markersize=4,
                color=colors[idx], label=SCENARIO_DISPLAY[scen])
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('NCRT Margin over NCRN (dB)')
    ax.set_ylim(-1.0, 7.5)
    ax.grid(True)
    # Put legend cleanly above with 5 columns so it never touches curves
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3,
              fontsize=7.5, frameon=True, columnspacing=0.8)
    save_clean_figure(fig, 'fig5_ncrt_margin_vs_snr')

# ─── 4. Figure 6: Blended Curves Across All 10 Scenarios (Single Panels)
def plot_figure_6_single_panels():
    # 6A: NCRT curves
    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for idx, scen in enumerate(SCENARIO_NAMES):
        df = pd.read_csv(os.path.join(CSV_DIR, f'ncrt_vs_ncrn_{scen}_full_pilot.csv'))
        ax.plot(df['snr_db'], df['ncrt_nmse_db'], marker='s', linewidth=1.2, markersize=3.8,
                color=colors[idx], label=SCENARIO_DISPLAY[scen])
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('NCRT NMSE (dB)')
    ax.set_ylim(-36, 16)
    ax.grid(True)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3,
              fontsize=7.5, frameon=True, columnspacing=0.8)
    save_clean_figure(fig, 'fig6a_ncrt_all_10_scenarios')

    # 6B: NCRN curves
    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    for idx, scen in enumerate(SCENARIO_NAMES):
        df = pd.read_csv(os.path.join(CSV_DIR, f'ncrt_vs_ncrn_{scen}_full_pilot.csv'))
        ax.plot(df['snr_db'], df['ncrn_nmse_db'], marker='o', linewidth=1.2, markersize=3.8,
                color=colors[idx], label=SCENARIO_DISPLAY[scen])
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('NCRN NMSE (dB)')
    ax.set_ylim(-36, 16)
    ax.grid(True)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3,
              fontsize=7.5, frameon=True, columnspacing=0.8)
    save_clean_figure(fig, 'fig6b_ncrn_all_10_scenarios')

# ─── 5. The 10 Individual Scenario Plots (Single Plot per File) ───────
def plot_all_10_scenarios_individually():
    for scen in SCENARIO_NAMES:
        csv_p = os.path.join(CSV_DIR, f'ncrt_vs_ncrn_{scen}_full_pilot.csv')
        df = pd.read_csv(csv_p)
        fig, ax = plt.subplots(figsize=(4.8, 3.8))
        snr = df['snr_db'].values
        for m in ['ls', 'mmse', 'ncrn', 'ncrn_raw', 'ncrt', 'ncrt_raw']:
            col = f'{m}_nmse_db'
            if col in df:
                c, ls, mk, lbl = STYLE[m]
                ax.plot(snr, df[col], color=c, linestyle=ls, marker=mk, label=lbl)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('NMSE (dB)')
        ax.grid(True)
        ax.legend(loc='lower left', fontsize=7.5)
        save_clean_figure(fig, f'scenario_single_{scen}')

# ─── 6. Figure 8 & Legacy 64-Scenario Suite (Single Plots) ────────────
def plot_legacy_figures():
    # Pooled legacy 64 benchmark
    csv_p = os.path.join(CSV_DIR, 'tdl_64.csv')
    if os.path.exists(csv_p):
        df = pd.read_csv(csv_p)
        fig, ax = plt.subplots(figsize=(4.8, 3.8))
        snr = df['SNR (dB)'].values
        for m, col in [('ls', 'LS NMSE (dB)'), ('mmse', 'MMSE NMSE (dB)'),
                       ('sadlcs', 'Sa-DLCS NMSE (dB)'), ('ncrn', 'NCRN NMSE (dB)')]:
            if col in df:
                c, ls, mk, lbl = STYLE[m]
                ax.plot(snr, df[col], color=c, linestyle=ls, marker=mk, label=lbl)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('NMSE (dB)')
        ax.grid(True)
        ax.legend(loc='lower left', fontsize=8)
        save_clean_figure(fig, 'fig8_legacy_pooled_tdl_64')
        
    # Individual legacy scenarios (no titles, single plot)
    for scen in SCENARIO_NAMES:
        p = os.path.join(CSV_DIR, f'tdl_{scen}_64.csv')
        if os.path.exists(p):
            df = pd.read_csv(p)
            fig, ax = plt.subplots(figsize=(4.8, 3.8))
            snr = df['SNR (dB)'].values
            for m, col in [('ls', 'LS NMSE (dB)'), ('mmse', 'MMSE NMSE (dB)'),
                           ('sadlcs', 'Sa-DLCS NMSE (dB)'), ('ncrn', 'NCRN NMSE (dB)')]:
                if col in df:
                    c, ls, mk, lbl = STYLE[m]
                    ax.plot(snr, df[col], color=c, linestyle=ls, marker=mk, label=lbl)
            ax.set_xlabel('SNR (dB)')
            ax.set_ylabel('NMSE (dB)')
            ax.grid(True)
            ax.legend(loc='lower left', fontsize=7.5)
            save_clean_figure(fig, f'legacy_single_{scen}_64')

# ─── 7. Figure 12: Raw vs. Blended Comparison (Single Panels) ─────────
def plot_figure_12_single_panels():
    # Low SNR (-15 dB)
    vals_m15 = {'ncrn_raw': [], 'ncrn_blend': [], 'ncrt_raw': [], 'ncrt_blend': []}
    vals_p20 = {'ncrn_raw': [], 'ncrn_blend': [], 'ncrt_raw': [], 'ncrt_blend': []}
    for scen in SCENARIO_NAMES:
        df = pd.read_csv(os.path.join(CSV_DIR, f'ncrt_vs_ncrn_{scen}_full_pilot.csv'))
        r_m15 = df[df['snr_db'] == -15]
        r_p20 = df[df['snr_db'] == 20]
        vals_m15['ncrn_raw'].append(float(r_m15['ncrn_raw_nmse_db'].values[0]))
        vals_m15['ncrn_blend'].append(float(r_m15['ncrn_nmse_db'].values[0]))
        vals_m15['ncrt_raw'].append(float(r_m15['ncrt_raw_nmse_db'].values[0]))
        vals_m15['ncrt_blend'].append(float(r_m15['ncrt_nmse_db'].values[0]))
        
        vals_p20['ncrn_raw'].append(float(r_p20['ncrn_raw_nmse_db'].values[0]))
        vals_p20['ncrn_blend'].append(float(r_p20['ncrn_nmse_db'].values[0]))
        vals_p20['ncrt_raw'].append(float(r_p20['ncrt_raw_nmse_db'].values[0]))
        vals_p20['ncrt_blend'].append(float(r_p20['ncrt_nmse_db'].values[0]))

    labels = [s.replace('_', ' ') for s in SCENARIO_NAMES]
    x = np.arange(len(labels))
    w = 0.20
    
    # 12A: -15 dB
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.bar(x - 1.5*w, vals_m15['ncrn_raw'],   width=w, color='#EF5350', label='NCRN (raw)')
    ax.bar(x - 0.5*w, vals_m15['ncrn_blend'], width=w, color='#B71C1C', label='NCRN (blended)')
    ax.bar(x + 0.5*w, vals_m15['ncrt_raw'],   width=w, color='#42A5F5', label='NCRT (raw)')
    ax.bar(x + 1.5*w, vals_m15['ncrt_blend'], width=w, color='#0D47A1', label='NCRT (blended)')
    ax.set_ylabel('NMSE at −15 dB (dB)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha='right', fontsize=8.5)
    ax.set_ylim(-3, 18)
    ax.grid(True, axis='y')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=8.0, frameon=True)
    save_clean_figure(fig, 'fig12a_raw_vs_blended_m15db')

    # 12B: +20 dB
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.bar(x - 1.5*w, vals_p20['ncrn_raw'],   width=w, color='#EF5350', label='NCRN (raw)')
    ax.bar(x - 0.5*w, vals_p20['ncrn_blend'], width=w, color='#B71C1C', label='NCRN (blended)')
    ax.bar(x + 0.5*w, vals_p20['ncrt_raw'],   width=w, color='#42A5F5', label='NCRT (raw)')
    ax.bar(x + 1.5*w, vals_p20['ncrt_blend'], width=w, color='#0D47A1', label='NCRT (blended)')
    ax.set_ylabel('NMSE at +20 dB (dB)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha='right', fontsize=8.5)
    ax.set_ylim(-36, -14)
    ax.grid(True, axis='y')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=8.0, frameon=True)
    save_clean_figure(fig, 'fig12b_raw_vs_blended_p20db')

# ─── 8. MATLAB 5G, 6G & Combined Figures (Single Panels) ──────────────
def plot_matlab_figures():
    # 5G NR TDL
    df_5g = pd.read_csv(os.path.join(CSV_DIR, 'matlab_5g_eval.csv'))
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    snr = df_5g['snr_db'].values
    for m in ['ls', 'ncrn', 'ncrt', 'ncrt_syn']:
        col = f'{m}_nmse_db'
        if col in df_5g:
            c, ls, mk = ('#000', '--', '^') if m == 'ls' else ('#B71C1C', '-', 'o') if m == 'ncrn' else ('#0D47A1', '-', 's') if m == 'ncrt' else ('#1976D2', ':', 'v')
            lbl = 'LS' if m == 'ls' else 'NCRN (target)' if m == 'ncrn' else 'NCRT (target)' if m == 'ncrt' else 'NCRT (synthetic)'
            ax.plot(snr, df_5g[col], color=c, linestyle=ls, marker=mk, label=lbl)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('NMSE (dB)')
    ax.grid(True)
    ax.legend(loc='lower left', fontsize=8)
    save_clean_figure(fig, 'fig14_matlab_5g_eval')

    # 6G Sub-THz (142 GHz)
    df_6g = pd.read_csv(os.path.join(CSV_DIR, 'matlab_6g_eval.csv'))
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    for m in ['ls', 'ncrn', 'ncrt', 'ncrt_syn']:
        col = f'{m}_nmse_db'
        if col in df_6g:
            c, ls, mk = ('#000', '--', '^') if m == 'ls' else ('#B71C1C', '-', 'o') if m == 'ncrn' else ('#0D47A1', '-', 's') if m == 'ncrt' else ('#1976D2', ':', 'v')
            lbl = 'LS' if m == 'ls' else 'NCRN (target)' if m == 'ncrn' else 'NCRT (target)' if m == 'ncrt' else 'NCRT (synthetic)'
            ax.plot(snr, df_6g[col], color=c, linestyle=ls, marker=mk, label=lbl)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('NMSE (dB)')
    ax.grid(True)
    ax.legend(loc='lower left', fontsize=8)
    save_clean_figure(fig, 'fig15_matlab_6g_subthz_eval')

    # Combined 5G+6G
    df_comb = pd.read_csv(os.path.join(CSV_DIR, 'matlab_5g_6g_combined_eval.csv'))
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    for m in ['ls', 'ncrn', 'ncrt', 'ncrt_syn']:
        col = f'{m}_nmse_db'
        if col in df_comb:
            c, ls, mk = ('#000', '--', '^') if m == 'ls' else ('#B71C1C', '-', 'o') if m == 'ncrn' else ('#0D47A1', '-', 's') if m == 'ncrt' else ('#1976D2', ':', 'v')
            lbl = 'LS' if m == 'ls' else 'NCRN (target)' if m == 'ncrn' else 'NCRT (target)' if m == 'ncrt' else 'NCRT (synthetic)'
            ax.plot(snr, df_comb[col], color=c, linestyle=ls, marker=mk, label=lbl)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('NMSE (dB)')
    ax.grid(True)
    ax.legend(loc='lower left', fontsize=8)
    save_clean_figure(fig, 'fig16_combined_5g_6g_eval')

# ─── 9. Hexa-X 142 GHz Sub-THz Benchmark ──────────────────────────────
def plot_hexax_figure():
    df = pd.read_csv(os.path.join(CSV_DIR, 'hexa_x_6g_subthz.csv'))
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    snr = df['snr_db'].values
    for col, lbl, c, ls, mk in [('noisy_nmse_db', 'LS (Noisy)', '#000000', '--', '^'),
                                ('ncrn_nmse_db', 'NCRN', '#B71C1C', '-', 'o'),
                                ('trust_net_nmse_db', 'NCRT', '#0D47A1', '-', 's')]:
        if col in df:
            ax.plot(snr, df[col], color=c, linestyle=ls, marker=mk, label=lbl)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('NMSE (dB)')
    ax.grid(True)
    ax.legend(loc='lower left', fontsize=8)
    save_clean_figure(fig, 'fig17_hexax_142ghz_subthz')

# ─── 10. Figure 18: Application UAV & V2V Single Panels ───────────────
def plot_application_figures():
    for tag, name in [('uav_a2g', 'application_uav_a2g_eval'),
                      ('v2v_mobility', 'application_v2v_mobility_eval'),
                      ('uav_v2x_combined', 'application_uav_v2x_combined_eval')]:
        csv_p = os.path.join(CSV_DIR, f'{name}.csv')
        if not os.path.exists(csv_p): continue
        df = pd.read_csv(csv_p)
        snr = df['snr_db'].values
        
        # Panel A: Raw vs. Blended
        fig, ax = plt.subplots(figsize=(4.8, 3.8))
        for col, lbl, c, ls, mk in [('ls_nmse_db', 'LS', '#000000', '--', '^'),
                                    ('ncrn_nmse_db', 'NCRN (blended)', '#B71C1C', '-', 'o'),
                                    ('ncrn_raw_nmse_db', 'NCRN (raw)', '#E57373', ':', 'x'),
                                    ('ncrt_app_nmse_db', 'NCRT (blended)', '#0D47A1', '-', 's'),
                                    ('ncrt_app_raw_nmse_db', 'NCRT (raw)', '#64B5F6', ':', '+')]:
            if col in df:
                ax.plot(snr, df[col], color=c, linestyle=ls, marker=mk, label=lbl)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('NMSE (dB)')
        ax.grid(True)
        ax.legend(loc='lower left', fontsize=7.5)
        save_clean_figure(fig, f'fig18_{tag}_raw_vs_blended')

        # Panel B: Everything Blended (Target vs Synthetic)
        fig, ax = plt.subplots(figsize=(4.8, 3.8))
        for col, lbl, c, ls, mk in [('ls_nmse_db', 'LS', '#000000', '--', '^'),
                                    ('ncrn_nmse_db', 'NCRN (target-trained)', '#B71C1C', '-', 'o'),
                                    ('ncrt_app_nmse_db', 'NCRT (target-trained)', '#0D47A1', '-', 's'),
                                    ('ncrt_base_nmse_db', 'NCRT (synthetic-trained)', '#1976D2', ':', 'v')]:
            if col in df:
                ax.plot(snr, df[col], color=c, linestyle=ls, marker=mk, label=lbl)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('NMSE (dB)')
        ax.grid(True)
        ax.legend(loc='lower left', fontsize=7.5)
        save_clean_figure(fig, f'fig18_{tag}_everything_blended')

# ─── 11. F1 and Classification Clean Figures (No Titles, 1 Plot/File) ──
def plot_f1_clean_figures():
    # 1. Classification Curves vs SNR (4 single plots)
    csv_snr = os.path.join(CSV_DIR, 'f1_metrics_vs_snr.csv')
    if os.path.exists(csv_snr):
        df = pd.read_csv(csv_snr)
        metrics = [
            ('accuracy_pct', 'Accuracy (%)', 'f1_clean_accuracy_vs_snr'),
            ('precision_pct', 'Precision (%)', 'f1_clean_precision_vs_snr'),
            ('recall_pct', 'Recall (%)', 'f1_clean_recall_vs_snr'),
            ('f1_score', 'F1 Score', 'f1_clean_f1_score_vs_snr')
        ]
        methods_order = ['LS', 'MMSE', 'NCRN (blended)', 'NCRN (raw)', 'NCRT (blended)', 'NCRT (raw)']
        style_map = {
            'LS': ('#000000', '--', '^'),
            'MMSE': ('#1B5E20', '-.', 'D'),
            'NCRN (blended)': ('#B71C1C', '-', 'o'),
            'NCRN (raw)': ('#E57373', ':', 'x'),
            'NCRT (blended)': ('#0D47A1', '-', 's'),
            'NCRT (raw)': ('#64B5F6', ':', '+')
        }
        for col_name, y_label, file_name in metrics:
            fig, ax = plt.subplots(figsize=(5.4, 4.3))
            for m in methods_order:
                sub = df[df['method'] == m]
                if not sub.empty:
                    c, ls, mk = style_map[m]
                    ax.plot(sub['snr_db'], sub[col_name], color=c, linestyle=ls, marker=mk, label=m)
            ax.set_xlabel('SNR (dB)')
            ax.set_ylabel(y_label)
            ax.grid(True)
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=7.5, frameon=True, columnspacing=0.8)
            save_clean_figure(fig, file_name)

    # 2. 2D Spatial F1 Maps (4 single plots, no titles)
    for tag, name in [('ncrt', 'spatial_f1_map_ncrt'), ('ncrn', 'spatial_f1_map_ncrn'), ('delta', 'spatial_f1_map_delta')]:
        p = os.path.join(CSV_DIR, f'{name}.csv')
        if os.path.exists(p):
            mat = np.loadtxt(p, delimiter=',')
            fig, ax = plt.subplots(figsize=(5.0, 4.2))
            cmap = 'coolwarm' if tag == 'delta' else 'viridis'
            vmin, vmax = (-0.05, 0.05) if tag == 'delta' else (0.5, 0.9)
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            ax.set_xlabel('Tx Antenna Index')
            ax.set_ylabel('Rx Antenna Index')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            save_clean_figure(fig, f'spatial_f1_map_clean_{tag}')

    # 3. Clean Annotated Heatmaps (No title inside plot, clean white lines)
    import seaborn as sns
    for mat_name, cbar_lbl, file_out, cmap, vmin, vmax, fmt in [
        ('delta_f1_gain_matrix.csv', r'$\Delta$F1 (NCRT − NCRN)', 'heatmap_clean_delta_f1_gain', 'coolwarm', -0.05, 0.55, '+.2f'),
        ('ncrt_f1_matrix.csv', 'NCRT F1 Score', 'heatmap_clean_ncrt_f1_values', 'Blues', 0.50, 0.96, '.2f'),
        ('ncrn_f1_matrix.csv', 'NCRN F1 Score', 'heatmap_clean_ncrn_f1_values', 'Blues', 0.20, 0.96, '.2f'),
        ('ncrt_nmse_matrix.csv', 'NCRT NMSE (dB)', 'heatmap_clean_ncrt_nmse_values', 'RdYlBu', -35.0, 15.0, '.1f')
    ]:
        p = os.path.join(CSV_DIR, mat_name)
        if os.path.exists(p):
            df_mat = pd.read_csv(p, index_col=0)
            fig, ax = plt.subplots(figsize=(9.5, 6.5))
            sns.heatmap(df_mat, annot=True, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax,
                        linewidths=1.2, linecolor='white',
                        cbar_kws={'label': cbar_lbl, 'shrink': 0.85, 'pad': 0.03},
                        annot_kws={'size': 9, 'weight': 'bold'}, ax=ax)
            ax.set_xlabel('Operating SNR (dB)', fontweight='bold')
            ax.set_ylabel('3GPP TDL Propagation Scenario', fontweight='bold')
            plt.xticks(rotation=0, fontsize=9, fontweight='bold')
            plt.yticks(rotation=0, fontsize=9, fontweight='bold')
            save_clean_figure(fig, file_out, figsize=(9.5, 6.5))

# ─── 12. Zip All Clean Figures ────────────────────────────────────────
def create_clean_zip():
    zip_path = os.path.join(ROOT, 'outputs', 'figures_clean.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for fname in sorted(os.listdir(OUT_DIR)):
            if fname.endswith(('.png', '.pdf')):
                z.write(os.path.join(OUT_DIR, fname), arcname=fname)
    print(f"\n[DONE] Created clean figure archive: {zip_path}")
    import shutil
    shutil.copyfile(zip_path, os.path.join(ARTIFACT_DIR, 'figures_clean.zip'))

def main():
    print("=" * 70)
    print("GENERATING 100% JOURNAL-COMPLIANT CLEAN FIGURES (NO TITLES, 1 PLOT/FILE)")
    print("=" * 70)
    plot_figure_2_pooled()
    plot_figure_4_single_panels()
    plot_figure_5_margin_vs_snr()
    plot_figure_6_single_panels()
    plot_all_10_scenarios_individually()
    plot_legacy_figures()
    plot_figure_12_single_panels()
    plot_matlab_figures()
    plot_hexax_figure()
    plot_application_figures()
    plot_f1_clean_figures()
    create_clean_zip()
    print("\nAll single-panel, title-free figures generated successfully!")

if __name__ == '__main__':
    main()

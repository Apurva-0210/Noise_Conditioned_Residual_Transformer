"""
eval_daugt_vs_ncrn_FIXED.py  (rewired for DAUGT V2)
────────────────────────────────────────────────────────────────────────
Run this AFTER training both models:

    python stage2_tdl/ncrn/train_ncrn_tdl.py                 # -> models/ncrn_tdl_64.keras
    python stage2_tdl/daugt/train_daugt_v2_FIXED.py --array 64  # -> models/daugt_v2_64.keras

This script does NOT fabricate numbers -- it will skip any method whose
.keras file isn't present in models/, and it prints [SKIP] rather than
plotting a placeholder. Run it, then read outputs/csv/daugt_vs_ncrn_*.csv
for the numbers to put in a paper.

Two regimes are evaluated, because DAUGT and NCRN aren't tested under
the same conditions:

  FULL-PILOT : every antenna-grid RE observed each frame (NCRN's native
               assumption). Fairest apples-to-apples comparison at
               matched information.
  SPARSE-PILOT: only `PILOT_DENSITY` fraction of REs observed (DAUGT's
               design regime). NCRN was never trained for this and is
               evaluated with the missing REs zero-filled, exactly like
               it would fail in a fielded system with realistic pilot
               overhead -- this is the regime the DAUGT paper should
               lead with, since it's the one where the architectural
               difference should show up most clearly.

FIX (this revision): rewired for DAUGT V2 (train_daugt_v2_FIXED.py),
which is architecturally unrelated to the older train_daugt_tdl.py this
script previously targeted:

  - V2 takes THREE separate model inputs -- h_seq (T,Nr,Nt,2), a
    pilot_mask_seq (T,Nr,Nt), and snr_db (1,) -- not a single stitched
    13-channel tensor the way the old apply_pilot_and_snr_v2() built.
    Calling `daugt.predict(x_in[None,...])` on a single 13-channel
    tensor here would fail immediately with an input-count mismatch
    (the model expects a 3-element list, not one array) -- or, if you
    tried to hand it a single ndarray, Keras would raise something like
    "Model was constructed with N inputs, but N were not provided" and
    tell you the required input names.
  - V2's custom Keras layers are entirely different: DopplerPhaseEncoder,
    SNRConditioning, UncertaintyGuidedAttention, TransformerBlock -- not
    PhaseInnovation/LastFrame/TemporalAttentionPool/PatchTokenize/
    PatchUntokenize from the old script. Loading the V2 checkpoint with
    the old custom_objects dict raises:
      ValueError: Unknown layer: 'DopplerPhaseEncoder'. Please ensure
      you are using a `keras.utils.custom_object_scope`...
  - V2 saves to models/daugt_v2_{suffix}.keras (see train_daugt_v2_FIXED.py
    main()), not models/daugt_tdl_corrected_64.keras -- the old filename
    is never produced by V2 and would silently [SKIP] DAUGT forever.

  DAUGT V2's output is still the same (B,Nr,Nt,8) DER-H format
  [gamma_r,nu_r,alpha_r,beta_r,gamma_i,nu_i,alpha_i,beta_i], so
  ubmmse_fuse() (already fixed to self-estimate signal power -- see the
  DAUGT-v1 CHANGELOG) needs no changes and is reused as-is.
"""
import os, sys, csv, argparse, warnings

# Windows' cmd/PowerShell consoles default to a legacy codepage that can't
# encode the box-drawing/unicode characters used in this script's output
# (UnicodeEncodeError). Force UTF-8 stdout/stderr on Windows only.
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.channels   import gen_tdl_random, TIME_STEPS
from shared.baselines  import (ls_estimate, mmse_estimate,
                                sadlcs_postprocess, add_snr_channel,
                                snr_aware_blend)
from shared.pilot      import gen_pilot_mask
from shared.evidential import ubmmse_fuse, decompose_uncertainty, der_h_loss, der_h_mae

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage2_tdl', 'ncrn'))
# NCRN defines its own LastFrame layer (independent of DAUGT's old
# architecture, which no longer exists in V2 anyway -- V2 uses Lambda
# layers for frame extraction instead of a named custom layer).
from train_ncrn_tdl import LastFrame as NCRN_LastFrame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage2_tdl', 'daugt'))
# FIX: import DAUGT V2's actual custom layers (train_daugt_v2_FIXED.py),
# not the older train_daugt_tdl.py's PhaseInnovation/LastFrame/
# TemporalAttentionPool/PatchTokenize/PatchUntokenize -- V2 is a
# different architecture and doesn't define any of those classes.
from train_daugt_v2_FIXED import (DopplerPhaseEncoder, SNRConditioning,
                                   UncertaintyGuidedAttention, TransformerBlock,
                                   SNRSpatialExpand)

HERE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, '..', 'models')
OUT_FIG    = os.path.join(HERE, '..', 'outputs', 'figures')
OUT_CSV    = os.path.join(HERE, '..', 'outputs', 'csv')

SNR_RANGE     = [-15, -10, -5, 0, 5, 10, 15, 20]
PILOT_DENSITY = 0.35     # sparse-pilot regime density

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 9,
    'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
    'legend.frameon': True, 'legend.framealpha': 0.92,
    'axes.linewidth': 0.7, 'grid.linewidth': 0.4, 'grid.linestyle': ':',
    'grid.alpha': 0.45, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'savefig.dpi': 600,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

COLORS = {'ls': '#000000', 'mmse': '#1B5E20', 'sadlcs': '#0D3B66',
          'ncrn': '#7B1A1A', 'daugt': '#B8860B', 'daugt_w1': '#B8860B', 'ncrn_raw': '#7B1A1A'}
STYLES = {'ls': '--', 'mmse': '-.', 'sadlcs': '-', 'ncrn': '-', 'daugt': '-',
          'daugt_w1': ':', 'ncrn_raw': ':'}
MARKERS = {'ls': '^', 'mmse': 'D', 'sadlcs': 's', 'ncrn': 'o', 'daugt': 'P',
           'daugt_w1': 'x', 'ncrn_raw': 'x'}
LABELS = {'ls': 'LS', 'mmse': 'MMSE', 'sadlcs': 'Sa-DLCS',
          'ncrn': 'NCRN', 'daugt': 'DAUGT (ours)',
          'daugt_w1': 'DAUGT, no fusion (w=1)',
          'ncrn_raw': 'NCRN, no blend (alpha=1)'}
ORDER = ['ls', 'mmse', 'sadlcs', 'ncrn', 'ncrn_raw', 'daugt', 'daugt_w1']

# Candidate variance scales lambda for U-BMMSE fusion (see
# shared/evidential.py::ubmmse_fuse, var_scale). 1.0 = the network's own
# predicted variance, unchanged. Smaller = trust the correction more.
# Chosen per SNR on HELD-OUT samples, never on test samples.
VAR_SCALE_GRID = [1.0, 0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 1e-4, 1e-6]

# Candidate NCRN blend alphas (see shared/baselines.py::snr_aware_blend).
# snr_aware_blend's own alpha (a fixed hand-tuned sigmoid of SNR) is the
# equivalent of DAUGT's var_scale=1.0 default -- a reasonable but
# uncalibrated guess. Calibrating alpha the same way, on the same kind of
# held-out samples, is required for "DAUGT vs NCRN" to be a fair
# comparison of the two models' own best achievable output, not
# calibrated-fusion-vs-fixed-heuristic.
NCRN_ALPHA_GRID = [0.30, 0.45, 0.55, 0.70, 0.85, 0.90, 0.95, 1.0]

# FIX: filename now matches train_daugt_v2_FIXED.py's
# `model_path = os.path.join(out_dir, f'daugt_v2_{suffix}.keras')`
DAUGT_MODEL_NAME = 'daugt_v2_64.keras'


def load(path, custom=None):
    if not os.path.exists(path):
        return None
    try:
        return tf.keras.models.load_model(path, custom_objects=custom, safe_mode=False)
    except Exception as e:
        print(f"  [WARN] failed to load {path}: {e}")
        return None


def build_daugt_v2_inputs(hn, snr, mask, Nr, Nt):
    """
    Build DAUGT V2's three model inputs from a noisy channel sequence and
    pilot mask, matching exactly what train_daugt_v2_FIXED.py's
    make_dataset() constructs per sample:

      h_seq      : (T, Nr, Nt, 2) -- noisy channel, masked at every frame
                   with the SAME pilot pattern (mask doesn't vary across
                   the T-frame window here)
      mask_seq   : (T, Nr, Nt)    -- the pilot mask, repeated across T
      snr_db_arr : (1,)           -- known/measured SNR in dB

    hn   : (T, Nr, Nt, 2) noisy channel (noise already added, NOT yet
           masked)
    mask : (Nr, Nt) binary pilot mask
    """
    mask_seq = np.repeat(mask[None, ...], TIME_STEPS, axis=0).astype(np.float32)  # (T,Nr,Nt)
    h_seq = np.stack([
        (hn[:, :, :, 0] * mask_seq).astype(np.float32),
        (hn[:, :, :, 1] * mask_seq).astype(np.float32),
    ], axis=-1)  # (T,Nr,Nt,2)
    snr_db_arr = np.array([snr], dtype=np.float32)  # (1,)
    return h_seq, mask_seq, snr_db_arr


def evaluate(Nr, Nt, n_samples, sparse, calibrate=False, n_cal=40):
    ncrn = load(os.path.join(MODELS_DIR, 'ncrn_tdl_64.keras'), {'LastFrame': NCRN_LastFrame})
    sadlcs = load(os.path.join(MODELS_DIR, 'sa_dlcs_tdl_64.keras'))
    daugt_custom = {'DopplerPhaseEncoder': DopplerPhaseEncoder,
                     'SNRConditioning': SNRConditioning,
                     'UncertaintyGuidedAttention': UncertaintyGuidedAttention,
                     'TransformerBlock': TransformerBlock,
                     'SNRSpatialExpand': SNRSpatialExpand,
                     '_loss': der_h_loss(), 'der_h_mae': der_h_mae}
    daugt = load(os.path.join(MODELS_DIR, DAUGT_MODEL_NAME), daugt_custom)

    if ncrn is None:
        print("  [SKIP] ncrn_tdl_64.keras not found -- train it first.")
    if daugt is None:
        print(f"  [SKIP] {DAUGT_MODEL_NAME} not found -- train it first "
              f"(python stage2_tdl/daugt/train_daugt_v2_FIXED.py --array 64).")

    res = {m: {'nmse_lin': []} for m in ORDER}
    res['_var_scale'] = {}

    def _draw(snr):
        h = gen_tdl_random(Nr, Nt)
        sp = float(np.mean(h ** 2)) + 1e-10
        ns = np.sqrt(sp * 10 ** (-snr / 10))
        hn = (h + np.random.normal(0, ns, h.shape)).astype(np.float32)
        mask = (gen_pilot_mask(Nr, Nt, PILOT_DENSITY) if sparse
                else np.ones((Nr, Nt), dtype=np.float32))
        return h, hn, mask

    def _daugt_raw(hn, mask, snr):
        h_seq_in, mask_seq_in, snr_in = build_daugt_v2_inputs(hn, snr, mask, Nr, Nt)
        return daugt.predict(
            [h_seq_in[None, ...], mask_seq_in[None, ...], snr_in[None, ...]],
            verbose=0)[0]

    def _ncrn_raw(hn, mask, snr):
        hn_obs_seq = hn * mask[None, ..., None]
        needs_snr = ncrn.input_shape[-1] == 3
        ncin = add_snr_channel(hn_obs_seq, snr) if needs_snr else hn_obs_seq
        return ncrn.predict(ncin[None, ...], verbose=0)[0]

    for snr in SNR_RANGE:
        errs = {m: [] for m in ORDER}
        denom_sum = 0.0

        # ---- held-out calibration of the fusion variance scale ----
        # Uses its OWN fresh draws (independent of the test draws below),
        # so the chosen lambda is not tuned on the samples that get
        # reported.
        best_scale = 1.0
        if calibrate and daugt is not None:
            cal = []
            for _ in range(n_cal):
                h, hn, mask = _draw(snr)
                raw_c = _daugt_raw(hn, mask, snr)
                cal.append((h[-1], hn[-1] * mask[..., None], mask, raw_c,
                            decompose_uncertainty(raw_c[None, ...])))
            scores = {}
            for lam in VAR_SCALE_GRID:
                num = den = 0.0
                for hr_c, hl_c, mask_c, raw_c, unc_c in cal:
                    hh, _ = ubmmse_fuse(raw_c, hl_c, snr, pilot_mask=mask_c,
                                        var_scale=lam, unc=unc_c)
                    num += np.sum((hr_c - hh) ** 2)
                    den += np.sum(hr_c ** 2) + 1e-12
                scores[lam] = num / den
            best_scale = min(scores, key=scores.get)
            print(f"  [calibrate] SNR {snr:>4} dB -> var_scale = {best_scale:g}  "
                  f"(cal NMSE {10*np.log10(scores[best_scale]+1e-12):+.2f} dB vs "
                  f"{10*np.log10(scores[1.0]+1e-12):+.2f} dB at scale=1)")
        res['_var_scale'][snr] = best_scale

        # ---- held-out calibration of NCRN's blend alpha (fair match) ----
        best_alpha = None  # None => use snr_aware_blend's own fixed curve
        if calibrate and ncrn is not None:
            cal_n = []
            for _ in range(n_cal):
                h, hn, mask = _draw(snr)
                cal_n.append((h[-1], hn[-1] * mask[..., None],
                              _ncrn_raw(hn, mask, snr)))
            scores_n = {}
            for a in NCRN_ALPHA_GRID:
                num = den = 0.0
                for hr_c, hl_c, raw_c in cal_n:
                    pred_c = hl_c + a * raw_c
                    num += np.sum((hr_c - pred_c) ** 2)
                    den += np.sum(hr_c ** 2) + 1e-12
                scores_n[a] = num / den
            fixed_num = fixed_den = 0.0
            for hr_c, hl_c, raw_c in cal_n:
                pred_c = snr_aware_blend(raw_c, hl_c, snr_db_known=snr)
                fixed_num += np.sum((hr_c - pred_c) ** 2)
                fixed_den += np.sum(hr_c ** 2) + 1e-12
            fixed_score = fixed_num / fixed_den
            best_alpha_cand = min(scores_n, key=scores_n.get)
            if scores_n[best_alpha_cand] < fixed_score:
                best_alpha = best_alpha_cand
            print(f"  [calibrate] SNR {snr:>4} dB -> NCRN alpha = "
                  f"{'fixed curve' if best_alpha is None else best_alpha}  "
                  f"(cal NMSE {10*np.log10(min(scores_n[best_alpha_cand], fixed_score)+1e-12):+.2f} dB "
                  f"vs fixed-curve {10*np.log10(fixed_score+1e-12):+.2f} dB)")
        res.setdefault('_ncrn_alpha', {})[snr] = best_alpha

        for _ in range(n_samples):
            h = gen_tdl_random(Nr, Nt)
            # Noise scaled by each sample's own signal power, matching
            # both training generators' convention -- see DAUGT-v1
            # CHANGELOG for why this matters.
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

            errs['ls'].append(np.sum((hr - ls_estimate(hl_obs)) ** 2))
            errs['mmse'].append(np.sum((hr - mmse_estimate(hl_obs, noise_var=ns**2, Nt=Nt, sp=sp)) ** 2))

            if sadlcs is not None:
                pr = sadlcs.predict(hn[None, ...], verbose=0)[0]
                errs['sadlcs'].append(np.sum((hr - sadlcs_postprocess(pr, hl_obs)) ** 2))

            if ncrn is not None:
                hn_obs_seq = hn * mask[None, ..., None]
                needs_snr = ncrn.input_shape[-1] == 3
                ncin = add_snr_channel(hn_obs_seq, snr) if needs_snr else hn_obs_seq
                raw = ncrn.predict(ncin[None, ...], verbose=0)[0]
                pred = (hl_obs + best_alpha * raw if best_alpha is not None
                        else snr_aware_blend(raw, hl_obs, snr_db_known=snr))
                errs['ncrn'].append(np.sum((hr - pred) ** 2))
                # Matched ablation: NCRN with NO hand-tuned damping (alpha=1).
                # snr_aware_blend caps NCRN's trust in its own residual at
                # 0.30-0.55 below 5 dB, so comparing DAUGT-without-fusion to
                # NCRN-with-damping would not be apples to apples.
                errs['ncrn_raw'].append(np.sum((hr - (hl_obs + raw)) ** 2))

            if daugt is not None:
                # FIX: DAUGT V2 takes three separate inputs, not one
                # stitched 13-channel tensor -- build and pass them as a
                # list in the same order the model's tf.keras.Input(...)
                # calls declared them: [h_seq, pilot_mask_seq, snr_db].
                h_seq_in, mask_seq_in, snr_in = build_daugt_v2_inputs(hn, snr, mask, Nr, Nt)
                raw = daugt.predict(
                    [h_seq_in[None, ...], mask_seq_in[None, ...], snr_in[None, ...]],
                    verbose=0
                )[0]
                unc_t = decompose_uncertainty(raw[None, ...])
                h_hat, _ = ubmmse_fuse(raw, hl_obs, snr, pilot_mask=mask,
                                       var_scale=best_scale, unc=unc_t)
                errs['daugt'].append(np.sum((hr - h_hat) ** 2))
                # Ablation: trust the network's correction fully (w = 1)
                gam = unc_t['residual_mean'][0]
                errs['daugt_w1'].append(np.sum((hr - (hl_obs + gam)) ** 2))

        for m in ORDER:
            if errs[m]:
                res[m]['nmse_lin'].append(sum(errs[m]) / denom_sum)
            else:
                res[m]['nmse_lin'].append(None)
    return res


def plot_and_save(res, tag):
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    for m in ORDER:
        vals = res[m]['nmse_lin']
        valid = [(s, v) for s, v in zip(SNR_RANGE, vals) if v is not None and v > 0]
        if not valid:
            continue
        xs, ys = zip(*valid)
        ax.semilogy(xs, ys, color=COLORS[m], linestyle=STYLES[m],
                    marker=MARKERS[m], label=LABELS[m], linewidth=1.3, markersize=4)
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('NMSE')
    ax.grid(True, which='both'); ax.minorticks_on()
    ax.legend(loc='upper right', fontsize=7)
    plt.tight_layout(pad=0.3)
    os.makedirs(OUT_FIG, exist_ok=True)
    fig.savefig(os.path.join(OUT_FIG, f'daugt_vs_ncrn_{tag}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT_FIG, f'daugt_vs_ncrn_{tag}.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)

    os.makedirs(OUT_CSV, exist_ok=True)
    with open(os.path.join(OUT_CSV, f'daugt_vs_ncrn_{tag}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        ms = [m for m in ORDER if any(v is not None for v in res[m]['nmse_lin'])]
        w.writerow(['SNR (dB)'] + [f'{LABELS[m]} NMSE (dB)' for m in ms])
        for i, snr in enumerate(SNR_RANGE):
            row = [snr]
            for m in ms:
                v = res[m]['nmse_lin'][i]
                row.append(f'{10*np.log10(v+1e-12):+.3f}' if v else 'N/A')
            w.writerow(row)
    print(f"  -> daugt_vs_ncrn_{tag}.pdf/.png/.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_samples', type=int, default=300)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--calibrate_fuse', action='store_true',
                   help='choose the U-BMMSE variance scale per SNR on '
                        'held-out calibration samples (never on test samples)')
    p.add_argument('--n_cal', type=int, default=40,
                   help='held-out calibration samples per SNR point')
    args = p.parse_args()
    n = args.n_samples // 4 if args.quick else args.n_samples

    print("=" * 60)
    print("  DAUGT vs NCRN — full-pilot regime (fair, matched-info)")
    print("=" * 60)
    res_full = evaluate(64, 64, n, sparse=False, calibrate=args.calibrate_fuse, n_cal=args.n_cal)
    plot_and_save(res_full, 'full_pilot')

    print("\n" + "=" * 60)
    print(f"  DAUGT vs NCRN — sparse-pilot regime (density={PILOT_DENSITY})")
    print("=" * 60)
    res_sparse = evaluate(64, 64, n, sparse=True, calibrate=args.calibrate_fuse, n_cal=args.n_cal)
    plot_and_save(res_sparse, 'sparse_pilot')


if __name__ == '__main__':
    main()
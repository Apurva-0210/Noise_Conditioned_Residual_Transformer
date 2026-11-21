"""
shared/pilot.py
────────────────────────────────────────────────────────────────────────
Pilot-resource-element (RE) masking utilities for DAUGT.

The 64x64 antenna-domain grid used elsewhere in this repo (Nr x Nt) is
treated as a proxy resource grid: a subset of positions are "observed"
(pilot-bearing) each coherence block, the rest must be inferred. This
is the standard sparse-pilot channel-estimation setting in massive
MIMO, expressed at the antenna-domain level to stay consistent with
the rest of the pipeline (shared/channels.py, shared/baselines.py).

Contents
  gen_pilot_mask          - base pilot mask at a target density
  stochastic_pilot_dropout- training-time augmentation (PCST)
  ugapa_next_mask         - Uncertainty-Guided Adaptive Pilot Allocation
                            (inference-time closed-loop pilot placement)
"""
import numpy as np


# ─── Base pilot mask ───────────────────────────────────────────────────
def gen_pilot_mask(Nr, Nt, density, rng=None, structured=True):
    """
    Returns a binary (Nr, Nt) float32 mask, 1 = pilot RE, 0 = data RE.

    density     : fraction of REs carrying pilots, in (0, 1]
    structured  : if True, lay pilots on a jittered rectangular lattice
                  (closer to real DMRS/CSI-RS grids than pure i.i.d.
                  random sampling, which under-samples high-frequency
                  spatial correlation); if False, i.i.d. Bernoulli mask.
    """
    if rng is None:
        rng = np.random.default_rng()
    if density >= 1.0:
        return np.ones((Nr, Nt), dtype=np.float32)

    if not structured:
        return (rng.random((Nr, Nt)) < density).astype(np.float32)

    # Jittered lattice: pick a stride that gives ~`density` coverage,
    # then jitter each lattice point by +/-1 RE to avoid a perfectly
    # periodic (and therefore trivially interpolable) pattern.
    stride = max(1, int(round(1.0 / np.sqrt(density))))
    mask = np.zeros((Nr, Nt), dtype=np.float32)
    for i in range(0, Nr, stride):
        for j in range(0, Nt, stride):
            di = rng.integers(-1, 2)
            dj = rng.integers(-1, 2)
            ii = int(np.clip(i + di, 0, Nr - 1))
            jj = int(np.clip(j + dj, 0, Nt - 1))
            mask[ii, jj] = 1.0

    # Top up / trim to hit the requested density exactly (keeps the
    # curriculum below well-defined instead of "approximately right").
    cur = mask.sum()
    target = density * Nr * Nt
    flat_idx = np.arange(Nr * Nt)
    if cur < target:
        zeros = flat_idx[mask.flatten() == 0]
        add = rng.choice(zeros, size=int(target - cur), replace=False)
        mask.flat[add] = 1.0
    elif cur > target:
        ones = flat_idx[mask.flatten() == 1]
        rem = rng.choice(ones, size=int(cur - target), replace=False)
        mask.flat[rem] = 0.0
    return mask


# ─── PCST training augmentation ────────────────────────────────────────
def stochastic_pilot_dropout(mask, drop_prob, rng=None):
    """
    Randomly zeroes out a further `drop_prob` fraction of the *existing*
    pilot REs in `mask`. This is what makes PCST robust to pilot-density
    variation instead of overfitting to one fixed grid: at train time we
    show it many random sub-samples of an already-sparse grid so the
    transformer's mask-token pathway (see train_daugt_tdl.py) is
    exercised across the full range of realistic pilot budgets.
    """
    if rng is None:
        rng = np.random.default_rng()
    if drop_prob <= 0:
        return mask
    keep = (rng.random(mask.shape) >= drop_prob).astype(np.float32)
    return mask * keep


# ─── UGAPA: Uncertainty-Guided Adaptive Pilot Allocation ──────────────
def ugapa_next_mask(epistemic_map, current_mask, budget, min_spacing=1):
    """
    Greedy closed-loop pilot placement for the *next* coherence block,
    driven by the epistemic-uncertainty map DER-H produces for the
    current block (see shared/evidential.py: `decompose_uncertainty`).

    Idea: epistemic uncertainty (the "nu" / evidence term) marks
    regions the model has not been able to pin down from data it has
    seen so far -- exactly the regions where *more measurement*, not
    more inference, is the right fix. UGAPA re-allocates a fixed pilot
    budget away from already-confident REs and onto the highest
    epistemic-uncertainty REs, subject to a minimum spacing constraint
    so pilots don't clump (which would waste budget on redundant
    information at nearly-identical channel taps).

    Parameters
    ----------
    epistemic_map : (Nr, Nt) float array, higher = more uncertain
    current_mask  : (Nr, Nt) binary mask used for the block just estimated
    budget        : int, number of pilot REs to allocate next block
    min_spacing   : int, Chebyshev distance enforced between new pilots

    Returns
    -------
    next_mask : (Nr, Nt) float32 binary mask for the next block
    """
    Nr, Nt = epistemic_map.shape
    order = np.argsort(epistemic_map.flatten())[::-1]  # most uncertain first
    chosen = []
    next_mask = np.zeros((Nr, Nt), dtype=np.float32)
    occupied = np.zeros((Nr, Nt), dtype=bool)

    for idx in order:
        if len(chosen) >= budget:
            break
        i, j = divmod(int(idx), Nt)
        lo_i, hi_i = max(0, i - min_spacing), min(Nr, i + min_spacing + 1)
        lo_j, hi_j = max(0, j - min_spacing), min(Nt, j + min_spacing + 1)
        if occupied[lo_i:hi_i, lo_j:hi_j].any():
            continue
        next_mask[i, j] = 1.0
        occupied[lo_i:hi_i, lo_j:hi_j] = True
        chosen.append((i, j))

    # If spacing constraints starved us of the full budget, relax and
    # fill remaining slots by pure uncertainty rank (no spacing check).
    if len(chosen) < budget:
        remaining = budget - len(chosen)
        flat_mask = next_mask.flatten()
        free = order[~np.isin(order, [i * Nt + j for i, j in chosen])]
        fill = free[:remaining]
        flat_mask[fill] = 1.0
        next_mask = flat_mask.reshape(Nr, Nt)

    return next_mask


def uniform_baseline_mask(Nr, Nt, budget, rng=None):
    """Random-allocation baseline for benchmarking UGAPA (no uncertainty info)."""
    if rng is None:
        rng = np.random.default_rng()
    mask = np.zeros((Nr, Nt), dtype=np.float32)
    idx = rng.choice(Nr * Nt, size=budget, replace=False)
    mask.flat[idx] = 1.0
    return mask


# ─── Combine pilot masking + SNR conditioning into model input ────────
def apply_pilot_and_snr(h_noisy_seq, snr_db, pilot_mask_seq):
    """
    h_noisy_seq    : (T, Nr, Nt, 2) real/imag noisy channel sequence
    pilot_mask_seq : (T, Nr, Nt)    binary pilot mask, one per frame
    Returns (T, Nr, Nt, 4): [real*mask, imag*mask, snr_norm, mask]
    Unobserved (non-pilot) REs are zeroed -- the model has no
    measurement there and must rely entirely on PCST/DPTE inference,
    exactly like a real receiver would.
    """
    T, Nr, Nt, _ = h_noisy_seq.shape
    m = pilot_mask_seq[..., None].astype(np.float32)          # (T,Nr,Nt,1)
    masked = h_noisy_seq * m                                   # zero non-pilot REs
    snr_norm = np.full((T, Nr, Nt, 1), snr_db / 30.0, dtype=np.float32)
    return np.concatenate([masked, snr_norm, m], axis=-1)

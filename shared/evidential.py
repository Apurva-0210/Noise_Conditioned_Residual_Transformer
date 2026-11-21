"""
shared/evidential.py
────────────────────────────────────────────────────────────────────────
DER-H  : Deep Evidential Regression + Heteroscedastic head, loss, and
         uncertainty decomposition.
U-BMMSE: Bayes-optimal, uncertainty-weighted fusion that replaces the
         fixed-curve `snr_aware_blend` heuristic in shared/baselines.py.

Evidential regression (Amini et al., NeurIPS 2020) places a Normal
Inverse-Gamma (NIG) prior over an unknown Gaussian's (mean, variance):

    y ~ N(mu, sigma^2),  mu ~ N(gamma, sigma^2/nu),  sigma^2 ~ Inv-Gamma(alpha, beta)

A single forward pass predicts (gamma, nu, alpha, beta) per output
element. This yields, in closed form and without sampling or ensembles:

    prediction        = gamma
    aleatoric var     = E[sigma^2]        = beta / (alpha - 1)
    epistemic var     = Var[mu]           = beta / (nu * (alpha - 1))
    total predictive   = aleatoric + epistemic = beta*(1+1/nu)/(alpha-1)

We apply this per (real, imag) residual channel, giving DAUGT a
calibrated, per-antenna-element uncertainty map at zero extra
inference cost -- exactly what UGAPA (shared/pilot.py) and U-BMMSE
(below) both consume.
"""
import tensorflow as tf
import numpy as np


# ─── Output parameterisation ───────────────────────────────────────────
def split_nig_params(raw):
    """
    raw: (..., 4) -> (gamma, nu, alpha, beta), each (..., 1)
    Positivity constraints via softplus; alpha additionally offset by
    +1 so alpha > 1 (required for the variance to be finite).
    """
    gamma, nu_r, alpha_r, beta_r = tf.split(raw, 4, axis=-1)
    nu = tf.nn.softplus(nu_r) + 1e-6
    alpha = tf.nn.softplus(alpha_r) + 1.0 + 1e-6
    beta = tf.nn.softplus(beta_r) + 1e-6
    return gamma, nu, alpha, beta


# ─── NIG negative log-likelihood + evidence regularizer ───────────────
def nig_nll(y, gamma, nu, alpha, beta):
    """Closed-form marginal NLL of y under the Student-t predictive
    distribution implied by the NIG prior (Amini et al., Eq. 8)."""
    two_beta_lambda = 2.0 * beta * (1.0 + nu)
    t1 = 0.5 * tf.math.log(np.pi / nu)
    t2 = -alpha * tf.math.log(two_beta_lambda)
    t3 = (alpha + 0.5) * tf.math.log(nu * tf.square(y - gamma) + two_beta_lambda)
    t4 = tf.math.lgamma(alpha) - tf.math.lgamma(alpha + 0.5)
    return t1 + t2 + t3 + t4


def nig_evidence_reg(y, gamma, nu, alpha):
    """Penalizes evidence (nu, alpha) accumulated on samples where the
    prediction is far from the target -- keeps the model from claiming
    high confidence in regions it is actually getting wrong."""
    evidence = 2.0 * nu + alpha
    return tf.abs(y - gamma) * evidence


def der_h_loss(coeff_reg=0.01):
    """
    Returns a Keras-compatible loss(y_true, y_pred) where y_pred has
    8 channels = [gamma_re, nu_re, alpha_re, beta_re,
                  gamma_im, nu_im, alpha_im, beta_im] and y_true has 2
    channels = [residual_re, residual_im].
    """
    def _loss(y_true, y_pred):
        raw_re, raw_im = tf.split(y_pred, 2, axis=-1)
        y_re, y_im = tf.split(y_true, 2, axis=-1)

        g_re, n_re, a_re, b_re = split_nig_params(raw_re)
        g_im, n_im, a_im, b_im = split_nig_params(raw_im)

        nll = nig_nll(y_re, g_re, n_re, a_re, b_re) + \
              nig_nll(y_im, g_im, n_im, a_im, b_im)
        reg = nig_evidence_reg(y_re, g_re, n_re, a_re) + \
              nig_evidence_reg(y_im, g_im, n_im, a_im)
        return tf.reduce_mean(nll + coeff_reg * reg)
    return _loss


def der_h_mae(y_true, y_pred):
    """Point-estimate MAE (on gamma) for monitoring, comparable to the
    plain 'mae' metric NCRN reports."""
    raw_re, raw_im = tf.split(y_pred, 2, axis=-1)
    y_re, y_im = tf.split(y_true, 2, axis=-1)
    g_re, _, _, _ = split_nig_params(raw_re)
    g_im, _, _, _ = split_nig_params(raw_im)
    return tf.reduce_mean(tf.abs(y_re - g_re) + tf.abs(y_im - g_im))


# ─── Decompose uncertainty from a raw 8-channel DER-H output ──────────
def decompose_uncertainty(raw_8ch):
    """
    raw_8ch: numpy array (..., 8) as produced by the DER-H head.
    Returns dict with per-element:
      residual_mean       (..., 2)   -- [gamma_re, gamma_im]
      aleatoric_var       (..., 2)
      epistemic_var       (..., 2)
      total_var           (..., 2)
      epistemic_scalar    (...,)     -- summed over re/im, for UGAPA
    """
    raw = tf.convert_to_tensor(raw_8ch, dtype=tf.float32)
    raw_re, raw_im = tf.split(raw, 2, axis=-1)
    g_re, n_re, a_re, b_re = split_nig_params(raw_re)
    g_im, n_im, a_im, b_im = split_nig_params(raw_im)

    ale_re = b_re / (a_re - 1.0)
    ale_im = b_im / (a_im - 1.0)
    epi_re = b_re / (n_re * (a_re - 1.0))
    epi_im = b_im / (n_im * (a_im - 1.0))

    out = {
        'residual_mean': tf.concat([g_re, g_im], axis=-1).numpy(),
        'aleatoric_var': tf.concat([ale_re, ale_im], axis=-1).numpy(),
        'epistemic_var': tf.concat([epi_re, epi_im], axis=-1).numpy(),
        'total_var': tf.concat([ale_re + epi_re, ale_im + epi_im], axis=-1).numpy(),
    }
    out['epistemic_scalar'] = out['epistemic_var'].sum(axis=-1)
    return out


# ─── U-BMMSE: Bayes-optimal uncertainty-aware fusion ──────────────────
def ubmmse_fuse(raw_8ch, h_noisy_last, snr_db_known, ref_var=1.0, pilot_mask=None,
                corr=0.0, var_scale=1.0, unc=None):
    """
    Replaces `snr_aware_blend` (shared/baselines.py) -- that function
    picks a fusion weight from a hand-fit sigmoid curve indexed only by
    scalar SNR. U-BMMSE instead derives a *per-element* fusion weight
    from the network's own predicted variance, giving a proper
    minimum-mean-square-error combination between "trust the residual
    correction" and "trust the raw noisy pilot observation":

        w = sigma_ref^2 / (sigma_ref^2 + sigma_pred_total^2)

    sigma_pred_total^2 -> 0   (confident correction)  => w -> 1, use gamma fully
    sigma_pred_total^2 -> big (uncertain correction)   => w -> 0, fall back to
                                                           the raw noisy estimate

    BUGFIX: sigma_ref used to be computed as `10 ** (-snr_db/10)`,
    silently assuming every channel realization has *unit* signal
    power. Measured signal power (mean(h**2)) actually ranges ~0.4 to
    12+ across TDL profiles -- a ~30x spread -- so the "how noisy is
    the raw observation" reference scale was wrong by up to that same
    factor for most samples, systematically miscalibrating the fusion
    weight regardless of how well the network itself was trained (this
    fusion step only runs at inference; training's loss never touches
    it, so no amount of additional training could ever have fixed it).

    Since DAUGT doesn't get oracle knowledge of the true signal power
    at inference (unlike the MMSE baseline, which is allowed to as a
    genie-aided oracle), we self-estimate it from the observed noisy
    pilot data: for pilot REs, h_noisy = h_true + noise are
    independent, so E[|h_noisy|^2] over pilot locations ~= sp + noise_var,
    and with noise_var = sp * 10**(-snr/10) (the same relationship used
    to *generate* training noise), this solves to
    sp_hat = mean_pilot_power / (1 + 10**(-snr/10)).

    Parameters
    ----------
    raw_8ch        : (Nr, Nt, 8) DER-H output for one sample
    h_noisy_last   : (Nr, Nt, 2) noisy channel at the last frame
                     (zero at non-pilot REs, matching how the eval/
                     training pipelines build this array)
    snr_db_known   : float, known/measured SNR in dB
    ref_var        : optional override for the reference variance scale
    pilot_mask     : optional (Nr, Nt) binary mask of pilot locations.
                     If omitted, inferred as the REs where h_noisy_last
                     is nonzero (matches how callers currently build it).
    var_scale      : multiplier lambda applied to the network's predicted
                     total variance before fusion (b = lambda * sigma_tot).
                     1.0 reproduces the original behavior exactly.
                     POST-HOC VARIANCE RECALIBRATION: if the network's
                     predicted variance is systematically too large
                     (under-confident) at some SNR, the fusion discards a
                     correction that is actually accurate. Measured on the
                     full-pilot sweep: the raw network output (w=1) beat
                     the fused output by ~100x at -15 dB, while the fused
                     output was better than w=1 at 15-20 dB -- i.e. the
                     predicted variance is badly inflated at low SNR and
                     roughly right at high SNR. lambda is chosen PER SNR on
                     HELD-OUT samples (temperature-scaling style), never on
                     the samples used to report test NMSE. No retraining.
    unc            : optional precomputed decompose_uncertainty() result
                     for raw_8ch[None]; lets calibration loops evaluate many
                     lambda values without recomputing the decomposition.
    corr           : assumed error correlation rho in [0, 1) between the
                     raw observation and the corrected estimate.
                     Default 0.0 reproduces the original independent-
                     error weight exactly (no behavior change).

    CORRELATED-ERROR WEIGHT (corr > 0):
    The original weight w = a/(a+b), a=sigma_ref, b=sigma_tot, is the
    MMSE combination of two estimators with INDEPENDENT errors. Here the
    two estimators are h_noisy and h_noisy + gamma -- the second is
    computed FROM the first, so their errors are positively correlated.
    With covariance c = rho*sqrt(a*b) the MMSE weight on the corrected
    estimate becomes
        w = (a - c) / (a + b - 2c),
    which is >= a/(a+b) whenever the corrected estimate is the more
    accurate one (b < a) -- i.e. the independent-error formula
    systematically under-trusts a good correction. The effect is
    negligible at low SNR (a >> b, so w ~ 1 either way) and largest at
    high SNR, where the raw observation is already fairly clean. rho is
    a scalar that should be CALIBRATED ON HELD-OUT SAMPLES (see
    calibrate_fuse_corr in plotting/eval_daugt_vs_ncrn_v2.py), never
    tuned on the same samples used to report test NMSE.

    Returns
    -------
    h_hat  : (Nr, Nt, 2) fused channel estimate
    w_map  : (Nr, Nt, 2) per-element fusion weight (for diagnostics)
    """
    if unc is None:
        unc = decompose_uncertainty(raw_8ch[None, ...])
    gamma = unc['residual_mean'][0]      # (Nr, Nt, 2)
    sigma_tot = unc['total_var'][0] * float(var_scale)   # (Nr, Nt, 2)

    noise_ratio = 10 ** (-float(snr_db_known) / 10.0)  # noise_var / sp

    if ref_var != 1.0:
        sigma_ref = ref_var
    else:
        if pilot_mask is None:
            pilot_mask = (np.abs(h_noisy_last[..., 0]) + np.abs(h_noisy_last[..., 1])) > 0
        pilot_mask = pilot_mask.astype(bool)
        if pilot_mask.any():
            obs_power = np.mean(h_noisy_last[pilot_mask] ** 2)
        else:
            obs_power = np.mean(h_noisy_last ** 2)
        sp_hat = obs_power / (1.0 + noise_ratio)
        sigma_ref = sp_hat * noise_ratio
    sigma_ref = max(float(sigma_ref), 1e-8)

    a = sigma_ref
    b = sigma_tot
    rho = float(np.clip(corr, 0.0, 0.99))
    c = rho * np.sqrt(a * b)
    w = (a - c) / (a + b - 2.0 * c + 1e-12)
    w = np.clip(w, 0.0, 1.0)
    h_hat = h_noisy_last + w * gamma
    return h_hat.astype(np.float32), w.astype(np.float32)
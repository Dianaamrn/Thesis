"""Monitoring channels, window statistics, thresholds and the alarm rule.

Every channel works the same way:
  1. fit on the reference's fit half (R_fit): what "normal" looks like
  2. give every triple a value (its type, its MONDO categories, its score, its outlier flag, ...)
  3. for each window of WINDOW consecutive stream triples, compute one number: how different the
     window is from R_fit (bigger = more different)
  4. raise an alarm when that number is above the channel's threshold

Thresholds are either
  fixed       ADAZOR's z_{1-alpha/2} / sqrt(2n) (Phase 1), or
  calibrated  chosen so that, on simulated no-drift streams resampled from the reference's
              calibration half (R_cal), the chance of ANY alarm within HORIZON triples is BETA
              (Cobb et al. 2022: calibrate the stream, not the single test).

Alarm rule (same for every channel): test at every `stride`-th triple once a full window is
available; after an alarm the window starts empty, so one event is not counted 250 times.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import settings

N = settings.WINDOW


# ---------------------------------------------------------------------------
# Window statistics. x = per-triple values of one sequence, in order.
# Return the statistic at window ends t = N-1, N-1+stride, ... (sequence positions).
# ---------------------------------------------------------------------------

def positions(length, stride):
    return np.arange(N - 1, length, stride)


def _window_sums(x, pos):
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    return c[pos + 1] - c[pos + 1 - N]


def stat_rate(x, stride, p_ref, one_sided=False):
    """|window rate - reference rate| (ADAZOR's statistic); one_sided: only an increase counts (C2)."""
    pos = positions(len(x), stride)
    diff = _window_sums(x.astype(float), pos) / N - p_ref
    return pos, diff if one_sided else np.abs(diff)


def stat_multinomial_g(codes, stride, p_ref):
    """G = 2 sum o ln(o/e) of the window's category counts against reference proportions (C1)."""
    pos = positions(len(codes), stride)
    onehot = np.zeros((len(codes), len(p_ref)))
    onehot[np.arange(len(codes)), codes] = 1
    o = _window_sums(onehot, pos)
    e = N * p_ref[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        g = np.where(o > 0, o * np.log(o / e), 0.0)
    return pos, 2 * g.sum(axis=1)


def stat_multilabel_g(m, stride, q_ref):
    """Sum over categories of the binomial G statistic (a triple can be in several categories) (C3)."""
    pos = positions(len(m), stride)
    o = _window_sums(m.astype(float), pos)
    e = N * q_ref[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(o > 0, o * np.log(o / e), 0.0)
        b = np.where(N - o > 0, (N - o) * np.log((N - o) / (N - e)), 0.0)
    return pos, 2 * (a + b).sum(axis=1)


def stat_ks(x, stride, ref_sorted):
    """Two-sample Kolmogorov-Smirnov distance between the window's scores and R_fit's (C4)."""
    pos = positions(len(x), stride)
    m = len(ref_sorted)
    out = np.empty(len(pos))
    for j, t in enumerate(pos):
        w = np.sort(x[t - N + 1:t + 1])
        allv = np.concatenate([w, ref_sorted])
        out[j] = np.max(np.abs(np.searchsorted(w, allv, 'right') / N - np.searchsorted(ref_sorted, allv, 'right') / m))
    return pos, out


class MMD:
    """Squared MMD (biased) with an RBF kernel, bandwidth = median distance in R_fit (C4)."""

    def __init__(self, ref):
        self.ref = ref
        d2 = self._d2(ref, ref)
        self.gamma = 1.0 / max(np.median(d2[np.triu_indices(len(ref), 1)]), 1e-12)
        self.k_rr = np.exp(-self.gamma * d2).mean()

    @staticmethod
    def _d2(a, b):
        return np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T, 0.0)

    def stat(self, x, stride):
        pos = positions(len(x), stride)
        out = np.empty(len(pos))
        for j, t in enumerate(pos):
            w = x[t - N + 1:t + 1]
            out[j] = (np.exp(-self.gamma * self._d2(w, w)).mean() + self.k_rr
                      - 2 * np.exp(-self.gamma * self._d2(w, self.ref)).mean())
        return pos, out


# ---------------------------------------------------------------------------
# OCSVM outlier flags: in-sample (as in Phase 1) and conformal (Bates et al. 2023)
# ---------------------------------------------------------------------------

def fit_ocsvm(X_fit):
    scaler = StandardScaler() if settings.OCSVM['standardize'] else None
    Xs = scaler.fit_transform(X_fit) if scaler is not None else X_fit
    model = OneClassSVM(kernel='rbf', nu=settings.OCSVM['nu'], gamma=settings.OCSVM['gamma']).fit(Xs)
    return model, scaler


def ocsvm_scores(model, scaler, X):
    """Decision function: negative = outside the boundary. Lower = more unusual."""
    return model.decision_function(scaler.transform(X) if scaler is not None else X)


def conformal_pvalues(scores, cal_scores):
    """p(x) = (1 + #{calibration scores <= s(x)}) / (m + 1). Under no drift P(p <= a) <= a."""
    cal = np.sort(cal_scores)
    return (1 + np.searchsorted(cal, scores, side='right')) / (len(cal) + 1)


def conformal_pvalues_loo(cal_scores):
    """Leave-one-out p-values of the calibration points themselves (for simulating no-drift streams)."""
    cal = np.sort(cal_scores)
    return np.searchsorted(cal, cal_scores, side='right') / len(cal)


# ---------------------------------------------------------------------------
# Channel instances for one reference split
# ---------------------------------------------------------------------------

class ChannelInstance:
    """values: per-pool-row values; boot_values: values to resample for calibration (R_cal rows);
    stat(x) -> (positions, statistic); fixed_threshold or None (= calibrate)."""

    def __init__(self, name, stride, values, boot_values, stat, fixed_threshold=None, info=None):
        self.name, self.stride = name, stride
        self.values, self.boot_values, self.stat = values, boot_values, stat
        self.fixed_threshold = fixed_threshold
        self.info = info or {}


def fixed_ztest_threshold(alpha=settings.ALPHA):
    """ADAZOR ZTestDriftDetector threshold (adazor/src/adazor/detectors): z_{1-alpha/2}/sqrt(2n)."""
    return norm.ppf(1 - alpha / 2) / math.sqrt(2 * N)


def build_channels(pool_data, model_data, ref_fit, ref_cal, models, compositions, channel_names):
    """All requested channel instances for one split. pool_data: model-independent arrays;
    model_data[model]: arrays from step 1."""
    out = []
    want = set(channel_names)
    cfg = settings.CHANNELS
    if 'C1_types' in want:
        codes = pool_data['type_codes']
        k = codes.max() + 1
        p = (np.bincount(codes[ref_fit], minlength=k) + 0.5) / (len(ref_fit) + 0.5 * k)
        out.append(ChannelInstance('C1_types', cfg['C1_types']['stride'], codes, codes[ref_cal],
                                   lambda x, s, p=p: stat_multinomial_g(x, s, p)))
    if 'C2_novelty' in want:
        nov = pool_data['new_entity']
        p = nov[ref_fit].mean()
        out.append(ChannelInstance('C2_novelty', cfg['C2_novelty']['stride'], nov, nov[ref_cal],
                                   lambda x, s, p=p: stat_rate(x, s, p, one_sided=True),
                                   info={'reference_rate': float(p)}))
    if 'C3_ontology' in want and pool_data.get('categories') is not None:
        m = pool_data['categories']
        q = (m[ref_fit].sum(0) + 0.5) / (len(ref_fit) + 1)
        out.append(ChannelInstance('C3_ontology', cfg['C3_ontology']['stride'], m, m[ref_cal],
                                   lambda x, s, q=q: stat_multilabel_g(x, s, q)))
    for model in models:
        md = model_data[model]
        if 'C4_plausibility' in want:
            sc = md['score']
            ref_sorted = np.sort(sc[ref_fit])
            out.append(ChannelInstance(f'C4_plausibility|{model}|-', cfg['C4_plausibility']['stride'], sc, sc[ref_cal],
                                       lambda x, s, r=ref_sorted: stat_ks(x, s, r)))
        for comp in compositions:
            X = md[comp]
            if 'C4_mmd' in want:
                scaler = StandardScaler().fit(X[ref_fit])
                pca = PCA(n_components=min(cfg['C4_mmd']['pca'], X.shape[1]), random_state=0).fit(scaler.transform(X[ref_fit]))
                Z = pca.transform(scaler.transform(X))
                mmd = MMD(Z[ref_fit])
                out.append(ChannelInstance(f'C4_mmd|{model}|{comp}', cfg['C4_mmd']['stride'], Z, Z[ref_cal],
                                           lambda x, s, m_=mmd: m_.stat(x, s)))
            c5 = [c for c in want if c.startswith('C5_')]
            if not c5:
                continue
            ocsvm, scaler = fit_ocsvm(X[ref_fit])
            s_all = ocsvm_scores(ocsvm, scaler, X)
            nu = settings.OCSVM['nu']
            if 'C5_insample_fixed' in want:
                flags = (s_all < 0).astype(np.int8)    # OCSVM predict == -1
                p_ref = flags[ref_fit].mean()           # in-sample, as Phase 1
                out.append(ChannelInstance(f'C5_insample_fixed|{model}|{comp}', 1, flags, flags[ref_cal],
                                           lambda x, s, p=p_ref: stat_rate(x, s, p), fixed_threshold=fixed_ztest_threshold(),
                                           info={'reference_rate': float(p_ref),
                                                 'calibration_half_rate': float(flags[ref_cal].mean())}))
            if 'C5_conformal_fixed' in want or 'C5_conformal_calibrated' in want:
                pv = conformal_pvalues(s_all, s_all[ref_cal])
                flags = (pv <= nu).astype(np.int8)
                boot = (conformal_pvalues_loo(s_all[ref_cal]) <= nu).astype(np.int8)
                for name, thr in (('C5_conformal_fixed', fixed_ztest_threshold()), ('C5_conformal_calibrated', None)):
                    if name in want:
                        out.append(ChannelInstance(f'{name}|{model}|{comp}', 1, flags, boot,
                                                   lambda x, s: stat_rate(x, s, nu), fixed_threshold=thr,
                                                   info={'reference_rate': nu}))
    return out


# ---------------------------------------------------------------------------
# Calibration and alarms
# ---------------------------------------------------------------------------

def calibrate(ch, rng, horizon=None, beta=None, n_boot=None):
    """Threshold with P(any alarm within `horizon` no-drift triples) = beta, by resampling R_cal."""
    c = settings.CALIBRATION
    horizon, beta = horizon or c['horizon'], beta or c['beta']
    n_boot = n_boot or settings.CHANNELS.get(ch.name.split('|')[0], {}).get('n_boot', c['n_boot'])
    maxima = np.empty(n_boot)
    length = horizon + N - 1
    for b in range(n_boot):
        idx = rng.integers(0, len(ch.boot_values), length)
        _, s = ch.stat(ch.boot_values[idx], ch.stride)
        maxima[b] = s.max()
    return float(np.quantile(maxima, 1 - beta))


def run_alarms(pos, stat, threshold):
    """Alarm rule: test when a full window is available; after an alarm the window starts empty."""
    next_ok, tested, alarms = N - 1, 0, []
    for t, v in zip(pos, stat):
        if t < next_ok:
            continue
        tested += 1
        if v > threshold:
            alarms.append(int(t))
            next_ok = t + N
    return tested, alarms


def score_run(alarms, tested, length, switch):
    """Per-run metrics. Positions are 0-based stream positions (alarm = last triple of the window)."""
    a = np.asarray(alarms, dtype=int)
    out = {'stream_length': int(length), 'tested': int(tested), 'alarms': int(len(a)),
           'alarms_per_1000': 1000 * len(a) / length if length else np.nan,
           'first_alarm': int(a[0]) if len(a) else np.nan, 'any_alarm': bool(len(a))}
    if switch is not None:
        pre, post = a[a < switch], a[a >= switch]
        out.update({'switch': int(switch), 'false_alarms': int(len(pre)),
                    'false_alarms_per_1000': 1000 * len(pre) / switch if switch else np.nan,
                    'detected': bool(len(post)),
                    'delay': int(post[0] - switch) if len(post) else np.nan,
                    'alarms_after_switch': int(len(post))})
    return out

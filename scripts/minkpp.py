"""Min-K%++ and every aggregation variant tested in this study.

One design decision governs the whole project: the model is run **once** per
(model, split) and three numbers are cached per token —

    logp_t   = log p(x_t | x_<t)                        observed-token log-probability
    mu_t     = E_{v ~ p(.|x_<t)} [ log p(v | x_<t) ]    = -H(p), the negative entropy
    sigma_t  = sqrt( E_{v ~ p} [ (log p(v) - mu)^2 ] )  spread of log-prob under p

from which the Min-K%++ per-token score is the z-score

    z_t = (logp_t - mu_t) / sigma_t .

Every scorer below is a different function of the same cached arrays, so the
baseline and all variants are compared at **exactly matched compute** by
construction: no variant can be advantaged by a different forward pass.

Scoring convention: higher score = more likely a training member, so AUROC is
computed directly against label 1 = member.
"""

from __future__ import annotations

import zlib

import numpy as np

# --------------------------------------------------------------------------- #
# per-token quantities
# --------------------------------------------------------------------------- #


def zscores(logp: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Min-K%++ per-token score. Guards sigma = 0, which happens on a one-hot
    next-token distribution (the model is certain), where the z-score is
    undefined rather than infinite."""
    s = np.where(sigma > 1e-12, sigma, np.nan)
    return (logp - mu) / s


def _lowest_frac(v: np.ndarray, k: float) -> np.ndarray:
    """The lowest ceil(k * n) entries of v, NaNs dropped."""
    v = v[np.isfinite(v)]
    if v.size == 0:
        return v
    n = max(1, int(np.ceil(k * v.size)))
    return np.sort(v)[:n]


# --------------------------------------------------------------------------- #
# published baselines
# --------------------------------------------------------------------------- #


def loss_score(logp, mu, sigma, text=None):
    """Loss / perplexity attack: the mean log-probability."""
    return float(np.nanmean(logp))


def zlib_score(logp, mu, sigma, text=None):
    """Carlini et al.: log-probability sum calibrated by zlib entropy.

    Negated so that higher = more likely a member, matching every other scorer
    here. The raw ratio is (negative sum) / bytes, which is lower for members.
    """
    n_bytes = len(zlib.compress(text.encode("utf-8"))) if text else 1
    return float(np.nansum(logp) / max(n_bytes, 1))


def mink(logp, mu, sigma, text=None, k: float = 0.2):
    """Min-K% Prob (Shi et al., ICLR 2024): mean of the lowest k% of logp."""
    sel = _lowest_frac(logp, k)
    return float(sel.mean()) if sel.size else np.nan


def minkpp(logp, mu, sigma, text=None, k: float = 0.2):
    """Min-K%++ (Zhang et al., ICLR 2025): mean of the lowest k% of z. BASELINE."""
    sel = _lowest_frac(zscores(logp, mu, sigma), k)
    return float(sel.mean()) if sel.size else np.nan


# --------------------------------------------------------------------------- #
# H1 / H6 — the two forms already refuted on the target model; kept so the
# refutation is reproduced here rather than taken on trust
# --------------------------------------------------------------------------- #


def h1_window_max(logp, mu, sigma, text=None, k=0.2, w: int = 32,
                  stride: int | None = None, pool: str = "max"):
    """H1: Min-K%++ inside sliding windows of width w, pooled across windows.

    For a sequence shorter than w this reduces exactly to the baseline, which
    `self_check` verifies.
    """
    z = zscores(logp, mu, sigma)
    n = z.size
    if n <= w:
        sel = _lowest_frac(z, k)
        return float(sel.mean()) if sel.size else np.nan
    step = stride or max(1, w // 2)
    vals = []
    for a in range(0, n - w + 1, step):
        sel = _lowest_frac(z[a:a + w], k)
        if sel.size:
            vals.append(sel.mean())
    if not vals:
        return np.nan
    return float(np.max(vals) if pool == "max" else np.percentile(vals, 90))


def h6_chunk_vote(logp, mu, sigma, text=None, k=0.2, K: int = 3, pool: str = "max"):
    """H6: split into exactly K chunks, score each, pool. Fixed chunk count
    removes the 'more windows -> higher max' confound of H1."""
    z = zscores(logp, mu, sigma)
    if z.size < K:
        sel = _lowest_frac(z, k)
        return float(sel.mean()) if sel.size else np.nan
    vals = []
    for part in np.array_split(z, K):
        sel = _lowest_frac(part, k)
        if sel.size:
            vals.append(sel.mean())
    if not vals:
        return np.nan
    return float(np.max(vals) if pool == "max" else np.mean(vals))


# --------------------------------------------------------------------------- #
# H2 — fixed-count selection
# --------------------------------------------------------------------------- #


def h2_fixed_m(logp, mu, sigma, text=None, m: int = 12):
    """H2: keep the lowest m z-scores rather than the lowest k%, so the number
    of averaged terms no longer drifts with passage length."""
    z = zscores(logp, mu, sigma)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return np.nan
    return float(np.sort(z)[:min(m, z.size)].mean())


# --------------------------------------------------------------------------- #
# H3 — length-normalised aggregation
# --------------------------------------------------------------------------- #


def h3_length_norm(logp, mu, sigma, text=None, k=0.2):
    """H3: standardise the min-k% mean by what the same statistic would be under
    a length-matched null.

    The null here is the *within-item* z distribution: the expected mean of the
    lowest k% of n i.i.d. draws from this item's own empirical z distribution.
    Dividing by it removes the part of the statistic that is driven purely by
    how many tokens the item has, which is the mechanism C1 names.
    """
    z = zscores(logp, mu, sigma)
    z = z[np.isfinite(z)]
    if z.size < 2:
        return np.nan
    sel = np.sort(z)[:max(1, int(np.ceil(k * z.size)))]
    # expected lowest-k% mean of a normal with this item's mean/sd, via the
    # order-statistic approximation E[min-k% mean] = mu - sd * phi(q)/q
    from math import exp, pi, sqrt
    q = max(sel.size / z.size, 1e-6)
    # inverse normal CDF at q, Acklam-style rational approximation is overkill:
    # use the empirical quantile of a standard normal sample instead, cached
    zq = _STD_NORMAL_QUANTILE(q)
    phi = exp(-0.5 * zq * zq) / sqrt(2 * pi)
    expected = z.mean() - z.std(ddof=0) * (phi / q)
    denom = abs(expected) if abs(expected) > 1e-9 else 1e-9
    return float(sel.mean() / denom)


_NORM_CACHE: dict[int, np.ndarray] = {}


def _STD_NORMAL_QUANTILE(q: float) -> float:
    """Standard-normal quantile from a fixed large sample. Deterministic and
    dependency-free; accurate to ~1e-3 in the range used here."""
    if 0 not in _NORM_CACHE:
        _NORM_CACHE[0] = np.sort(np.random.default_rng(0).standard_normal(2_000_000))
    s = _NORM_CACHE[0]
    return float(s[min(int(q * s.size), s.size - 1)])


# --------------------------------------------------------------------------- #
# H4 — magnitude-weighted aggregation
# --------------------------------------------------------------------------- #


def h4_magnitude_weighted(logp, mu, sigma, text=None, k=0.2, power: float = 1.0):
    """H4: weight each selected token by |z|^power, so depth below the mode
    counts rather than every selected token counting equally."""
    sel = _lowest_frac(zscores(logp, mu, sigma), k)
    if sel.size == 0:
        return np.nan
    wts = np.abs(sel) ** power
    tot = wts.sum()
    if tot < 1e-12:
        return float(sel.mean())
    return float((sel * wts).sum() / tot)


# --------------------------------------------------------------------------- #
# H5 — order-statistic aggregation
# --------------------------------------------------------------------------- #


def h5_order_statistic(logp, mu, sigma, text=None, q: float = 0.1):
    """H5: use a single quantile of the z distribution instead of the mean of a
    tail. Robust to how many tokens happen to fall in the tail."""
    z = zscores(logp, mu, sigma)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return np.nan
    return float(np.quantile(z, q))


# --------------------------------------------------------------------------- #
# H8 — position-weighted aggregation
# --------------------------------------------------------------------------- #


def h8_position_weighted(logp, mu, sigma, text=None, k=0.2, skip: int = 8,
                         mode: str = "skip"):
    """H8: early tokens have little conditioning context, so their z-scores are
    noisy. Either drop the first `skip` positions or downweight by position."""
    z = zscores(logp, mu, sigma)
    n = z.size
    if mode == "skip":
        if n <= skip + 2:
            return np.nan
        sel = _lowest_frac(z[skip:], k)
        return float(sel.mean()) if sel.size else np.nan
    # linear ramp: weight 0 at position 0, 1 from `skip` onwards
    ramp = np.clip(np.arange(n) / max(skip, 1), 0.0, 1.0)
    ok = np.isfinite(z)
    zz, rr = z[ok], ramp[ok]
    if zz.size == 0:
        return np.nan
    order = np.argsort(zz)[:max(1, int(np.ceil(k * zz.size)))]
    wts = rr[order]
    if wts.sum() < 1e-12:
        return float(zz[order].mean())
    return float((zz[order] * wts).sum() / wts.sum())


# --------------------------------------------------------------------------- #
# H9 — two-sided aggregation
# --------------------------------------------------------------------------- #


def h9_two_sided(logp, mu, sigma, text=None, k=0.2, beta: float = 0.5):
    """H9: the paper's own theory says training tokens sit *at* modes, which is a
    high-z signal the one-sided statistic discards. Combine both tails:

        score = (1 - beta) * mean(lowest k%) + beta * mean(highest k%)
    """
    z = zscores(logp, mu, sigma)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return np.nan
    n = max(1, int(np.ceil(k * z.size)))
    s = np.sort(z)
    return float((1 - beta) * s[:n].mean() + beta * s[-n:].mean())


# --------------------------------------------------------------------------- #
# H10 — early termination (efficiency, not accuracy)
# --------------------------------------------------------------------------- #


def h10_prefix(logp, mu, sigma, text=None, k=0.2, frac: float = 0.5):
    """H10: score only the first `frac` of the tokens. Measures how much of the
    sequence is actually needed, i.e. the accuracy/compute trade-off."""
    z = zscores(logp, mu, sigma)
    n = max(2, int(np.ceil(frac * z.size)))
    sel = _lowest_frac(z[:n], k)
    return float(sel.mean()) if sel.size else np.nan


# --------------------------------------------------------------------------- #
# NEW variants, derived from the measured failures (see REPORT.md §5)
# --------------------------------------------------------------------------- #


def n1_k_ensemble(logp, mu, sigma, text=None, ks=(0.1, 0.2, 0.3, 0.4, 0.5)):
    """N1: average the Min-K%++ statistic over a grid of k instead of choosing
    one. Motivated by the measurement that k moves AUROC more than any
    aggregation variant, while the paper does not say which k it used."""
    z = zscores(logp, mu, sigma)
    vals = []
    for k in ks:
        sel = _lowest_frac(z, k)
        if sel.size:
            vals.append(sel.mean())
    return float(np.mean(vals)) if vals else np.nan


def n2_length_adaptive_k(logp, mu, sigma, text=None, k_short=0.5, k_long=0.15,
                         n_short=40, n_long=200):
    """N2: k as a function of token count.

    Derived, not invented: the per-item difference between fixed-m and the
    baseline correlates with token count at r = -0.71, i.e. a fixed count helps
    short passages and hurts long ones. A fraction that *shrinks* with length
    is the continuous version of that observation.
    """
    z = zscores(logp, mu, sigma)
    n = z.size
    t = np.clip((n - n_short) / max(n_long - n_short, 1), 0.0, 1.0)
    k = k_short + t * (k_long - k_short)
    sel = _lowest_frac(z, k)
    return float(sel.mean()) if sel.size else np.nan


def n3_rank_normalised(logp, mu, sigma, text=None, k=0.2):
    """N3: replace the z *value* by its within-item rank before aggregating.

    If part of the length dependence is that longer items have heavier tails in
    absolute z, a rank-based statistic is immune to it while keeping the global
    selection that the failure analysis says is already well targeted.
    """
    z = zscores(logp, mu, sigma)
    ok = np.isfinite(z)
    zz = z[ok]
    if zz.size < 2:
        return np.nan
    ranks = np.empty(zz.size)
    ranks[np.argsort(zz)] = np.arange(zz.size) / (zz.size - 1)
    n = max(1, int(np.ceil(k * zz.size)))
    return float(np.sort(ranks)[:n].mean())


def n4_trimmed_tail(logp, mu, sigma, text=None, k=0.2, trim=0.1):
    """N4: trim the most extreme tokens before averaging the tail.

    The failure analysis says global selection is well targeted; this asks
    whether its *worst* members are outliers (tokenizer artefacts, rare
    Unicode) that add variance rather than signal.
    """
    z = zscores(logp, mu, sigma)
    z = z[np.isfinite(z)]
    if z.size < 4:
        return np.nan
    s = np.sort(z)
    n = max(2, int(np.ceil(k * z.size)))
    tail = s[:n]
    cut = int(np.floor(trim * tail.size))
    kept = tail[cut:] if cut < tail.size - 1 else tail
    return float(kept.mean())


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

def build_registry() -> dict:
    """name -> (callable, kwargs). Every entry is scored from the same cache."""
    reg: dict[str, tuple] = {
        # published baselines
        "loss": (loss_score, {}),
        "zlib": (zlib_score, {}),
    }
    for k in (0.1, 0.2, 0.3, 0.4, 0.5):
        reg[f"mink_{k}"] = (mink, {"k": k})
        reg[f"minkpp_{k}"] = (minkpp, {"k": k})

    # H1 sliding window
    for w in (16, 32, 64):
        for pool in ("max", "p90"):
            reg[f"h1_w{w}_{pool}"] = (h1_window_max, {"w": w, "pool": pool})
    # H6 fixed chunk count
    for K in (2, 3, 4, 6):
        reg[f"h6_K{K}_max"] = (h6_chunk_vote, {"K": K, "pool": "max"})
    reg["h6_K3_mean"] = (h6_chunk_vote, {"K": 3, "pool": "mean"})
    # H2 fixed count
    for m in (5, 8, 12, 20):
        reg[f"h2_m{m}"] = (h2_fixed_m, {"m": m})
    # H3 length normalisation
    reg["h3_lennorm"] = (h3_length_norm, {})
    # H4 magnitude weighting
    for p in (0.5, 1.0, 2.0):
        reg[f"h4_pow{p}"] = (h4_magnitude_weighted, {"power": p})
    # H5 order statistic
    for q in (0.02, 0.05, 0.1, 0.2, 0.3):
        reg[f"h5_q{q}"] = (h5_order_statistic, {"q": q})
    # H8 position
    for s in (4, 8, 16):
        reg[f"h8_skip{s}"] = (h8_position_weighted, {"skip": s, "mode": "skip"})
    reg["h8_ramp8"] = (h8_position_weighted, {"skip": 8, "mode": "ramp"})
    # H9 two sided
    for b in (0.25, 0.5, 0.75, 1.0):
        reg[f"h9_beta{b}"] = (h9_two_sided, {"beta": b})
    # H10 prefix
    for f in (0.25, 0.5, 0.75):
        reg[f"h10_frac{f}"] = (h10_prefix, {"frac": f})
    # new variants
    reg["n1_kens"] = (n1_k_ensemble, {})
    reg["n2_adaptk"] = (n2_length_adaptive_k, {})
    reg["n3_rank"] = (n3_rank_normalised, {})
    for t in (0.1, 0.25):
        reg[f"n4_trim{t}"] = (n4_trimmed_tail, {"trim": t})
    return reg


BASELINE = "minkpp_0.2"

# Which hypothesis each variant family belongs to, for the report tables.
FAMILY = {
    "loss": "baseline", "zlib": "baseline", "mink": "baseline", "minkpp": "baseline",
    "h1": "H1 sliding window", "h2": "H2 fixed count", "h3": "H3 length norm",
    "h4": "H4 magnitude weight", "h5": "H5 order statistic",
    "h6": "H6 chunk and vote", "h8": "H8 position weight",
    "h9": "H9 two-sided", "h10": "H10 prefix / early stop",
    "n1": "N1 k-ensemble", "n2": "N2 length-adaptive k",
    "n3": "N3 rank-normalised", "n4": "N4 trimmed tail",
}


def family_of(name: str) -> str:
    head = name.split("_")[0]
    return FAMILY.get(head, "other")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC by the rank formula, NaN scores dropped. Ties handled by average
    rank, which matches sklearn."""
    ok = np.isfinite(scores)
    y, s = np.asarray(labels)[ok], np.asarray(scores)[ok]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, float)
    ranks[order] = np.arange(1, s.size + 1)
    # average ranks within ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr: float = 0.05) -> float:
    ok = np.isfinite(scores)
    y, s = np.asarray(labels)[ok], np.asarray(scores)[ok]
    neg = np.sort(s[y == 0])
    if neg.size == 0 or (y == 1).sum() == 0:
        return np.nan
    thr = neg[max(0, int(np.ceil((1 - fpr) * neg.size)) - 1)]
    return float((s[y == 1] > thr).mean())


def paired_bootstrap(labels, score_a, score_b, n_boot=2000, seed=0):
    """Paired bootstrap over items of AUROC(a) - AUROC(b).

    The resampling indices are shared between the two methods, which is what
    makes the comparison paired and buys a materially narrower interval than
    two independent intervals would.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(labels)
    n = y.size
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            diffs[i] = np.nan
            continue
        diffs[i] = auroc(y[idx], np.asarray(score_a)[idx]) - \
                   auroc(y[idx], np.asarray(score_b)[idx])
    d = diffs[np.isfinite(diffs)]
    if d.size == 0:
        return np.nan, np.nan, np.nan
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi), float((d > 0).mean())


# --------------------------------------------------------------------------- #
# self-check — refuses to let the study run on a broken implementation
# --------------------------------------------------------------------------- #


def self_check(verbose: bool = True) -> bool:
    rng = np.random.default_rng(1)
    n = 120
    logp = -rng.exponential(2.0, n)
    mu = -rng.uniform(3, 6, n)
    sigma = rng.uniform(0.5, 2.0, n)
    ok = True

    def chk(name, cond):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # 1. z must not collapse onto logp -- the one bug that silently turns
    #    Min-K%++ back into Min-K% while still looking plausible.
    z = zscores(logp, mu, sigma)
    chk("z differs from logp", not np.allclose(z, logp))
    chk("z uses sigma", not np.allclose(z, logp - mu))

    # 2. a window at least as wide as the sequence reproduces the baseline exactly
    base = minkpp(logp, mu, sigma, k=0.2)
    chk("window w>=n == baseline",
        np.isclose(h1_window_max(logp, mu, sigma, k=0.2, w=n + 10), base))

    # 3. one chunk reproduces the baseline exactly
    chk("chunk K=1 == baseline",
        np.isclose(h6_chunk_vote(logp, mu, sigma, k=0.2, K=1), base))

    # 4. prefix with frac=1 reproduces the baseline
    chk("prefix frac=1 == baseline",
        np.isclose(h10_prefix(logp, mu, sigma, k=0.2, frac=1.0), base))

    # 5. magnitude weighting with power 0 reproduces the baseline
    chk("magnitude power=0 == baseline",
        np.isclose(h4_magnitude_weighted(logp, mu, sigma, k=0.2, power=0.0), base))

    # 6. two-sided with beta=0 reproduces the baseline
    chk("two-sided beta=0 == baseline",
        np.isclose(h9_two_sided(logp, mu, sigma, k=0.2, beta=0.0), base))

    # 7. k=1 selects everything, so the statistic is the plain mean of z
    chk("minkpp k=1 == mean(z)",
        np.isclose(minkpp(logp, mu, sigma, k=1.0), np.nanmean(z)))

    # 8. AUROC sanity: perfect separation = 1, reversed = 0, constant = 0.5
    y = np.array([0] * 50 + [1] * 50)
    chk("auroc perfect", np.isclose(auroc(y, np.arange(100.0)), 1.0))
    chk("auroc reversed", np.isclose(auroc(y, -np.arange(100.0)), 0.0))
    chk("auroc constant", np.isclose(auroc(y, np.ones(100)), 0.5))

    # 9. sigma = 0 must not produce inf
    chk("sigma=0 -> nan not inf",
        not np.isinf(zscores(np.array([-1.0]), np.array([-2.0]),
                             np.array([0.0]))).any())
    if verbose:
        print(f"\nself-check: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_check() else 1)

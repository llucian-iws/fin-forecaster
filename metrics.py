#!/usr/bin/env python3
"""
Proper scoring rules and economic evaluation for the forecaster's backtest.

Pure functions only (numpy / pandas / scipy.stats) — no network, no file I/O —
so this module is standalone-testable and safe to import anywhere. It extends
backtest.py beyond directional hit-rate / MAE / coverage with:

  - PROPER scoring rules (CRPS, pinball) that reward calibration AND sharpness
    in a single number, so a probabilistic forecast can be ranked honestly.
  - An ECONOMIC evaluation (vol-targeted long/short P&L) that turns a drift
    forecast into a trade and reports the risk/return it would have produced.
  - STATISTICAL inference for the backtest report: a binomial CI for hit-rate /
    coverage proportions and a Diebold-Mariano loss-differential test (with the
    Harvey-Leybourne-Newbold small-sample correction), so metric differences can
    be read against their sampling noise instead of taken at face value.

Everything is NaN-/empty-safe: degenerate inputs return NaN (scoring rules) or a
dict of NaNs (strategy_eval) rather than raising, mirroring volatility.py's
graceful-degradation style.
"""

import numpy as np
from scipy.stats import norm, t as student_t

_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)


def crps_ensemble(samples, y):
    """Empirical CRPS of a Monte-Carlo sample array against scalar outcome `y`.

    Uses the energy form of the Continuous Ranked Probability Score:

        CRPS = E|X - y| - 0.5 * E|X - X'|

    where X, X' are i.i.d. draws from the forecast distribution (here the
    ensemble `samples`). Lower is better; CRPS rewards calibration AND sharpness
    in a single number and reduces to MAE for a deterministic forecast.

    The first term is mean(|samples - y|). The second term is computed in
    O(n log n) (sort + vectorized weights) rather than the naive O(n^2) pairwise
    sum: for samples sorted ascending x_(1..n) (1-based),

        E|X - X'| = (2 / n^2) * sum_{i=1..n} (2*i - n - 1) * x_(i).

    Returns NaN if `y` is NaN or no finite samples remain.
    """
    s = np.asarray(samples, dtype=float).ravel()
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0 or not np.isfinite(y):
        return float('nan')

    mad = np.mean(np.abs(s - y))                 # E|X - y|
    xs = np.sort(s)                              # x_(1..n) ascending
    i = np.arange(1, n + 1)                      # 1-based index
    coef = 2.0 * i - n - 1                        # (2*i - n - 1)
    e_xx = (2.0 / n ** 2) * np.sum(coef * xs)    # E|X - X'|, O(n log n)
    return float(mad - 0.5 * e_xx)


def crps_gaussian(mu, sigma, y):
    """Closed-form CRPS of a Gaussian forecast N(mu, sigma) against outcome `y`.

        CRPS = sigma * ( z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) )

    with z = (y - mu) / sigma and Phi/phi the standard-normal CDF/PDF. This is
    the analytic limit of crps_ensemble as the sample size grows, so the two are
    cross-checked against each other in the smoke test.

    Degenerate spread (sigma <= 0, or non-finite mu/sigma/y) collapses the
    forecast to a point mass at mu, for which CRPS is simply |y - mu|.
    """
    if not (np.isfinite(mu) and np.isfinite(sigma) and np.isfinite(y)):
        return float('nan')
    if sigma <= 0:
        return float(abs(y - mu))
    z = (y - mu) / sigma
    return float(sigma * (z * (2.0 * norm.cdf(z) - 1.0)
                          + 2.0 * norm.pdf(z) - _INV_SQRT_PI))


def pinball_loss(quantile_preds, y):
    """Mean pinball (quantile) loss of a set of quantile forecasts against `y`.

    `quantile_preds` is a dict {q: predicted_value}, q in (0, 1). For each
    quantile the pinball loss is

        (y - pred) * q          if y >= pred
        (pred - y) * (1 - q)    otherwise

    averaged over the provided quantiles. For the single median (q=0.5) this is
    0.5*|y - pred|. Lower is better; it is the proper scoring rule for quantile
    forecasts and, summed over a dense quantile grid, approximates the CRPS.

    Returns NaN if `y` is NaN or no quantile has a finite prediction.
    """
    if not np.isfinite(y) or not quantile_preds:
        return float('nan')
    losses = []
    for q, pred in quantile_preds.items():
        if not (np.isfinite(q) and np.isfinite(pred)):
            continue
        diff = y - pred
        losses.append(diff * q if diff >= 0 else -diff * (1.0 - q))
    if not losses:
        return float('nan')
    return float(np.mean(losses))


def interval_coverage(lower, upper, y):
    """Indicator (1/0) that outcome `y` falls within the closed band [lower, upper].

    Returns 0 if any of lower/upper/y is non-finite (the band can't be verified).
    """
    if not (np.isfinite(lower) and np.isfinite(upper) and np.isfinite(y)):
        return 0
    return int(lower <= y <= upper)


def coverage_rate(lowers, uppers, ys):
    """Empirical coverage: mean of the interval_coverage indicator over arrays.

    For a well-calibrated nominal-(1-alpha) band this should sit near (1-alpha).
    Returns NaN if there are no rows where all three of lower/upper/y are finite.
    """
    lowers = np.asarray(lowers, dtype=float).ravel()
    uppers = np.asarray(uppers, dtype=float).ravel()
    ys = np.asarray(ys, dtype=float).ravel()
    if not (lowers.size == uppers.size == ys.size) or lowers.size == 0:
        return float('nan')
    mask = np.isfinite(lowers) & np.isfinite(uppers) & np.isfinite(ys)
    if not mask.any():
        return float('nan')
    inside = (lowers[mask] <= ys[mask]) & (ys[mask] <= uppers[mask])
    return float(np.mean(inside))


def binomial_ci(k, n, level=0.90):
    """Wilson score confidence interval for a binomial proportion k/n.

    Used to put sampling error around hit-rates and empirical coverage: at the
    backtest's typical n=150 folds, a proportion near 0.5 carries a ~+/-7pp
    90% interval — differences inside that window are noise, not edge.

    `k` is the SUCCESS COUNT (not the rate). Returns (lo, hi) clipped to
    [0, 1], or (NaN, NaN) on degenerate input (n <= 0, non-finite, k outside
    [0, n]).
    """
    if k is None or n is None:
        return (float('nan'), float('nan'))
    k, n = float(k), float(n)
    if not (np.isfinite(k) and np.isfinite(n)) or n <= 0 or k < 0 or k > n:
        return (float('nan'), float('nan'))
    p = k / n
    z = norm.ppf(0.5 + level / 2.0)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def dm_test(loss_a, loss_b, h=1):
    """Diebold-Mariano equal-predictive-accuracy test, HLN-corrected.

    Tests H0: E[loss_a - loss_b] = 0 on per-fold loss arrays (e.g. absolute
    point errors, or per-fold CRPS). The loss differential's long-run variance
    uses a rectangular kernel over lags < h (forecasts h steps ahead overlap
    h-1 neighbors); `h` is in FOLD units, so non-overlapping daily folds use
    h=1. The statistic is scaled by the Harvey-Leybourne-Newbold small-sample
    factor and referred to a t(n-1) distribution.

    Returns (stat, p_value), two-sided. stat > 0 means loss_a is WORSE
    (higher loss) than loss_b. Degenerate input — fewer than 10 aligned finite
    pairs, an identical-loss differential, or a non-positive long-run variance
    — returns (NaN, NaN) rather than fabricating significance.
    """
    a = np.asarray(loss_a, dtype=float).ravel()
    b = np.asarray(loss_b, dtype=float).ravel()
    if a.size != b.size:
        return (float('nan'), float('nan'))
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    n = d.size
    if n < 10:
        return (float('nan'), float('nan'))

    h = max(1, int(h))
    dbar = float(d.mean())
    dc = d - dbar
    lrv = float(np.mean(dc * dc))                  # gamma_0
    for lag in range(1, min(h, n)):
        lrv += 2.0 * float(np.mean(dc[lag:] * dc[:-lag]))
    if not np.isfinite(lrv) or lrv <= 0.0:
        return (float('nan'), float('nan'))

    dm = dbar / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n)
    stat = float(hln * dm)
    p = float(2.0 * student_t.sf(abs(stat), df=n - 1))
    return (stat, p)


def strategy_eval(pred_drifts, realized_log_returns, vols, fee_bps=5.0,
                  ann_factor=None):
    """Vol-targeted long/short backtest turning drift forecasts into a P&L.

    A deliberately simple economic evaluation: size each position by the
    forecast drift scaled by its volatility, cap leverage to [-1, 1], and charge
    a turnover fee on every change in position.

        position_t = clip(pred_drift_t / vol_t, -1, +1)
        pnl_t      = position_t * realized_log_return_t
                     - |position_t - position_{t-1}| * fee_bps / 1e4

    The first position has no prior, so its turnover is measured against 0 (i.e.
    entering the book costs a fee). Rows where any of drift/return/vol is
    non-finite, or vol <= 0, are dropped before sizing.

    Returns a dict:
        sharpe            mean(pnl)/std(pnl) * sqrt(ann_factor)
        total_return      sum of per-step P&L (log-return space; additive)
        max_drawdown      largest peak-to-trough drop of the cumulative-sum
                          equity curve (<= 0; 0.0 if it never draws down)
        hit_rate          fraction of nonzero-position steps with pnl > 0
        n_traded          number of nonzero-position steps (hit_rate's sample
                          size, for a binomial CI on the trade hit-rate)
        n_flips           number of steps whose position sign changed
        avg_abs_position  mean(|position|), i.e. realized leverage usage

    Annualization convention: `ann_factor` is the number of periods per year for
    the Sharpe scaling (e.g. ~8760 for hourly, 252 for daily). When None it
    DEFAULTS to len(pnl), which annualizes "per backtest length" — convenient
    for relative comparison across runs of equal length, but NOT a calendar-year
    Sharpe unless the sample happens to span one period-unit. Pass an explicit
    factor for a calendar-correct figure. Returns all-NaN on empty/degenerate
    input, and sharpe=NaN when P&L has zero variance.
    """
    d = np.asarray(pred_drifts, dtype=float).ravel()
    r = np.asarray(realized_log_returns, dtype=float).ravel()
    v = np.asarray(vols, dtype=float).ravel()

    nan_result = {
        'sharpe': float('nan'),
        'total_return': float('nan'),
        'max_drawdown': float('nan'),
        'hit_rate': float('nan'),
        'n_traded': 0,
        'n_flips': 0,
        'avg_abs_position': float('nan'),
    }
    if not (d.size == r.size == v.size) or d.size == 0:
        return nan_result

    mask = np.isfinite(d) & np.isfinite(r) & np.isfinite(v) & (v > 0)
    d, r, v = d[mask], r[mask], v[mask]
    if d.size == 0:
        return nan_result

    position = np.clip(d / v, -1.0, 1.0)
    prev = np.concatenate(([0.0], position[:-1]))     # entering the book costs a fee
    turnover = np.abs(position - prev)
    pnl = position * r - turnover * (fee_bps / 1e4)

    if ann_factor is None:
        ann_factor = pnl.size

    std = pnl.std()
    sharpe = float(pnl.mean() / std * np.sqrt(ann_factor)) if std > 0 else float('nan')

    equity = np.cumsum(pnl)                            # additive log-return equity
    running_max = np.maximum.accumulate(equity)
    max_drawdown = float(np.min(equity - running_max))  # <= 0

    traded = position != 0
    hit_rate = float(np.mean(pnl[traded] > 0)) if traded.any() else float('nan')
    n_flips = int(np.sum(np.sign(position[1:]) != np.sign(position[:-1]))) if position.size > 1 else 0

    return {
        'sharpe': sharpe,
        'total_return': float(pnl.sum()),
        'max_drawdown': max_drawdown,
        'hit_rate': hit_rate,
        'n_traded': int(traded.sum()),
        'n_flips': n_flips,
        'avg_abs_position': float(np.mean(np.abs(position))),
    }


if __name__ == '__main__':
    # Synthetic, no-network smoke test that cross-checks correctness.
    rng = np.random.default_rng(0)

    # (a) Ensemble CRPS should converge to the closed-form Gaussian CRPS.
    mu, sigma = 1.5, 2.0
    samples = rng.normal(mu, sigma, size=400_000)
    print("CRPS cross-check (ensemble vs Gaussian closed-form):")
    max_err_pct = 0.0
    for y in [mu, mu + sigma, mu - 1.5 * sigma, mu + 3.0 * sigma]:
        emp = crps_ensemble(samples, y)
        exact = crps_gaussian(mu, sigma, y)
        err_pct = abs(emp - exact) / exact * 100
        max_err_pct = max(max_err_pct, err_pct)
        print(f"  y={y:+6.2f}  ensemble={emp:.5f}  gaussian={exact:.5f}  err={err_pct:.3f}%")
        assert err_pct < 2.0, f"CRPS mismatch at y={y}: {err_pct:.3f}%"
    print(f"  -> max CRPS cross-check error: {max_err_pct:.3f}% (tolerance 2%)\n")

    # (b) Median pinball loss == 0.5 * |y - median|.
    median = 10.0
    for y in [7.0, 10.0, 14.5]:
        pb = pinball_loss({0.5: median}, y)
        expected = 0.5 * abs(y - median)
        assert abs(pb - expected) < 1e-12, f"pinball mismatch: {pb} vs {expected}"
    print(f"Pinball median check OK: pinball({{0.5:{median}}}, 14.5) "
          f"= {pinball_loss({0.5: median}, 14.5):.4f} == 0.5*|14.5-10| = {0.5*4.5:.4f}\n")

    # Multi-quantile pinball sanity (just exercises the averaging path).
    qpreds = {0.1: 8.0, 0.5: 10.0, 0.9: 12.5}
    print(f"Pinball multi-quantile {qpreds} vs y=11.0 -> {pinball_loss(qpreds, 11.0):.4f}\n")

    # (c) Coverage of a known-normal 90% band should land near nominal 0.90.
    n = 200_000
    truth_mu, truth_sigma = 0.0, 1.0
    ys = rng.normal(truth_mu, truth_sigma, size=n)
    half = norm.ppf(0.95) * truth_sigma            # two-sided 90%
    lowers = np.full(n, truth_mu - half)
    uppers = np.full(n, truth_mu + half)
    cov = coverage_rate(lowers, uppers, ys)
    print(f"Coverage check: empirical={cov:.4f} vs nominal=0.9000 "
          f"(single point: {interval_coverage(lowers[0], uppers[0], ys[0])})")
    assert abs(cov - 0.90) < 0.01, f"coverage off: {cov}"
    print()

    # (d) binomial_ci: centered on p, contains the truth, degenerate -> NaN.
    lo, hi = binomial_ci(75, 150, level=0.90)
    print(f"binomial_ci(75, 150) 90% -> [{lo:.4f}, {hi:.4f}]")
    assert lo < 0.5 < hi and 0.42 < lo < 0.46 and 0.54 < hi < 0.58, (lo, hi)
    lo60, hi60 = binomial_ci(12, 20, level=0.95)         # the 60%-over-20-folds trap
    print(f"binomial_ci(12, 20)  95% -> [{lo60:.4f}, {hi60:.4f}]  (60% on 20 folds)")
    assert lo60 < 0.40 and hi60 > 0.78, (lo60, hi60)
    assert all(np.isnan(v) for v in binomial_ci(5, 0))
    assert all(np.isnan(v) for v in binomial_ci(10, 5))   # k > n
    print()

    # (e) dm_test: detects a real loss gap, stays quiet on equal losses,
    #     NaNs out on a degenerate (identical) differential.
    base_losses = np.abs(rng.normal(0.0, 1.0, size=150))
    stat, p = dm_test(base_losses + 0.5, base_losses + rng.normal(0, 0.05, 150), h=1)
    print(f"dm_test(worse vs better)  stat={stat:+.2f}  p={p:.2e}")
    assert stat > 0 and p < 1e-6, (stat, p)
    la = np.abs(rng.normal(0.0, 1.0, size=150))
    lb = np.abs(rng.normal(0.0, 1.0, size=150))
    stat_eq, p_eq = dm_test(la, lb, h=1)
    print(f"dm_test(equal dists)      stat={stat_eq:+.2f}  p={p_eq:.3f}")
    assert np.isfinite(p_eq) and p_eq > 0.01, (stat_eq, p_eq)
    stat_d, p_d = dm_test(la, la, h=1)                   # identical -> degenerate
    assert np.isnan(stat_d) and np.isnan(p_d)
    stat_h, p_h = dm_test(base_losses + 0.5,
                          base_losses + rng.normal(0, 0.05, 150), h=3)  # h>1 path
    assert np.isfinite(p_h) and p_h < 1e-6
    print(f"dm_test(h=3 overlap)      stat={stat_h:+.2f}  p={p_h:.2e}\n")

    # (f) Strategy backtest on synthetic, mildly predictable returns.
    n = 5_000
    vols = np.full(n, 0.01)
    true_drift = rng.normal(0.0, 0.002, size=n)
    noise = rng.normal(0.0, vols)
    realized = true_drift + noise                  # signal + noise
    pred_drifts = true_drift + rng.normal(0.0, 0.001, size=n)  # noisy but informative
    result = strategy_eval(pred_drifts, realized, vols, fee_bps=5.0)
    print("strategy_eval (synthetic, ann_factor=len(pnl)):")
    for k, val in result.items():
        print(f"  {k:18s} {val}")

    print("\nAll smoke-test assertions passed.")

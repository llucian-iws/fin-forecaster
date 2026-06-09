#!/usr/bin/env python3
"""
Data-driven post-processing for the forecaster's point/drift + scenario layer.

Motivation: a walk-forward backtest showed the 24h point/drift forecast does NOT
beat a random walk (lite engine ~47% directional hit-rate over 150 folds; CNN
~60% over 20 folds; both lose to persistence on MAE; the CNN carries a ~+0.88%
systematic bullish bias). The production scenario layer also used HARDCODED
probabilities (0.35/0.40/0.25) and shocks (+0.04/0/-0.06). This module replaces
those heuristics with principled, data-driven post-processing.

Pure functions only: numpy / pandas / scipy. No network, no file I/O, no global
state. Every function is defensive against NaN / empty / degenerate input.
"""

import numpy as np
import pandas as pd


def fit_shrinkage(model_drifts, realized_drifts):
    """Optimal shrinkage of the predicted drift toward ZERO (the random walk).

    Finds the alpha in [0, 1] minimizing MSE of (alpha * model_drift) vs the
    realized drift. The unconstrained least-squares solution is the closed form
    alpha = sum(m*r) / sum(m*m); we clamp it to [0, 1] so the result is a true
    shrinkage factor (0 = trust the random walk fully, 1 = trust the model).

    NaN-aligned pairs are dropped. Empty or degenerate input (sum(m*m) == 0)
    returns 0.0, i.e. full shrink to the random walk.
    """
    m = pd.Series(model_drifts, dtype=float).reset_index(drop=True)
    r = pd.Series(realized_drifts, dtype=float).reset_index(drop=True)
    mask = m.notna() & r.notna()
    m = m[mask].values
    r = r[mask].values
    if len(m) == 0:
        return 0.0

    denom = float(np.sum(m * m))
    if denom == 0.0 or not np.isfinite(denom):
        return 0.0
    num = float(np.sum(m * r))
    if not np.isfinite(num):
        return 0.0

    alpha = num / denom
    return float(np.clip(alpha, 0.0, 1.0))


def apply_shrinkage(drift, alpha):
    """Shrink a single drift toward zero by the factor alpha: alpha * drift."""
    if drift is None or alpha is None:
        return 0.0
    drift = float(drift)
    alpha = float(alpha)
    if not (np.isfinite(drift) and np.isfinite(alpha)):
        return 0.0
    return alpha * drift


def rolling_bias_correction(past_errors, window=30):
    """Trailing mean of the most recent `window` signed forecast errors.

    Errors are defined as (forecast - realized); the returned value is meant to
    be SUBTRACTED from the next forecast to remove systematic bias (e.g. the
    CNN's ~+0.88% bullish skew). With fewer than `window` observations, all
    available errors are used. Empty input returns 0.0 (no correction).
    """
    e = pd.Series(past_errors, dtype=float).dropna()
    if len(e) == 0:
        return 0.0
    recent = e.iloc[-int(window):] if window and window > 0 else e
    bias = float(recent.mean())
    return bias if np.isfinite(bias) else 0.0


def regime_scenarios(trans_matrix, current_state, state_labels, regime_mean_ret,
                     regime_vol, horizon_hours):
    """Build a data-driven scenario set from an HMM's next-step transitions.

    Replaces the hardcoded 0.35/0.40/0.25 probabilities and +0.04/0/-0.06 shocks.
    The next-step probabilities are read directly from the transition matrix row
    for the current state, trans_matrix[current_state] (which sums to 1). Each
    reachable state s contributes one scenario:
        prob        = trans_matrix[current_state, s]
        drift_per_hr = regime_mean_ret[s]
        vol_per_hr   = regime_vol[s]
        label        = state_labels[s]

    `horizon_hours` is accepted for interface symmetry with the caller; the
    per-hour rates returned here are compounded over the horizon downstream.

    Returns a list of dicts with keys {'label', 'prob', 'drift_per_hr',
    'vol_per_hr'}, one per reachable state, ordered by descending probability.
    Degenerate input (bad current_state, empty/NaN row) returns an empty list.
    """
    T = np.asarray(trans_matrix, dtype=float)
    if T.ndim != 2 or T.shape[0] == 0:
        return []
    n_states = T.shape[0]
    if current_state is None:
        return []
    current_state = int(current_state)
    if current_state < 0 or current_state >= n_states:
        return []

    row = T[current_state].astype(float)
    if row.size == 0 or not np.all(np.isfinite(row)):
        return []
    total = float(row.sum())
    if total <= 0.0 or not np.isfinite(total):
        return []
    probs = row / total  # renormalize defensively so probs sum to 1

    scenarios = []
    for s in range(probs.size):
        p = float(probs[s])
        if p <= 0.0:
            continue
        label = state_labels[s] if state_labels is not None and s < len(state_labels) else f"state_{s}"
        drift = float(regime_mean_ret[s]) if regime_mean_ret is not None and s < len(regime_mean_ret) else 0.0
        vol = float(regime_vol[s]) if regime_vol is not None and s < len(regime_vol) else 0.0
        scenarios.append({
            'label': label,
            'prob': p,
            'drift_per_hr': drift if np.isfinite(drift) else 0.0,
            'vol_per_hr': vol if np.isfinite(vol) else 0.0,
        })

    scenarios.sort(key=lambda d: d['prob'], reverse=True)
    return scenarios


def empirical_quantile_band(samples, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)):
    """Asymmetric prediction band straight from Monte-Carlo `samples`.

    Computes np.percentile for each requested quantile, so the band reflects the
    sampled distribution's skew and fat tails rather than a symmetric Gaussian.
    Quantiles may be given in [0, 1] or as percentages (0-100). NaNs are dropped;
    empty input maps every quantile to NaN.

    Returns a dict {q: value} keyed by the original quantile arguments.
    """
    s = pd.Series(samples, dtype=float).dropna().values
    out = {}
    if s.size == 0:
        for q in quantiles:
            out[q] = float('nan')
        return out

    for q in quantiles:
        pct = float(q) * 100.0 if 0.0 <= float(q) <= 1.0 else float(q)
        pct = float(np.clip(pct, 0.0, 100.0))
        out[q] = float(np.percentile(s, pct))
    return out


def ensemble_combine(preds, oos_mae):
    """Inverse-error weighted average of model predictions.

    Each model's weight is 1 / mae (lower out-of-sample error -> more weight).
    A single prediction returns its own value. Any model whose MAE is missing,
    non-finite, or <= 0 is treated as having no usable error signal; if NONE of
    the models has a usable MAE, the function falls back to an equal-weight mean.
    Models with non-finite predictions are skipped.

    Returns the combined point value, or NaN if no finite predictions exist.
    """
    if not preds:
        return float('nan')

    names, values, weights = [], [], []
    for name, val in preds.items():
        if val is None:
            continue
        v = float(val)
        if not np.isfinite(v):
            continue
        mae = oos_mae.get(name) if oos_mae else None
        w = None
        if mae is not None:
            mae = float(mae)
            if np.isfinite(mae) and mae > 0.0:
                w = 1.0 / mae
        names.append(name)
        values.append(v)
        weights.append(w)

    if not values:
        return float('nan')
    if len(values) == 1:
        return float(values[0])

    if all(w is None for w in weights) or any(w is None for w in weights):
        # No usable error signal for at least one model -> equal weights.
        return float(np.mean(values))

    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    wsum = float(weights.sum())
    if wsum <= 0.0 or not np.isfinite(wsum):
        return float(np.mean(values))
    return float(np.sum(weights * values) / wsum)


if __name__ == '__main__':
    rng = np.random.default_rng(42)

    # 1. fit_shrinkage: realized = 0.3 * model (+ small noise) -> alpha ~ 0.3.
    model = rng.normal(0.0, 0.01, size=500)
    realized = 0.3 * model + rng.normal(0.0, 1e-4, size=500)
    alpha = fit_shrinkage(model, realized)
    assert np.isfinite(alpha) and 0.0 <= alpha <= 1.0
    assert abs(alpha - 0.3) < 0.05, alpha
    assert fit_shrinkage([], []) == 0.0
    assert fit_shrinkage([0.0, 0.0], [1.0, 2.0]) == 0.0  # degenerate sum(m*m)==0
    assert fit_shrinkage([np.nan, 0.01], [np.nan, 0.003]) >= 0.0

    # 2. apply_shrinkage: scales linearly; non-finite -> 0.0.
    assert abs(apply_shrinkage(0.02, 0.5) - 0.01) < 1e-12
    assert apply_shrinkage(0.02, 0.0) == 0.0
    assert apply_shrinkage(np.nan, 0.5) == 0.0

    # 3. rolling_bias_correction: known +0.5 bias over the window.
    errors = np.concatenate([np.full(10, -2.0), np.full(30, 0.5)])
    bias = rolling_bias_correction(errors, window=30)
    assert np.isfinite(bias) and abs(bias - 0.5) < 1e-9, bias
    assert rolling_bias_correction([]) == 0.0
    assert np.isfinite(rolling_bias_correction([0.1, 0.2], window=30))  # short history

    # 4. regime_scenarios: 3x3 HMM, probs must sum to ~1.
    trans = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.4, 0.3],
        [0.1, 0.3, 0.6],
    ])
    labels = ['BULL', 'CHOP', 'BEAR']
    means = [0.0008, 0.0, -0.0009]
    vols = [0.004, 0.006, 0.011]
    scen = regime_scenarios(trans, current_state=0, state_labels=labels,
                            regime_mean_ret=means, regime_vol=vols, horizon_hours=24)
    assert len(scen) == 3
    psum = sum(d['prob'] for d in scen)
    assert abs(psum - 1.0) < 1e-9, psum
    assert all(np.isfinite(d['drift_per_hr']) and np.isfinite(d['vol_per_hr']) for d in scen)
    assert scen[0]['prob'] >= scen[-1]['prob']  # sorted desc
    assert regime_scenarios(trans, current_state=9, state_labels=labels,
                            regime_mean_ret=means, regime_vol=vols, horizon_hours=24) == []

    # 5. empirical_quantile_band: monotonic non-decreasing in q.
    samples = rng.standard_t(df=3, size=20000) * 0.01  # fat-tailed
    band = empirical_quantile_band(samples)
    qs = sorted(band.keys())
    vals = [band[q] for q in qs]
    assert all(np.isfinite(v) for v in vals)
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)), vals
    nan_band = empirical_quantile_band([])
    assert all(np.isnan(v) for v in nan_band.values())

    # 6. ensemble_combine: lower MAE pulls the result toward that model.
    preds = {'cnn': 1.0, 'lite': 3.0}
    mae = {'cnn': 1.0, 'lite': 3.0}  # weights 1.0 vs 0.333
    combo = ensemble_combine(preds, mae)
    assert np.isfinite(combo)
    assert 1.0 < combo < 2.0, combo  # closer to the lower-error 'cnn'
    assert ensemble_combine({'only': 2.5}, {}) == 2.5
    eq = ensemble_combine({'a': 1.0, 'b': 3.0}, {'a': 0.0, 'b': np.nan})  # bad MAEs
    assert abs(eq - 2.0) < 1e-12, eq
    assert np.isnan(ensemble_combine({}, {}))

    print("forecast_post smoke test: all assertions passed")
    print(f"  fit_shrinkage(realized=0.3*model)      = {alpha:.4f}  (expected ~0.30)")
    print(f"  apply_shrinkage(0.02, 0.5)             = {apply_shrinkage(0.02, 0.5):.4f}")
    print(f"  rolling_bias_correction(+0.5 window)   = {bias:.4f}")
    print(f"  regime_scenarios prob sum              = {psum:.6f}")
    print(f"  regime_scenarios[0]                    = {scen[0]}")
    print(f"  empirical_quantile_band                = "
          f"{{ {', '.join(f'{q}: {band[q]:+.4f}' for q in qs)} }}")
    print(f"  ensemble_combine(1.0@mae1, 3.0@mae3)   = {combo:.4f}  (expected 1<c<2)")

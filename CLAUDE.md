# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

# Project: fin-forecaster

Quant forecasting stack for **price** and **volatility**, crypto + stocks. See
`README.md` for full usage.

## Layout
- `btc_forecast.py` — full stack: CNN-LSTM + HMM + MC-dropout + conformal +
  10k-path scenario Monte Carlo, plus the `--volatility-only` branch. Includes
  the funding feature, drift **shrinkage** toward the random walk (`SHRINK_ALPHA`),
  **HMM data-driven regime scenarios** (replaces the old hardcoded 35/40/25), an
  **asymmetric quantile band**, and the **forward-vol shock** (GARCH(1,1) ⊕ DVOL).
- `volatility.py` — asset-agnostic vol math (numpy/pandas/requests only, **no
  TensorFlow**): Deribit DVOL fetch, realized vol, EWMA, GARCH(1,1), report.
- `exogenous.py` — keyless Binance funding-rate / OI / basis fetchers +
  look-ahead-free funding feature builder. (Funding history is deep; OI ~30d
  only; basis is a live snapshot — so only **funding** is backtestable.)
- `forecast_post.py` — pure post-processing: drift shrinkage, bias correction,
  HMM regime scenarios, empirical quantile bands, inverse-error ensemble.
- `metrics.py` — CRPS (ensemble + Gaussian), pinball, coverage, economic
  (vol-targeted long/short) strategy eval.
- `btc_forecast_lite.py` — Gradient Boosting fallback (no TensorFlow).
- `backtest.py` — walk-forward backtest: compares model variants (base /
  +funding / vol-standardized / ensemble / shrunk) vs persistence on hit-rate &
  MAE, scores shock variants with coverage + **CRPS + pinball**, and runs an
  economic eval. Engines `lite` (GB, fast) and `cnnlstm` (slow).
- `Dockerfile` / `docker-compose.yml` — Python 3.11 + TensorFlow runtime.

## How it runs
- **The price path needs Docker** (TensorFlow 2.13+ / Python 3.11). Build once
  with `docker build -t fin-forecaster:latest .`, run with
  `-v "$(pwd)/results:/app/results"` so outputs reach the host.
- `OUTPUT_DIR = /app/results` is hard-coded — scripts only run cleanly inside
  the container (the path is read-only on the host).
- TensorFlow is imported **lazily** in Section 3, so `--volatility-only` never
  loads it.
- Run exactly **one container at a time**; clean up with `docker ps`/`docker kill`.

## Flags / env
`--target-date`/`TARGET_DATE` (`YYYY-MM-DD` or `next-<weekday>`),
`--target-hour`/`TARGET_HOUR`, `--runs`/`RUNS` (average N retrains, distinct
seeds), `MODEL_EPOCHS` (default 5), `SHRINK_ALPHA` (drift shrinkage toward RW,
default 0.3), `--volatility-only`/`VOLATILITY_ONLY`, `--asset`/`ASSET`
(`crypto|stock`), `--ticker`/`TICKER`.
`backtest.py`: `--engine lite|cnnlstm`, `--max-folds`, `--step`, `--horizon`.

## Gotchas (learned the hard way)
- **`--runs` must vary the seed per run** — a fixed seed makes averaging a no-op.
- **Stdout is buffered when piped**; the Dockerfile sets `PYTHONUNBUFFERED=1` so
  progress past the TF training bar is visible.
- **MC dropout is batched** (replicate input → one forward pass), not N
  sequential eager calls — same distribution, ~200× faster.
- **Single-name option-chain IV only works during US market hours** (Yahoo IV is
  stale after-hours); it's gated via pytz and degrades to realized-vol-only.
- The model has a **24h horizon**; longer targets compound the hourly rate.
- **Neither engine beats a random-walk on the 24h point forecast.** In
  `backtest.py`, both lite and cnnlstm lose to a persistence baseline on MAE.
  Directional hit-rate is sample-dependent and NOT a reliable edge (lite 47%
  over 150 daily folds; cnnlstm 60% over only 20 weekly folds — different fold
  sets, so not comparable, and 20 folds is noisy). Don't sell the point
  forecast as predictive — its value is the scenario distribution, not direction.
- **The forward-vol integration is what the backtest validates**, and both
  engines agree. Pre-integration realized-vol bands under-cover (lite 0.86,
  cnnlstm 0.75 at nominal 0.90 — overconfident); the GARCH+DVOL forward shock
  is best-calibrated (lite 0.93, cnnlstm 0.90 exact). Re-run `backtest.py`
  before claiming any future calibration change.
- **Cross-engine comparisons need the SAME folds.** Lite defaults to daily
  steps (150 folds); cnnlstm was run with `--step 168` (20 weekly folds). Their
  persistence baselines differ because the samples differ — match `--step`/
  `--max-folds` before comparing engines head-to-head.
- **Funding is the one signal with real directional content** — it lifts the
  24h hit-rate 47%→52% in the backtest (still no MAE edge). That's why the live
  drift is shrunk toward the random walk (`SHRINK_ALPHA`, backtest α→0) and the
  point forecast lands ≈spot. The product's value is the calibrated asymmetric
  band + a slight funding tilt, NOT the point. **Bias-correction was measured to
  HURT** at 24h (noise, not stable bias) — it's in `forecast_post.py` but is
  deliberately NOT applied in production.
- **Only funding is backtestable among the exogenous feeds.** Binance OI history
  is ~30d and basis is a live snapshot, so they're live-only; don't try to use
  them as walk-forward features.

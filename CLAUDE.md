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
  **HMM data-driven regime scenarios** (illustrative per-regime views only), and
  the **forward-vol shock**: **GK-HAR-RV ⊕ DVOL** (GARCH fallback when the HAR
  fit is degenerate; realized vol_24h if the whole vol model fails). The
  report's **headline 90% band is the validated single forward-vol shock at
  the shrunk drift** — the regime-composite aggregate was removed after
  failing validation on every backtest run.
- `volatility.py` — asset-agnostic vol math (numpy/pandas/requests only, **no
  TensorFlow**): Deribit DVOL fetch, realized vol, EWMA, GARCH(1,1), **HAR-RV**
  (`har_rv_forecast`, same None-fallback contract as GARCH), **Garman-Klass** /
  **bipower** per-bar realized measures (feed HAR via `rv=`), report. Pure
  functions have a no-network `__main__` smoke test.
- `exogenous.py` — keyless Binance funding-rate / OI / basis fetchers +
  look-ahead-free funding feature builder. (Funding history is deep; the REST
  endpoints give only ~30d of OI and a live basis snapshot, but
  **data.binance.vision bulk archives** carry 5-min OI / taker / premium back
  to 2021 — so deep history IS fetchable for backtesting; nothing is wired in
  yet because the directional gates were refuted on statistical power.)
- `forecast_post.py` — pure post-processing: drift shrinkage, bias correction,
  HMM regime scenarios, empirical quantile bands, **vincentization**
  (probability-weighted quantile averaging), inverse-error ensemble.
- `metrics.py` — CRPS (ensemble + Gaussian), pinball, coverage, **binomial CI**
  (Wilson), **Diebold-Mariano test** (HLN-corrected), economic (vol-targeted
  long/short) strategy eval.
- `btc_forecast_lite.py` — Gradient Boosting fallback (no TensorFlow).
- `backtest.py` — walk-forward backtest: compares model variants (base /
  +funding / vol-standardized / ensemble / shrunk) vs persistence on hit-rate &
  MAE, scores band variants (realized / GARCH / GARCH+DVOL / HAR / HAR+DVOL /
  GK-HAR+DVOL; the refuted regime-composite + vincentized pair only behind
  `--composite`) with coverage + **CRPS + pinball** under **common random
  numbers**, and prints the rigor layer: binomial CIs, HLN-DM p-values vs
  persistence and vs the dvol incumbent, a per-side **fee sweep** (0/5/10/20
  bps), and a trial counter. Engines `lite` (GB, fast) and `cnnlstm` (slow).
- `Dockerfile` / `docker-compose.yml` — Python 3.11 + TensorFlow runtime.

## How it runs
- **The price path needs Docker** (TensorFlow 2.13+ / Python 3.11). Build once
  with `docker build -t fin-forecaster:latest .`, run with
  `-v "$(pwd)/results:/app/results"` so outputs reach the host.
- `OUTPUT_DIR = /app/results` is hard-coded in `btc_forecast.py` (Docker-only),
  but `backtest.py` honors the `OUTPUT_DIR` env var — the lite engine runs on
  the host with `OUTPUT_DIR=./results python3 backtest.py --engine lite` if
  numpy/pandas/scipy/sklearn/yfinance/hmmlearn are installed.
- The lite 150-fold backtest takes **~23 min** on this Mac since the
  regime-composite variant added a per-fold HMM fit (was a few minutes).
  Host stdout is block-buffered when redirected (progress lines appear late);
  the Docker image sets `PYTHONUNBUFFERED=1`.
- TensorFlow is imported **lazily** in Section 3, so `--volatility-only` never
  loads it.
- Run exactly **one container at a time**; clean up with `docker ps`/`docker kill`.

## Flags / env
`--target-date`/`TARGET_DATE` (`YYYY-MM-DD` or `next-<weekday>`),
`--target-hour`/`TARGET_HOUR`, `--runs`/`RUNS` (average N retrains, distinct
seeds), `MODEL_EPOCHS` (default 5), `SHRINK_ALPHA` (drift shrinkage toward RW,
default 0.3), `--volatility-only`/`VOLATILITY_ONLY`, `--asset`/`ASSET`
(`crypto|stock`), `--ticker`/`TICKER`.
`backtest.py`: `--engine lite|cnnlstm`, `--max-folds`, `--step`, `--horizon`,
`--composite` (re-score the refuted regime-composite bands; ~3x slower).

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
  `backtest.py`, both lite and cnnlstm lose to a persistence baseline on MAE —
  now formally tested: HLN-DM p ≥ 0.10 for every point variant (lite v3, 150
  folds, 2026-06). Directional hit-rate is sample-dependent and NOT a reliable
  edge (lite 47% over 150 daily folds; cnnlstm 60% over only 20 weekly folds —
  different fold sets, so not comparable, and 20 folds is noisy: 95% CI
  [39%, 78%]). Don't sell the point forecast as predictive — its value is the
  calibrated distribution, not direction.
- **The GARCH+DVOL band's validated claims (lite v4, 300 folds): coverage at
  nominal plus the project's FIRST significant sharpness win.** Coverage 0.917
  [CI 0.887, 0.939] (cnn matched folds: 0.900 exact). Raw realized-vol bands
  now PROVABLY under-cover: 0.830 [0.791, 0.863] — the CI excludes 0.90 at
  300 folds (150 folds couldn't resolve this). And blending DVOL into GARCH
  improves CRPS ~0.9% with HLN-DM p=0.001 under common random numbers — at
  150 folds this looked like noise (p=0.17); at 300 it's established. Claim
  exactly that and no more; the report prints CIs and DM p-values to hold the
  line, and comparisons are only valid within one run's fold set.
- **The regime-composite band FAILED its first validation** (it had never been
  scored before 2026-06). Mirrored production construction (per-fold HMM,
  regime drifts, shared GARCH+DVOL sigma, mixture-sampled): coverage 0.960 at
  ±10.4% width, CRPS $1,183 vs $1,001 for the plain single shock — significantly
  worse, DM p=0.002; the vincentized variant also loses (p=0.023). The cnnlstm
  matched-fold run replicates the failure (0.950 coverage at ±10.1% width,
  CRPS worse), and the 300-fold lite v4 run triple-confirms it (0.947 at
  ±8.4%, CRPS DM p<0.001; vinc p=0.022). Regime drifts inject dispersion, not information.
  Production therefore reports the **single GARCH+DVOL shock band as the
  headline** and keeps scenarios as illustrative probabilities only. (Caveat
  for readers of the cnn report: "best CRPS: vincentized comp" there is a
  20-fold best-of-8 artifact — its DM p=0.53 — the 150-fold lite run is the
  decisive sample and it loses there at p=0.023.) (The old composite was additionally buggy —
  per-path convex averaging shrank variance by √(Σp²)≈0.95; now fixed via
  mixture sampling, which is correct but still not the headline.)
- **GK-HAR+DVOL passes the pre-registered width gate at 300 folds** (lite v4):
  coverage 0.890 [0.857, 0.916] inside the [0.88, 0.92] window AND nearest to
  nominal of all variants, band 10% narrower than GARCH+DVOL (±3.6% vs ±4.0%),
  best CRPS of all 8 variants (not significant, p=0.127), width-stability
  replicated a third time (std −15%; −17% lite150, −25% cnn20). Plain HAR /
  HAR+DVOL stay within noise of GARCH+DVOL — no reason to prefer them.
  **PROMOTED to production 2026-06-11**: the shipped blend is exactly the
  validated construct — mean(GK-HAR, DVOL), GK-HAR REPLACING the GARCH leg,
  with the GARCH forecast as fallback when the HAR fit is degenerate. The
  "third averaged leg" (GARCH+GK-HAR+DVOL) was never scored — don't ship it
  without scoring it first.
- **Cross-engine comparisons need the SAME folds.** Lite defaults to daily
  steps (150 folds); cnnlstm was run with `--step 168` (20 weekly folds). Their
  persistence baselines differ because the samples differ — match `--step`/
  `--max-folds` before comparing engines head-to-head.
- **The funding tilt is NOT stable across windows.** On Jan-Jun 2026 (150
  folds) funding lifted the 24h hit-rate 47%→53% (CI [46, 59]); on the longer
  Aug 2025-Jun 2026 window (lite v4, 300 folds) the lift vanished — base 49.0%
  vs +funding 48.7% [44.0, 53.4] — and the funding variant's MAE is
  significantly WORSE than persistence (DM p=0.025). The economic eval agrees
  (Sharpe ≈ 0 before fees, negative after, on the 300-fold window). Treat
  funding as a regime-dependent tilt at most; never sell it as an edge. The
  drift stays shrunk toward the random walk (`SHRINK_ALPHA`, backtest α→0) —
  at 300 folds the un-shrunk model points are significantly WORSE than
  persistence (DM p≈0.03-0.04), which settles the shrink-to-spot design.
  **Bias-correction was measured to HURT** at 24h (noise, not stable bias) —
  it's in `forecast_post.py` but is deliberately NOT applied in production.
- **Event-day band widening (FOMC/CPI) measured REDUNDANT at 24h**
  (2026-06-12, see `RESEARCH_NEWS_EVENTS.md`): the big event-HOUR vol spike
  dilutes to a 1.05x |24h return| ratio; dvol band covers 15/16 event folds
  (0.938) with BETTER event-fold CRPS than baseline; the one miss needed +29%
  width while the event-hour spike justifies only ~+11%. Sentiment features
  and Fear&Greed are refuted outright. Don't relitigate at 24h - the event
  calendar is the first feature to add IF a 1-4h horizon product ever exists.
- **Only funding is WIRED as a backtestable exogenous feed.** The Binance REST
  endpoints give ~30d of OI and a live basis snapshot — but that limit is
  REST-only: **data.binance.vision bulk archives** (verified live) carry 5-min
  OI / taker ratio / premium index back to 2021, covering the full backtest
  window. Nothing is wired in because the directional promotion gates were
  refuted on statistical power (+1.5pp hit-rate ≈ 2 folds flipping at the
  150-fold SE of ~4.1pp), not because the data doesn't exist.

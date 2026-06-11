# fin-forecaster - Final Improvement Roadmap

Repo: `/Users/llucian/PROJECTS/predictions` (main @ 6ace854). All gates run on the lite engine, 150 daily folds (`backtest.py --engine lite --max-folds 150`, default step=horizon=24), with cross-engine confirmation only at matched folds.

---

## STATUS - implemented and gate-tested 2026-06-11

All three confirmed items were implemented and run through a fresh lite
150-fold backtest (`results/backtest_report_lite_v3.txt`, folds CSV archived
as `backtest_folds_lite_v3.csv`). Gate outcomes:

- **2.1 Rigor gates: SHIPPED, PASSED.** binomial_ci + dm_test (HLN) in
  metrics.py; CIs, DM p-values, fee sweep (0/5/10/20 bps), trial counter in
  the backtest report. DM confirms NO point variant beats persistence
  (all p >= 0.10). Per this item's own escape clause, the "best-calibrated
  band" claim was DOWNGRADED: GARCH+DVOL's CRPS edge over realized vol is NOT
  significant (DM p=0.56 under common random numbers) - its validated claim
  is coverage (0.907, CI [0.86, 0.94]), not sharpness.
- **2.2 HAR-RV + GK: SHIPPED to backtest, NOT promoted to production.**
  har/hardvol/gkdvol variants scored; none clears the CRPS/width promotion
  gate (CRPS within noise of GARCH+DVOL, DM p >= 0.16). GK-HAR+DVOL passed
  only the width-STABILITY arm (cross-fold width std -17%, CRPS +0.2%,
  coverage 0.880). Production fwd_components unchanged; cnnlstm matched-fold
  confirmation would be required before any promotion.
- **3.1 Composite fix: SHIPPED; headline band REVERTED per the failure
  branch.** Mixture sampling replaced the variance-collapsing per-path
  average in btc_forecast.py (defect verified: width shrank by exactly
  sqrt(sum p^2) ~ 0.95). The regime-composite band's FIRST validation failed
  decisively: coverage 0.960 at +/-10.4% width, CRPS $1,183 vs $1,001 single
  shock (DM p=0.002); vincentized variant also loses (p=0.023). Production
  now reports the single GARCH+DVOL band as the headline; scenarios are
  illustrative only.
- Funding tilt re-measured: 53% hit-rate, 90% CI [46, 59] - contains 50%.
- Doc correction recorded (CLAUDE.md): data.binance.vision bulk archives
  carry deep OI/taker/premium history; the ~30d limit is REST-only.

---

## 1. Executive summary

For this system, "unusually high accuracy" cannot mean beating the random walk on the 24h point forecast - the project's own walk-forward backtest and the 2024-2025 literature (M6: forecast accuracy and investment returns are near-uncorrelated; arXiv:2511.18578: 100M+ parameter TSFMs lose to naive baselines on short-horizon financial returns) agree that no such edge exists at this horizon, and the shrunk-to-spot point is the correct design, not a deficiency. The achievable edge is distributional: the GARCH(1,1)+DVOL forward-vol shock already delivers 0.900 empirical coverage at nominal 0.90 (CRPS $999.92, pinball $443.06 on 150 lite folds), and the realistic wins are (a) a few percent of band width at held coverage from better realized-vol inputs (HAR-RV, Garman-Klass), (b) fixing a verified variance-shrinking bug in the composite scenario band and validating the shipped headline band in the backtest for the first time, and (c) statistical machinery (binomial CIs, Diebold-Mariano tests, fee sweeps) so that no future "improvement" passes on noise. The funding tilt (47% to 52% hit-rate) stays as the only directional signal, applied linearly and shrunk - every attempt to sharpen it conditionally was refuted on statistical power. Expected headline movement is modest and honest: a validated, slightly sharper, mathematically correct band - not a prediction miracle.

---

## 2. Tier 1 - do now

### 2.1 Statistical-rigor gates in the backtest report (effort S; evidence 7 / feasibility 9)

**Change:** Add `binomial_ci(hits, n, level=0.90)` and `dm_test(loss_a, loss_b, h)` with the Harvey-Leybourne-Newbold small-sample correction to `metrics.py` (pure scipy, NaN-safe, `__main__` smoke tests). In `backtest.py`: print the 90% binomial CI next to every hit-rate line; DM p-values on MAE and CRPS loss differentials vs persistence; replace the single 5bps `strategy_eval` call (~line 297) with a fee sweep at 0/5/10/20 bps per side; print the count of variants scored on this fold set as Deflated-Sharpe context.

**Components:** `/Users/llucian/PROJECTS/predictions/metrics.py`, `/Users/llucian/PROJECTS/predictions/backtest.py` (report section ~lines 315-345, strategy_eval ~line 297).

**Why:** Bailey & Lopez de Prado (DSR, SSRN 2460551; PBO) show best-of-N variant selection on one fold set manufactures edges; Diebold-Mariano (1995) + HLN is the standard loss-differential test; M6 (arXiv:2310.13357) shows single-cost-point economic claims do not replicate. The binomial math is brutal: 60% over 20 folds has a 95% CI of roughly [38%, 81%] - the exact trap this project already hit with the cnnlstm number. This is the acceptance machinery for every other item below.

**Gate:** No forecast metric moves. Same-data re-run (same UTC hour, or recompute offline from the archived `backtest_folds.csv`) must reproduce headline numbers bit-identically (funding hit 52.0% now printed with 90% CI ~[45%, 59%]; GARCH+DVOL coverage 0.900 / CRPS $999.92; Sharpe +0.56 at the 5bps sweep point), and the DM test must report NO significant MAE edge vs persistence for any point variant. Implementation conditions: keep all new computation post-fold-loop (RNG stream untouched); `dm_test` returns NaN on degenerate differentials (alpha->0 makes shrunk = persistence); `h` in fold units (h=1 for non-overlapping folds); persistence CRPS = point-mass |cur - actual|; add `n_traded` to `strategy_eval` for the trade hit-rate CI.

### 2.2 HAR-RV third forward-vol leg + Garman-Klass / bipower realized measures (effort M; evidence 6 / feasibility 8.5)

**Change:** (a) `har_rv_forecast(log_ret, horizon, ann)` in `volatility.py` with the same signature/None-fallback contract as `garch11_forecast` (volatility.py:90): daily RV from 24 hourly squared log-returns, 3-coefficient HAR OLS on daily/weekly(5d)/monthly(22d) averages, pure numpy, no TF. (b) `gk_variance(open, high, low, close)` per hourly candle plus a bipower option (sum |r_t||r_{t-1}| scaled pi/2) feeding the GARCH and HAR fits (backtest.py already retains OHLC at line 160). Wire `('har', sig_har)`, `('hardvol', mean(har, dvol))` and GK-fed variants into the backtest shock loop (~line 263). Production `fwd_components` blend (`btc_forecast.py` ~lines 548-553) only after the gate passes, as a third averaged leg - never replacing the GARCH leg.

**Components:** `/Users/llucian/PROJECTS/predictions/volatility.py`, `/Users/llucian/PROJECTS/predictions/backtest.py`, `/Users/llucian/PROJECTS/predictions/btc_forecast.py` (gated).

**Why:** Bergsli et al. (RIBAF 2022) and repeated BTC horseraces: HAR-RV on intraday realized variance beats GARCH-family models, edge strongest at exactly the 1-day horizon; range estimators are ~5-8x more efficient vol proxies than squared close-to-close returns, and 24/7 BTC has no overnight gap so GK on existing hourly OHLC is near-free. A better vol level feeds directly into band width at calibrated coverage - the product.

**Gate (relative comparisons within ONE fresh lite 150-fold run; absolute dollar anchors drift with the rolling 700d window):** a HAR or hardvol variant holds coverage@90 in [0.88, 0.92] AND beats the same-run GARCH+DVOL variant by >=2% on CRPS or pinball, OR narrows mean width >=5% at same coverage. GK-only stabilization accept: coverage held, CRPS within +1%, cross-fold std of `wid_dvol` drops >=10% (computed from `backtest_folds.csv`). Treat the width arms as primary (measured CRPS spreads across vol variants are ~0.5%, so the 2% CRPS bar is unlikely). Conditions: sanity-check yfinance OHLC (High>=max(O,C), Low<=min(O,C)); scale/drop days with <24 bars in daily RV; build RV strictly from data <= fold time t; map daily HAR to hourly sigma via sqrt(RV_daily/24); confirm at cnnlstm matched folds (`--step 168`) before shipping.

---

## 3. Tier 2 - promising, needs prep

### 3.1 Fix the composite scenario distribution (mixture sampling primary) and backtest-validate the production band for the first time (effort L; evidence 7 / feasibility 8.5)

**Change:** Verified bug at `btc_forecast.py:621-623`: `comp_final += prob * scenario_paths[:, -1]` pointwise-averages INDEPENDENT scenario draws - a convolution (variance sum p^2*sigma^2) that shrinks variance and erases regime skew. Ship mixture sampling (probability-weighted concatenation/resampling of scenario draws) as the primary fix; add `vincentize_scenarios` to `forecast_post.py` as a secondary scored variant (expect it to flatten regime skew; decide by lower pinball at equal coverage). Critically, add the regime-composite band as a scored variant in `backtest.py` (per-fold HMM fit strictly on prior data, try/except with single-shock fallback per fold, scored via the existing coverage/CRPS/pinball loop ~lines 262-272) - today the report's headline asymmetric band has never been scored.

**Why:** Closes the largest validation gap in the product; the 2024-2025 combination literature (arXiv:2412.09430; Ranjan-Gneiting) proves combinations of calibrated members lose calibration, so the combined band must be re-validated - impossible today.

**Gate:** Lite 150 daily folds: composite coverage in [0.88, 0.92] (read with binomial SE ~0.0245 in mind - CRPS/pinball are primary) AND CRPS <= the same-run GARCH+DVOL variant. If the composite cannot match the single-shock band, the averaging fix still ships but the report's headline band reverts to the validated single GARCH+DVOL band. Regime mean drifts must not shift the composite median materially off the shrunk point without separate scoring.

**Prep to start now:**
- Land item 2.1 first - its DM/CI machinery is this item's acceptance layer.
- Regenerate the lite 150-fold `backtest_folds.csv` (the on-disk CSV is the 20-fold cnn run) and archive it per run.
- Decide the centering convention up front: make the backtest composite mirror production construction, or recenter production on the shrunk point - pick one so the validated band IS the shipped band.

### 3.2 Cross-cutting prep: grow statistical power

Most rejections below died on 150-fold noise (hit-rate SE ~4.1pp; coverage SE ~2.45pp). Cheap prep that unlocks future gates: extend `--max-folds` toward the ~300+ daily folds the 12h DVOL history (~500d) covers; archive every run's `backtest_folds.csv` so future comparisons are paired per-fold tests within one run rather than stale absolute anchors like $999.92.

---

## 4. Rejected - do not relitigate

- **Per-tail CQR conformalization of the 90% band:** premise empirically wrong - BTC 24h 5%/95% quantiles are nearly symmetric (skew lives in extremes a 90% band never touches); per-tail gates unmeasurable at 150 folds; same prior-fold-residual-correction pattern as the bias correction that measurably HURT. (Only the free per-tail diagnostic columns are worth adding, reporting-only.)
- **Regime-conditioned band dispersion / QRS residual draws:** shape-only change at anchored variance has a <1% CRPS ceiling vs a >=2% gate; CHOP ~95% weights make the mixture nearly degenerate; HMM regimes are redundant with what GARCH+DVOL already conditions on.
- **Fitted GARCH/DVOL blend weights + MZ/VRP debiasing:** the entire measured CRPS spread across all three vol sources is 0.49% - the 2% gate is ~4x the total effect of including DVOL at all; coverage is already exactly 0.900 (component biases cancel); per-quantile weights = ~15 parameters on ~120 autocorrelated folds, the bias-correction failure mode.
- **Funding-extremeness tail skew on the band:** BIS carry-crash evidence is weeks/month horizon, not 24h; measured on the actual fold window the conditional tail asymmetry is absent or SIGN-INVERTED; in-sample oracle improvement 0.15% vs a 2.5% gate.
- **NexCP lambda-blend trailing-residual width:** rho=0.99/hour mis-transferred (ESS ~8 effective residuals for a 90% quantile); redundant with GARCH (already exponentially recency-weighted on 24x more data); the production q_hat swap at btc_forecast.py:447 was never covered by backtest evidence.
- **Online ACI / P+I per-tail width feedback:** the static band already achieves the aggregate guarantee ACI provides (0.900 exact); a perfectly calibrated band trips the rolling-30 "miscoverage" trigger 84% of the time and the acceptance gate passes under the null ~44%; ~7 per-tail misses across 150 folds = no learning signal at safe gammas.
- **Conditional funding tilt at extremes:** central citation misattributed (arXiv:2506.08573 is a funding-mechanism-design paper, not contrarian-signal evidence); second-half active folds 11-67, requiring 59.7-64.5% active hit-rate vs a measured 52% overall (itself p~0.34 vs coin flip); production impact ~nil since the point already sits at spot.
- **Binance Vision positioning suite (taker/premium/OI):** directional promotion gate (+1.5pp) is ~2 folds flipping at 150-fold SE ~4.1pp; the OI-cascade widener fits 2 parameters on 3-8 events (overfitting class measured to hurt); cited evidence is weekly/cross-sectional. (The verified bulk-archive fact - data.binance.vision overturns the REST-only ~30d OI / snapshot-basis limits for backtesting - may be recorded as a doc correction, infrastructure only, no model wiring.)
- **Chronos/TiRex zero-shot band + Chronos-2 funding covariate:** gates statistically undecidable at 150 folds; Chronos-Bolt emits quantiles only on [0.1, 0.9] so the P05/P95 gate measures hand-rolled tail extrapolation, not the model; TSFMs documented to underperform naive baselines on financial returns (arXiv:2511.18578); the covariate channel scores directional hit-rate, which ground truth says is not a reliable edge.
- **LoRA-fine-tuned TimesFM 2.5 RV forecaster:** the cited result used linear probing on TimesFM 2.0 across 21 equity indices with 5-min RV (~59k obs), not LoRA/2.5 on one crypto series with ~550 leak-free days; the paper never evaluated calibration; marginal gain on top of an already-0.900-calibrated blend containing forward-looking DVOL is at the noise floor; L effort, second ML framework.

---

## 5. Execution order

1. **Rigor gates** (`metrics.py`: `binomial_ci`, `dm_test` + smoke tests; `backtest.py`: CIs, DM p-values, fee sweep 0/5/10/20 bps, trial counter). **Gate:** same-data run reproduces 52.0% funding hit / 0.900 coverage / $999.92 CRPS / +0.56 Sharpe at 5bps bit-identically; DM reports no significant MAE edge vs persistence for any point variant.
2. **Regenerate the lite 150-fold baseline** with the new reporting (`backtest.py --engine lite --max-folds 150`, OUTPUT_DIR env set) and archive `backtest_folds.csv`. **Gate:** report internally consistent (CIs/DM/sweep present); this run's GARCH+DVOL numbers become the in-run baseline for steps 3-4.
3. **HAR-RV + GK/bipower** in `volatility.py`, scored as `har`/`hardvol`/GK-fed tags in the shock loop. **Gate:** same-run coverage in [0.88, 0.92] AND (CRPS or pinball >=2% better than the in-run GARCH+DVOL, OR width >=5% narrower at same coverage, OR GK width-std down >=10% with CRPS within +1%); DM test on per-fold CRPS differentials from step 1 machinery; wire into production `fwd_components` only on pass, then confirm at cnnlstm matched folds (`--step 168 --max-folds 20`).
4. **Composite scenario fix + first composite-band validation**: mixture sampling at `btc_forecast.py` ~lines 621-623, `vincentize_scenarios` in `forecast_post.py`, regime-composite variant in `backtest.py` with strictly-prior per-fold HMM fits. **Gate:** composite coverage in [0.88, 0.92] with CRPS <= the in-run GARCH+DVOL variant (mixture vs vincentization decided by pinball); on failure, ship the averaging fix anyway and revert the report headline band to the validated single-shock band.
5. **Consolidate:** one final lite 150-fold run plus cnnlstm matched-fold run with all accepted variants; update `README.md` and `CLAUDE.md` gotchas with the new validated numbers (and only those that cleared their DM/CI gates); archive both reports and `backtest_folds.csv` in `results/`. **Gate:** every claim in the docs traces to a gated, in-run, significance-tested number.

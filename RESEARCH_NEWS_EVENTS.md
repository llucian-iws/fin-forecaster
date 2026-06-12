# Research: news / scheduled-event signals for the 24h BTC forecast

Deep-research run, 2026-06-12. 23 sources fetched, 111 claims extracted, 25
adversarially verified (3 votes each: 17 confirmed, 8 killed), synthesized to
the findings below. Binding constraints baked into the question: 24h horizon,
band channel (coverage/CRPS) preferred, 300-fold noise floor (~1.4% CRPS /
+/-1.7pp coverage / +/-4.7pp hit-rate), free deep-history data only.

## Verdict

**Scheduled macro events (FOMC, CPI) are the only exogenous signal class that
passes both the evidence bar and the free-data bar.** News-sentiment features
and the Fear & Greed Index are rejected on evidence; Kalshi prediction-market
signals are below the measurability floor; the implied-vol event crush is real
but small - its design implication (do not carry event vol forward) matters
more than its tradable size.

## Confirmed findings

1. **Scheduled US macro events significantly elevate BTC intraday vol**
   (3-0, 3-0, 3-0; HIGH). Two peer-reviewed papers (Ben Omrane et al., IREF
   2025, 5-min BTC/ETH 2016-2023; Ben Omrane, Houidi & Savaser, Applied
   Economics 2024) - response concentrated in the PRE-announcement period; US
   events dominate non-US; ETH more sensitive than BTC.
2. **The event-hour magnitude is large and persists into our own backtest
   window** (3-0; HIGH). Volmex Labs: CPI +107.3 / Fed +115.3 annualized vol
   points in the event hour (99% sig, 2022-2024). Independently REPLICATED by
   a verifier on free Binance 1h klines over Jan 2025-Jun 2026: FOMC event
   hours 155.9 vs 60.2 baseline (11/11 events elevated, up to 4.2x); CPI
   +66.6 excess (8/9). Order of magnitude above the noise floor AT THE EVENT
   HOUR. Note the attenuation: CPI +107 (2022-24) -> ~+67 (2025-26); do not
   hardcode point estimates.
3. **Post-event implied-vol crush is real but small** (3-0; MEDIUM). BVIV
   falls -0.94 pts after CPI, -1.32 after Fed (99% sig) - but that is only
   ~1.6-2.3% of mean BVIV, at our noise floor. Design implication: widen for
   the event window, do NOT carry elevated vol forward past the event. The
   sub-claim about persistence asymmetry (CPI 24h vs FOMC 8h) was REFUTED
   0-3 - the within-24h timing profile is unsupported.
4. **ETF approval day was a ~2x vol day** (2-1; MEDIUM, N=1). Supports the
   idea of widening on crypto-specific scheduled decision days, but one event
   cannot establish a rule.
5. **Kalshi macro prediction-market signals fail the floor** (3-0, 3-0;
   MEDIUM). KXFED predicts 5-day RV in-sample (t=3.63) but FAILS
   out-of-sample (MSFE 1.009 vs HAR); effect size ~2.4% of mean RV at the
   wrong horizon.
6. **No 2023-2026 sentiment study clears a proper baseline at <=48h** (six
   claims merged, mostly 3-0; HIGH). The field's best results benchmark
   against buy-and-hold or report raw accuracy with no significance tests;
   the FinBERT-LSTM paper's own feature importance concedes BTC is mostly
   explained by lagged price.
7. **Where sentiment IS rigorously DM-tested, it is model-dependent and HURTS
   our architecture class** (3-0; MEDIUM). Brauneis & Sahiner 2024 (6h RV,
   8 coins, DM + MCS): LightGBM/XGBoost/LSTM improve; **CNN-BiLSTM and MLP
   markedly deteriorate**; HAR not improved; the feed is proprietary anyway.
8. **Fear & Greed Index refuted as a next-day feature** (3-0, 3-0; MEDIUM).
   No Granger causality to returns, no OOS gain (2018-2025); causality runs
   in REVERSE (returns -> sentiment); the index is mechanically ~half
   price/volume-derived - a reactive transform of features the model already
   has.
9. **Backtestability synthesis** (HIGH): FOMC/CPI/NFP calendars are
   deterministic, free, hourly-alignable arbitrarily far back; free Binance
   1h klines cover the window (the replication used exactly this stack).
   Critical caveat: event days are ~2-3/month => ~25-30 of 300 folds, so the
   event-day SUBSET noise floor is ~3-4x the full-sample floor. The 1-hour
   spike diluted across a 24h window may or may not clear the gate - measure,
   don't assume.

## Refuted (do not relitigate)

- FOMC vol elevation "across pre/during/post phases" as stated (1-2).
- CPI-persists-24h / FOMC-fades-8h persistence asymmetry (0-3).
- "No BTC study beats naive baseline" as a quotable systematic-review claim (0-3).
- NLP features improving next-day BTC/ETH forecasts + Sharpe (0-3).
- Sentiment improving 6h RV forecasts as a universal claim (0-3) - it is
  model-dependent (see finding 7).
- "HAR-without-sentiment beats all ML on BTC QLIKE" as stated (0-3).
- "Free news history is NOT retroactively backtestable" (0-3) - refuted, but
  no surviving claim establishes that it IS; GDELT/CryptoPanic timestamp
  depth remains an OPEN question.

## Open questions

1. Does event-day band widening clear the 24h coverage/CRPS gate given only
   ~25-30 event-day folds? (Extend to ~700d window to double event-day N.)
2. Is GARCH+DVOL already capturing event effects endogenously (DVOL rises
   into events, crushes after)? Cheap first test: split the EXISTING
   backtest_folds CSV by event-day vs non-event-day folds and compare dvol
   coverage/CRPS before writing any new code.
3. At what lead time should widening activate (peer-reviewed evidence says
   the response concentrates PRE-announcement)?
4. GDELT / CryptoPanic timestamped deep history, ETF flow data, and on-chain
   free tiers were never resolved either way.

## Sources (primaries)

- Ben Omrane et al., Intl Review of Economics & Finance 103 (2025)
  https://www.sciencedirect.com/science/article/pii/S1059056025006720
- Ben Omrane, Houidi & Savaser, Applied Economics 56(38) (2024)
  https://ideas.repec.org/a/taf/applec/v56y2024i38p4594-4610.html
- Volmex Labs, Macro Announcements Fuel Volatility (2024)
  https://volmex.finance/Macroeconomic%20Announcements%20Fuel%20Volatility%20An%20In-Depth%20Analysis.pdf
- Mohanty & Krishnamachari (USC), arXiv 2604.01431 (2026)
- Borsa Istanbul Review ETF intraday event study (2025)
  https://www.sciencedirect.com/science/article/pii/S221484502500153X
- Brauneis & Sahiner, Asia-Pacific Financial Markets (2024)
  https://link.springer.com/article/10.1007/s10690-024-09510-6
- Finance Research Open, FGI Granger study (2026)
  https://www.sciencedirect.com/science/article/pii/S305070062600006X
- Gurgul et al. IJF 2025 (arXiv 2311.14759); Frontiers in Blockchain GPT
  headlines (2025); Wiley FinBERT-LSTM (J. Forecasting 2025)

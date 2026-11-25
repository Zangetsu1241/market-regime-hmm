# Market Regime HMM Research Lab

Test-driven quantitative research using a diagonal-Gaussian Hidden Markov Model to identify latent volatility regimes and compare a risk-off overlay with buy-and-hold.

## Features
- NumPy/SciPy Baum-Welch EM with log-space forward/backward recursions, random restarts and variance floors.
- Causal filtered probabilities, not smoothed future-aware states, for out-of-sample decisions.
- Expanding-window refits and standardization fitted on historical training rows only.
- Trailing-volatility and daily-log-return features with risk ranking based on training-only conditional squared returns.
- Cost-adjusted walk-forward evaluation, buy-and-hold benchmark and reproducible synthetic generator.
- Seven tests covering normalization, causality, filtering continuation, return alignment and input errors.

## Quickstart

```bash
python -m pip install -e '.[test]'
python -m pytest -q
regime-lab --synthetic-days 600 --train-size 250 --test-block 100 --output results
```

Or use your own daily `date,close` CSV with ascending unique dates and adjusted positive closing prices:

```bash
regime-lab --csv prices.csv --output results-real --train-size 500 --test-block 100
```

Output: `walk_forward.csv`, `metrics.json`, and `regimes.png`.

## Research limitations
- Synthetic performance is illustrative, never evidence of live tradability.
- Signal at close t is evaluated on the next close-to-close return. Executing at the observed close is not practical after observing it, so this is a research upper bound subject to executable-price validation.
- State numbers are arbitrary; training-only risk ranking reduces label-switching ambiguity.
- Gaussian emissions and proportional transaction costs are approximations. Brokerage costs, spread, impact, financing, and taxes are not comprehensively modeled.
- No performance hyperparameters should be selected on the reported held-out data.

## Modules
`src/regime_lab/hmm.py`: HMM and causal filter
`src/regime_lab/data.py`: synthetic prices, strict CSV loader and features
`src/regime_lab/research.py`: walk-forward backtest and metrics
`src/regime_lab/cli.py`: experiment entrypoint
`tests/test_hmm.py`: numerical and validation tests

MIT licensed.

## Real-data upgrade: adjusted OHLC + next-open paper execution (phase 1)

**Status:** Real-data adapter, paper-only execution simulator and 8 new tests have been implemented. Historical SPY data ingestion has **not been verified end-to-end in the authoring environment**, because external market-data endpoints are not reachable here. This is **not yet a live-trading system**.

The original `regime-lab` CLI remains a *research approximation* with close-to-close scoring. The upgraded `regime-real` command uses adjusted closing prices for HMM signals, but measures execution at the **next session open** and returns from that open to the following open.

Install the optional data provider, then download actual SPY data and record a reproducible input snapshot:

```bash
python -m pip install -e '.[test,market]'
python -m pytest -q
regime-real --download --ticker SPY --start 2005-01-01 --train-size 750 --test-block 63 --cost-bps 5 --output results-spy
```

Or provide an independently acquired, consistently adjusted `date,open,close` CSV. For an old historical dataset, explicitly disable the freshness check:

```bash
regime-real --ohlc-csv historical_spy.csv --historical-only --output results-historical
```

The CLI rejects missing/duplicate/out-of-order dates, nonpositive prices, suspicious daily jumps, insufficient history, and stale inputs unless `--historical-only` is supplied. It writes `adjusted_ohlc.csv` and `data_manifest.json` (including SHA-256), `model_signals_raw.csv`, `paper_open_to_open.csv`, and `paper_metrics.json`. Nonhistorical mode also writes a **non-actionable** `signal_status.json` marked `RESEARCH_ONLY_NO_ORDERS`; it does not submit orders.

**Known gaps before real production:** correct latest-day regime inference without the final one-day research lag, validated paid/licensed provider and corporate-action history, open-auction spreads/slippage and order-fill assumptions, persistent idempotent portfolio state, trading calendar and market-hour scheduling, alerts/fail-closed behavior, broker paper-account reconciliation, walk-forward parameter governance, and independently verified remote CI plus real-data results. Do not put real money behind these results.


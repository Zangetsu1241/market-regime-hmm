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
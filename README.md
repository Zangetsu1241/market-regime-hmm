# Market Regime HMM — reproducible research and paper simulation

A diagonal Gaussian HMM research system for SPY, QQQ and IWM, with versioned Yahoo/CSV data, causal walk-forward inference, next-open portfolio accounting and a persistent **paper-only** ledger. There is no brokerage client or live-order capability.

## Verified status

The original 15 tests passed before changes. The expanded suite, actual downloaded-data experiments and paper checks are recorded in [the verification report](docs/VERIFICATION.md). Real experiments use 2,011 daily observations per ETF, 2018-01-02 through 2025-12-31. Validation is 2021–2023; the final chronological test is 2024–2025 (502 sessions). These results do **not** demonstrate superior returns to buy-and-hold.

| Validation-selected strategy | Test CAGR | Buy/hold CAGR | Strategy max drawdown | Buy/hold max drawdown |
|---|---:|---:|---:|---:|
| SPY risk-scaled | 12.19% | 21.72% | -7.89% | -18.75% |
| QQQ risk-scaled | 9.77% | 23.78% | -9.84% | -22.77% |
| IWM binary | 0.07% | 12.41% | -23.43% | -27.50% |

See [saved reports](reports/verified-2018-2025/) for curves, candidate comparisons, parameters, diagnostics, cost sensitivity, graphs and provenance. Raw vendor snapshots remain in the ignored local cache; redistribution rights have not been assumed.

## Setup

Python 3.10+; the local verification environment used Python 3.14.6 on Windows. The CI matrix specifies 3.10 and 3.12; a workflow definition alone is not proof of a successful remote CI run.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e '.[test,market]'
.venv/Scripts/python.exe -m pytest -q
Copy-Item .env.example .env
```

The [environment snapshot](reports/requirements-verified.txt) records exact installed dependency versions for the verified Windows/Python 3.14 environment. It is not a universal cross-platform lock. Environment variables control cache, output and paper-state paths; shell values override the optional local environment file. No secrets are required. Never add brokerage credentials.

## Commands

Run from the repository root. Module commands work without relying on shell PATH:

```powershell
# Download all three assets; identical requests reuse checksummed local snapshots.
.venv/Scripts/python.exe -m regime_lab.system_cli download

# Compare 2/3/4 states, expanding/rolling windows and two allocations on validation only.
.venv/Scripts/python.exe -m regime_lab.system_cli research --output results-validation

# Explicitly release the final test after fixing the specification.
.venv/Scripts/python.exe -m regime_lab.system_cli research --release-test --output results-final

# One asset, or canonical CSVs exported by a licensed data source.
.venv/Scripts/python.exe -m regime_lab.system_cli research --ticker SPY --csv-directory vendor-export

# Current completed-session paper signal and reconciliation of the preceding pending signal.
.venv/Scripts/python.exe -m regime_lab.system_cli paper --ticker SPY --refresh
```

The installed `regime-system` entry point accepts the same arguments. `--config` selects a JSON specification; `--cache`, `--output` and `--paper-state` select independent directories. `--refresh` downloads and validates a new version before replacing the request pointer. Do not repeatedly tune against released test results. Use a new output directory for a new research specification and retain old experiment records.

## Data contract and provenance

[Data adapters](src/regime_lab/market_data.py) implement a `Provider.fetch(ticker, start, end)` protocol with exclusive end dates. Yahoo is an initial research source, not a production feed SLA. A licensed provider can implement the same interface, or export per-symbol CSVs with:

```text
date,open,high,low,close,adj_close,volume,dividends,splits
```

Dates are naive normalized US exchange session labels, sorted and unique. OHLC is as-traded, volume is in the corresponding share units, dividends are cash per share on ex-date, splits are new/old share factors (0 means no split), and adjusted close is a consistently total-return-adjusted series. Yahoo already split-adjusts nominal OHLC and dividends; the adapter reverses subsequent split factors, including available actions beyond a historical request's end, before trimming the response. CSV exports must already obey the canonical contract.

Missing XNYS sessions, duplicate dates, incomplete/nonfinite prices, zero volume, inconsistent high/low, and suspicious adjusted-price jumps fail closed. There is no forward filling. Provider requests retry three times with bounded backoff. SHA-256 content objects, request metadata and atomic pointers retain previous successful versions. A checksum mismatch is an error, not a silent redownload.

## Methodology

[Features and fitting](src/regime_lab/pipeline.py) use daily log returns, trailing 20-session volatility, trailing 20-session momentum and same-session log high/low range. All are observable after close. Feature means and scales use training rows only. The HMM uses log-space forward/backward Baum–Welch, seeded restarts and variance floors. Smoothed quantities are used only inside fitting; decisions use filtered probabilities with continuation from training history. States are ranked by training-only conditional squared returns, making risk labels interpretable across refits without claiming stable economic state identities.

[Experiment selection](src/regime_lab/experiments.py) compares 12 candidates: 2/3/4 states × expanding/504-row rolling training × binary/risk-scaled allocations. Refit cadence, feature lookback, seed, costs and 12% target volatility are fixed in [the configuration](configs/research.json). Validation Sharpe chooses the winner; final-test rows are never supplied to that selection. Later test predictions may refit on earlier test observations, as in an online expanding/rolling process; that does not retune hyperparameters. Fit dates, parameters, scalers, histories, convergence and occupancies are saved. Nonconvergence is disclosed in research; daily paper runs reject it.

Binary allocation is 100% ETF when filtered risk score is below 0.5, otherwise cash. Risk-scaled allocation is clipped to [0,1]: `(1 - risk_score) * 0.12 / trailing_annual_volatility`. Baselines are buy-and-hold, positive trailing 20-day momentum, and a simple 12% volatility target.

[Portfolio accounting](src/regime_lab/portfolio.py) observes close t, trades at open t+1, and marks holdings at close t+1. Existing holdings earn overnight returns; a new entry cannot capture the gap before its execution. A cash/share ledger handles portfolio drift, split share counts, dividends and fractional shares. Fees are solved against post-cost target equity. Default cost is 5 bps of traded notional: 1 commission + 1 half-spread + 3 slippage. Both strategies and benchmarks pay entry/reinvestment costs. No leverage, terminal liquidation, taxes, financing or cash interest are modeled. Dividends are credited on ex-date rather than pay-date, an explicitly optimistic cash-timing approximation.

Metrics include CAGR (252 sessions/year), simple-return Sharpe with zero cash hurdle, Sortino with downside RMS over all days, Calmar, drawdown including initial capital, annual volatility, turnover, exposure and costs. Undefined ratios are JSON null. Annual subperiods, causal bull/bear momentum groups, fixed 20% high/low-volatility groups and 0×/1×/2×/4× cost stress are reported. Conditional disconnected regimes get daily summaries, not misleading compounded annual returns.

## Paper operation and scheduling

[The paper ledger](src/regime_lab/paper.py) is a separate baseline: configured state count, fixed rolling training length, **daily refitting and binary allocation**. It does not automatically promote a validation winner or claim parity with the annually refitted research strategy. Promotion requires a separately specified and prospectively tested policy.

The first run starts with $100,000 cash and schedules a target for the next exchange session; it never invents an earlier fill. The next after-close run simulates that pending instruction at that session's recorded open. This is daily-bar paper simulation, not an auction-connected paper broker. Subsequent runs update cash, shares, fills, reports and the next signal atomically in one JSON state. The exact configuration is bound to the ledger. Duplicate-session reruns leave economic state unchanged; a configuration change requires a separate ledger.

A 30-minute close buffer, exchange holidays/early closes, freshness checks, convergence gates, single-writer lock, previous-state backup and health files protect updates. Missing a session requires explicit operator reconciliation; the system will not silently backfill missed trades. Stale locks require investigation before removal. Health status and process exit codes are local monitoring; no email/pager alert delivery is configured. Check status even when economic state is unchanged.

```powershell
# Run manually first.
powershell -NoProfile -File scripts/daily-paper.ps1
# Optional installation in Windows Task Scheduler (local machine time).
powershell -NoProfile -File scripts/register-task.ps1 -At '04:30'
```

04:30 Asia/Kolkata is safely after US close in both DST seasons. Adjust for the Windows machine's actual timezone. Task registration is opt-in and was not performed during verification. Keep the machine awake/online and use a persistent, backed-up local path. The workflow's exclusive file lock is intended for one local filesystem, not distributed execution.

## Compatibility and limitations

Legacy `regime-lab` and `regime-real` remain available. The former is a synthetic/close-to-close research demonstration with assumed same-close execution; the latter is the earlier adjusted-open evaluator and explicitly non-actionable signal status. Use the new system for executable timing and persistent paper state.

[The audit](docs/AUDIT.md) and [verification report](docs/VERIFICATION.md) distinguish implemented controls from remaining work. Yahoo revises history; downloaded snapshots are reproducible inputs, not a point-in-time vendor archive. A preselected surviving ETF universe does not eliminate survivorship/selection bias. Gaussian diagonal emissions, state instability, overlapping features, small sample sizes and repeated validation comparisons constrain inference. Annual model refits and other untested cadences are not optimality claims. No licensed production feed, intraday fill verification, real broker reconciliation, external alerting, disaster-recovery certification or live deployment approval is included. **Do not use this repository to place real-money orders.**

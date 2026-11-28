# Verification and operational readiness

## Observed verification

- Original repository: `1310019`; original suite **15 passed** before implementation.
- Final local suite: **34 passed** on Windows, Python 3.14.6. The run emitted 88
  pandas timezone-array deprecation warnings from calendar dependency calls.
  These are retained rather than hidden; future pandas/calendar compatibility
  needs monitoring.
- Numerical forward likelihood was checked against enumeration of every latent
  path on a small example. Filter continuation, invalid settings, negative initial
  probabilities, training-only scalers, latest-day predictions, expanding/rolling
  prefix invariance and future-price perturbations were tested.
- Provider/cache tests cover missing/duplicate sessions, holiday/early-close
  handling, immutable checksums, failed refresh preservation and split adjustment
  for actions beyond the requested historical end.
- Portfolio tests cover next-open entry timing, self-financing fractional
  allocations, splits and dividend cash. Paper tests cover duplicate reruns,
  missing data, nonconvergence, exclusive locks and unchanged economic state on
  failures. Selection tests verify final-test perturbations cannot affect ranking.
- Yahoo downloaded **2,011 sessions for each of SPY, QQQ and IWM**, covering
  2018-01-02–2025-12-31. Initial sandbox/cache access failures were resolved;
  a subsequent transient provider timeout recovered under the retry policy.
- Actual-data research completed for all three assets. The final repeat uses
  code commit `c70a5bc` and records source hashes, package versions, dataset hashes,
  model artifacts, validation comparisons and test performance in the
  [saved experiment directory](../reports/verified-2018-2025/).
- Latest SPY paper initialization completed for **2026-11-27**. It started in
  cash, with a cash target dated for **2026-11-30**; rerunning was idempotent.
  This is a local simulated instruction, not a broker order.
- A separate, explicitly historical SPY paper replay initialized on 2025-12-30,
  processed one next-open simulated fill on 2025-12-31, and verified that a repeat
  left the serialized economic state byte-for-byte unchanged. The replay ended
  with 145.45801853592218 simulated shares, zero cash and $99,190.72951416124 equity
  after $49.97501249375313 modeled costs. These are replay observations, not a
  prospective trading record.
- Both Windows automation scripts passed PowerShell parser validation. No
  scheduled task was registered. Editor diagnostics reported no errors; this
  does not substitute for a full static type checker.

## Research interpretation

The final test covers 502 sessions during 2024–2025. Candidate selection uses
only 2021–2023 validation returns; initial history is 2018–2020. Models continue
to refit causally through subsequent validation/test history. Annual cadence,
features, costs and target volatility were fixed before viewing final-test
performance. The repeat after the adapter/provenance fix did not retune them.

| Asset | Selected allocation | Net CAGR | Sharpe | Max drawdown | Buy/hold CAGR |
|---|---|---:|---:|---:|---:|
| SPY | Risk-scaled | 12.19% | 1.272 | -7.89% | 21.72% |
| QQQ | Risk-scaled | 9.77% | 0.942 | -9.84% | 23.78% |
| IWM | Binary | 0.07% | 0.091 | -23.43% | 12.41% |

These HMM overlays did not beat buy-and-hold returns. Lower drawdown for SPY/QQQ
comes with lower exposure and opportunity cost. IWM's binary result is particularly
sensitive to transaction costs. The held-out results are evidence for this one
specification and period, not evidence of a robust market edge. After inspection,
this test period must no longer be treated as unseen for future strategy redesign.

## Ready

- Reproducible research downloads/CSV imports, cached versions and provenance.
- Causal HMM inference, chronological selection, reproducible model artifacts.
- Daily-bar next-open cash/share backtests, cost scenarios, baseline comparisons.
- Local, single-process paper ledger with fail-closed checks and rerun protection.
- Windows scheduling scripts, local logs, status files and full test commands.

## Still required before any live deployment

1. A licensed production feed with corporate-action corrections, point-in-time
   history, pay dates, availability guarantees and independent data reconciliation.
2. Prospective evaluation of a frozen strategy. The operational paper baseline
   uses daily rolling binary decisions; it is not automatically the annually
   refitted validation winner. Policy promotion and prospective parity are not done.
3. Actual auction/slippage/impact validation, volume/participation limits, settlement,
   dividend pay-date cash handling, taxes, broker reconciliation and operational
   capital/risk limits. The daily simulator assumes complete fractional fills.
4. Remote alert delivery, redundant scheduling, durable off-machine backups,
   recovery drills, stale-lock remediation and missed-session reconciliation.
   Atomic local replacement is not a distributed transaction or disaster-recovery SLA.
5. More assets/time periods, seed/refit/lookback stability analysis, uncertainty
   intervals and selection-bias correction. The implemented state/window/cost
   sensitivity does not exhaust robustness analysis.
6. Independent review and deployment authorization. No live trading capability
   has been added, and no real-money order has been placed.

Remote CI status is separate from local test results. The repository workflow
defines Python 3.10/3.12 checks; only an observed remote run can verify them.

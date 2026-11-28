# Productionization audit

Baseline: commit `1310019`. The repository contains six implementation modules,
two test modules, packaging, README and CI; no notebooks or private datasets.

## Findings and implementation plan

1. HMM: log-space inference and filter continuation are sound. Invalid iteration,
   restart, variance and state settings are unchecked; initial filter distributions
   may contain negative/nonfinite probabilities. Track convergence explicitly and
   validate inputs. Rank states by training-only conditional return variance.
2. Legacy research: standardization is training-only. The old close-to-close
   accounting assumes unattainable same-close execution, and its final prediction
   needs a future return. Preserve that documented demonstration; introduce a
   signal-only path that includes the most recent completed observation.
3. Data: existing ingestion drops high/low/volume/actions and does not detect
   missing exchange sessions. Add a provider protocol, Yahoo and CSV adapters,
   strict calendar validation, retries, content-addressed snapshots and checksums.
   Yahoo prices are split-adjusted even when auto_adjust=False; reconstruct
   as-traded prices using subsequent split factors for share accounting. Retain
   original vendor fields and adjusted prices for reproducibility.
4. Execution: next-open timing in the existing adapter is correctly lagged, but
   fractional allocations, portfolio drift, benchmark entry costs and split/dividend
   accounting need a separate cash/share ledger. Include spread, slippage and
   commission, with no leverage and no broker integration.
5. Validation: no selection/test governance exists. Predeclare chronological
   boundaries; compare 2/3/4 states and rolling/expanding fits on validation only;
   freeze the winner before the final test. Keep test evaluation opt-in and record
   selection metadata. Refitting on earlier test observations is permitted online;
   tuning using test performance is not.
6. Operations: no persistent state, lock, calendar gate or atomic updates exist.
   Add a single-writer paper ledger, dated pending signals, transactional writes,
   idempotent fills and fail-closed monitoring. Historical replay remains explicit.
7. Reproducibility: save configurations, environment versions, input hashes,
   fitted scalers/parameters, convergence and regime diagnostics, folds and reports.

## Verification policy

Run the original suite before edits, then numerical, future-perturbation,
calendar, provider/cache, cash/share, failure-atomicity and idempotence regressions.
Attempt actual Yahoo downloads for SPY, QQQ and IWM. Never substitute synthetic
prices for a real-market result. Fixture prices are used only in tests.

## Structural limitations

These three ETFs are a preselected surviving universe. Conclusions do not establish
survivorship-free stock selection. Vendor history is revised and is not a
point-in-time archive. Daily bars cannot verify auction liquidity or actual fills.
Dividend cash is credited on ex-date for research (pay-date financing is omitted).
No live brokerage capability is implemented or authorized.

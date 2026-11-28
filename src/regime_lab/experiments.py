"""Validation-only selection with an explicit, opt-in final test release."""
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
from .pipeline import ModelConfig, signals
from .portfolio import Costs, backtest, metrics
from .storage import write_json


def evaluate(bars, spec, output, manifest, release_test=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    validation = pd.Timestamp(spec['validation_start'])
    test = pd.Timestamp(spec['test_start'])
    if not bars.index[0] < validation < test <= bars.index[-1]:
        raise ValueError('Invalid chronological train/validation/test boundaries')
    base = ModelConfig(**spec['model'])
    costs = Costs(**spec['costs'])
    # Never pass test rows into candidate generation or scoring.
    development = bars.loc[bars.index < test]
    candidates = []
    for states in (2,3,4):
        for window in (None, base.train_size):
            config = replace(base, states=states, window=window)
            decision, artifacts = signals(development, config)
            selected = decision.loc[decision.index >= validation]
            for variant in ('binary','risk_scaled'):
                curve = backtest(development, selected, variant, costs)
                summary = metrics(curve)
                candidate_id = f'{states}-{window or "expanding"}-{variant}'
                curve.to_csv(output/f'validation-{candidate_id}.csv')
                candidates.append({'id':candidate_id, 'config':asdict(config),
                    'variant':variant, 'metrics':summary,
                    'nonconverged_fits':sum(not a['converged'] for a in artifacts)})
            write_json(output/f'validation-models-{states}-{window or "expanding"}.json', artifacts)
    eligible = [c for c in candidates if c['metrics']['sharpe'] is not None]
    if not eligible:
        raise RuntimeError('No valid validation candidate')
    winner = max(eligible, key=lambda c:c['metrics']['sharpe'])
    selection = {'criterion':'maximum validation simple-return Sharpe; zero cash interest',
        'validation_start':str(validation.date()), 'test_start':str(test.date()),
        'candidates':candidates, 'winner':winner,
        'warning':'Multiple comparisons on validation; final test is descriptive and must not be retuned'}
    write_json(output/'selection.json', selection)
    provenance = {'spec':spec, 'dataset':manifest, 'python':platform.python_version(),
        'packages':{p:importlib.metadata.version(p) for p in
            ['numpy','pandas','scipy','exchange-calendars','yfinance']},
        'selection_sha256':hashlib.sha256((output/'selection.json').read_bytes()).hexdigest(),
        'final_test_released':release_test}
    write_json(output/'provenance.json', provenance)
    if not release_test:
        return {'selection':winner, 'test':'not evaluated'}
    config = ModelConfig(**winner['config'])
    decision, artifacts = signals(bars, config)
    decision.to_csv(output/'signals.csv')
    write_json(output/'models.json', artifacts)
    # First test session is executed from the preceding close. Validation selection
    # does not score this overnight/next-open interval.
    next_session = pd.Series(bars.index, index=bars.index).shift(-1).reindex(decision.index)
    heldout = decision.loc[next_session >= test]
    summaries, curves = {}, {}
    for variant in ('binary','risk_scaled','buy_hold','trend','vol_target'):
        curve = backtest(bars, heldout, variant, costs)
        curves[variant] = curve
        curve.to_csv(output/f'test-{variant}.csv')
        summaries[variant] = metrics(curve)
    chosen = curves[winner['variant']]
    subperiods = {str(year):metrics(frame) for year, frame in chosen.groupby(chosen.index.year)}
    # These are conditional daily summaries; disconnected groups are not a CAGR.
    regime_stats = {}
    for name, mask in {'bull':chosen.momentum > 0, 'bear':chosen.momentum <= 0,
                       'high_vol':chosen.volatility*np.sqrt(252) >= .20,
                       'low_vol':chosen.volatility*np.sqrt(252) < .20}.items():
        part = chosen.loc[mask]
        regime_stats[name] = {'days':len(part), 'mean_daily_return':float(part['return'].mean()) if len(part) else None,
                              'mean_exposure':float(part.exposure.mean()) if len(part) else None}
    # Cost sensitivity is prespecified; no test-based reselection occurs.
    sensitivity = {str(multiplier):metrics(backtest(bars, heldout, winner['variant'],
        Costs(**{k:v*multiplier for k,v in asdict(costs).items()}))) for multiplier in (0,1,2,4)}
    result = {'selected_variant':winner['variant'], 'test':summaries,
        'test_years':subperiods, 'conditional_regimes':regime_stats, 'cost_sensitivity':sensitivity,
        'state_switch_fraction':float((heldout.state.diff().dropna()!=0).mean()),
        'test_nonconverged_fits':sum(not a['converged'] for a in artifacts),
        'cash_rate':0, 'terminal_liquidation':False,
        'accounting':'next-open rebalance; close mark; ex-date dividends; fractional shares'}
    write_json(output/'results.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11,6))
    for variant, curve in curves.items():
        ax.plot(curve.index, curve.equity/100000, label=variant)
    ax.set(title='Final chronological test — simulated net equity', ylabel='Growth of initial capital')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/'test-equity.png', dpi=140)
    plt.close(fig)
    return result

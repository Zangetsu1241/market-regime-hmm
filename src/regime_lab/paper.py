"""Atomic, idempotent local paper portfolio. No live credentials or order API."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .market_data import latest_completed, sessions, validate_bars
from .pipeline import ModelConfig, signals
from .portfolio import Costs, actions, rebalance
from .storage import exclusive_lock, write_json


def update_paper(bars, directory, ticker='SPY', config=ModelConfig(), costs=Costs(),
                 now=None, dataset_hash=None, initial_cash=100000.):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    with exclusive_lock(root/'run.lock'):
        try:
            bars = validate_bars(bars)
            completed = latest_completed(now)
            if bars.index[-1] != completed:
                raise RuntimeError('Missing, stale or incomplete latest-session data')
            if ticker not in {'SPY','QQQ','IWM'}:
                raise ValueError('Unsupported paper asset')
            if not np.isfinite(initial_cash) or initial_cash <= 0:
                raise ValueError('Invalid initial paper capital')
            costs.rate
            # Refit a fixed rolling window through yesterday; filter today's close.
            daily_config = replace(config, window=config.train_size, refit_every=1)
            config_payload = {'model':asdict(daily_config), 'costs':asdict(costs), 'ticker':ticker}
            config_id = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode()).hexdigest()
            path = root/'state.json'
            state = json.loads(path.read_text()) if path.exists() else {
                'schema':1, 'mode':'PAPER_ONLY', 'ticker':ticker, 'config_id':config_id,
                'cash':initial_cash, 'shares':0., 'last_session':None,
                'pending':None, 'fills':[], 'reports':[]}
            if state['ticker'] != ticker or state['config_id'] != config_id:
                raise RuntimeError('Paper configuration changed; use a new ledger')
            day = str(completed.date())
            if state['last_session'] == day:
                write_json(root/'health.json', {'status':'OK_ALREADY_PROCESSED', 'at':now.isoformat()})
                return state
            if state['last_session'] and pd.Timestamp(state['last_session']) > completed:
                raise RuntimeError('Cannot move paper state backwards')
            decision, artifacts = signals(bars.iloc[-(config.train_size+21):], daily_config)
            latest = decision.iloc[-1]
            if decision.index[-1] != completed or not artifacts[-1]['converged']:
                raise RuntimeError('Latest model missing or not converged; state preserved')
            if (completed-pd.Timestamp(artifacts[-1]['fit_through'])).days > 7:
                raise RuntimeError('Stale model; state preserved')
            if not np.isfinite(latest.risk_score) or not 0 <= latest.risk_score <= 1:
                raise RuntimeError('Abnormal model prediction')
            fee, turnover = 0., 0.
            if state['pending']:
                pending = state['pending']
                if pending['execution_date'] != day:
                    raise RuntimeError('Missed paper session; explicit reconciliation required')
                bar = bars.iloc[-1]
                state['cash'], state['shares'] = actions(state['cash'], state['shares'], bar)
                state['cash'], state['shares'], fee, turnover = rebalance(
                    state['cash'], state['shares'], bar.open, pending['target'], costs)
                state['fills'].append({'id':f'{ticker}:{day}', 'signal_date':pending['signal_date'],
                    'execution_date':day, 'reference_open':float(bar.open), 'target':pending['target'],
                    'cost':fee, 'turnover':turnover, 'cash':state['cash'], 'shares':state['shares']})
            future = sessions(completed+pd.Timedelta(days=1), completed+pd.Timedelta(days=10))[0]
            state['pending'] = {'signal_date':day, 'execution_date':str(future.date()),
                'target':float(latest.binary), 'risk_score':float(latest.risk_score)}
            state['last_session'] = day
            state['model'] = artifacts[-1]
            state['dataset_sha256'] = dataset_hash
            state['reports'].append({'date':day, 'cash':state['cash'], 'shares':state['shares'],
                'equity':state['cash']+state['shares']*float(bars.close.iloc[-1]),
                'cost':fee, 'next_target':state['pending']['target']})
            # Entire economic state, fills and pending signal commit in one replace.
            if path.exists():
                write_json(root/'previous-state.json', json.loads(path.read_text()))
            write_json(path, state)
            write_json(root/'health.json', {'status':'OK', 'at':now.isoformat(), 'session':day})
            return state
        except Exception as exc:
            write_json(root/'health.json', {'status':'FAILED_CLOSED', 'at':now.isoformat(),
                       'error_type':type(exc).__name__, 'reason':str(exc)})
            raise

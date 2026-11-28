"""Long-only fractional-share simulation; never connects to a broker."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    commission_bps: float = 1.
    half_spread_bps: float = 1.
    slippage_bps: float = 3.

    @property
    def rate(self):
        parts = np.array([self.commission_bps, self.half_spread_bps, self.slippage_bps])
        if not np.isfinite(parts).all() or (parts < 0).any() or parts.sum() >= 1000:
            raise ValueError('Invalid execution costs')
        return float(parts.sum()/10000)


def rebalance(cash, shares, price, target, costs):
    if not np.isfinite([cash, shares, price, target]).all() or min(cash, shares) < -1e-8 or price <= 0 or not 0 <= target <= 1:
        raise ValueError('Invalid portfolio or target')
    equity = cash+shares*price
    old_value = shares*price
    # Solve target fraction of equity *after* proportional execution costs.
    delta = target*equity-old_value
    traded = delta/(1+target*costs.rate) if delta >= 0 else delta/(1-target*costs.rate)
    fee = abs(traded)*costs.rate
    cash = cash-traded-fee
    shares = shares+traded/price
    if cash < -1e-7 or shares < -1e-7:
        raise RuntimeError('Portfolio would borrow cash or shares')
    return max(cash, 0.), max(shares, 0.), fee, abs(traded)/equity if equity else 0.


def actions(cash, shares, bar):
    shares *= float(bar.splits) if bar.splits else 1.
    cash += shares*float(bar.dividends)
    return cash, shares


def backtest(bars, decisions, variant='binary', costs=Costs(), initial_cash=100000.):
    if variant not in {'binary','risk_scaled','buy_hold','trend','vol_target'}:
        raise ValueError('Unknown strategy')
    if initial_cash <= 0 or not np.isfinite(initial_cash):
        raise ValueError('Invalid initial cash')
    costs.rate
    if decisions.empty or decisions.index.has_duplicates or not decisions.index.is_monotonic_increasing:
        raise ValueError('Invalid decisions')
    locations = bars.index.get_indexer(decisions.index)
    if (locations < 0).any() or (np.diff(locations) != 1).any():
        raise ValueError('Decisions must cover contiguous market sessions')
    cash, shares, prior_equity = initial_cash, 0., initial_cash
    rows = []
    for day, signal in decisions.iterrows():
        loc = bars.index.get_loc(day)
        if loc+1 >= len(bars):
            continue
        bar = bars.iloc[loc+1]
        cash, shares = actions(cash, shares, bar)
        target = {'binary':signal.binary, 'risk_scaled':signal.risk_scaled,
            'buy_hold':1., 'trend':float(signal.momentum > 0),
            'vol_target':float(np.clip(.12/max(signal.volatility*np.sqrt(252),1e-8),0,1))}[variant]
        # Buy-and-hold reinvests ex-date dividends at the next available open;
        # other strategies rebalance to daily target weights.
        cash, shares, fee, turnover = rebalance(cash, shares, bar.open, target, costs)
        equity = cash+shares*bar.close
        if equity <= 0:
            raise RuntimeError('Bankrupt simulation')
        rows.append({'date':bars.index[loc+1], 'signal_date':day,
            'return':equity/prior_equity-1, 'equity':equity,
            'cash':cash, 'shares':shares, 'target':target,
            'exposure':shares*bar.close/equity, 'turnover':turnover,
            'cost':fee, 'cost_fraction':fee/prior_equity,
            'regime':int(signal.state), 'momentum':signal.momentum,
            'volatility':signal.volatility})
        prior_equity = equity
    if not rows:
        raise ValueError('No executable next-open signals')
    return pd.DataFrame(rows).set_index('date')


def metrics(frame):
    r = frame['return'].to_numpy(float)
    if not len(r) or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError('Invalid returns')
    growth = np.cumprod(1+r)
    peak = np.maximum.accumulate(np.r_[1., growth])[1:]
    dd = float(np.min(growth/peak-1))
    cagr = float(growth[-1]**(252/len(r))-1)
    vol = float(np.std(r, ddof=1)) if len(r)>1 else 0.
    downside = float(np.sqrt(np.mean(np.minimum(r, 0)**2)))
    return {'observations':len(r), 'total_return':float(growth[-1]-1), 'cagr':cagr,
        'sharpe':float(np.mean(r)/vol*np.sqrt(252)) if vol else None,
        'sortino':float(np.mean(r)/downside*np.sqrt(252)) if downside else None,
        'calmar':cagr/abs(dd) if dd else None, 'max_drawdown':dd,
        'annual_volatility':vol*np.sqrt(252),
        'annual_turnover':float(frame.turnover.mean()*252),
        'exposure':float(frame.exposure.mean()), 'transaction_costs':float(frame.cost.sum()),
        'cost_fraction_sum':float(frame.cost_fraction.sum())}

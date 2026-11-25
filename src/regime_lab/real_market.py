"""OHLC market-data validation and lagged-open paper execution for Market Regime HMM.

No broker connectivity. After observing close t, a signal enters at next session's
open t+1 and exits or rebalances at session t+2 open. In particular it must not
claim the close-to-close t+1 return as attainable after knowing close t.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def validate_ohlc(frame: pd.DataFrame, *, min_rows: int = 600) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not {'open', 'close'}.issubset(frame.columns):
        raise ValueError('OHLC data need open and close columns')
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError('OHLC index must contain dates')
    if frame.index.hasnans or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError('Dates must be valid, unique, and increasing')
    if len(frame) < min_rows:
        raise ValueError(f'At least {min_rows} complete sessions are required')
    out = frame[['open', 'close']].astype(float).copy()
    array = out.to_numpy()
    if not np.isfinite(array).all() or (array <= 0).any():
        raise ValueError('Open and close must be positive finite prices')
    # Guard against obviously broken dividend/split adjustments, symbols, and bad rows.
    if (out['close'].pct_change().abs().dropna() > 0.40).any():
        raise ValueError('Suspicious >40% daily price change: inspect adjustment/data')
    return out


def load_adjusted_ohlc(csv_path: str | Path, min_rows: int = 600) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if not {'date', 'open', 'close'}.issubset(df.columns):
        raise ValueError('CSV requires date,open,close; prices must be consistently adjusted')
    index = pd.to_datetime(df['date'], utc=True, errors='raise')
    data = df[['open', 'close']].copy()
    data.index = pd.DatetimeIndex(index).tz_localize(None)
    return validate_ohlc(data, min_rows=min_rows)


def download_yahoo_ohlc(ticker='SPY', start='2005-01-01', end=None) -> pd.DataFrame:
    """Fetch fully completed daily bars through yfinance (optional dependency).

    Yahoo access is not guaranteed; callers must fail closed on empty/stale data.
    end is exclusive and defaults to UTC today's date to avoid partial-day bars.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install optional provider with: pip install '.[market]'") from exc
    if not ticker or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-^' for c in ticker.upper()):
        raise ValueError('Ticker contains invalid characters')
    if end is None:
        end = pd.Timestamp.now(tz='UTC').date().isoformat()
    bars = yf.download(ticker, start=start, end=end, interval='1d',
                       auto_adjust=True, progress=False, threads=False, multi_level_index=False)
    if bars is None or bars.empty:
        raise RuntimeError('Market provider returned no historical data')
    # yfinance versions vary in flat vs multi-index shape.
    if isinstance(bars.columns, pd.MultiIndex):
        bars.columns = bars.columns.get_level_values(0)
    out = bars.rename(columns={'Open': 'open', 'Close': 'close'})
    out = out[['open', 'close']]
    dates = pd.DatetimeIndex(out.index)
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    out.index = dates.normalize()
    return validate_ohlc(out)


def executable_next_open(signals: pd.DataFrame, prices: pd.DataFrame, cost_bps: float = 5.) -> pd.DataFrame:
    """Apply the *existing* HMM close-t positions at open t+1, valued at open t+2.

    signals index is t+1 close and signal_date column is close t as in
    research.walk_forward. The final signal is discarded unless t+2 exists.
    All outputs are paper-trading only, with proportional turnover fee.
    """
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError('Cost must be finite and nonnegative')
    ohlc = validate_ohlc(prices, min_rows=3)
    if not isinstance(signals, pd.DataFrame) or not {'signal_date', 'position'}.issubset(signals.columns):
        raise ValueError('Signals need signal_date and position columns')
    dates = ohlc.index
    opens = ohlc.open.to_numpy(dtype=float)
    position_prior = 0.
    rows=[]
    previous_signal=None
    for index, (_, item) in enumerate(signals.iterrows()):
        signal_at = pd.Timestamp(item['signal_date'])
        if signal_at.tzinfo is not None:
            signal_at = signal_at.tz_localize(None)
        if previous_signal is not None and signal_at <= previous_signal:
            raise ValueError('Signals must be in strictly increasing date order')
        previous_signal=signal_at
        loc = dates.get_indexer([signal_at])[0]
        if loc < 0:
            raise ValueError('Missing signal date in OHLC data')
        if loc + 2 >= len(dates):
            continue  # cannot evaluate this decision without a future valuation open
        position = float(item['position'])
        if not np.isfinite(position) or position not in (0., 1.):
            raise ValueError('This implementation supports SPY/cash positions 0 or 1')
        buy_open=opens[loc+1]
        next_open=opens[loc+2]
        simple_return=next_open/buy_open - 1.
        turnover=abs(position-position_prior)
        fee=turnover*cost_bps/10000
        strategy_simple=position*simple_return-fee
        if strategy_simple <= -1 or simple_return <= -1:
            raise ValueError('Bankrupt or invalid open-to-open return')
        rows.append({'signal_date':signal_at,'execution_date':dates[loc+1],
                     'valuation_date':dates[loc+2],'position':position,
                     'turnover':turnover,'trade_cost_fraction':fee,
                     'asset_open_to_open_return':simple_return,
                     'strategy_log_return':float(np.log1p(strategy_simple)),
                     'benchmark_log_return':float(np.log1p(simple_return))})
        position_prior=position
    if not rows:
        raise ValueError('No fully observable open-to-open execution periods')
    result=pd.DataFrame(rows).set_index('valuation_date')
    if result.index.has_duplicates:
        raise ValueError('Duplicate execution valuation dates')
    result['strategy_equity']=np.exp(result['strategy_log_return'].cumsum())
    result['benchmark_equity']=np.exp(result['benchmark_log_return'].cumsum())
    return result


def write_snapshot(prices: pd.DataFrame, dest: str | Path, source: str) -> dict:
    """Store reproducible historical market snapshot and SHA-256 provenance."""
    ohlc = validate_ohlc(prices)
    dest=Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    payload=ohlc.rename_axis('date').to_csv(index=True, float_format='%.12g').encode()
    digest=hashlib.sha256(payload).hexdigest()
    (dest/'adjusted_ohlc.csv').write_bytes(payload)
    metadata={'source':source,'rows':len(ohlc),'first_date':str(ohlc.index[0].date()),
              'last_date':str(ohlc.index[-1].date()),'sha256':digest,
              'price_type':'dividend/split-adjusted open and close'}
    (dest/'data_manifest.json').write_text(json.dumps(metadata,indent=2))
    return metadata

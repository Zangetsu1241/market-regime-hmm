"""Versioned daily US ETF data. Providers must supply the canonical raw schema."""
from dataclasses import dataclass
import hashlib
import io
import json
import logging
from pathlib import Path
import time
from typing import Protocol
import exchange_calendars as xcals
import numpy as np
import pandas as pd
from .storage import atomic_bytes, write_json

FIELDS = ['open', 'high', 'low', 'close', 'adj_close', 'volume', 'dividends', 'splits']
LOG = logging.getLogger(__name__)


def calendar(start, end):
    return xcals.get_calendar('XNYS', start=pd.Timestamp(start)-pd.Timedelta(days=10),
                              end=pd.Timestamp(end)+pd.Timedelta(days=10))


def sessions(start, end):
    result = calendar(start, end).sessions_in_range(start, end)
    return result.tz_localize(None) if result.tz is not None else result


def latest_completed(now=None):
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError('now must be timezone-aware')
    cal = calendar(now.date()-pd.Timedelta(days=15), now.date())
    schedule = cal.schedule
    # A 30-minute publication buffer prevents treating partial/delayed bars as final.
    closes = pd.to_datetime(schedule['close'], utc=True)
    eligible = schedule.index[closes <= now-pd.Timedelta(minutes=30)]
    if not len(eligible):
        raise RuntimeError('No completed exchange session')
    return pd.Timestamp(eligible[-1]).tz_localize(None)


def validate_bars(frame):
    if not set(FIELDS).issubset(frame.columns) or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError('Canonical OHLCV/action schema and DatetimeIndex required')
    out = frame[FIELDS].astype(float).copy()
    if out.empty or out.index.tz is not None or out.index.hasnans or out.index.has_duplicates:
        raise ValueError('Empty, timezone-aware, missing or duplicate session dates')
    if not out.index.is_monotonic_increasing or not out.index.equals(out.index.normalize()):
        raise ValueError('Session dates must be normalized and increasing')
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError('Missing or nonfinite market values')
    if (out[['open','high','low','close','adj_close']] <= 0).any().any():
        raise ValueError('Prices must be positive')
    if (out[['volume','dividends','splits']] < 0).any().any() or (out.volume == 0).any():
        raise ValueError('Invalid volume or corporate actions')
    if ((out.high < out[['open','close','low']].max(axis=1)) |
            (out.low > out[['open','close','high']].min(axis=1))).any():
        raise ValueError('Inconsistent OHLC bounds')
    expected = sessions(out.index[0], out.index[-1])
    if not out.index.equals(expected):
        raise ValueError('Missing or unexpected exchange sessions; never forward-fill prices')
    if (np.log(out.adj_close).diff().abs().dropna() > np.log(1.5)).any():
        raise ValueError('Suspicious adjusted-price jump; inspect corporate actions')
    return out


class Provider(Protocol):
    name: str
    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame: ...


@dataclass
class YahooProvider:
    name: str = 'yahoo-yfinance'
    attempts: int = 3

    def fetch(self, ticker, start, end):
        import yfinance as yf
        yf.set_tz_cache_location(str(Path('data-cache/vendor').resolve()))
        if ticker not in {'SPY', 'QQQ', 'IWM'}:
            raise ValueError('Supported research universe: SPY, QQQ, IWM')
        failure = None
        for attempt in range(self.attempts):
            try:
                data = yf.download(ticker, start=start, end=end, auto_adjust=False,
                    actions=True, progress=False, threads=False, timeout=20,
                    multi_level_index=False)
                if data is None or data.empty:
                    raise RuntimeError('Provider returned no bars')
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data = data.rename(columns={'Open':'open','High':'high','Low':'low',
                    'Close':'close','Adj Close':'adj_close','Volume':'volume',
                    'Dividends':'dividends','Stock Splits':'splits'})
                data.index = pd.DatetimeIndex(data.index).tz_localize(None).normalize()
                # Yahoo OHLC and dividends are already split-adjusted. Undo future
                # splits within this snapshot to obtain a consistent share ledger.
                split = data.splits.replace(0, 1)
                future = split.iloc[::-1].cumprod().iloc[::-1] / split
                data[['open','high','low','close','dividends']] = data[
                    ['open','high','low','close','dividends']].mul(future, axis=0)
                data['volume'] = data.volume / future
                return validate_bars(data)
            except Exception as exc:
                failure = exc
                LOG.warning('Provider attempt %s/%s failed for %s: %s',
                            attempt+1, self.attempts, ticker, type(exc).__name__)
                if attempt+1 < self.attempts:
                    time.sleep(2**attempt)
        raise RuntimeError(f'Yahoo failed for {ticker} after {self.attempts} attempts') from failure


@dataclass
class CSVProvider:
    directory: Path
    name: str = 'canonical-csv'

    def fetch(self, ticker, start, end):
        frame = pd.read_csv(Path(self.directory)/f'{ticker}.csv', index_col='date', parse_dates=True)
        frame = frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index < pd.Timestamp(end))]
        return validate_bars(frame)


def dataset(provider, ticker, start, end, cache, refresh=False):
    """Immutable content blobs plus atomic request pointers; corrupted caches fail closed."""
    request = {'provider':provider.name, 'ticker':ticker, 'start':start, 'end':end}
    if isinstance(provider, CSVProvider):
        request['csv_sha256'] = hashlib.sha256((Path(provider.directory)/f'{ticker}.csv').read_bytes()).hexdigest()
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    root = Path(cache)
    pointer = root/'requests'/f'{key}.json'
    if pointer.exists() and not refresh:
        meta = json.loads(pointer.read_text())
        payload = (root/'objects'/f"{meta['sha256']}.csv").read_bytes()
        if hashlib.sha256(payload).hexdigest() != meta['sha256']:
            raise RuntimeError('Dataset checksum mismatch')
        frame = pd.read_csv(io.BytesIO(payload), index_col='date', parse_dates=True)
        return validate_bars(frame), meta
    frame = validate_bars(provider.fetch(ticker, start, end))
    wanted = sessions(start, pd.Timestamp(end)-pd.Timedelta(days=1))
    if not frame.index.equals(wanted):
        raise ValueError('Provider response does not cover requested sessions')
    payload = frame.rename_axis('date').to_csv(float_format='%.17g').encode()
    digest = hashlib.sha256(payload).hexdigest()
    meta = {**request, 'sha256':digest, 'rows':len(frame),
            'downloaded_at':pd.Timestamp.now(tz='UTC').isoformat(),
            'first_session':str(frame.index[0].date()), 'last_session':str(frame.index[-1].date()),
            'schema':1, 'adjustments':'as-traded OHLC/actions; vendor total-return adjusted close'}
    atomic_bytes(root/'objects'/f'{digest}.csv', payload)
    write_json(root/'objects'/f'{digest}.json', meta)
    write_json(pointer, meta)
    return frame, meta

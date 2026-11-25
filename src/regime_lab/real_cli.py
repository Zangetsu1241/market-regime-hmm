"""Reproducible real-data SPY research + paper execution. Never submits orders."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .data import make_features
from .research import Config, walk_forward, performance
from .real_market import (download_yahoo_ohlc, executable_next_open,
                          load_adjusted_ohlc, write_snapshot)


def main(argv=None):
    p=argparse.ArgumentParser(description='Paper-only HMM regime evaluation on real adjusted daily OHLC')
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--download', action='store_true', help='Download SPY adjusted OHLC from Yahoo Finance')
    source.add_argument('--ohlc-csv', help='Local date,open,close CSV with consistently adjusted prices')
    p.add_argument('--ticker', default='SPY', help='Only used with --download')
    p.add_argument('--start', default='2005-01-01')
    p.add_argument('--end', help='Exclusive date for Yahoo download')
    p.add_argument('--train-size', type=int, default=750)
    p.add_argument('--test-block', type=int, default=63)
    p.add_argument('--cost-bps', type=float, default=5.)
    p.add_argument('--output', default='results-real')
    p.add_argument('--max-stale-days', type=int, default=7,
                   help='Latest adjusted close must be no older than this for a fresh paper signal')
    p.add_argument('--historical-only', action='store_true',
                   help='Allow older snapshots for backtests; never generate a fresh signal')
    args=p.parse_args(argv)
    if args.max_stale_days < 0:
        p.error('--max-stale-days must be nonnegative')
    source_name=f'Yahoo Finance adjusted daily bars: {args.ticker.upper()}' if args.download else f'User-supplied adjusted OHLC: {args.ohlc_csv}'
    prices=(download_yahoo_ohlc(args.ticker.upper(),args.start,args.end) if args.download
            else load_adjusted_ohlc(args.ohlc_csv))
    latest=pd.Timestamp(prices.index[-1]).date()
    age=(pd.Timestamp.now(tz='UTC').date()-latest).days
    if age < 0:
        raise RuntimeError('Latest data date is in the future')
    if age>args.max_stale_days and not args.historical_only:
        raise RuntimeError(f'Data are stale ({age} calendar days); use --historical-only for research')
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    manifest=write_snapshot(prices,out,source_name)
    # Existing model is fitted on adjusted close only; open is reserved for execution.
    close=pd.DataFrame({'close':prices.close},index=prices.index)
    cfg=Config(train_size=args.train_size,test_block=args.test_block,
               cost_bps=args.cost_bps)
    original=walk_forward(close,cfg)
    paper=executable_next_open(original,prices,cost_bps=args.cost_bps)
    paper.to_csv(out/'paper_open_to_open.csv')
    original.to_csv(out/'model_signals_raw.csv')
    results={'asset':args.ticker.upper() if args.download else 'CSV',
             'source':source_name,'data':manifest,
             'strategy':performance(paper.strategy_log_return),
             'benchmark':performance(paper.benchmark_log_return),
             'trades':int(paper.turnover.sum()),'observations':len(paper),
             'caveats':['Research simulation only; not broker executions',
                        'Trading is modeled at next-day adjusted open and valued next open',
                        'Adjusted OHLC must be point-in-time consistent; Yahoo revisions can alter history',
                        'No spread, impact, financing, tax, or settlement simulation']}
    (out/'paper_metrics.json').write_text(json.dumps(results,indent=2))
    if not args.historical_only:
        features=make_features(close)
        last=original.iloc[-1]
        signal={'generated_at_utc':pd.Timestamp.now(tz='UTC').isoformat(),
                'latest_market_date':str(latest),
                'latest_model_signal_date':str(pd.Timestamp(last.signal_date).date()),
                'model_position':int(last.position),
                'disposition':'RESEARCH_ONLY_NO_ORDERS',
                'warning':'The latest walk-forward signal may lag the latest bar; this file is not an actionable order.'}
        (out/'signal_status.json').write_text(json.dumps(signal,indent=2))
    print(json.dumps(results,indent=2))
    return results

if __name__=='__main__':main()

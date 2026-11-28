"""Reproducible research and isolated paper-only command line."""
import argparse
import json
import logging
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from .experiments import evaluate
from .market_data import CSVProvider, YahooProvider, dataset, latest_completed
from .paper import update_paper
from .pipeline import ModelConfig
from .portfolio import Costs
from .storage import write_json


def main(argv=None):
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description='Research and PAPER ONLY simulation; no brokerage support')
    parser.add_argument('command', choices=['download','research','paper'])
    parser.add_argument('--config', default='configs/research.json')
    parser.add_argument('--cache', default=os.getenv('REGIME_CACHE','data-cache'))
    parser.add_argument('--output', default=os.getenv('REGIME_OUTPUT','results-system'))
    parser.add_argument('--paper-state', default=os.getenv('REGIME_PAPER_STATE','paper-state'))
    parser.add_argument('--csv-directory')
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--release-test', action='store_true')
    parser.add_argument('--ticker', choices=['SPY','QQQ','IWM'])
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(out/'run.log', encoding='utf-8')])
    spec = json.loads(Path(args.config).read_text())
    provider = CSVProvider(Path(args.csv_directory)) if args.csv_directory else YahooProvider()
    tickers = [args.ticker] if args.ticker else spec['tickers']
    failed = []
    for ticker in tickers:
        try:
            end = spec['end']
            if args.command == 'paper':
                end = str((latest_completed()+pd.Timedelta(days=1)).date())
            bars, meta = dataset(provider, ticker, spec['start'], end, args.cache, args.refresh)
            if args.command == 'research':
                result = evaluate(bars, spec, out/ticker, meta, args.release_test)
                logging.info('%s research complete: %s', ticker, result.get('selected_variant','validation only'))
            elif args.command == 'paper':
                update_paper(bars, Path(args.paper_state)/ticker, ticker,
                    ModelConfig(**spec['model']), Costs(**spec['costs']), dataset_hash=meta['sha256'])
            write_json(out/f'{ticker}-status.json', {'status':'OK', 'command':args.command, 'data':meta})
        except Exception as exc:
            failed.append(ticker)
            logging.exception('%s failed closed', ticker)
            write_json(out/f'{ticker}-status.json', {'status':'FAILED_CLOSED',
                'command':args.command, 'type':type(exc).__name__, 'reason':str(exc)})
    if failed:
        raise SystemExit(f'Failed assets: {", ".join(failed)}. Previous paper economic state preserved.')


if __name__ == '__main__':
    main()

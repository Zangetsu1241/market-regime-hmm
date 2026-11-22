"""Run market regime experiment with synthetic or user supplied daily closes."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import synthetic_prices,load_prices
from .research import Config,walk_forward,performance

def main(argv=None):
    p=argparse.ArgumentParser(description='Leakage-aware market regime research')
    p.add_argument('--csv',help='CSV with date and positive close columns')
    p.add_argument('--output',default='results')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--train-size',type=int,default=500)
    p.add_argument('--test-block',type=int,default=100)
    p.add_argument('--cost-bps',type=float,default=5.0)
    p.add_argument('--states',type=int,default=2)
    p.add_argument('--synthetic-days',type=int,default=1800)
    args=p.parse_args(argv)
    prices=load_prices(args.csv) if args.csv else synthetic_prices(args.synthetic_days,args.seed)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    cfg=Config(train_size=args.train_size,test_block=args.test_block,n_states=args.states,cost_bps=args.cost_bps,seed=args.seed)
    result=walk_forward(prices,cfg)
    stats={'model':performance(result['strategy_return']),'buy_and_hold':performance(result['benchmark_return']),
           'observations':len(result),'trades':int(result['turnover'].sum()),
           'exposure':float(result['position'].mean()),'dataset':'external CSV' if args.csv else 'synthetic',
           'assumptions':{'signal':'after close, same-close execution assumed for next close-to-close exposure',
                          'returns':'log','cost_bps_per_position_unit':args.cost_bps,
                          'selection':'expanding-window refit; no evaluation metrics used to select model'}}
    result.to_csv(out/'walk_forward.csv')
    (out/'metrics.json').write_text(json.dumps(stats,indent=2))
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    result[['strategy_equity','benchmark_equity']].plot(ax=axes[0],title='Out-of-sample equity (illustrative)')
    result['risk_score'].plot(ax=axes[1],title='Filtered market risk score',color='tab:orange')
    axes[1].set_ylabel('Risk score')
    fig.tight_layout()
    fig.savefig(out/'regimes.png',dpi=160)
    plt.close(fig)
    print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
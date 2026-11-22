"""Expanding-window HMM evaluation with next-day positions and costs."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .hmm import GaussianHMM
from .data import make_features

@dataclass
class Config:
    train_size:int=500
    test_block:int=100
    n_states:int=2
    cost_bps:float=5.0
    seed:int=42
    max_iter:int=120
    n_init:int=3

class Standardizer:
    def fit(self,x):
        self.mean=np.mean(x,axis=0)
        self.std=np.std(x,axis=0)
        self.std=np.where(self.std<1e-10,1.0,self.std)
        return self
    def transform(self,x):return (x-self.mean)/self.std

def walk_forward(prices,cfg=Config()):
    """Refit using only past observations. Signal at close t targets t+1 return."""
    if cfg.n_states<2 or cfg.train_size<cfg.n_states*10 or cfg.test_block<1 or cfg.cost_bps<0:
        raise ValueError('invalid configuration')
    feat=make_features(prices)
    if len(feat)<=cfg.train_size+1:raise ValueError('Need more observations than train_size+1')
    X=feat[['return','volatility']].to_numpy()
    ret=feat['return'].to_numpy()
    rows=[]
    for start in range(cfg.train_size,len(feat)-1,cfg.test_block):
        end=min(start+cfg.test_block,len(feat)-1)
        scaler=Standardizer().fit(X[:start])
        x_train=scaler.transform(X[:start])
        model=GaussianHMM(cfg.n_states,n_init=cfg.n_init,max_iter=cfg.max_iter,random_state=cfg.seed).fit(x_train)
        train_prob=model.filter(x_train)
        risks=(train_prob.T@(ret[:start]**2))/np.maximum(train_prob.sum(axis=0),1e-8)
        weights=np.argsort(np.argsort(risks))/(cfg.n_states-1)
        prob=model.filter(scaler.transform(X[start:end]),initial=train_prob[-1])
        risk_score=prob@weights
        position=(risk_score<0.5).astype(float)
        for j,t in enumerate(range(start,end)):
            rows.append((feat.index[t],feat.index[t+1],float(ret[t+1]),position[j],risk_score[j],int(np.argmax(prob[j])),start))
    out=pd.DataFrame(rows,columns=['signal_date','return_date','next_return','position','risk_score','state','train_samples']).set_index('return_date')
    out['prior_position']=out['position'].shift(1).fillna(0)
    out['turnover']=(out['position']-out['prior_position']).abs()
    out['strategy_return']=out['position']*out['next_return']-out['turnover']*cfg.cost_bps/10000
    out['benchmark_return']=out['next_return']
    out['strategy_equity']=np.exp(out['strategy_return'].cumsum())
    out['benchmark_equity']=np.exp(out['benchmark_return'].cumsum())
    return out

def performance(returns,periods=252):
    """Daily log returns; geometric annual return and log-return Sharpe."""
    x=np.asarray(returns,float)
    if x.ndim!=1 or len(x)==0 or not np.isfinite(x).all():raise ValueError('invalid returns')
    eq=np.exp(np.cumsum(x))
    maxeq=np.maximum.accumulate(np.r_[1.,eq])[1:]
    vol=float(np.std(x,ddof=1)) if len(x)>1 else 0.
    return {'total_return':float(eq[-1]-1),'annual_return':float(np.exp(x.mean()*periods)-1),
            'annual_volatility':float(vol*np.sqrt(periods)),
            'sharpe':float(x.mean()/vol*np.sqrt(periods)) if vol>0 else 0.,
            'max_drawdown':float(np.min(eq/maxeq-1))}
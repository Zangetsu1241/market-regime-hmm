import numpy as np
import pytest
from regime_lab.hmm import GaussianHMM
from regime_lab.data import synthetic_prices,make_features,load_prices
from regime_lab.research import Config,walk_forward,performance

def test_hmm_probabilities_and_likelihood():
    rng=np.random.default_rng(10)
    x=np.r_[rng.normal(-3,.5,(150,2)),rng.normal(3,.5,(150,2))]
    m=GaussianHMM(n_init=2,max_iter=80).fit(x)
    p=m.filter(x)
    assert np.allclose(p.sum(axis=1),1,atol=1e-10)
    assert np.all(p>=0)
    assert np.isfinite(m.score(x))
    assert m.log_likelihood_>m.history_[0]

def test_filter_is_causal():
    x=np.random.default_rng(3).normal(size=(150,2))
    m=GaussianHMM(n_init=1,max_iter=30).fit(x)
    assert np.allclose(m.filter(x)[:60],m.filter(x[:60]),atol=1e-10)

def test_filter_continuation():
    x=np.random.default_rng(4).normal(size=(150,2))
    m=GaussianHMM(n_init=1,max_iter=30).fit(x)
    together=m.filter(x)
    continued=m.filter(x[100:],initial=m.filter(x[:100])[-1])
    assert np.allclose(together[100:],continued,atol=1e-10)

def test_walk_forward_and_alignment():
    prices=synthetic_prices(480)
    r=walk_forward(prices,Config(train_size=180,test_block=70,max_iter=25,n_init=1))
    assert len(r)>200
    assert np.all(r['signal_date'].values<r.index.values)
    assert np.all(r['train_samples'].values<=180+((np.arange(len(r))//70)*70))
    assert np.allclose(r['strategy_return'],r['position']*r['next_return']-r['turnover']*.0005)
    assert np.isfinite(r[['strategy_equity','benchmark_equity']]).all().all()

def test_prefix_invariance_first_block():
    p=synthetic_prices(550)
    cfg=Config(train_size=180,test_block=60,max_iter=25,n_init=1)
    a=walk_forward(p.iloc[:350],cfg)
    b=walk_forward(p,cfg)
    assert np.allclose(a['risk_score'].iloc[:60],b['risk_score'].iloc[:60])

def test_no_invalid_inputs(tmp_path):
    with pytest.raises(ValueError): make_features(synthetic_prices(100),span=1)
    with pytest.raises(ValueError): walk_forward(synthetic_prices(200),Config(train_size=500))
    file=tmp_path/'bad.csv'
    file.write_text('date,close\n2022-01-01,12\n2022-01-01,15\n')
    with pytest.raises(ValueError):load_prices(file)

def test_performance_zero():
    v=performance(np.zeros(100))
    assert v['sharpe']==0 and v['max_drawdown']==0
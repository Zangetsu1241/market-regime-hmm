from dataclasses import replace
import itertools
import json
import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from regime_lab.hmm import GaussianHMM
from regime_lab.market_data import sessions, latest_completed, validate_bars, dataset
from regime_lab.pipeline import ModelConfig, signals, features
from regime_lab.portfolio import Costs, backtest, rebalance, metrics
from regime_lab.paper import update_paper
from regime_lab.storage import exclusive_lock


def test_yahoo_reverses_splits_after_requested_end(monkeypatch):
    import yfinance
    from regime_lab.market_data import YahooProvider
    dates = sessions('2020-01-02','2020-01-10')[:6]
    vendor = pd.DataFrame({'Open':50.,'High':51.,'Low':49.,'Close':50.,
        'Adj Close':50.,'Volume':200.,'Dividends':0.,'Stock Splits':0.},index=dates)
    vendor.loc[dates[-1],'Stock Splits'] = 2.
    vendor.loc[dates[1],'Dividends'] = .5
    monkeypatch.setattr(yfinance,'download',lambda *a,**k:vendor.copy())
    monkeypatch.setattr(yfinance,'set_tz_cache_location',lambda *a:None)
    got = YahooProvider().fetch('SPY',str(dates[0].date()),str(dates[4].date()))
    assert len(got) == 4
    assert np.allclose(got.open,100)
    assert np.allclose(got.volume,100)
    assert got.dividends.iloc[1] == 1.


def fixture_bars(n=400):
    dates = sessions('2020-01-01','2023-12-31')[:n]
    rng = np.random.default_rng(8)
    close = 100*np.exp(np.cumsum(rng.normal(.0003,.01,n)))
    return pd.DataFrame({'open':close*.998, 'high':close*1.01, 'low':close*.99,
        'close':close, 'adj_close':close, 'volume':1e6, 'dividends':0., 'splits':0.}, index=dates)


@pytest.mark.parametrize('kwargs', [{'n_init':0}, {'max_iter':0}, {'min_var':0}, {'tol':float('nan')}, {'n_states':1}])
def test_hmm_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        GaussianHMM(**kwargs).fit(np.ones((100,2)))


def test_filter_invalid_initial():
    m = GaussianHMM(n_init=1,max_iter=5).fit(np.random.default_rng(1).normal(size=(100,2)))
    for initial in ([2,-1], [np.nan,np.nan], [1,0,0]):
        with pytest.raises(ValueError):
            m.filter(np.zeros((1,2)), initial=initial)


def test_forward_matches_enumerated_paths():
    m = GaussianHMM(n_init=1,max_iter=5).fit(np.random.default_rng(3).normal(size=(100,2)))
    x = np.array([[0.,.1],[.2,.3],[-.1,0.]])
    b = m._emissions(x)
    paths = [np.log(m.start_[s[0]])+sum(b[t,s[t]] for t in range(3))+
             sum(np.log(m.trans_[s[t-1],s[t]]) for t in range(1,3))
             for s in itertools.product(range(2), repeat=3)]
    assert m.score(x) == pytest.approx(logsumexp(paths))


def test_calendar_missing_duplicate_and_holiday():
    b = fixture_bars()
    validate_bars(b)
    assert pd.Timestamp('2020-01-01') not in b.index
    with pytest.raises(ValueError,match='sessions'):
        validate_bars(b.drop(b.index[20]))
    with pytest.raises(ValueError,match='duplicate'):
        validate_bars(pd.concat([b.iloc[:1],b]))
    assert latest_completed('2024-11-29T18:20:00Z') == pd.Timestamp('2024-11-27')
    assert latest_completed('2024-11-29T18:31:00Z') == pd.Timestamp('2024-11-29')


def test_cache_and_failed_refresh_preserve_snapshot(tmp_path):
    class Fake:
        name = 'fixture-only'
        calls = 0
        fail = False
        def fetch(self, *args):
            self.calls += 1
            if self.fail:
                raise RuntimeError('offline')
            return fixture_bars(100)
    provider = Fake()
    b = fixture_bars(100)
    args = (provider,'SPY',str(b.index[0].date()),str((b.index[-1]+pd.Timedelta(days=1)).date()),tmp_path)
    _, meta = dataset(*args)
    dataset(*args)
    assert provider.calls == 1
    provider.fail = True
    with pytest.raises(RuntimeError):
        dataset(*args, refresh=True)
    _, again = dataset(*args)
    assert again == meta
    (tmp_path/'objects'/f"{meta['sha256']}.csv").write_text('corrupt')
    with pytest.raises(RuntimeError,match='checksum'):
        dataset(*args)


@pytest.mark.parametrize('window',[None,100])
def test_future_perturbation_all_earlier_signals(window):
    b = fixture_bars(260)
    cfg = ModelConfig(train_size=100, refit_every=50, window=window,max_iter=8,n_init=1)
    a, artifacts = signals(b,cfg)
    changed = b.copy()
    changed.loc[changed.index[210:],['open','high','low','close','adj_close']] *= 1.05
    c, _ = signals(changed,cfg)
    pd.testing.assert_frame_equal(a.loc[a.index < b.index[210]],c.loc[c.index < b.index[210]])
    assert a.index[-1] == b.index[-1]
    assert (a.fit_through < a.index).all()
    assert artifacts[0]['mean'] == pytest.approx(features(b).iloc[:100].mean().to_numpy())


def test_fractional_rebalance_self_financing():
    cash, shares, fee, _ = rebalance(100.,0.,10.,.5,Costs(10,0,0))
    equity = cash+shares*10
    assert equity == pytest.approx(100-fee)
    assert shares*10/equity == pytest.approx(.5)
    assert cash >= 0
    cash, shares, fee, _ = rebalance(cash,shares,10.,1.,Costs(10,0,0))
    assert cash == pytest.approx(0,abs=1e-12)


def simple_decisions(b):
    return pd.DataFrame({'binary':1., 'risk_scaled':.5, 'momentum':.1,
        'volatility':.01, 'state':0},index=b.index[:-1])


def test_next_open_and_split_dividend_accounting():
    b = fixture_bars(4)
    b[['open','close']] = 100.
    b.loc[b.index[2]:,['open','close']] = 50.
    b.loc[b.index[2],'splits'] = 2.
    b.loc[b.index[3],['open','close']] = 49.
    b.loc[b.index[3],'dividends'] = 1.
    out = backtest(b,simple_decisions(b),costs=Costs(0,0,0),initial_cash=100)
    assert out.index[0] == b.index[1]
    assert np.allclose(out.equity,100)
    assert out.shares.iloc[1] == 2
    assert metrics(out)['total_return'] == 0


def test_signal_does_not_capture_overnight_entry_gap():
    b = fixture_bars(3)
    b[['open','close']] = 100.
    b.loc[b.index[1]:,['open','close']] = 110.
    out = backtest(b,simple_decisions(b),costs=Costs(0,0,0))
    assert np.allclose(out['return'],0)


def test_paper_idempotence_failure_and_cash(tmp_path,monkeypatch):
    import regime_lab.paper as paper
    b = fixture_bars(160)
    def fake_signals(frame,config):
        row = pd.DataFrame({'binary':[1.], 'risk_score':[.2]},index=frame.index[-1:])
        return row, [{'converged':True,'fit_through':str(frame.index[-2].date())}]
    monkeypatch.setattr(paper,'signals',fake_signals)
    now = b.index[-2].tz_localize('UTC')+pd.Timedelta(hours=23)
    update_paper(b.iloc[:-1],tmp_path,now=now,config=ModelConfig(train_size=100))
    now = b.index[-1].tz_localize('UTC')+pd.Timedelta(hours=23)
    state = update_paper(b,tmp_path,now=now,config=ModelConfig(train_size=100))
    payload = (tmp_path/'state.json').read_bytes()
    again = update_paper(b,tmp_path,now=now,config=ModelConfig(train_size=100))
    assert len(again['fills']) == 1
    assert state['shares'] > 0 and state['cash'] >= 0
    assert payload == (tmp_path/'state.json').read_bytes()
    with pytest.raises(RuntimeError,match='stale'):
        update_paper(b.iloc[:-1],tmp_path,now=now,config=ModelConfig(train_size=100))
    assert payload == (tmp_path/'state.json').read_bytes()
    assert json.loads((tmp_path/'health.json').read_text())['status'] == 'FAILED_CLOSED'
    with exclusive_lock(tmp_path/'run.lock'):
        with pytest.raises(RuntimeError,match='lock'):
            update_paper(b,tmp_path,now=now)


def test_unconverged_paper_model_fails_closed(tmp_path,monkeypatch):
    import regime_lab.paper as paper
    b = fixture_bars(160)
    monkeypatch.setattr(paper,'signals',lambda frame,cfg:(
        pd.DataFrame({'binary':[1.],'risk_score':[.1]},index=frame.index[-1:]),
        [{'converged':False,'fit_through':str(frame.index[-2].date())}]))
    with pytest.raises(RuntimeError,match='converged'):
        update_paper(b,tmp_path,now=b.index[-1].tz_localize('UTC')+pd.Timedelta(hours=23),
                     config=ModelConfig(train_size=100))
    assert not (tmp_path/'state.json').exists()

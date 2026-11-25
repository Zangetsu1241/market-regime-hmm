import json
import numpy as np
import pandas as pd
import pytest
from regime_lab.real_market import validate_ohlc,load_adjusted_ohlc,executable_next_open,write_snapshot


def bars(n=800):
    dates=pd.bdate_range('2018-01-01',periods=n)
    opens=100*np.exp(.0003*np.arange(n))
    return pd.DataFrame({'open':opens,'close':opens*1.001},index=dates)


def signals(prices):
    d=prices.index
    return pd.DataFrame({'signal_date':d[10:20], 'position':[1,1,0,0,1,1,0,0,1,1]},index=d[11:21])


def test_execution_uses_two_future_opens_not_signal_close():
    p=bars()
    got=executable_next_open(signals(p),p,cost_bps=0)
    row=got.iloc[0]
    assert row.execution_date==p.index[11]
    assert got.index[0]==p.index[12]
    assert np.isclose(row.asset_open_to_open_return,p.open.iloc[12]/p.open.iloc[11]-1)


def test_signal_close_change_does_not_change_same_execution_return():
    p=bars();s=signals(p)
    a=executable_next_open(s,p)
    modified=p.copy()
    modified.loc[modified.index[10],'close']*=1.05
    b=executable_next_open(s,modified)
    pd.testing.assert_frame_equal(a,b)


def test_turnover_and_costs():
    p=bars();a=executable_next_open(signals(p),p,cost_bps=10)
    assert np.allclose(a.turnover.to_numpy(),[1,0,1,0,1,0,1,0,1,0])
    assert np.isclose(a.iloc[0].trade_cost_fraction,.001)
    assert np.isclose(a.iloc[0].strategy_log_return,
                      np.log1p(a.iloc[0].asset_open_to_open_return-.001))


def test_cannot_score_without_following_open():
    p=bars();s=signals(p).iloc[-1:]
    s['signal_date']=p.index[-2]
    s.index=p.index[-1:]
    with pytest.raises(ValueError,match='No fully observable'):
        executable_next_open(s,p)


def test_invalid_prices_and_missing_dates():
    p=bars();p.iloc[20,0]=np.nan
    with pytest.raises(ValueError,match='positive finite'):
        validate_ohlc(p)
    p=bars().drop(bars().index[11])
    with pytest.raises(ValueError,match='Missing signal'):
        executable_next_open(signals(bars()),p)


def test_snapshot_and_csv_load(tmp_path):
    p=bars();meta=write_snapshot(p,tmp_path,'synthetic-test')
    assert meta['sha256']==__import__('hashlib').sha256((tmp_path/'adjusted_ohlc.csv').read_bytes()).hexdigest()
    back=load_adjusted_ohlc(tmp_path/'adjusted_ohlc.csv')
    assert np.allclose(back.open,p.open)
    assert json.loads((tmp_path/'data_manifest.json').read_text())['rows']==800


def test_reject_unsorted_ohlc_csv(tmp_path):
    df=bars()
    df=df.iloc[::-1].rename_axis('date')
    path=tmp_path/'bad.csv';df.to_csv(path)
    with pytest.raises(ValueError,match='increasing'):
        load_adjusted_ohlc(path)


def test_future_prices_do_not_change_earlier_executions():
    p=bars();s=signals(p)
    a=executable_next_open(s,p)
    later=p.copy();later.loc[later.index[100:],'open']*=1.1
    b=executable_next_open(s,later)
    pd.testing.assert_frame_equal(a,b)

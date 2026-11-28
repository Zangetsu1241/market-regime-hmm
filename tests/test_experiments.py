import json
import numpy as np
import pandas as pd
from regime_lab.experiments import evaluate
from test_system import fixture_bars


def test_selection_never_receives_test_rows(tmp_path, monkeypatch):
    import regime_lab.experiments as experiment
    bars = fixture_bars(400)
    boundary = bars.index[300]
    observed = []
    def fixture_signals(data, config):
        observed.append(data.index[-1])
        index = data.index[120:]
        frame = pd.DataFrame({'binary':1., 'risk_scaled':.5, 'momentum':.1,
            'volatility':.01, 'state':0, 'fit_through':index-pd.Timedelta(days=1)},index=index)
        return frame, [{'converged':True}]
    monkeypatch.setattr(experiment, 'signals', fixture_signals)
    spec = {'validation_start':str(bars.index[200].date()), 'test_start':str(boundary.date()),
            'model':{'train_size':100}, 'costs':{}}
    result = evaluate(bars,spec,tmp_path,{'source':'TEST FIXTURE'},release_test=False)
    assert len(observed) == 6 and all(date < boundary for date in observed)
    assert result['test'] == 'not evaluated'
    assert not (tmp_path/'results.json').exists()
    first = json.loads((tmp_path/'selection.json').read_text())
    changed = bars.copy()
    changed.loc[boundary:,['open','high','low','close','adj_close']] *= 1.2
    evaluate(changed,spec,tmp_path,{'source':'TEST FIXTURE'},release_test=False)
    assert first == json.loads((tmp_path/'selection.json').read_text())


def test_complete_research_report_contract(tmp_path,monkeypatch):
    import regime_lab.experiments as experiment
    bars = fixture_bars(400)
    def fixture_signals(data, config):
        index = data.index[120:]
        return pd.DataFrame({'binary':1.,'risk_scaled':.5,'momentum':.1,
            'volatility':.01,'state':0},index=index), [{'converged':True}]
    monkeypatch.setattr(experiment,'signals',fixture_signals)
    spec = {'validation_start':str(bars.index[200].date()), 'test_start':str(bars.index[300].date()),
            'model':{'train_size':100}, 'costs':{}}
    report = evaluate(bars,spec,tmp_path,{'source':'TEST FIXTURE'},release_test=True)
    assert len(report['test']) == 5
    assert report['test']['binary']['observations'] == 100
    assert (tmp_path/'test-equity.png').stat().st_size > 1000
    assert json.loads((tmp_path/'provenance.json').read_text())['final_test_released']

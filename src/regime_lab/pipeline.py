"""Causal signals, train-only scalers and auditable rolling/expanding fits."""
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from .hmm import GaussianHMM
from .research import Standardizer


@dataclass(frozen=True)
class ModelConfig:
    states: int = 2
    train_size: int = 504
    refit_every: int = 63
    window: int | None = None
    max_iter: int = 150
    n_init: int = 3
    seed: int = 42
    target_vol: float = .12

    def validate(self):
        if (self.states not in (2,3,4) or self.train_size < 100 or self.refit_every < 1
                or (self.window is not None and self.window < self.train_size)
                or not np.isfinite(self.target_vol) or self.target_vol <= 0):
            raise ValueError('Invalid model configuration')


def features(bars):
    r = np.log(bars.adj_close).diff()
    # All quantities are dimensionless and trailing; no absolute adjusted-price level.
    out = pd.DataFrame({'return':r, 'volatility':r.rolling(20).std(),
        'momentum':r.rolling(20).sum(),
        'range':np.log(bars.high/bars.low)}, index=bars.index).dropna()
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError('Nonfinite features')
    return out


def signals(bars, config=ModelConfig()):
    config.validate()
    feat = features(bars)
    if len(feat) <= config.train_size:
        raise ValueError('Insufficient training history')
    x = feat.to_numpy()
    returns = feat['return'].to_numpy()
    rows, artifacts = [], []
    for start in range(config.train_size, len(feat), config.refit_every):
        lo = 0 if config.window is None else max(0, start-config.window)
        stop = min(start+config.refit_every, len(feat))
        scaler = Standardizer().fit(x[lo:start])
        train = scaler.transform(x[lo:start])
        model = GaussianHMM(config.states, n_init=config.n_init, max_iter=config.max_iter,
                            random_state=config.seed).fit(train)
        posterior = model.filter(train)
        risk = posterior.T @ (returns[lo:start]**2) / np.maximum(posterior.sum(axis=0), 1e-12)
        order = np.argsort(risk, kind='stable')
        rank = np.argsort(order)
        probs = model.filter(scaler.transform(x[start:stop]), initial=posterior[-1])
        score = probs @ (rank/(config.states-1))
        artifact = {'config':asdict(config), 'feature_names':list(feat.columns),
            'fit_through':str(feat.index[start-1].date()), 'fit_start':str(feat.index[lo].date()),
            'samples':start-lo, 'mean':scaler.mean.tolist(), 'std':scaler.std.tolist(),
            'start':model.start_.tolist(), 'transition':model.trans_.tolist(),
            'means':model.means_.tolist(), 'variances':model.vars_.tolist(),
            'risk_order':order.tolist(), 'risk':risk.tolist(),
            'log_likelihood':model.log_likelihood_, 'iterations':model.n_iter_,
            'converged':bool(model.converged_), 'history':model.history_,
            'occupancy':posterior.mean(axis=0).tolist()}
        artifacts.append(artifact)
        for j, t in enumerate(range(start, stop)):
            row = {'date':feat.index[t], 'risk_score':float(score[j]),
                'binary':float(score[j] < .5),
                'risk_scaled':float(np.clip((1-score[j])*config.target_vol /
                    max(feat.volatility.iloc[t]*np.sqrt(252), 1e-8), 0, 1)),
                'state':int(rank[np.argmax(probs[j])]), 'fit_through':feat.index[start-1],
                'converged':bool(model.converged_), 'volatility':feat.volatility.iloc[t],
                'momentum':feat.momentum.iloc[t]}
            row.update({f'p_risk_{k}':float(probs[j,order[k]]) for k in range(config.states)})
            rows.append(row)
    return pd.DataFrame(rows).set_index('date'), artifacts

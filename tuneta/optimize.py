import optuna
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import pandas_ta as pta
from finta import TA as fta
import talib as tta


def _weighted_pearson(y, y_pred, w):
    """Calculate the weighted Pearson correlation coefficient."""
    with np.errstate(divide='ignore', invalid='ignore'):
        y_pred_demean = y_pred - np.average(y_pred, weights=w)
        y_demean = y - np.average(y, weights=w)
        corr = ((np.sum(w * y_pred_demean * y_demean) / np.sum(w)) /
                np.sqrt((np.sum(w * y_pred_demean ** 2) *
                         np.sum(w * y_demean ** 2)) /
                        (np.sum(w) ** 2)))
    if np.isfinite(corr):
        return np.abs(corr)
    return 0.


def _weighted_spearman(y, y_pred, w):
    """Calculate the weighted Spearman correlation coefficient."""
    y_pred_ranked = np.apply_along_axis(rankdata, 0, y_pred)
    y_ranked = np.apply_along_axis(rankdata, 0, y)
    return _weighted_pearson(y_pred_ranked, y_ranked, w)


def objective(self, trial, X, y, weights):
    # import joblib
    # joblib.dump([X, y], 'state/Xy.job')
    # Xold, yold = joblib.load('state/Xy.job')
    # y == yold
    res = eval(self.function)
    if isinstance(res, tuple):
        res = pd.DataFrame(res).T
    res = pd.DataFrame(res, index=X.index)  # Convert to dataframe
    res = res.iloc[:, self.idx]  # Only tune on one column (maximize)
    res_y = res.reindex(y.index).to_numpy().flatten()  # Reduce to y and convert to array
    self.res_y.append(res_y)
    ws = _weighted_spearman(np.array(y), res_y, weights)
    return ws


def trial(self, trial, X):
    res = eval(self.function)
    if isinstance(res, tuple):
        res = pd.DataFrame(res).T
    res = pd.DataFrame(res, index=X.index)
    return res


class Optimize():

    def __init__(self, function, n_trials=100):
        self.function = function
        self.n_trials = n_trials
        self.res_y = []

    def fit(self, X, y=None, weights=None, idx=0, verbose=False):
        if weights is None:
            weights = np.ones(len(y))
        self.idx = idx
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.ERROR)
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(lambda trial: objective(self, trial, X, y, weights), n_trials=self.n_trials)
        return self

    def transform(self, X):
        features = trial(self, self.study.best_trial, X)
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        # for p, v in self.study.best_trial.params.items():
            # col = f"{col}_{p}_{v}"
        return features


if __name__ == "__main__":
    import joblib
    X, y_shb, y_ret, weights = joblib.load('state/Xyw.job')
    y = y_ret
    fn = "pta.thermo(X.high, X.low, length=trial.suggest_int('length', 2, 10000), )"
    opt = Optimize(function=fn, n_trials=5)
    opt.fit(X, y, idx=2)
    features = opt.transform(X)

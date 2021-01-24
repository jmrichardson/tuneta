import optuna
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import pandas_ta as pta
from finta import TA as fta
import talib as tta
import re


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
    res = eval(self.function)
    if isinstance(res, tuple):
        res = pd.DataFrame(res).T
    if len(res) != len(X):
        raise RuntimeError("Unequal indicator result")
    res = pd.DataFrame(res, index=X.index).iloc[:, self.idx]  # Convert to dataframe
    res_y = res.reindex(y.index).to_numpy().flatten()  # Reduce to y and convert to array
    self.res_y.append(res_y)
    idx = ~np.logical_or(np.isnan(res_y), np.isnan(y))  # Drop NAs w/boolean mask
    y = np.compress(idx, np.array(y))
    res_y = np.compress(idx, res_y)
    weights = np.compress(idx, weights)
    if self.spearman:
        ws = _weighted_spearman(y, res_y, weights)
    else:
        ws = _weighted_pearson(y, res_y, weights)
    return ws


def trial(self, trial, X):
    res = eval(self.function)
    if isinstance(res, tuple):
        res = pd.DataFrame(res).T
    res = pd.DataFrame(res, index=X.index)
    if res.iloc[:, 0].name == 0:
        fn = self.function.split("(")[0]
        params = re.sub('[^0-9a-zA-Z_:]', '', str(self.study.best_params))
        res.columns = [f"{fn}_{params}_{col}" for col in res.columns]
    return res


class Optimize():

    def __init__(self, function, n_trials=100, spearman=True):
        self.function = function
        self.n_trials = n_trials
        self.res_y = []
        self.spearman = spearman

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
        return features


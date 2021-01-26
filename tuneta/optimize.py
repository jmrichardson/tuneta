import optuna
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import pandas_ta as pta
from finta import TA as fta
import talib as tta
import re
import warnings
warnings.filterwarnings("ignore")


def col_name(function, study_best_params):
    function_name = function.split("(")[0].replace(".", "_")
    params = re.sub('[^0-9a-zA-Z_:,]', '', str(study_best_params)).replace(",", "-").replace(":", "_")
    col = f"{function_name}-{params}"
    return col


def _weighted_pearson(y, y_pred, w=None, pearson=True):
    """Calculate the weighted Pearson correlation coefficient."""
    if pearson:
        if w is None:
            w = np.ones(len(y))
        idx = ~np.logical_or(np.isnan(y_pred), np.isnan(y))  # Drop NAs w/boolean mask
        y = np.compress(idx, np.array(y))
        y_pred = np.compress(idx, np.array(y_pred))
        w = np.compress(idx, w)
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


def _weighted_spearman(y, y_pred, w=None):
    """Calculate the weighted Spearman correlation coefficient."""
    if w is None:
        w = np.ones(len(y))
    idx = ~np.logical_or(np.isnan(y_pred), np.isnan(y))  # Drop NAs w/boolean mask
    y = np.compress(idx, np.array(y))
    y_pred = np.compress(idx, np.array(y_pred))
    w = np.compress(idx, w)
    y_pred_ranked = np.apply_along_axis(rankdata, 0, y_pred)
    y_ranked = np.apply_along_axis(rankdata, 0, y)
    return _weighted_pearson(y_pred_ranked, y_ranked, w, pearson=False)


def objective(self, trial, X, y, weights):
    try:
        res = eval(self.function)
    except:
        raise RuntimeError(f"Optuna execution error: {self.function}")
    if isinstance(res, tuple):
        res = pd.DataFrame(res).T
    if len(res) != len(X):
        raise RuntimeError(f"Optuna unequal indicator result: {self.function}")
    res = pd.DataFrame(res, index=X.index).iloc[:, self.idx]  # Convert to dataframe
    res_y = res.reindex(y.index).to_numpy().flatten()  # Reduce to y and convert to array
    self.res_y.append(res_y)  # Save results
    if np.isnan(res_y).sum() / len(res_y) > .95:  # Most or all NANs
        print(f"INFO: Optimization trial produced mostly NANs: {self.function}")
        return False
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
    name = col_name(self.function, self.study.best_params)
    if len(res.columns) > 1:
        res.columns = [f"{name}_{i}" for i in range(len(res.columns))]
    else:
        res.columns = [f"{name}"]
    return res


class Optimize():
    def __init__(self, function, n_trials=100, spearman=True):
        self.function = function
        self.n_trials = n_trials
        self.res_y = []
        self.spearman = spearman

    def fit(self, X, y=None, weights=None, idx=0, verbose=False):

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


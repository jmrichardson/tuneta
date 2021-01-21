import pandas as pd
import numpy as np
from pathos.multiprocessing import ProcessPool
import multiprocessing
import inspect
from scipy.stats import rankdata
from tuneta.config import *
from tuneta.optimize import Optimize
import pandas_ta as pta
from finta import TA as fta
import talib as tta

import warnings
import re


class TuneTA():

    def __init__(self, n_jobs=multiprocessing.cpu_count() - 1, verbose=False):
        self.result = []
        self.fitted = []
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, trials=5, indicators=indicators, ranges=ranges, tune_series=tune_series,
            tune_params=tune_params, tune_column=tune_column):
        X.columns = X.columns.str.lower()  # columns must be lower case

        pool = ProcessPool(nodes=self.n_jobs)

        for low, high in ranges:
            if low <=1:
                raise ValueError("Range low must be > 1")
            if high >= len(X):
                raise ValueError(f"Range high:{high} must be > length of X:{len(X)}")
            for ind in indicators:
                idx = 0
                if ":" in ind:
                    idx = int(ind.split(":")[1])
                    ind = ind.split(":")[0]
                fn = f"{ind}("
                if ind[0:3] == "tta":
                    usage = eval(f"{ind}.__doc__").split(")")[0].split("(")[1]
                    params = re.sub('[^0-9a-zA-Z_\s]', '', usage).split()
                else:
                    sig = inspect.signature(eval(ind))
                    params = sig.parameters.values()

                for param in params:
                    param = re.split(':|=', str(param))[0].strip()
                    if param == "open_":
                        param = "open"
                    if param == "real":
                        fn += f"X.close, "
                    elif param == "ohlc":
                        fn += f"X, "
                    elif param in tune_series:
                        fn += f"X.{param}, "
                    elif param in tune_params:
                        fn += f"{param}=trial.suggest_int('{param}', {low}, {high}), "
                fn += ")"
                self.fitted.append(pool.apipe(Optimize(function=fn, n_trials=trials).fit, X, y, idx=idx, verbose=self.verbose))
        self.fitted = [fit.get() for fit in self.fitted]  # Get results of jobs

    def prune(self, top=2, studies=1):

        if top >= len(self.fitted) or studies >= len(self.fitted):
            print("Cannot prune because top or studies is >= tuned indicators")
            return
        if top <= studies:
            raise ValueError(f"top {top} must be > studies {studies}")

        fitness = []
        for t in self.fitted:
            fitness.append(t.study.best_value)  # get fitness of study
        fitness = np.array(fitness)  # Fitness of each fitted study

        fitness = fitness.argsort()[::-1][:top]  # Get sorted fitness indices of HOF

        # Gets best trial feature of each study in HOF
        features = []
        top_studies = [self.fitted[i] for i in fitness]  # Get HOF studies
        for study in top_studies:
            features.append(study.res_y[study.study.best_trial.number])
        features = np.array(features)  # Features of HOF studies

        # Correlation of HOF features
        eval = np.apply_along_axis(rankdata, 1, features)
        with np.errstate(divide='ignore', invalid='ignore'):
            correlations = np.abs(np.corrcoef(eval))
        np.fill_diagonal(correlations, 0.)

        # Iteratively remove least fit individual of most correlated pairs of studies
        components = list(range(top))
        indices = list(range(top))
        while len(components) > studies:
            most_correlated = np.unravel_index(np.argmax(correlations), correlations.shape)
            worst = max(most_correlated)
            components.pop(worst)
            indices.remove(worst)
            correlations = correlations[:, indices][indices, :]
            indices = list(range(len(components)))

        self.fitted = [self.fitted[i] for i in fitness[components]]

    def transform(self, X):
        X.columns = X.columns.str.lower()  # columns must be lower case
        pool = ProcessPool(nodes=self.n_jobs)
        self.result = []
        for ind in self.fitted:
            self.result.append(pool.apipe(ind.transform, X))
        self.result = [res.get() for res in self.result]
        res = pd.concat(self.result, axis=1)
        return res


if __name__ == "__main__":
    import joblib
    X, y = joblib.load('state/Xy.job')

    inds = TuneTA(verbose=True)
    # inds.fit(X, y, indicators=["fta.SMA"], ranges=[(1, 100)], trials=5)
    # inds.fit(X, y, indicators=["pta.sma"], ranges=[(1, 100)], trials=5)
    # inds.fit(X, y, indicators=['tta.BBANDS', 'tta.DEMA', 'tta.EMA',], ranges=[(0, 100)], trials=5)
    # inds.fit(X, y, indicators=['tta.BBANDS:1', 'fta.SMA', 'pta.sma', 'tta.DEMA'], ranges=[(1, 100)], trials=5)
    # """
    # inds.fit(X, y, indicators=["fta.SMM"], ranges=[(2, 100)], trials=5)
    # """
    indicators = ['fta.SMA', 'fta.SMM', 'fta.SSMA', 'fta.EMA', 'fta.DEMA', 'fta.TEMA', 'fta.TRIMA', 'fta.TRIX', 'fta.VAMA', 'fta.ER', 'fta.KAMA', 'fta.ZLEMA',
    'fta.WMA', 'fta.HMA', 'fta.EVWMA', 'fta.VWAP', 'fta.SMMA', 'fta.FRAMA', 'fta.MACD', 'fta.PPO', 'fta.VW_MACD', 'fta.EV_MACD', 'fta.MOM', 'fta.ROC', 'fta.RSI', 'fta.IFT_RSI',
    'fta.TR', 'fta.ATR', 'fta.SAR', 'fta.BBANDS', 'fta.BBWIDTH', 'fta.MOBO', 'fta.PERCENT_B', 'fta.KC', 'fta.DO', 'fta.DMI', 'fta.ADX', 'fta.PIVOT', 'fta.PIVOT_FIB', 'fta.STOCH',
    'fta.STOCHD', 'fta.STOCHRSI', 'fta.WILLIAMS', 'fta.UO', 'fta.AO', 'fta.MI', 'fta.VORTEX', 'fta.KST', 'fta.TSI', 'fta.TP', 'fta.ADL', 'fta.CHAIKIN', 'fta.MFI', 'fta.OBV', 'fta.WOBV',
    'fta.VZO', 'fta.PZO', 'fta.EFI', 'fta.CFI', 'fta.EBBP', 'fta.EMV', 'fta.CCI', 'fta.COPP', 'fta.BASP', 'fta.BASPN', 'fta.CMO', 'fta.CHANDELIER', 'fta.QSTICK', 'fta.TMF',
    'fta.WTO', 'fta.FISH', 'fta.APZ', 'fta.SQZMI', 'fta.VPT', 'fta.FVE', 'fta.VFI', 'fta.MSD', 'fta.STC',]
    for i in indicators:
        print(i)
        inds = TuneTA(verbose=True)
        inds.fit(X, y, indicators=[i], ranges=[(2, 100)], trials=5)


    # """
    # inds.fit(X, y, indicators=['tta.BETA'], ranges=[(5, 100)], trials=5)

    # inds.prune(top=3, studies=2)
    out = inds.transform(X)
    print("done")


    # pta.vp(X.close, X.volume)
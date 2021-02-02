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
import re
from tabulate import tabulate
from tuneta.optimize import col_name



class TuneTA():

    def __init__(self, n_jobs=multiprocessing.cpu_count() - 1, verbose=False):
        self.result = []
        self.fitted = []
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, trials=5, indicators=indicators, ranges=ranges, tune_series=tune_series,
            tune_params=tune_params, spearman=True, weights=None, early_stop=99999, split=None):
        self.fitted = []
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
                    elif param == "ohlcv":
                        fn += f"X, "
                    elif param in tune_series:
                        fn += f"X.{param}, "
                    elif param in tune_params:
                        fn += f"{param}=trial.suggest_int('{param}', {low}, {high}), "
                fn += ")"
                self.fitted.append(pool.apipe(Optimize(function=fn, n_trials=trials, spearman=spearman).fit, X, y,
                                              idx=idx, verbose=self.verbose, weights=weights, early_stop=early_stop, lookback=max(max(ranges)), split=split), )
        self.fitted = [fit.get() for fit in self.fitted]  # Get results of jobs

    def report(self, target_corr=True, features_corr=True):
        fns = []
        cor = []
        features = []
        for fit in self.fitted:
            if fit.split is None:
                fns.append(col_name(fit.function, fit.study.best_params))
                cor.append(round(fit.study.best_value, 6))
            else:
                fns.append(col_name(fit.function, fit.study.top_params))
                cor.append(round(fit.study.top_value, 6))
            features.append(fit.res_y)
        fitness = pd.DataFrame(cor, index=fns, columns=['Correlation']).sort_values(by=['Correlation'], ascending=False)
        if target_corr:
            print("\nTarget Correlation:\n")
            print(tabulate(fitness, headers=fitness.columns, tablefmt="simple"))

        eval = np.apply_along_axis(rankdata, 1, features)
        with np.errstate(divide='ignore', invalid='ignore'):
            correlations = np.abs(np.corrcoef(eval))
        correlations = pd.DataFrame(correlations, columns=fns, index=fns)
        if features_corr:
            print("\nFeature Correlation:\n")
            print(tabulate(correlations, headers=correlations.columns, tablefmt="simple"))


    def prune(self, top=2, studies=1):

        if top > len(self.fitted) or studies > len(self.fitted):
            print("Cannot prune because top or studies is >= tuned indicators")
            return
        if top < studies:
            raise ValueError(f"top {top} must be >= studies {studies}")

        fitness = []
        for t in self.fitted:
            if t.split is None:
                fitness.append(t.study.best_trial.value)  # get fitness of study
            else:
                fitness.append(sum(t.study.trials[t.study.top_trial].values))  # get fitness of study
        fitness = np.array(fitness)  # Fitness of each fitted study

        fitness = fitness.argsort()[::-1][:top]  # Get sorted fitness indices of HOF

        # Gets best trial feature of each study in HOF
        features = []
        top_studies = [self.fitted[i] for i in fitness]  # Get HOF studies
        for study in top_studies:
            features.append(study.res_y)
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
    inds.fit(X, y, indicators=['tta.BBANDS:1', 'fta.SMA', 'pta.sma', 'tta.RSI', 'tta.RSI', 'pta.kst'], ranges=[(2, 100)], trials=5)
    # inds.fit(X, y, indicators=['pta.kst'], ranges=[(2, 100)], trials=5)
    inds.report()
    inds.prune(top=5, studies=4)
    inds.report()
    features = inds.transform(X)
    X_train = pd.concat([X, features], axis=1)
    print("done")


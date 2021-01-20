import pandas as pd
import numpy as np
from pathos.multiprocessing import ProcessPool
import multiprocessing
import inspect
from scipy.stats import rankdata
from tuneta.config import *
from tuneta.optimize import Optimize
import pandas_ta as pta
import warnings


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
            if high > len(X):
                raise ValueError(f"Range high:{high} > length of X:{len(X)}")
            for ind in indicators:
                fn = f"{ind}("
                sig = inspect.signature(eval(ind))
                for param in sig.parameters.values():
                    param = str(param).split("=")[0].strip()
                    if param == "open_":
                        param = "open"
                    if param in tune_series:
                        fn += f"X.{param}, "
                    elif param in tune_params:
                        fn += f"{param}=trial.suggest_int('{param}', {low}, {high}), "
                fn += ")"
                idx = 0
                if ind in tune_column:
                    idx = tune_column.get(ind)
                self.fitted.append(pool.apipe(Optimize(function=fn, n_trials=trials).fit, X, y, idx=idx, verbose=self.verbose))
        self.fitted = [fit.get() for fit in self.fitted]  # Get results of jobs

    def prune(self, top=2, studies=1):

        if len(self.fitted) <= studies:
            warnings.warn(f"Existing studies {len(self.fitted)} is <= {studies}.  Abort prune", RuntimeWarning)
            return
        if top <= studies:
            raise ValueError(f"top:{top} must be > studies:{studies}")
        if top >= len(self.fitted):
            raise ValueError(f"top:{top} must be < fitted studies:{len(self.fitted)}")

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
        return pd.concat(self.result, axis=1)


if __name__ == "__main__":
    import joblib
    X, y_shb, y, weights = joblib.load('state/Xyw.job')

    inds = TuneTA(verbose=False)
    inds.fit(X, y, indicators=["pta.rsi", "pta.macd", "pta.cci", "pta.adx"], trials=5)
    inds.prune(top=3, studies=2)
    out = inds.transform(X)
    print("done")

import pandas as pd
import numpy as np
from pathos.multiprocessing import ProcessPool
import multiprocessing
import inspect
from scipy.stats import rankdata
from tuneta.config import *
from tuneta.optimize import Optimize
import pandas_ta as pta


class TuneTA():

    def __init__(self):
        self.result = []
        self.fitted = []

    def fit(self, X, y, trials=5, indicators=indicators, ranges=ranges, tune_series=tune_series,
            tune_params=tune_params, tune_column=tune_column):

        pool = ProcessPool(nodes=multiprocessing.cpu_count() - 2)

        for low, high in ranges:
            for ind in indicators:
                fn = f"{ind}("
                sig = inspect.signature(eval(ind))
                for param in sig.parameters.values():
                    param = str(param).split("=")[0].strip()
                    if param == "open_": param = "open"
                    print(param)
                    if param in tune_series:
                        fn += f"X.{param}, "
                    elif param in tune_params:
                        fn += f"{param}=trial.suggest_int('{param}', {low}, {high}), "
                fn += ")"
                idx = 0
                if ind in tune_column:
                    idx = tune_column.get(ind)
                self.fitted.append(pool.apipe(Optimize(function=fn, n_trials=trials).fit, X, y, idx=idx))
        self.fitted = [fit.get() for fit in self.fitted]  # Get results of jobs

    def prune(self, top=.3, studies=2):

        fitness = []
        for t in self.fitted:
            fitness.append(t.study.best_value)  # get fitness of study
        fitness = np.array(fitness)  # Fitness of each fitted study

        hof = round(top * len(fitness))  # Hall of fame - top x of studies
        self.n_components = 2  # Least correlated studies in hall of fame
        fitness = fitness.argsort()[::-1][:hof]  # Get sorted fitness indices of HOF

        # Gets best trial feature of each study in HOF
        features = []
        hof_studies = [self.fitted[i] for i in fitness]  # Get HOF studies
        for study in hof_studies:
            features.append(study.res_y[study.study.best_trial.number])
        features = np.array(features)  # Features of HOF studies

        # Correlation of HOF features
        eval = np.apply_along_axis(rankdata, 1, features)
        with np.errstate(divide='ignore', invalid='ignore'):
            correlations = np.abs(np.corrcoef(eval))
        np.fill_diagonal(correlations, 0.)

        # Iteratively remove least fit individual of most correlated pairs of studies
        components = list(range(hof))
        indices = list(range(hof))
        while len(components) > studies:
            most_correlated = np.unravel_index(np.argmax(correlations), correlations.shape)
            worst = max(most_correlated)
            components.pop(worst)
            indices.remove(worst)
            correlations = correlations[:, indices][indices, :]
            indices = list(range(len(components)))

        self.fitted = [self.fitted[i] for i in fitness[components]]

    def transform(self, X, y):

        pool = ProcessPool(nodes=multiprocessing.cpu_count() - 2)
        self.result = []
        for ind in self.fitted:
            self.result.append(pool.apipe(ind.transform, X, y))
        self.result = [res.get() for res in self.result]
        return pd.concat(self.result, axis=1)


if __name__ == "__main__":

    import joblib
    X, y_shb, y, weights = joblib.load('state/Xyw.job')

    inds = TuneTA()
    inds.fit(X, y, indicators=["pta.thermo", "pta.macd", "pta.ao"], trials=5)
    inds.prune()
    out = inds.transform(X, y)


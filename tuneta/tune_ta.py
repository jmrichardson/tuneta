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
from collections import OrderedDict


class TuneTA():

    def __init__(self, n_jobs=multiprocessing.cpu_count() - 1, verbose=False):
        self.fitted = []
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, trials=5, indicators=['tta'], ranges=[(3, 180)],
        spearman=True, weights=None, early_stop=99999, split=None):
        """
        Optimize indicator parameters to maximize correlation
        :param X: Historical dataset
        :param y: Target used to measure correlation.  Can be a subset index of X
        :param trials: Number of optimization trials per indicator set
        :param indicators: List of indicators to optimize
        :param ranges: Parameter search space
        :param spearman: Perform spearman vs pearson correlation
        :param weights: Optional weights sharing the same index as y
        :param early_stop: Max number of optimization trials before stopping
        :param split: Index cut points defining time periods
        """

        self.fitted = []  # List containing each indicator completed study
        X.columns = X.columns.str.lower()  # columns must be lower case
        pool = ProcessPool(nodes=self.n_jobs)  # Set parallel cores

        # Package level optimization
        if 'tta' in indicators:
            indicators = indicators + talib_indicators
            indicators.remove('tta')
        if 'pta' in indicators:
            indicators = indicators + pandas_ta_indicators
            indicators.remove('pta')
        if 'fta' in indicators:
            indicators = indicators + finta_indicatrs
            indicators.remove('fta')
        if 'all' in indicators:
            indicators = talib_indicators + pandas_ta_indicators + finta_indicatrs
        indicators = list(OrderedDict.fromkeys(indicators))

        # Create textual representation of function in Optuna format
        # Example: 'tta.RSI(X.close, length=trial.suggest_int(\'timeperiod1\', 2, 1500))'
        # Utilizes the signature of the indicator (ie user parameters) if available
        # TTA uses help docstrings as signature is not available in C bindings
        # Parameters contained in config.py are tuned

        # Iterate user defined search space ranges
        for low, high in ranges:
            if low <= 1:
                raise ValueError("Range low must be > 1")
            if high >= len(X):
                raise ValueError(f"Range high:{high} must be > length of X:{len(X)}")

            # Iterate indicators per range
            for ind in indicators:

                # Index column to optimize if indicator returns dataframe
                idx = 0
                if ":" in ind:
                    idx = int(ind.split(":")[1])
                    ind = ind.split(":")[0]
                fn = f"{ind}("

                # If TTA indicator, use doc strings for lack of better way to
                # get indicator arguments (C binding)
                if ind[0:3] == "tta":
                    usage = eval(f"{ind}.__doc__").split(")")[0].split("(")[1]
                    params = re.sub('[^0-9a-zA-Z_\s]', '', usage).split()

                # Pandas-TA and FinTA both can be inspected for parameters
                else:
                    sig = inspect.signature(eval(ind))
                    params = sig.parameters.values()

                # Format function string
                suggest = False
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
                        suggest = True
                        if param in ['mamode']:
                            fn += f"{param}=trial.suggest_categorical('{param}', {tune_ta_mm}), "
                        else:
                            fn += f"{param}=trial.suggest_int('{param}', {low}, {high}), "

                    elif param in tune_params:
                        suggest = True
                        fn += f"{param}=trial.suggest_categorical('{param}', {tune_ta_mm}), "

                fn += ")"

                # Only optimize indicators that contain tunable parameters
                if suggest:
                    self.fitted.append(pool.apipe(Optimize(function=fn, n_trials=trials,
                        spearman=spearman).fit, X, y, idx=idx, verbose=self.verbose,
                        weights=weights, early_stop=early_stop, split=split), )
                else:
                    self.fitted.append(pool.apipe(Optimize(function=fn, n_trials=1,
                        spearman=spearman).fit, X, y, idx=idx, verbose=self.verbose,
                        weights=weights, early_stop=early_stop, split=split), )

        # Blocking wait to retrieve results
        # if item comes back as non-numerical dont add
        self.fitted = [fit.get() for fit in self.fitted if isinstance(fit.get().res_y_corr,(float,int))]

        # Some items might come back as an array
        # if they are cant be a float skip
        for i in self.fitted:
            try:
                float(i.res_y_corr)
            except:
                continue

    def prune(self, top=2, studies=1):
        """
        Select most correlated with target, least intercorrelated
        :param top: Selects top x most correlated with target
        :param studies: From top x, keep y least intercorelated
        :return:
        """

        # Error checking
        if top > len(self.fitted) or studies > len(self.fitted):
            raise ValueError("Cannot prune because top or studies is >= tuned indicators")
            return
        if top < studies:
            raise ValueError(f"top {top} must be >= studies {studies}")

        # Create fitness array that maps to the correlation of each indicator study
        fitness = []
        for t in self.fitted:
            if t.split is None:
                fitness.append(t.study.best_trial.value)
            else:
                fitness.append(sum(t.study.trials[t.study.top_trial].values))
        fitness = np.array(fitness)

        # Select top x indices with most correlation to target
        fitness = fitness.argsort()[::-1][:top]  # Get sorted fitness indices of HOF

        # Gets best trial feature of each study in HOF
        features = []
        top_studies = [self.fitted[i] for i in fitness]  # Get fitness mapped studies
        for study in top_studies:
            features.append(study.res_y) # Get indicator values stored from optimization
        features = np.array(features)  # Features of HOF studies / actual indicator results

        # Correlation of HOF features
        # Create correlation table of features
        eval = np.apply_along_axis(rankdata, 1, features)
        with np.errstate(divide='ignore', invalid='ignore'):
            correlations = np.abs(np.corrcoef(eval))
        np.fill_diagonal(correlations, 0.)

        # Iteratively removes least fit individual of most correlated pairs of studies
        # IOW, finds most correlated pairs, removes lest correlated to target until x studies
        components = list(range(top))
        indices = list(range(top))
        while len(components) > studies:
            most_correlated = np.unravel_index(np.argmax(correlations), correlations.shape)
            worst = max(most_correlated)
            components.pop(worst)
            indices.remove(worst)
            correlations = correlations[:, indices][indices, :]
            indices = list(range(len(components)))

        # Save only fitted studies (overwriting all studies)
        self.fitted = [self.fitted[i] for i in fitness[components]]

    def transform(self, X, columns=None):
        """
        Given X, create features of fitted studies
        :param X: Dataset with features used to create fitted studies
        :return:
        """

        # Remove trailing identifier in column list if present
        if columns is not None:
            columns = [re.sub(r'_[0-9]+$', '', s) for s in columns]

        X.columns = X.columns.str.lower()  # columns must be lower case
        pool = ProcessPool(nodes=self.n_jobs)  # Number of jobs
        self.result = []

        # Iterate fitted studies and calculate TA with fitted parameter set
        for ind in self.fitted:
            # Create field if no columns or is in columns list
            if columns is None or ind.res_y.name in columns:
                self.result.append(pool.apipe(ind.transform, X))

        # Blocking wait for asynchronous results
        self.result = [res.get() for res in self.result]

        # Combine results into dataframe to return
        res = pd.concat(self.result, axis=1)
        return res

    def report(self, target_corr=True, features_corr=True):
        fns = []  # Function names
        cor = []  # Target Correlation
        moc = []  # Multi-Time Period Correlation
        mean_moc = []
        std_moc = []  # Multi STD
        features = []
        for fit in self.fitted:

            if fit.split is None:
                fns.append(col_name(fit.function, fit.study.best_params))
            else:
                fns.append(col_name(fit.function, fit.study.top_params))
                moc.append(fit.study.trials[fit.study.top_trial].values)
                mean_moc.append(np.mean(fit.study.trials[fit.study.top_trial].values))
                std_moc.append(np.std(fit.study.trials[fit.study.top_trial].values))


            cor.append(np.round(fit.res_y_corr, 6))
            features.append(fit.res_y)

        if fit.split is None:
            fitness = pd.DataFrame(cor, index=fns, columns=['Correlation']).sort_values(by=['Correlation'], ascending=False)
        else:
            fitness = pd.DataFrame(zip(cor, mean_moc, std_moc, moc), index=fns, columns=['Correlation', 'Split Mean', 'Split STD', 'Split Correlation']).sort_values(by=['Correlation'], ascending=False)

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

    def fit_times(self):
        times = [fit.time for fit in self.fitted]
        inds = [fit.function.split('(')[0] for fit in self.fitted]
        df = pd.DataFrame({'Indicator': inds, 'Times': times}).sort_values(by='Times', ascending=False)
        print(tabulate(df, headers=df.columns, tablefmt="simple"))

    def get_indicator_params(self,type_='PTA',objective='single'):
        # WIP for other indicators
        self.indicator_params = []
        if type_ == 'PTA':
            if objective == 'single':
                for ind in self.fitted:
                    study_dict = ind.study.best_params
                    study_dict['kind'] = ind.function.split('(')[0]
                    self.indicator_params.append(study_dict)
            else:
                for ind in self.fitted:
                    study_dict = ind.study.top_params
                    study_dict['kind'] = ind.function.split('(')[0]
                    self.indicator_params.append(study_dict)


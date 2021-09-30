import pandas as pd
import numpy as np
from pathos.multiprocessing import ProcessPool
import multiprocessing
import inspect
from tuneta.config import *
from tuneta.optimize import Optimize
import pandas_ta as pta
from finta import TA as fta
import talib as tta
import re
from tabulate import tabulate
from tuneta.utils import col_name
from tuneta.utils import distance_correlation
from collections import OrderedDict
from scipy.spatial.distance import squareform
import itertools


class TuneTA():

    def __init__(self, n_jobs=multiprocessing.cpu_count() - 1, verbose=False):
        self.fitted = []
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, trials=5, indicators=['tta'], ranges=[(3, 180)], early_stop=99999):
        """
        Optimize indicator parameters to maximize correlation
        :param X: Historical dataset
        :param y: Target used to measure correlation.  Can be a subset index of X
        :param trials: Number of optimization trials per indicator set
        :param indicators: List of indicators to optimize
        :param ranges: Parameter search space
        :param early_stop: Max number of optimization trials before stopping
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
                        fn += f"{param}=trial.suggest_int('{param}', {low}, {high}), "
                fn += ")"

                # Only optimize indicators that contain tunable parameters
                if suggest:
                    self.fitted.append(pool.apipe(Optimize(function=fn, n_trials=trials).fit, X, y, idx=idx,
                        verbose=self.verbose, early_stop=early_stop))
                else:
                    self.fitted.append(pool.apipe(Optimize(function=fn, n_trials=1).fit, X, y, idx=idx,
                        verbose=self.verbose, early_stop=early_stop))

        # Blocking wait to retrieve results
        self.fitted = [fit.get() for fit in self.fitted]

        # Fits must contain best trial data
        self.fitted = [f for f in self.fitted if len(f.study.user_attrs) > 0]

        # Order fits by correlation (Descending)
        self.fitted = sorted([f for f in self.fitted], key=lambda x:x.study.user_attrs['best_trial'].value, reverse=True)

    def prune(self, top=2):
        """
        Select most correlated with target, least intercorrelated
        :param top: Selects top x most correlated with target
        :param studies: From top x, keep y least intercorelated
        :return:
        """
        # Error checking
        if top > len(self.fitted):
            raise ValueError("Cannot prune: top must be <= tuned indicators")

        if not hasattr(self, 'f_corr'):
            self.features_corr()

        # Iteratively removes least fit individual of most correlated pairs of studies
        # IOW, finds most correlated pairs, removes lest correlated to target until x studies
        components = list(range(len(self.fitted)))
        indices = list(range(len(self.fitted)))
        correlations = np.array(self.f_corr)
        while len(components) > top:
            most_correlated = np.unravel_index(np.argmax(correlations), correlations.shape)
            worst = max(most_correlated)
            components.pop(worst)
            indices.remove(worst)
            correlations = correlations[:, indices][indices, :]
            indices = list(range(len(components)))

        # Remove most correlated fits
        self.fitted = [self.fitted[i] for i in components]

        # Recalculate correlation of fits
        self.target_corr()
        self.features_corr()

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

    def target_corr(self):
        fns = []  # Function names
        cor = []  # Target Correlation
        for fit in self.fitted:
            fns.append(col_name(fit.function, fit.study.user_attrs['best_trial'].params))
            cor.append(np.round(fit.study.user_attrs['best_trial'].value, 6))

        # Target correlation
        self.t_corr = pd.DataFrame(cor, index=fns, columns=['Correlation']).sort_values(by=['Correlation'], ascending=False)

    def features_corr(self):
        fns = []  # Function names
        cor = []  # Target Correlation
        features = []
        for fit in self.fitted:
            fns.append(col_name(fit.function, fit.study.user_attrs['best_trial'].params))
            cor.append(np.round(fit.study.user_attrs['best_trial'].value, 6))
            features.append(fit.study.user_attrs['best_trial'].user_attrs['res_y'])

        # Feature must be same size for correlation
        start = max([f.first_valid_index() for f in features])
        features = [f[f.index >= start] for f in features]

        # Inter Correlation
        pair_order_list = itertools.combinations(features, 2)
        correlations = [distance_correlation(p[0], p[1]) for p in pair_order_list]
        correlations = squareform(correlations)
        self.f_corr = pd.DataFrame(correlations, columns=fns, index=fns)

    def report(self, target_corr=True, features_corr=True):
        if target_corr:
            if not hasattr(self, 't_corr'):
                self.target_corr()
            print("\nIndicator Correlation to Target:\n")
            print(tabulate(self.t_corr, headers=self.t_corr.columns, tablefmt="simple"))

        if features_corr:
            if not hasattr(self, 'f_corr'):
                self.features_corr()
            print("\nIndicator Correlation to Each Other:\n")
            print(tabulate(self.f_corr, headers=self.f_corr.columns, tablefmt="simple"))

    def fit_times(self):
        times = [fit.time for fit in self.fitted]
        inds = [fit.function.split('(')[0] for fit in self.fitted]
        df = pd.DataFrame({'Indicator': inds, 'Times': times}).sort_values(by='Times', ascending=False)
        print(tabulate(df, headers=df.columns, tablefmt="simple"))


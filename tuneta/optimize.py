import optuna
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import pandas_ta as pta
from finta import TA as fta
import talib as tta
import re
import warnings
import pareto
warnings.filterwarnings("ignore")


def col_name(function, study_best_params):
    """
    Create consistent column names given string function and params
    :param function:  Function represented as string
    :param study_best_params:  Params for function
    :return:
    """

    # Optuna string of indicator
    function_name = function.split("(")[0]

    # Optuna string of parameters
    params = re.sub('[^0-9a-zA-Z_:,]', '', str(study_best_params)).replace(",", "_")

    # Concatenate name and params to define
    col = f"{function_name}_{params}"
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
    idx = ~np.logical_or(np.isnan(y_pred), np.isnan(y))  # Drop NAs w/boolean mask
    y = np.compress(idx, np.array(y))
    y_pred = np.compress(idx, np.array(y_pred))
    w = np.compress(idx, w)
    y_pred_ranked = np.apply_along_axis(rankdata, 0, y_pred)
    y_ranked = np.apply_along_axis(rankdata, 0, y)
    return _weighted_pearson(y_pred_ranked, y_ranked, w, pearson=False)


def _trial(self, trial, X):
    """
    Calculate indicator using best fitted trial over X
    :param self:  Optuna study
    :param trial:  Optuna trial
    :param X:  dataset
    :return:
    """

    # Evaluate TA defined as optuna trial string
    res = eval(self.function)

    # If return is tuple, convert to DF
    if isinstance(res, tuple):
        res = pd.DataFrame(res).T

    # Index result with X index
    res = pd.DataFrame(res, index=X.index)

    # Create consistent column names with function string and params
    name = col_name(self.function, trial.params)

    # Append integer identifier to DF with multiple columns
    if len(res.columns) > 1:
        res.columns = [f"{name}_{i}" for i in range(len(res.columns))]
    else:
        res.columns = [f"{name}"]
    return res


# Minimize difference, Maximize Total
def _min_max(study):
    """
    Multi-objective function to find best trial index with minimum deviation and max correlation
    :param study: Optuna study
    :return:
    """

    # Iterate pareto-front trials storing mean correlation and std dev
    df = []
    for trial in study.best_trials:
        df.append([trial.number, np.mean(trial.values), np.std(trial.values)])

    #from scipy.stats import skew
    #print(trial.values)
    #print(np.std(trial.values))
    #print(skew(trial.values))

    # Sort dataframe ascending by mean correlation
    df = pd.DataFrame(df).sort_values(by=2, ascending=True)

    # Sort df with best trial in first row
    if len(df) > 1:

        # Create second pareto to maximize correlation and minimize stddev
        # Epsilons define precision, ie dominance over other candidates
        # Dominance is defined as x percent of stddev of stddev
        nd = pareto.eps_sort([list(df.itertuples(False))], objectives=[1, 2],
            epsilons=[1e-09, np.std(df[1])*.5], maximize=[1])

        # Sort remaining candidates
        nd = pd.DataFrame(nd).sort_values(by=2, ascending=True)

    # Only 1st trial so return it
    else:
        nd = df

    # Return "best" trial index
    return nd.iloc[0, 0]


def _multi_early_stopping_opt(study, trial):
    """
    Callback for to stop Optuna early with improvement
    :param study: Optuna study
    :param trial: Optuna trial
    :return:
    """

    # Get index of this trial
    this_trial = trial.number

    # Function to find "best" trial
    # Returns index of best trial (always 0 initially)
    best_trial = _min_max(study)

    # If this trial is best trial then
    # store trial index and params
    # reset early stop counter
    if this_trial == best_trial:
        study.top_trial = this_trial
        study.top_params = study.trials[this_trial].params
        study.top_value = np.mean(study.trials[this_trial].values)
        study.early_stop_count = 0

    # If best_trial (index of trials) is not the best then
    # stop if stop_count great than user defined early_stop
    # else increment count
    else:
        if study.early_stop_count > study.early_stop:
            study.early_stop_count = 0
            raise optuna.exceptions.OptunaError
        else:
            study.early_stop_count = study.early_stop_count + 1
    return


def _single_early_stopping_opt(study, trial):
    """
    Callback function to stop optuna trials early for single objective
    :param study:  Optuna study
    :param trial:  Optuna trial
    :return:
    """

    # Record first as best score
    if study.best_score is None:
        study.best_score = study.best_value

    # Record better value and reset count
    if study.best_value > study.best_score:
        study.best_score = study.best_value
        study.early_stop_count = 0

    # If count greater than user defined stop, raise error to stop
    else:
        if study.early_stop_count > study.early_stop:
            study.early_stop_count = 0
            raise optuna.exceptions.OptunaError
        else:
            study.early_stop_count = study.early_stop_count+1
    return


def _objective(self, trial, X, y, weights=None, split=None):
    """
    Objective function used in Optuna trials
    :param self:  Optuna study
    :param trial:  Optuna trial
    :param X: Entire dataset
    :param y: Target
    :param weights: Optional weights
    :param split: Split cut points
    :return:
    """

    # Generate even weights if none
    if weights is None:
        weights = np.ones(len(y))

    # Execute trial function
    try:
        res = eval(self.function)
    except:
        raise RuntimeError(f"Optuna execution error: {self.function}")

    # If indicator result is tuple, select the one of interest
    if isinstance(res, tuple):
        res = res[self.idx]

    # Ensure result is a dataframe with same index as X
    res = pd.DataFrame(res, index=X.index)

    # If indicator result is dataframe, select the one of interest
    if len(res.columns) > 1:
        res = pd.DataFrame(res.iloc[:, self.idx])

    # y may be a subset of X, so reduce result to y and convert to series
    res_y = res.reindex(y.index).iloc[:, 0].replace([np.inf, -np.inf], np.nan)

    # Save all trial results for pruning and reporting
    # Only the best trial will eventually be saved to limit storage requirements
    self.res_y.append(res_y)  # Save results

    # Indicator result may be all NANs based on parameter set
    # Return FALSE and alert
    if np.isnan(res_y).sum() / len(res_y) > .95:  # Most or all NANs
        print(f"INFO: Optimization trial produced mostly NANs: {self.function}")
        self.res_y_corr.append(np.zeros(len(y)))
        return False

    # y and res_y must be arrays
    y = np.array(y)
    res_y = np.array(res_y)

    # Obtain correlation for entire dataset
    if self.spearman:
        corr = _weighted_spearman(y, res_y, weights)
    else:
        corr = _weighted_pearson(y, res_y, weights)

    # Save correlation for res_y
    self.res_y_corr.append(corr)

    # Multi-objective optimization
    # Obtain correlation to target for each split for Optuna to maximize
    if split is not None:
        mo = []
        for i, e in enumerate(split):
            if i == 0:
                s = e
                continue
            y_se = y[s:e]
            res_y_se = res_y[s:e]
            weights_se = weights[s:e]

            # Too man NANs in split
            if np.isnan(res_y_se).sum() / len(res_y_se) > .98:
                raise ValueError(f"Too many NANs in split {i}")

            if self.spearman:
                mo.append(_weighted_spearman(y_se, res_y_se, weights_se))
            else:
                mo.append(_weighted_pearson(y_se, res_y_se, weights_se))
            s = e
        return tuple(mo)

    # Single objective optimization return corr for entire dataset
    else:
        return corr


class Optimize():
    def __init__(self, function, n_trials=100, spearman=True):
        self.function = function
        self.n_trials = n_trials
        self.res_y = []
        self.res_y_corr = []
        self.spearman = spearman

    def fit(self, X, y, weights=None, idx=0, verbose=False, early_stop=None, split=None):
        """
        Optimize a technical indicator
        :param X: Historical dataset
        :param y: Target
        :param weights: Optional weights
        :param idx: Column to maximize if TA returnes multiple
        :param verbose: Verbosity level
        :param early_stop: Number of trials to perform without improvement before stopping
        :param split: Split points for multi-objective optimization
        :return:
        """
        self.idx = idx
        self.split = split

        # Display Optuna trial messages
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.ERROR)

        # Single objective optimization
        if split is None:

            # Create optuna study maximizing correlation
            self.study = optuna.create_study(direction='maximize')

            # Set required early stopping variables
            self.study.early_stop = early_stop
            self.study.early_stop_count = 0
            self.study.best_score = None

            # Start optimization trial
            try:
                self.study.optimize(lambda trial: _objective(self, trial, X, y, weights),
                    n_trials=self.n_trials, callbacks=[_single_early_stopping_opt])

            # Early stopping (not officially supported by Optuna)
            except optuna.exceptions.OptunaError:
                pass

            # Keep only results of best trial for prune and reporting
            self.res_y = self.res_y[self.study.best_trial.number]
            self.res_y_corr = self.res_y_corr[self.study.best_trial.number]

        # Multi objective optimization
        else:

            # Create study to maximize eash split
            sampler = optuna.samplers.NSGAIISampler()
            self.study = optuna.create_study(directions=(len(split)-1) * ['maximize'], sampler=sampler)

            # Early stopping variables
            self.study.early_stop = early_stop
            self.study.early_stop_count = 0

            # Custom best trial variables (Optuna "best" are immutable)
            self.study.top_trial = 0
            self.study.top_params = None
            self.study.top_value = None

            # Start optimization trial
            try:
                self.study.optimize(lambda trial: _objective(self, trial, X, y, weights, split),
                    n_trials=self.n_trials, callbacks=[_multi_early_stopping_opt])

            # Early stopping (not officially supported by Optuna)
            except optuna.exceptions.OptunaError:
                pass

            # Validation checks
            if np.mean(self.study.trials[self.study.top_trial].values) != self.study.top_value:
                raise RuntimeError("Top trial score invalid")
            if len(self.res_y) != len(self.study.trials) or len(self.res_y) != len(self.res_y_corr):
                raise RuntimeError("Total results does not equal trials")

            # Keep only results of best trial for prune and reporting
            self.res_y = self.res_y[self.study.top_trial]
            self.res_y_corr = self.res_y_corr[self.study.top_trial]

        return self

    def transform(self, X):
        """
        Calculate TA indicator using fittted best trial
        :param X: Datset
        :return:
        """

        # Calculate and store in features, replacing any potential non-finites (not sure needed)
        if self.split is None:
            features = _trial(self, self.study.best_trial, X)
        else:
            features = _trial(self, self.study.trials[self.study.top_trial], X)
        return features

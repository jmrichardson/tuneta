import optuna
import pandas as pd
import numpy as np
import pandas_ta as pta
from finta import TA as fta
import talib as tta
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from optuna.trial import TrialState
from timeit import default_timer as timer
from tuneta.utils import col_name
from tuneta.utils import distance_correlation
from yellowbrick.cluster import KElbowVisualizer
from joblib import delayed, Parallel
import json
import warnings
warnings.filterwarnings("ignore")


# Apply trial on multi index
def trial_results(X, function, trial, sym=None):
    if sym:
        X = X.droplevel(1)
    try:
        res = eval(function) # Eval contains reference to best trial (in argument) to re-use original parameters
    except Exception as e:
        raise Exception(e)
    if isinstance(res, tuple):
        res = pd.DataFrame(res).T
    res = pd.DataFrame(res, index=X.index)  # Ensure result aligns with X
    if sym:
        res['sym'] = sym
        res.set_index('sym', append=True, inplace=True)
    return res


def _trial(self, trial, X):
    """
    Calculate indicator using best fitted trial over X
    :param self:  Optuna study
    :param trial:  Optuna trial
    :param X:  dataset
    :return:
    """
    if X.index.nlevels == 2:  # support 2 level inddex (data/symbol)
        res = [trial_results(X, self.function, trial, sym=sym) for sym, X in X.groupby(level=1)]
        res = pd.concat(res, axis=0).sort_index()
    else:
        res = trial_results(X, self.function, trial)

    # Create consistent column names with function string and params
    name = col_name(self.function, trial.params)

    # Append integer identifier to DF with multiple columns
    if len(res.columns) > 1:
        res.columns = [f"{name}_{i}" for i in range(len(res.columns))]
    else:
        res.columns = [f"{name}"]
    return res


def _early_stopping_opt(study, trial):
    """
    Callback function to stop optuna trials early for single objective
    :param study:  Optuna study
    :param trial:  Optuna trial
    :return:
    """

    # Don't early stop on trial 0, also avoids ValueError when accessing study too soon
    if trial.number == 0:
        return

    if len([t.state for t in study.trials if t.state == TrialState.COMPLETE]) == 0:
        return

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


# Apply best trial parameters on multi-index dataframe
def eval_res(X, function, idx, trial, sym=None):
    if sym:
        X = X.droplevel(1)
    try:
        res = eval(function)
    except Exception as e:
        print(f"Error:  Function: {function}  Parameters: {trial.params}")
        raise Exception(e)
    if isinstance(res, tuple):
        res = res[idx]
    res = pd.DataFrame(res, index=X.index)
    if len(res.columns) > 1:
        res = pd.DataFrame(res.iloc[:, idx])
    if sym:
        res['sym'] = sym
        res.set_index('sym', append=True, inplace=True)
    return res


def _objective(self, trial, X, y):
    """
    Objective function used in Optuna trials
    :param self:  Optuna study
    :param trial:  Optuna trial
    :param X: Entire dataset
    :param y: Target
    :return:
    """

    # Execute trial function
    # try:
    if X.index.nlevels == 2:
        res = [eval_res(X, self.function, self.idx, trial, sym=sym) for sym, X in X.groupby(level=1)]
        res = pd.concat(res, axis=0).sort_index()
    else:
        res = eval_res(X, self.function, self.idx, trial)
    # except:
        # raise RuntimeError(f"Optuna execution error: {self.function}")

    # y may be a subset of X, so reduce result to y and convert to series
    res_y = res.reindex(y.index).iloc[:, 0].replace([np.inf, -np.inf], np.nan)

    # Obtain distance correlation
    if sum(np.isnan(res_y)) >= len(res_y)*.9:  # If mostly nans in result, return nan correlation
        correlation = np.nan
    else:
        # Ensure results and target are aligned with target by index
        res_tgt = pd.concat([res_y, y], axis=1)
        res_tgt.columns = ['results', 'target']

        # Measure Correlation
        fvi = res_tgt['results'].first_valid_index()
        res_tgt = res_tgt[res_tgt.index >= fvi]
        correlation = distance_correlation(np.array(res_tgt.target), np.array(res_tgt.results))

    # Save results
    trial.set_user_attr("correlation", correlation)
    trial.set_user_attr("res_y", res_y)

    return correlation


class Optimize():
    def __init__(self, function, n_trials=100, n_jobs=1):
        self.function = function
        self.n_trials = n_trials
        self.n_jobs = n_jobs

    def fit(self, X, y, idx=0, verbose=False, early_stop=None, n_jobs=1):
        """
        Optimize a technical indicator
        :param X: Historical dataset
        :param y: Target
        :param idx: Column to maximize if TA returnes multiple
        :param verbose: Verbosity level
        :param early_stop: Number of trials to perform without improvement before stopping
        :return:
        """

        start_time = timer()
        self.idx = idx

        # Display Optuna trial messages
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.ERROR)

        # Create optuna study maximizing correlation
        sampler = optuna.samplers.TPESampler(seed=123)
        self.study = optuna.create_study(direction='maximize', study_name=self.function, sampler=sampler)

        # Set required early stopping variables
        self.study.early_stop = early_stop
        self.study.early_stop_count = 0
        self.study.best_score = None

        # Start optimization trial
        try:
            self.study.optimize(lambda trial: _objective(self, trial, X, y),
                n_trials=self.n_trials, callbacks=[_early_stopping_opt])

        # Early stopping (not officially supported by Optuna)
        except optuna.exceptions.OptunaError:
            pass

        if len([t for t in self.study.trials if t.state == TrialState.COMPLETE]) == 0:  # Min 1 complete trial
            return self
        elif self.n_trials == 1:  # Indicators with no parameters
            best_trial = 0
        else:
            # Unique trials (converts params to json for duplicate comparison)
            trials = pd.DataFrame([[t.number, t.user_attrs['correlation'], t.params, json.dumps(t.params)] for t in self.study.trials if t.state == TrialState.COMPLETE])
            trials.columns = ['trial', 'correlation', 'params', 'json']
            trials.set_index('trial', drop=True, inplace=True)
            trials = trials[~trials.json.duplicated(keep='first')]
            trials = trials.drop(columns=['json'])

            # Scaler for cluster scoring
            mms = MinMaxScaler()

            # Trial correlations
            correlations = np.array(list(trials.correlation)).reshape(-1, 1)

            # Clusters of trial correlations
            if len(correlations) <= 7:
                num_clusters = 1
            else:
                max_clusters = int(min([20, len(correlations)/2]))
                ke = KElbowVisualizer(KMeans(random_state=123), k=(1, max_clusters))
                ke.fit(correlations)
                num_clusters = ke.elbow_value_
                if num_clusters is None:
                    num_clusters = int(len(correlations) * .2)
            kmeans = KMeans(n_clusters=num_clusters, random_state=123).fit(correlations.reshape(-1, 1))

            # Mean correlation per cluster, membership and score
            cluster_mean_correlation = [np.mean(trials[(kmeans.labels_ == c)].correlation) for c in range(num_clusters)]
            cluster_members = [(kmeans.labels_ == c).sum() for c in range(num_clusters)]
            clusters = pd.DataFrame([cluster_mean_correlation, cluster_members]).T
            clusters.columns = ['mean_correlation', 'members']

            # Choose best cluster
            df = pd.DataFrame(mms.fit_transform(clusters), index=clusters.index, columns=clusters.columns)
            clusters['score'] = df['mean_correlation'] + df['members']
            clusters = clusters.sort_values(by='score', ascending=False)
            cluster = clusters.score.idxmax()

            # Trials of best cluster
            trials = trials[kmeans.labels_ == cluster]

            # Trial parameters of best cluster
            num_params = len(self.study.best_params)
            params = []
            for i in range(0, num_params):
                params.append([list(p.values())[i] for p in trials.params])
            params = np.nan_to_num(np.array(params).T)

            if len(params) <= 7:
                num_clusters = 1
            else:
                # Clusters of trial parameters for best correlation cluster
                max_clusters = int(min([20, len(params)/2]))
                ke = KElbowVisualizer(KMeans(random_state=123), k=(1, max_clusters))
                ke.fit(params)
                num_clusters = ke.elbow_value_
                if num_clusters is None:
                    num_clusters = int(len(params) * .2)
            kmeans = KMeans(n_clusters=num_clusters, random_state=123).fit(params)

            # Mean correlation per cluster, membership and score
            cluster_mean_correlation = [np.mean(trials[(kmeans.labels_ == c)].correlation) for c in range(num_clusters)]
            cluster_members = [(kmeans.labels_ == c).sum() for c in range(num_clusters)]
            clusters = pd.DataFrame([cluster_mean_correlation, cluster_members]).T
            clusters.columns = ['mean_correlation', 'members']

            # Choose best cluster
            df = pd.DataFrame(mms.fit_transform(clusters), index=clusters.index, columns=clusters.columns)
            clusters['score'] = df['mean_correlation'] + df['members']
            clusters = clusters.sort_values(by='score', ascending=False)
            cluster = clusters.score.idxmax()

            # Choose center of cluster
            center = kmeans.cluster_centers_[cluster]
            center_matrix = np.vstack((center, params))
            distances = pd.DataFrame(euclidean_distances(center_matrix)[1:, :], index=trials.index)[0]
            best_trial = distances.sort_values().index[0]

            # look = trials[kmeans.labels_ == cluster]

        self.study.set_user_attr("best_trial_number", best_trial)
        self.study.set_user_attr("best_trial", self.study.trials[best_trial])
        self.study.set_user_attr("name", col_name(self.function, self.study.trials[best_trial].params))

        end_time = timer()
        self.time = round(end_time - start_time, 2)

        return self

    def transform(self, X):
        """
        Calculate TA indicator using fittted best trial
        :param X: Dataset
        :return:
        """

        # Calculate and store in features, replacing any potential non-finites (not sure needed)
        features = _trial(self, self.study.user_attrs['best_trial'], X)
        return features

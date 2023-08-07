import json
import warnings
from timeit import default_timer as timer

import numpy as np
import optuna
import pandas as pd
import pandas_ta as pta
import talib as tta
from finta import TA as fta
from kmodes.kprototypes import KPrototypes
from optuna.trial import TrialState
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer

from tuneta.config import pandas_ta_mamodes
from tuneta.utils import (
    col_name,
    distance_correlation,
    remove_consecutive_duplicates_and_nans,
)

warnings.filterwarnings("ignore")


# Apply trial on multi index
def trial_results(X, function, trial, sym=None):
    if sym:
        level_name = X.index.names[1]
        X = X.droplevel(1)
    try:
        res = eval(
            function
        )  # Eval contains reference to best trial (in argument) to re-use original parameters
    except Exception as e:
        raise Exception(e)
    if isinstance(res, tuple):
        try:
            res = pd.DataFrame(res).T
        except Exception as e:
            print("Error:")
            print(f"Function: {function}")
            print(f"X Length:  {len(X)}")
            for k, v in enumerate(res):
                u, c = np.unique(v.index, return_counts=True)
                dup = u[c > 1]
                print(f"Series {k} duplicates: {len(dup)}")
                print(v)
            raise Exception(e)

    res = pd.DataFrame(res, index=X.index)  # Ensure result aligns with X
    if sym:
        res[level_name] = sym
        res.set_index(level_name, append=True, inplace=True)
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
        res = [
            trial_results(X, self.function, trial, sym=sym)
            for sym, X in X.groupby(level=1)
        ]
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
            study.early_stop_count = study.early_stop_count + 1
    return


# Apply best trial parameters on multi-index dataframe
def eval_res(X, function, idx, trial, sym=None):
    if sym:
        level_name = X.index.names[1]
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
        res[level_name] = sym
        res.set_index(level_name, append=True, inplace=True)
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
        res = [
            eval_res(X, self.function, self.idx, trial, sym=sym)
            for sym, X in X.groupby(level=1)
        ]
        res = pd.concat(res, axis=0).sort_index()
    else:
        res = eval_res(X, self.function, self.idx, trial)

    # y may be a subset of X, so reduce result to y and convert to series
    if res.empty:
        res_y = None
        correlation = np.nan
    else:
        res_y = res.reindex(y.index).iloc[:, 0].replace([np.inf, -np.inf], np.nan)

        if self.remove_consecutive_duplicates:
            res_y = remove_consecutive_duplicates_and_nans(res_y)

        # Obtain distance correlation
        # Ensure results and target are aligned with target by index
        res_tgt = pd.concat([res_y, y], axis=1)
        res_tgt.columns = ["results", "target"]

        # Measure Correlation
        fvi = res_tgt["results"].first_valid_index()
        if fvi is None:
            correlation = np.nan
        else:
            res_tgt = res_tgt[res_tgt.index >= fvi]
            res_tgt.dropna(inplace=True)
            if np.all((np.array(res_tgt.results) == 0)):
                correlation = np.nan
            else:
                correlation = distance_correlation(
                    np.array(res_tgt.target), np.array(res_tgt.results)
                )

    # Save results
    trial.set_user_attr("correlation", correlation)
    trial.set_user_attr("res_y", res_y)

    return correlation


class Optimize:
    def __init__(self, function, n_trials=100, remove_consecutive_duplicates=False):
        self.function = function
        self.n_trials = n_trials
        self.remove_consecutive_duplicates = remove_consecutive_duplicates

    def fit(self, X, y, idx=0, max_clusters=10, verbose=False, early_stop=None):
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
        sampler = optuna.samplers.TPESampler(seed=42)
        self.study = optuna.create_study(
            direction="maximize", study_name=self.function, sampler=sampler
        )

        # Set required early stopping variables
        self.study.early_stop = early_stop
        self.study.early_stop_count = 0
        self.study.best_score = None

        # Start optimization trial
        try:
            self.study.optimize(
                lambda trial: _objective(self, trial, X, y),
                n_trials=self.n_trials,
                callbacks=[_early_stopping_opt],
                n_jobs=1,
            )

        # Early stopping (not officially supported by Optuna)
        except optuna.exceptions.OptunaError:
            pass

        if (
            len([t for t in self.study.trials if t.state == TrialState.COMPLETE]) == 0
        ):  # Min 1 complete trial
            return self
        elif self.n_trials == 1:  # Indicators with no parameters
            best_trial = 0
        else:
            # Unique trials (converts params to json for duplicate comparison)
            trials = pd.DataFrame(
                [
                    [
                        t.number,
                        t.user_attrs["correlation"],
                        t.params,
                        json.dumps(t.params),
                    ]
                    for t in self.study.trials
                    if t.state == TrialState.COMPLETE
                ]
            )
            trials.columns = ["trial", "correlation", "params", "json"]
            trials.set_index("trial", drop=True, inplace=True)
            trials = trials[~trials.json.duplicated(keep="first")]
            trials = trials.drop(columns=["json"])

            # Scaler for cluster scoring
            mms = MinMaxScaler()

            # Trial correlations
            correlations = np.array(list(trials.correlation)).reshape(-1, 1)

            # Clusters of trial correlations
            if len(correlations) <= 7:
                num_clusters = 1
            else:
                num_clusters = int(min([max_clusters * 2, len(correlations) / 2]))
                ke = KElbowVisualizer(KMeans(random_state=42), k=(1, num_clusters))
                ke.fit(correlations)
                num_clusters = ke.elbow_value_
                if num_clusters is None:
                    num_clusters = int(len(correlations) * 0.2)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(
                correlations.reshape(-1, 1)
            )

            # Mean correlation per cluster, membership and score
            cluster_mean_correlation = [
                np.mean(trials[(kmeans.labels_ == c)].correlation)
                for c in range(num_clusters)
            ]
            cluster_members = [(kmeans.labels_ == c).sum() for c in range(num_clusters)]
            clusters = pd.DataFrame([cluster_mean_correlation, cluster_members]).T
            clusters.columns = ["mean_correlation", "members"]

            # Choose best cluster
            df = pd.DataFrame(
                mms.fit_transform(clusters),
                index=clusters.index,
                columns=clusters.columns,
            )
            clusters["score"] = df["mean_correlation"] + df["members"]
            clusters = clusters.sort_values(by="score", ascending=False)
            cluster = clusters.score.idxmax()

            # Trials of best cluster
            trials = trials[kmeans.labels_ == cluster]

            # Trial parameters of best cluster
            num_params = len(self.study.best_params)
            params = []
            for i in range(0, num_params):
                params.append([list(p.values())[i] for p in trials.params])
            params = np.nan_to_num(np.array(params).T)
            if "mamode" in list(
                trials.params[trials.params.first_valid_index()].keys()
            ):
                index = list(
                    trials.params[trials.params.first_valid_index()].keys()
                ).index("mamode")
                params[:, index] = np.vectorize(pandas_ta_mamodes.get)(params[:, index])
                params = params.astype(float)
            if len(params) <= 7:
                num_clusters = 1
            else:
                try:
                    # Clusters of trial parameters for best correlation cluster
                    num_clusters = int(min([max_clusters, len(params) / 2]))
                    if "index" in locals():
                        ke = KElbowVisualizer(
                            KPrototypes(random_state=42), k=(1, num_clusters)
                        )
                        ke.fit(params, categorical=[index])
                    else:
                        ke = KElbowVisualizer(KMeans(random_state=42), k=(1, num_clusters))
                        ke.fit(params)
                    num_clusters = ke.elbow_value_
                except Exception as e:
                    num_clusters = None
                if num_clusters is None:
                    num_clusters = int(len(params) * 0.2)
            if "index" in locals():
                kmeans = KPrototypes(n_clusters=num_clusters, random_state=42).fit(
                    params, categorical=[index]
                )
            else:
                kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(params)

            # Mean correlation per cluster, membership and score
            cluster_mean_correlation = [
                np.mean(trials[(kmeans.labels_ == c)].correlation)
                for c in range(num_clusters)
            ]
            cluster_members = [(kmeans.labels_ == c).sum() for c in range(num_clusters)]
            clusters = pd.DataFrame([cluster_mean_correlation, cluster_members]).T
            clusters.columns = ["mean_correlation", "members"]

            # Choose best cluster
            df = pd.DataFrame(
                mms.fit_transform(clusters),
                index=clusters.index,
                columns=clusters.columns,
            )
            clusters["score"] = df["mean_correlation"] + df["members"]
            clusters = clusters.sort_values(by="score", ascending=False)
            cluster = clusters.score.idxmax()

            # Choose center of cluster
            if "index" in locals():
                center = kmeans.cluster_centroids_[cluster]
            else:
                center = kmeans.cluster_centers_[cluster]
            center_matrix = np.vstack((center, params))
            distances = pd.DataFrame(
                euclidean_distances(center_matrix)[1:, :], index=trials.index
            )[0]
            best_trial = distances.sort_values().index[0]

            # look = trials[kmeans.labels_ == cluster]

        self.study.set_user_attr("best_trial_number", best_trial)
        self.study.set_user_attr("best_trial", self.study.trials[best_trial])
        self.study.set_user_attr(
            "name", col_name(self.function, self.study.trials[best_trial].params)
        )

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
        features = _trial(self, self.study.user_attrs["best_trial"], X)
        return features

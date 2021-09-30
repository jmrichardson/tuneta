# from tuneta.optimize import _weighted_spearman
import pandas as pd
from tabulate import tabulate
import numpy as np
import re
from scipy.spatial.distance import squareform, pdist


def col_name(function, study_best_params):
    """
    Create consistent column names given string function and params
    :param function:  Function represented as string
    :param study_best_params:  Params for function
    :return:
    """

    # Optuna string of indicator
    function_name = function.split("(")[0].replace(".", "_")

    # Optuna string of parameters
    params = re.sub('[^0-9a-zA-Z_:,]', '', str(study_best_params)).replace(",", "_").replace(":", "_")

    # Concatenate name and params to define
    col = f"{function_name}_{params}"
    return col


def distance_correlation(x: np.array, y: np.array) -> float:
    """
    mlfinlab distance correlation function
    Returns distance correlation between two vectors. Distance correlation captures both linear and non-linear
    dependencies.
    Formula used for calculation:
    Distance_Corr[X, Y] = dCov[X, Y] / (dCov[X, X] * dCov[Y, Y])^(1/2)
    dCov[X, Y] is the average Hadamard product of the doubly-centered Euclidean distance matrices of X, Y.
    Read Cornell lecture notes for more information about distance correlation:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes.
    :param x: (np.array/pd.Series) X vector.
    :param y: (np.array/pd.Series) Y vector.
    :return: (float) Distance correlation coefficient.
    """

    x = x[:, None]
    y = y[:, None]

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    a = squareform(pdist(x))
    b = squareform(pdist(y))

    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    d_cov_xx = (A * A).sum() / (x.shape[0] ** 2)
    d_cov_xy = (A * B).sum() / (x.shape[0] ** 2)
    d_cov_yy = (B * B).sum() / (x.shape[0] ** 2)

    coef = np.sqrt(d_cov_xy) / np.sqrt(np.sqrt(d_cov_xx) * np.sqrt(d_cov_yy))

    return coef



# import seaborn as sns
# import matplotlib.pyplot as plt

def gen_plot(indicators, title):
    data = pd.DataFrame()
    for fitted in indicators.fitted:
        fitted.fitness = []
        fitted.length = []
        for trial in fitted.study.trials:
            print(trial)
            fitted.fitness.append(trial.value)
            fitted.length.append(trial.params['length'])
        fitted.fitness = pd.Series(fitted.fitness, name="Correlation")
        fitted.length = pd.Series(fitted.length, name="Length")
        fitted.data = pd.DataFrame([fitted.fitness, fitted.length]).T
        fitted.fn = fitted.function.split('(')[0]
        fitted.data['Indicator'] = fitted.fn
        data = pd.concat([data, fitted.data])
        fitted.x = fitted.study.best_params['length']
        fitted.y = fitted.study.best_value

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Length", y="Correlation", data=data, hue="Indicator")
    plt.title(title)
    for fit in indicators.fitted:
        plt.vlines(x=fit.x, ymin=0, ymax=fit.y, linestyles='dotted')


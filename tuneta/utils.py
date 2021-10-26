# from tuneta.optimize import _weighted_spearman
import pandas as pd
from tabulate import tabulate
import numpy as np
import re
from scipy.spatial.distance import squareform, pdist
import dcor


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

    # Remove any trailing underscores
    col = re.sub(r'_$', '', col)
    return col


def distance_correlation(a, b):
    return dcor.distance_correlation(a, b)


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


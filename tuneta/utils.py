from tuneta.optimize import _weighted_spearman
import pandas as pd
from tabulate import tabulate


def corr_xy(X, y):
    X = X.loc[:, ~X.columns.duplicated()]  # Remove duplicate columns
    fitness = []
    for col in X.columns:
        fitness.append(_weighted_spearman(y, X[col]))
    df = pd.DataFrame([X.columns, fitness]).T
    df.columns = ['Features', 'Correlation']
    df = df.sort_values(by=['Correlation'], ascending=False)
    print(tabulate(df, headers=df.columns, tablefmt="simple"))
    return df

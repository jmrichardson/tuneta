import pandas as pd
from pandas_ta import percent_return
from sklearn.model_selection import train_test_split
import yfinance as yf
from catch22 import catch22_all
from tuneta.tune_ta import TuneTA


if __name__ == "__main__":
    # Download data set from yahoo
    X = yf.download("SPY", period="10y", interval="1d", auto_adjust=True)

    # Add catch22 30 day rolling features for demonstration purposes
    c22 = [pd.Series(catch22_all(r)['values']) for r in X.Close.rolling(30) if len(r) == 30]
    features = pd.concat(c22, axis=1).T
    features.columns = catch22_all(X.Close.tail(30)).get('names')
    X = X.tail(len(features))
    features.index = X.index
    X = pd.concat([X, features], axis=1)

    # Add target return
    y = percent_return(X.Close, offset=-1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, shuffle=False)

    # Initialize
    tt = TuneTA(n_jobs=4, verbose=True)

    # Features to keep
    feature_names = tt.prune_df(X_train, y_train, max_correlation=.7, report=True)

    # Filter datasets
    X_train = X_train[feature_names]
    X_test = X_test[feature_names]

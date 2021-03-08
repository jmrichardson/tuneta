import yfinance as yf
import pandas as pd
from pandas_ta import percent_return
from tuneta.tune_ta import TuneTA
from sklearn.model_selection import train_test_split
import numpy as np


if __name__ == "__main__":
    # Download data set from yahoo, calculate next day return and split into train and test
    X = yf.download("SPY", period="10y", interval="1d", auto_adjust=True)
    y = percent_return(X.Close, offset=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, shuffle=False)

    # Initialize with x cores and show trial results
    tt = TuneTA(n_jobs=2, verbose=True)

    # Optimize indicators
    tt.fit(X_train, y_train,
        # List of indicators to optimize.  You can speficy all packages ('all'),
        # specific packages ('tta', 'pta', 'fta') and/or specific indicator(ie, 'tta.MACD', 'pta.ema')
        # ":1" optimizes column 1 instead of default 0 if indicator returns dataframe
        # Examples: ['all'], ['tta', 'fta'], ['pta', 'tta.MACD', 'tta.ARRON:1]
        indicators=['tta'],  # Optimize all TA-Lib indicators
        ranges=[(3, 180)],  # Period range(s) to tune for each indicator
        trials=200,  # Number of optimization trials per indicator per range
        split=None,  # Optimize across all of X_train (single-objective)
        # Split points are used for multi-objective optimization
        # 3 split points (num=3) below defines two splits (begin, middle, end)
        # split=np.linspace(0, len(X_train), num=3).astype(int),
        early_stop=30,  # Stop after x number of trials without improvement
        spearman=True,  # Type of correlation metric (Set False for Pearson)
        weights=None,  # Optional weights for correlation evaluation
    )

    # Show time duration in seconds per indicator
    tt.fit_times()

    # Show correlation of indicators to target
    tt.report(target_corr=True, features_corr=False)

    # Take top x tuned indicators, and select y with the least intercorrelation
    tt.prune(top=30, studies=10)

    # Show correlation of indicators to target and among themselves
    tt.report(target_corr=True, features_corr=True)

    # Add indicators to X_train
    features = tt.transform(X_train)
    X_train = pd.concat([X_train, features], axis=1)

    # Add same indicators to X_test
    features = tt.transform(X_test)
    X_test = pd.concat([X_test, features], axis=1)

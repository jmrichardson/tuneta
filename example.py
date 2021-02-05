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

    tta = TuneTA(n_jobs=2, verbose=True)  # Initialize with 2 cores and show trial results
    tta.fit(X_train, y_train,
                   # Indicators to tune / optimize
                   # ":1" means optimize column index 1 vs default 0 if indicator returns dataframe
                   indicators=["tta.MACD", "tta.ULTOSC", "tta.AROON:1", "pta.rsi", "pta.kst", "pta.apo", "pta.zlma", "fta.ADX"],
                   # indicators=["tta.MACD", "tta.ULTOSC", "pta.kst"],
                   ranges=[(2, 260)],  # Period range(s) to tune for each indicator
                   trials=100,  # Number of optimization trials per indicator per range
                   split=np.linspace(0, len(X_train), num=3).astype(int),  # Define split points
                   early_stop=20,  # Stop after number of trials without improvement
                   spearman=True,  # Type of correlation metric (Set False for Pearson)
                   weights=None,  # Optional weights for correlation evaluation
                   )
    tta.report(target_corr=True, features_corr=True)  # Show correlation report

    # Take top x tuned indicators, and select y with the least intercorrelation
    tta.prune(top=6, studies=4)
    tta.report(target_corr=True, features_corr=True)  # Show correlation report

    # Add indicators to X_train
    features = tta.transform(X_train)
    X_train = pd.concat([X_train, features], axis=1)

    # Add same indicators to X_test
    features = tta.transform(X_test)
    X_test = pd.concat([X_test, features], axis=1)
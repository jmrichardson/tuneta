import yfinance as yf
import pandas as pd
from pandas_ta import percent_return
from tuneta.tune_ta import TuneTA
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # Download data set from yahoo, calculate next day return and split into train and test
    X = yf.download("SPY", period="10y", interval="1d", auto_adjust=True)
    y = percent_return(X.close)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

    # Default configuration is to tune all supported indicators
    indicators = TuneTA(n_jobs=2)
    indicators.fit(X_train, y_train,
                   indicators=["pta.macd", "pta.ao"],  # Indicators to tune/optimize
                   ranges=[(0, 20), (21, 100)],  # Period ranges to tune for each indicator
                   trials=5  # Number of optimization trials per indicator per range
                   )

    # Take 10 tuned indicators, and select the 3 least correlated with each other
    indicators.prune(top=10, studies=3)

    # Add indicators to X_train
    features = indicators.transform(X_train)
    X_train = pd.concat([X_train, features], axis=1)

    # Add same indicators to X_test
    features = indicators.transform(X_test)
    X_test = pd.concat([X_test, features], axis=1)

    print("done")

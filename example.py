import yfinance as yf
import pandas as pd
from pandas_ta import percent_return
from tuneta.tune_ta import TuneTA
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # Download data set from yahoo, calculate next day return and split into train and test
    X = yf.download("SPY", period="10y", interval="1d", auto_adjust=True)
    y = percent_return(X.Close, offset=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, shuffle=False)

    indicators = TuneTA(n_jobs=2, verbose=True)  # Initialize with 2 cores and show trial results
    indicators.fit(X_train, y_train,
                   # Indicators to tune / optimize
                   # ":1" means optimize on column index 1 vs default of 0 if indicator returns dataframe
                   indicators=["tta.MACD", "tta.ULTOSC", "tta.AROON:1", "pta.rsi", "fta.OBV"],
                   ranges=[(2, 180)],  # Period range(s) to tune for each indicator
                   trials=30  # Number of optimization trials per indicator per range
                   )

    # Take top 3 tuned indicators, and select the 2 least correlated with each other
    indicators.prune(top=3, studies=2)

    # Add indicators to X_train
    features = indicators.transform(X_train)
    X_train = pd.concat([X_train, features], axis=1)

    # Add same indicators to X_test
    features = indicators.transform(X_test)
    X_test = pd.concat([X_test, features], axis=1)
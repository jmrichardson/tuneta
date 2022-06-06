from tuneta.tune_ta import TuneTA
import pandas as pd
from pandas_ta import percent_return
from sklearn.model_selection import train_test_split
import yfinance as yf


if __name__ == "__main__":
    # Download data set from yahoo, calculate next day return and split into train and test
    X = yf.download("SPY", period="10y", interval="1d", auto_adjust=True)
    y = percent_return(X.Close, offset=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, shuffle=False)

    # Initialize with x cores and show trial results
    tt = TuneTA(n_jobs=4, verbose=True)

    # Optimize indicators
    tt.fit(X_train, y_train,
        indicators=['tta.RSI'],
        ranges=[(4, 30), (31, 180)],
        trials=300,
        early_stop=50,
    )

    # Show correlation of indicators
    tt.report(target_corr=True, features_corr=True)

    # Add indicators to X_train
    features = tt.transform(X_train)
    X_train = pd.concat([X_train, features], axis=1)

    # Add same indicators to X_test
    features = tt.transform(X_test)
    X_test = pd.concat([X_test, features], axis=1)


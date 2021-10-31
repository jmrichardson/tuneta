from tuneta.tune_ta import TuneTA
import pandas as pd
from pandas_ta import percent_return
from sklearn.model_selection import train_test_split
import yfinance as yf
import joblib


if __name__ == "__main__":
    # Download data, calculate next day return and symbol index
    aapl = yf.download("AAPL", period="10y", interval="1d", auto_adjust=True)
    aapl['sym'] = "AAPL"
    aapl.set_index('sym', append=True, inplace=True)
    aapl['return'] = percent_return(aapl.Close, offset=-1)

    msft = yf.download("MSFT", period="10y", interval="1d", auto_adjust=True)
    msft['sym'] = "MSFT"
    msft.set_index('sym', append=True, inplace=True)
    msft['return'] = percent_return(msft.Close, offset=-1)
    
    goog = yf.download("GOOG", period="10y", interval="1d", auto_adjust=True)
    goog['sym'] = "GOOG"
    goog.set_index('sym', append=True, inplace=True)
    goog['return'] = percent_return(goog.Close, offset=-1)

    X = pd.concat([aapl, msft, goog], axis=0).sort_index()
    y = X['return']
    X = X.drop(columns=['return'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, shuffle=False)

    # Initialize with x cores and show trial results
    tt = TuneTA(n_jobs=4, verbose=True)

    # Optimize indicators
    tt.fit(X_train, y_train,
        indicators=['tta.RSI', 'tta.MACD', 'tta.SMA', 'tta.CMO'],
        ranges=[(2, 30), (31, 60)],
        trials=300,
        early_stop=50,
    )

    # Show time duration in seconds per indicator
    tt.fit_times()

    # Show correlation of indicators to target and among themselves
    tt.report(target_corr=True, features_corr=True)

    # Select features with at most x correlation between each other
    tt.prune(max_inter_correlation=.85)

    # Show correlation of indicators to target and among themselves
    tt.report(target_corr=True, features_corr=True)

    # Add indicators to X_train
    features = tt.transform(X_train)
    X_train = pd.concat([X_train, features], axis=1)

    # Add same indicators to X_test
    features = tt.transform(X_test)
    X_test = pd.concat([X_test, features], axis=1)


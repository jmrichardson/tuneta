<p align="center">
  <a href="https://github.com/jmrichardson/tuneta">
    <img src="images/logo.png" alt="tuneTA">
  </a>
</p>

TuneTA optimizes a broad set of technical indicators to maximize its correlation to a user defined target variable.  The set of tuned indicators can be reduced to the most correlated with the target with the least correlation with each other. TuneTA maintains its state of optimized and reduced indicators which can be used to easily add to multiple data sets (train, validation, test).

### Features

* Given financial prices (OHLCV) and return (X and y respectively), optimize technical indicator parameters to maximize the correlation to return.  Multiple ranges can be defined to target specific periods of time
* Select the top X optimized indicators with most correlation to return and select X with the least correlation to each other with the goal of improving downstream ML models with uncorrelated features
* Persists state to create identical indicators on multiple datasets (train/test).
* Parallel processing

### Overview

"A picture is worth a thousand words"

<p align="center">
  <a href="https://github.com/jmrichardson/tuneta">
    <img src="images/ico.jpg" alt="tuneTA">
  </a>
</p>

<p align="center">
  <a href="https://github.com/jmrichardson/tuneta">
    <img src="images/ic2.jpg" alt="tuneTA">
  </a>
</p>

### Install

### Install

```python
pip install tuneta
```

### Example Usage

```python
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

    # Take top 10 correlated indicators to return, and then select the 3 least correlated with each other
    indicators.prune(top=10, studies=3)

    # Add indicators to X_train
    features = indicators.transform(X_train)
    X_train = pd.concat([X_train, features], axis=1)

    # Add same indicators to X_test
    features = indicators.transform(X_test)
    X_test = pd.concat([X_test, features], axis=1)
```





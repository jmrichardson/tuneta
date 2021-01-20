<p align="center">
  <a href="https://github.com/jmrichardson/tuneta">
    <img src="images/logo.png" alt="tuneTA">
  </a>
</p>

### Features

* Given financial prices (OHLCV) and return (X and y respectively), optimize technical indicator parameters to maximize correlation to return.  Multiple ranges can be defined to target specific periods of time.
* Select top indicators with most correlation to return with least correlation to each other with the goal of improving ML models with uncorrelated features
* Persists state to create identical indicators on multiple datasets (train/test)
* Parallel processing

### Install

```python
pip install tuneta
```

### Example Usage

```python
import yfinance as yf
ohlcv = yf.download("SPY", period="ytd", interval="1d", auto_adjust=True)


```





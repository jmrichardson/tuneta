<p align="center">
  <a href="https://github.com/jmrichardson/tuneta">
    <img src="images/logo.png" alt="tuneTA">
  </a>
</p>

TuneTA intelligently optimizes one or many technical indicators to optimize its distance correlation to a user defined target variable.  Indicator parameter(s) are selected using KMeans clustering to avoid "peak" or "lucky" values.  The set of tuned indicators can further be reduced by choosing the most correlated with the target while minimizing correlation with each other. TuneTA maintains its state to add all tuned indicators to multiple data sets (train, validation, test).

### Features

* Given financial prices (OHLCV) and a target variable such as return, TuneTa optimizes the parameter(s) of each technical indicator to optimize the distance correlation to the target variable. Distance correlation captures linear and non-linear strength versus the widely used Pearson correlation.
* Optimal indicator parameters are selected in a multi-step clustering process to avoid values which are not consistent with neighboring values providing a more robust selection criteria than many other optimization frameworks.
* Select top X indicators with the least correlation to each other.  This is helpful for machine learning models which generally perform better with minimal feature intercorrelation.
* Persist state to generate identical indicators on multiple datasets (train, validation, test)
* Early stopping
* Correlation report of target and features
* Parallel processing
* Supports technical indicators produced from the following packages.  See config.py for indicators supported.
  * [Pandas TA](https://github.com/twopirllc/pandas-ta)
  * [TA-Lib](https://github.com/mrjbq7/ta-lib)
  * [FinTA](https://github.com/peerchemist/finta)

### Overview

TuneTA simplifies the process of optimizing technical indicators while avoiding "peak" values and selecting the best with minimal correlation between each other (optional). At a high level, TuneTA performs the following steps:

1.  For each indicator, [Optuna](https://optuna.org) searches for the parameter(s) which maximizes its correlation to the user defined target (ie next x day return).
2.  After the specified Optuna trials are complete, a 3-step KMeans clustering method is used to select the best parameter(s):

    1. Each trial is placed in its nearest neighbor cluster based on its distance correlation to the target.  The optimal number of clusters is determined using the elbow method.  The cluster with the highest average correlation is selected with respect to its membership.  In other words, a weighted score is used to select the cluster with highest correlation but also with the most trials.
    2. After the best correlation cluster is selected, the parameter(s) of the trials within that cluster are also clustered. Again, the best cluster of parameter(s) is selected with respect to its membership.
    3. Finally, the centered best trial is selected from the best parameter cluster.
3.  Optionally, the tuned parameters can be reduced by selecting the top x indicators with the least intercorrelation.
4.  Finally, TuneTA will generate each indicator with the best parameters.

### Installation

Install the latest code (recommended):

```python
pip install -U git+https://github.com/jmrichardson/tuneta
```

Install the latest release:

```python
pip install -U tuneta
```

### Example Usage

```
python example.py 
```

<p align="center">
  <a href="https://github.com/jmrichardson/tuneta">
    <img src="images/logo.png" alt="tuneTA">
  </a>
</p>

TuneTA optimizes technical indicators using [distance correlation](https://towardsdatascience.com/introducing-distance-correlation-a-superior-correlation-metric-d569dc8900c7) to a user defined target variable.  Indicator parameter(s) are selected using clustering techniques to avoid "peak" or "lucky" values.  The set of tuned indicators can further be reduced by choosing the most correlated with the target while minimizing correlation with each other. TuneTA maintains its state to add all tuned indicators to multiple data sets (train, validation, test).

### Features

* Given financial prices (OHLCV) and a target variable such as return, TuneTA optimizes the parameter(s) of each technical indicator using distance correlation to the target variable. Distance correlation captures both linear and non-linear strength.
* Optimal indicator parameters are selected in a multi-step clustering process to avoid values which are not consistent with neighboring values providing a more robust selection.
* Selects indicators with the least correlation to each other.  This is helpful for machine learning models which generally perform better with minimal feature intercorrelation.
* Persists state to generate identical indicators on multiple datasets (train, validation, test)
* Correlation report of target and features
* Parallel processing
* Supports technical indicators produced from the following packages.  See config.py for indicators supported.
  * [Pandas TA](https://github.com/twopirllc/pandas-ta)
  * [TA-Lib](https://github.com/mrjbq7/ta-lib)
  * [FinTA](https://github.com/peerchemist/finta)
* Early stopping

### Overview

TuneTA simplifies the process of optimizing many technical indicators while avoiding "peak" values, and selecting the best with minimal correlation between each other (optional). At a high level, TuneTA performs the following steps:

1.  For each indicator, [Optuna](https://optuna.org) searches for the parameter(s) which maximize its correlation to the user defined target (for example, next day return).
2.  After the specified Optuna trials are complete, a 3-step KMeans clustering method is used to select the optimal parameter(s):

    1. Each trial is placed in its nearest neighbor cluster based on its distance correlation to the target.  The optimal number of clusters is determined using the elbow method.  The cluster with the highest average correlation is selected with respect to its membership.  In other words, a weighted score is used to select the cluster with highest correlation but also with the most trials.
    2. After the best correlation cluster is selected, the parameters of the trials within the cluster are also clustered. Again, the best cluster of indicator parameter(s) are selected with respect to its membership.
    3. Finally, the centered best trial is selected from the best parameter cluster.
    
3.  Optionally, the tuned indicators can be pruned by selecting the indicators with a maximum correlation to the other indicators.
4.  Finally, TuneTA generates each indicator feature with the best parameters.
---

### Installation

Install the latest code (recommended):

```python
pip install -U git+https://github.com/jmrichardson/tuneta
```

Install the latest release:

```python
pip install -U tuneta
```

---

### Examples

Under construction...
* [Tune RSI Indicator](#tune-rsi-indicator)
* [Tune Multiple Indicators](#tune-multiple-indicators)



#### Tune RSI Indicator

For simplicity, lets optimize a single indicator:

* RSI Indicator
* Two time periods (short and long term): 2-30 and 31-180
* Maximum of 500 trials per time period to search for the best indicator parameter
* Stop after 100 trials per time period without improvement

    Note: **config.py** contains the list of indicators supported from each TA package
    
The following is a snippet of the complete example found in the examples directory:

```python
# Initialize with x cores and show trial results
tt = TuneTA(n_jobs=4, verbose=True)

# Optimize indicators
tt.fit(X_train, y_train,
    indicators=['tta.RSI'],
    ranges=[(2, 30), (31, 180)],
    trials=500,
    early_stop=100,
)
```

Two studies are created for each time period with up to 200 trials to test different indicator length values.  The correlation values are displayed based on the trial parameter.  The best trial with its respective parameter value is saved for both time ranges. 

To view the correlation of both indicators to the target return as well as each other:
```python
# Show correlation of indicators
tt.report(target_corr=True, features_corr=True)
```
```csharp
Indicator Correlation to Target:

                         Correlation
---------------------  -------------
tta_RSI_timeperiod_19       0.23393
tta_RSI_timeperiod_36       0.227434

Indicator Correlation to Each Other:

                         tta_RSI_timeperiod_19    tta_RSI_timeperiod_36
---------------------  -----------------------  -----------------------
tta_RSI_timeperiod_19                  0                        0.93175
tta_RSI_timeperiod_36                  0.93175                  0
```
To generate both RSI indicators on a data set:
```python
features = tt.transform(X_train)
```

```csharp
            tta_RSI_timeperiod_19  tta_RSI_timeperiod_36
Date                                                    
2011-10-03                    NaN                    NaN
2011-10-04                    NaN                    NaN
2011-10-05                    NaN                    NaN
2011-10-06                    NaN                    NaN
2011-10-07                    NaN                    NaN
...                           ...                    ...
2018-09-25              62.173261              60.713051
2018-09-26              59.185666              59.362731
2018-09-27              61.026238              60.210235
2018-09-28              61.094793              60.241806
2018-10-01              63.384824              61.305540
```

### Tune Multiple Indicators

Lets optimize a handful of indicators:

* Basket of indicators from multiple packages
* One time periods: 2-60
* Maximum of 200 trials per time period to search for the best indicator parameter
* Stop after 50 trials per time period without improvement

Note: config.py contains the list of indicators supported for each TA package

```python
    tt = TuneTA(n_jobs=2, verbose=True)
    tt.fit(X_train, y_train, indicators=['pta.stoch', 'pta.slope', 'tta.MOM', 'fta.SMA'], ranges=[(2, 60)], trials=200, early_stop=50)
```

View each indicator's distance correlation to target:

```csharp
                                  Correlation
------------------------------  -------------
pta_stoch_k_51_d_11_smooth_k_4       0.216212
tta_MOM_timeperiod_14                0.212336
pta_slope_length_22                  0.201844
fta_SMA_period_6                     0.099375
```

As in the previous example, we can easily create features:

```python
features = tt.transform(X_train)
```

```csharp
            pta_stoch_k_51_d_11_smooth_k_4_0  pta_stoch_k_51_d_11_smooth_k_4_1  tta_MOM_timeperiod_14  pta_slope_length_22  fta_SMA_period_6
Date                                                                                                                                        
2011-10-03                               NaN                               NaN                    NaN                  NaN               NaN
2011-10-04                               NaN                               NaN                    NaN                  NaN               NaN
2011-10-05                               NaN                               NaN                    NaN                  NaN               NaN
2011-10-06                               NaN                               NaN                    NaN                  NaN               NaN
2011-10-07                               NaN                               NaN                    NaN                  NaN               NaN
...                                      ...                               ...                    ...                  ...               ...
2018-09-25                         90.393165                         91.254190               2.867035             0.269386        275.953323
2018-09-26                         85.712117                         91.337471               2.863281             0.158049        275.997752
2018-09-27                         83.432356                         91.128202               4.160370             0.095541        276.121440
2018-09-28                         82.802412                         90.553986               3.716797             0.090829        275.878464
2018-10-01                         84.230122                         89.847810               3.777649             0.067411        275.837362
```

### Tune and Prune all Ta-Lib Indicators

Lets optimize all ta-lib indicators:

* All ta-lib indicators
* 3 time periods: 2-30, 31-90, 91-280
* Maximum of 200 trials per time period to search for the best indicator parameter
* Stop after 50 trials per time period without improvement
* Keep best indicators with the least intercorrelation

Note: config.py contains the list of indicators supported for each TA package

```python
    tt = TuneTA(n_jobs=6, verbose=True)
    tt.fit(X_train, y_train, indicators=['tta'], ranges=[(2, 30), (31, 90), (91, 280)], trials=200, early_stop=50)
```

To view the correlation to the target and to each other.  For brevity, not displaying results:

```python
tt.report(target_corr=True, features_corr=True)
```

To keep the best indicators with the least intercorrelation:
```python
tt.prune(top=30) 
```


### Complete Example Code

A working example is provided in example.py

*** Under construction...









# Search ranges
ranges = [(1, 90), (91, 270)]

# Dont use ichimoku or DPO (data leakage)
indicators = ["pta.rsi", "pta.mom", "pta.macd"]

# Series data
tune_series = ['open', 'high', 'low', 'close', 'volume']

# Parameters to tune
tune_params = ['fast', 'slow', 'signal', 'length', 'short', 'drift', 'max_lookback', 'min_lookback', 'initial',
               'lower_length', 'upper_length', 'lookback', 'medium', 'slow_w', 'roc1', 'roc2', 'roc3', 'roc4',
               'sma1', 'sma2', 'sma3', 'sma4', 'offset', 'scalar', 'width', 'long', 'atr_length', 'high_length',
               'low_length', 'kc_length', 'bb_length', 'mom_length', 'swma_length', 'rvi_length', 'period',
               'er', 'ema_fast', 'ema_slow', 'period_fast', 'period_slow', 'rsi_period', 'wma_period', 'atr_period',
               'upper_period', 'lower_period', 'stoch_period', 'slow_period', 'fast_period', 'r1', 'r2', 'r3', 'r4',
               'long', 'short', 'signal', 'short_period', 'long_period', 'channel_lenght', 'channel_length',
               'average_lenght', 'average_length', 'tenkan_period', 'kijun_period', 'senkou_period', 'chikou_period',
               'k_period', 'timeperiod', 'timeperiod1', 'timeperiod2', 'timeperiod3', 'slowlimit', 'fastlimit',
               'minperiod', 'maxperiod', 'acceleration', 'accelerationlong', 'accelerationshort', 'signalperiod',
               'slowperiod', 'signalperiod', 'fastperiod', 'fastk_period', 'slowk_period', 'slowd_period',
               'fastd_period']

# tune_params = ['th']

# Index of column to maximize if indicator returns multiple
tune_column = {
    "pta.aobv": 5,
    "pta.thermo": 2,
    "pta.vp": 5,
    "pta.bbands": 3,
    "pta.aroon": 2,
    "pta.macd": 2,
}

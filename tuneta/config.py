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
               'low_length', 'kc_length', 'bb_length', 'mom_length', 'swma_length', 'rvi_length']
tune_params = ['length']

# Index of column to maximize if indicator returns multiple
tune_column = {
    "pta.aobv": 5,
    "pta.thermo": 2,
    "pta.vp": 5,
    "pta.bbands": 3,
    "pta.aroon": 2,
    "pta.macd": 2,
}

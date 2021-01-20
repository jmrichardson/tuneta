# Search ranges
# ranges = [(0, 500), (501, 3600), (3601, 15000)]
ranges = [(0, 10), (11, 50), (51, 100)]

# Dont use ichimoku or DPO (data leakage)
indicators = [
    'pta.accbands',
    'pta.ao',
    'pta.aobv',
    'pta.amat',
    'pta.bias',
    'pta.brar',
    'pta.cmf',
    'pta.decay',
    'pta.donchian',
    'pta.entropy',
    'pta.eom',
    'pta.eri',
    'pta.er',
    'pta.efi',
    'pta.kdj',
    'pta.kst',
    'pta.kurtosis',
    'pta.macd',
    'pta.massi',
    'pta.nvi',
    'pta.pdist',
    'pta.pvi',
    'pta.pvo',
    'pta.pvt',
    'pta.qstick',
    'pta.rma',
    'pta.rvi',
    'pta.smi',
    'pta.thermo',
    'pta.thermo',
    'pta.vwma',
    'pta.vwap',
    'pta.wcp',
    'pta.uo',
    'pta.zlma',
]

### Advanced configuration below ###

# Series data
tune_series = ['open', 'high', 'low', 'close', 'volume']

# Parameters to tune
tune_params = ['fast', 'slow', 'signal', 'length', 'short', 'drift', 'max_lookback', 'min_lookback', 'initial',
               'lower_length', 'upper_length', 'lookback', 'medium', 'slow_w', 'roc1', 'roc2', 'roc3', 'roc4',
               'sma1', 'sma2', 'sma3', 'sma4', 'offset', 'scalar']

# Index of column to maximize if indicator returns multiple
tune_column = {
    "pta.aobv": 5,
    "pta.thermo": 2,
}

# Series data
tune_series = ['open', 'high', 'low', 'close', 'volume']

# Parameters to tune
tune_params = ['width', 'fast', 'slow', 'signal', 'length', 'max_lookback', 'min_lookback', 'initial',
               'lower_length', 'upper_length', 'lookback', 'medium', 'slow_w', 'roc1', 'roc2', 'roc3', 'roc4',
               'sma1', 'sma2', 'sma3', 'sma4', 'scalar', 'long', 'atr_length', 'high_length',
               'low_length', 'kc_length', 'bb_length', 'mom_length', 'swma_length', 'rvi_length', 'period',
               'er', 'ema_fast', 'ema_slow', 'period_fast', 'period_slow', 'rsi_period', 'wma_period', 'atr_period',
               'upper_period', 'lower_period', 'stoch_period', 'slow_period', 'fast_period', 'r1', 'r2', 'r3', 'r4',
               'short', 'signal', 'short_period', 'long_period', 'channel_lenght', 'channel_length',
               'average_lenght', 'average_length', 'tenkan_period', 'kijun_period', 'senkou_period', 'chikou_period',
               'k_period', 'timeperiod', 'timeperiod1', 'timeperiod2', 'timeperiod3', 'slowlimit', 'fastlimit',
               'minperiod', 'maxperiod', 'acceleration', 'accelerationlong', 'accelerationshort', 'signalperiod',
               'slowperiod', 'signalperiod', 'fastperiod', 'fastk_period', 'slowk_period', 'slowd_period',
               'fastd_period','drift','bb_std','ddof','std','tclength','mom_smooth']

tune_ma_params = [
        "dema", "ema", "fwma", "hma", "linreg", "midpoint", "pwma", "rma","sinwma", "sma", "swma", "t3", "tema", "trima", "vidya", "wma", "zlma"
    ]

talib_indicators = ['tta.BBANDS', 'tta.DEMA', 'tta.EMA', 'tta.HT_TRENDLINE', 'tta.KAMA', 'tta.MA',
                        'tta.MIDPOINT', 'tta.MIDPRICE', 'tta.SAR', 'tta.SAREXT', 'tta.SMA', 'tta.T3', 'tta.TEMA',
                        'tta.TRIMA',
                        'tta.WMA', 'tta.ADX', 'tta.ADXR', 'tta.APO', 'tta.AROON:1', 'tta.AROONOSC', 'tta.BOP',
                        'tta.CCI', 'tta.CMO',
                        'tta.DX', 'tta.MACD', 'tta.MACDEXT', 'tta.MACDFIX', 'tta.MFI', 'tta.MINUS_DI', 'tta.MINUS_DM',
                        'tta.MOM',
                        'tta.PLUS_DI', 'tta.PLUS_DM', 'tta.PPO', 'tta.ROC', 'tta.ROCP', 'tta.ROCR', 'tta.ROCR100',
                        'tta.RSI', 'tta.STOCH',
                        'tta.STOCHF', 'tta.STOCHRSI', 'tta.TRIX', 'tta.ULTOSC', 'tta.WILLR', 'tta.AD', 'tta.ADOSC',
                        'tta.OBV',
                        'tta.HT_DCPERIOD', 'tta.HT_DCPHASE', 'tta.HT_PHASOR', 'tta.HT_SINE', 'tta.HT_TRENDMODE',
                        'tta.AVGPRICE', 'tta.MEDPRICE',
                        'tta.TYPPRICE', 'tta.WCLPRICE', 'tta.ATR', 'tta.NATR', 'tta.TRANGE', 'tta.CDL2CROWS',
                        'tta.CDL3BLACKCROWS',
                        'tta.CDL3INSIDE', 'tta.CDL3LINESTRIKE', 'tta.CDL3OUTSIDE', 'tta.CDL3STARSINSOUTH',
                        'tta.CDL3WHITESOLDIERS',
                        'tta.CDLABANDONEDBABY', 'tta.CDLADVANCEBLOCK', 'tta.CDLBELTHOLD', 'tta.CDLBREAKAWAY',
                        'tta.CDLCLOSINGMARUBOZU',
                        'tta.CDLCONCEALBABYSWALL', 'tta.CDLCOUNTERATTACK', 'tta.CDLDARKCLOUDCOVER', 'tta.CDLDOJI',
                        'tta.CDLDOJISTAR',
                        'tta.CDLDRAGONFLYDOJI', 'tta.CDLENGULFING', 'tta.CDLEVENINGDOJISTAR', 'tta.CDLEVENINGSTAR',
                        'tta.CDLGAPSIDESIDEWHITE',
                        'tta.CDLGRAVESTONEDOJI', 'tta.CDLHAMMER', 'tta.CDLHANGINGMAN', 'tta.CDLHARAMI',
                        'tta.CDLHARAMICROSS',
                        'tta.CDLHIGHWAVE', 'tta.CDLHIKKAKE', 'tta.CDLHIKKAKEMOD', 'tta.CDLHOMINGPIGEON',
                        'tta.CDLIDENTICAL3CROWS',
                        'tta.CDLINNECK', 'tta.CDLINVERTEDHAMMER', 'tta.CDLKICKING', 'tta.CDLKICKINGBYLENGTH',
                        'tta.CDLLADDERBOTTOM',
                        'tta.CDLLONGLEGGEDDOJI', 'tta.CDLLONGLINE', 'tta.CDLMARUBOZU', 'tta.CDLMATCHINGLOW',
                        'tta.CDLMATHOLD',
                        'tta.CDLMORNINGDOJISTAR', 'tta.CDLMORNINGSTAR', 'tta.CDLONNECK', 'tta.CDLPIERCING',
                        'tta.CDLRICKSHAWMAN',
                        'tta.CDLRISEFALL3METHODS', 'tta.CDLSEPARATINGLINES', 'tta.CDLSHOOTINGSTAR', 'tta.CDLSHORTLINE',
                        'tta.CDLSPINNINGTOP',
                        'tta.CDLSTALLEDPATTERN', 'tta.CDLSTICKSANDWICH', 'tta.CDLTAKURI', 'tta.CDLTASUKIGAP',
                        'tta.CDLTHRUSTING', 'tta.CDLTRISTAR',
                        'tta.CDLUNIQUE3RIVER', 'tta.CDLUPSIDEGAP2CROWS', 'tta.CDLXSIDEGAP3METHODS', 'tta.LINEARREG',
                        'tta.LINEARREG_ANGLE', 'tta.LINEARREG_INTERCEPT', 'tta.LINEARREG_SLOPE', 'tta.STDDEV',
                        'tta.TSF', 'tta.VAR',]

pandas_ta_indicators = ['pta.ebsw', 'pta.stdev', 'pta.mad', 'pta.quantile', 'pta.skew', 'pta.tos_stdevall', 'pta.variance', 'pta.zscore', 'pta.median', 'pta.kurtosis',
    'pta.entropy', 'pta.eri', 'pta.bias', 'pta.cg', 'pta.willr', 'pta.psl', 'pta.rvgi', 'pta.kdj', 'pta.smi', 'pta.fisher', 'pta.macd', 'pta.mom', 'pta.stochrsi',
    'pta.apo', 'pta.squeeze_pro', 'pta.inertia', 'pta.cmo', 'pta.slope', 'pta.roc', 'pta.rsx', 'pta.dm', 'pta.cci', 'pta.cti', 'pta.er', 'pta.uo', 'pta.bop', 'pta.cfo',
    'pta.stc', 'pta.tsi', 'pta.ao', 'pta.stoch', 'pta.qqe', 'pta.coppock', 'pta.rsi', 'pta.trix', 'pta.ppo', 'pta.pgo', 'pta.squeeze', 'pta.brar', 'pta.td_seq', 'pta.kst',
    'pta.pvo', 'pta.amat', 'pta.adx', 'pta.vortex', 'pta.qstick', 'pta.psar', 'pta.decay', 'pta.cksp', 'pta.chop', 'pta.ttm_trend', 'pta.vhf',
    'pta.aroon', 'pta.dpo', 'pta.donchian', 'pta.accbands', 'pta.ui', 'pta.hwc', 'pta.natr', 'pta.atr', 'pta.true_range', 'pta.massi', 'pta.thermo', 'pta.aberration', 'pta.kc',
    'pta.bbands', 'pta.pdist', 'pta.rvi', 'pta.pvt', 'pta.efi', 'pta.adosc', 'pta.aobv', 'pta.kvo', 'pta.pvi', 'pta.nvi', 'pta.vp', 'pta.mfi', 'pta.obv', 'pta.pvol', 'pta.eom',
    'pta.pvr', 'pta.ad', 'pta.cmf',"pta.dema", "pta.ema", "pta.fwma", "pta.hma", "pta.linreg", "pta.midpoint", "pta.pwma", "pta.rma","pta.sinwma", "pta.sma", "pta.swma", "pta.t3",
    "pta.tema", "pta.trima", "pta.vidya", "pta.wma", "pta.zlma"]

finta_indicatrs = ['fta.SMA', 'fta.SMM', 'fta.SSMA', 'fta.EMA', 'fta.DEMA', 'fta.TEMA', 'fta.TRIMA', 'fta.TRIX',
                   'fta.VAMA', 'fta.ER', 'fta.KAMA', 'fta.ZLEMA', 'fta.WMA', 'fta.HMA', 'fta.EVWMA', 'fta.VWAP',
                   'fta.SMMA', 'fta.MACD', 'fta.PPO', 'fta.VW_MACD', 'fta.EV_MACD', 'fta.MOM', 'fta.ROC', 'fta.RSI',
                   'fta.IFT_RSI', 'fta.TR', 'fta.ATR', 'fta.SAR', 'fta.BBANDS', 'fta.BBWIDTH', 'fta.MOBO',
                   'fta.PERCENT_B', 'fta.KC', 'fta.DO', 'fta.DMI', 'fta.ADX', 'fta.PIVOT', 'fta.PIVOT_FIB',
                   'fta.STOCH', 'fta.STOCHD', 'fta.STOCHRSI', 'fta.WILLIAMS', 'fta.UO', 'fta.AO', 'fta.MI',
                   'fta.VORTEX', 'fta.KST', 'fta.TSI', 'fta.TP', 'fta.ADL', 'fta.CHAIKIN', 'fta.MFI', 'fta.OBV',
                   'fta.WOBV', 'fta.VZO', 'fta.PZO', 'fta.EFI', 'fta.CFI', 'fta.EBBP', 'fta.EMV', 'fta.CCI',
                   'fta.COPP', 'fta.BASP', 'fta.BASPN', 'fta.CMO', 'fta.CHANDELIER', 'fta.WTO', 'fta.FISH', 'fta.APZ',
                   'fta.SQZMI', 'fta.VPT', 'fta.FVE', 'fta.VFI', 'fta.MSD', 'fta.STC']




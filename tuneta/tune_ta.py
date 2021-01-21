import pandas as pd
import numpy as np
from pathos.multiprocessing import ProcessPool
import multiprocessing
import inspect
from scipy.stats import rankdata
from tuneta.config import *
from tuneta.optimize import Optimize
import pandas_ta as pta
from finta import TA as fta
import talib as tta

import warnings
import re


class TuneTA():

    def __init__(self, n_jobs=multiprocessing.cpu_count() - 1, verbose=False):
        self.result = []
        self.fitted = []
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, trials=5, indicators=indicators, ranges=ranges, tune_series=tune_series,
            tune_params=tune_params, tune_column=tune_column):
        self.fitted = []
        X.columns = X.columns.str.lower()  # columns must be lower case

        pool = ProcessPool(nodes=self.n_jobs)

        for low, high in ranges:
            if low <=1:
                raise ValueError("Range low must be > 1")
            if high >= len(X):
                raise ValueError(f"Range high:{high} must be > length of X:{len(X)}")
            for ind in indicators:
                idx = 0
                if ":" in ind:
                    idx = int(ind.split(":")[1])
                    ind = ind.split(":")[0]
                fn = f"{ind}("
                if ind[0:3] == "tta":
                    usage = eval(f"{ind}.__doc__").split(")")[0].split("(")[1]
                    params = re.sub('[^0-9a-zA-Z_\s]', '', usage).split()
                else:
                    sig = inspect.signature(eval(ind))
                    params = sig.parameters.values()

                for param in params:
                    param = re.split(':|=', str(param))[0].strip()
                    if param == "open_":
                        param = "open"
                    if param == "real":
                        fn += f"X.close, "
                    elif param == "ohlc":
                        fn += f"X, "
                    elif param == "ohlcv":
                        fn += f"X, "
                    elif param in tune_series:
                        fn += f"X.{param}, "
                    elif param in tune_params:
                        fn += f"{param}=trial.suggest_int('{param}', {low}, {high}), "
                fn += ")"
                self.fitted.append(pool.apipe(Optimize(function=fn, n_trials=trials).fit, X, y, idx=idx, verbose=self.verbose))
        self.fitted = [fit.get() for fit in self.fitted]  # Get results of jobs

    def prune(self, top=2, studies=1):

        if top >= len(self.fitted) or studies >= len(self.fitted):
            print("Cannot prune because top or studies is >= tuned indicators")
            return
        if top <= studies:
            raise ValueError(f"top {top} must be > studies {studies}")

        fitness = []
        for t in self.fitted:
            fitness.append(t.study.best_value)  # get fitness of study
        fitness = np.array(fitness)  # Fitness of each fitted study

        fitness = fitness.argsort()[::-1][:top]  # Get sorted fitness indices of HOF

        # Gets best trial feature of each study in HOF
        features = []
        top_studies = [self.fitted[i] for i in fitness]  # Get HOF studies
        for study in top_studies:
            features.append(study.res_y[study.study.best_trial.number])
        features = np.array(features)  # Features of HOF studies

        # Correlation of HOF features
        eval = np.apply_along_axis(rankdata, 1, features)
        with np.errstate(divide='ignore', invalid='ignore'):
            correlations = np.abs(np.corrcoef(eval))
        np.fill_diagonal(correlations, 0.)

        # Iteratively remove least fit individual of most correlated pairs of studies
        components = list(range(top))
        indices = list(range(top))
        while len(components) > studies:
            most_correlated = np.unravel_index(np.argmax(correlations), correlations.shape)
            worst = max(most_correlated)
            components.pop(worst)
            indices.remove(worst)
            correlations = correlations[:, indices][indices, :]
            indices = list(range(len(components)))

        self.fitted = [self.fitted[i] for i in fitness[components]]

    def transform(self, X):
        X.columns = X.columns.str.lower()  # columns must be lower case
        pool = ProcessPool(nodes=self.n_jobs)
        self.result = []
        for ind in self.fitted:
            self.result.append(pool.apipe(ind.transform, X))
        self.result = [res.get() for res in self.result]
        res = pd.concat(self.result, axis=1)
        return res


if __name__ == "__main__":
    import joblib
    X, y = joblib.load('state/Xy.job')

    inds = TuneTA(verbose=True)
    # inds.fit(X, y, indicators=["fta.SMA"], ranges=[(1, 100)], trials=5)
    # inds.fit(X, y, indicators=["pta.sma"], ranges=[(1, 100)], trials=5)
    # inds.fit(X, y, indicators=['tta.BBANDS', 'tta.DEMA', 'tta.EMA',], ranges=[(0, 100)], trials=5)
    # inds.fit(X, y, indicators=['tta.BBANDS:1', 'fta.SMA', 'pta.sma', 'tta.DEMA'], ranges=[(1, 100)], trials=5)
    # """
    #inds.fit(X, y, indicators=["fta.VAMA"], ranges=[(2, 100)], trials=5)
    # """
    indicators = ['tta.BBANDS', 'tta.DEMA', 'tta.EMA', 'tta.HT_TRENDLINE', 'tta.KAMA', 'tta.MA',
'tta.MIDPOINT', 'tta.MIDPRICE', 'tta.SAR', 'tta.SAREXT', 'tta.SMA', 'tta.T3', 'tta.TEMA', 'tta.TRIMA',
'tta.WMA', 'tta.ADX', 'tta.ADXR', 'tta.APO', 'tta.AROON:1', 'tta.AROONOSC', 'tta.BOP', 'tta.CCI', 'tta.CMO',
'tta.DX', 'tta.MACD', 'tta.MACDEXT', 'tta.MACDFIX', 'tta.MFI', 'tta.MINUS_DI', 'tta.MINUS_DM', 'tta.MOM',
'tta.PLUS_DI', 'tta.PLUS_DM', 'tta.PPO', 'tta.ROC', 'tta.ROCP', 'tta.ROCR', 'tta.ROCR100', 'tta.RSI', 'tta.STOCH',
'tta.STOCHF', 'tta.STOCHRSI', 'tta.TRIX', 'tta.ULTOSC', 'tta.WILLR', 'tta.AD', 'tta.ADOSC', 'tta.OBV',
'tta.HT_DCPERIOD', 'tta.HT_DCPHASE', 'tta.HT_PHASOR', 'tta.HT_SINE', 'tta.HT_TRENDMODE', 'tta.AVGPRICE', 'tta.MEDPRICE',
'tta.TYPPRICE', 'tta.WCLPRICE', 'tta.ATR', 'tta.NATR', 'tta.TRANGE', 'tta.CDL2CROWS', 'tta.CDL3BLACKCROWS',
'tta.CDL3INSIDE', 'tta.CDL3LINESTRIKE', 'tta.CDL3OUTSIDE', 'tta.CDL3STARSINSOUTH', 'tta.CDL3WHITESOLDIERS',
'tta.CDLABANDONEDBABY', 'tta.CDLADVANCEBLOCK', 'tta.CDLBELTHOLD', 'tta.CDLBREAKAWAY', 'tta.CDLCLOSINGMARUBOZU',
'tta.CDLCONCEALBABYSWALL', 'tta.CDLCOUNTERATTACK', 'tta.CDLDARKCLOUDCOVER', 'tta.CDLDOJI', 'tta.CDLDOJISTAR',
'tta.CDLDRAGONFLYDOJI', 'tta.CDLENGULFING', 'tta.CDLEVENINGDOJISTAR', 'tta.CDLEVENINGSTAR', 'tta.CDLGAPSIDESIDEWHITE',
'tta.CDLGRAVESTONEDOJI', 'tta.CDLHAMMER', 'tta.CDLHANGINGMAN', 'tta.CDLHARAMI', 'tta.CDLHARAMICROSS',
'tta.CDLHIGHWAVE', 'tta.CDLHIKKAKE', 'tta.CDLHIKKAKEMOD', 'tta.CDLHOMINGPIGEON', 'tta.CDLIDENTICAL3CROWS',
'tta.CDLINNECK', 'tta.CDLINVERTEDHAMMER', 'tta.CDLKICKING', 'tta.CDLKICKINGBYLENGTH', 'tta.CDLLADDERBOTTOM',
'tta.CDLLONGLEGGEDDOJI', 'tta.CDLLONGLINE', 'tta.CDLMARUBOZU', 'tta.CDLMATCHINGLOW', 'tta.CDLMATHOLD',
'tta.CDLMORNINGDOJISTAR', 'tta.CDLMORNINGSTAR', 'tta.CDLONNECK', 'tta.CDLPIERCING', 'tta.CDLRICKSHAWMAN',
'tta.CDLRISEFALL3METHODS', 'tta.CDLSEPARATINGLINES', 'tta.CDLSHOOTINGSTAR', 'tta.CDLSHORTLINE', 'tta.CDLSPINNINGTOP',
'tta.CDLSTALLEDPATTERN', 'tta.CDLSTICKSANDWICH', 'tta.CDLTAKURI', 'tta.CDLTASUKIGAP', 'tta.CDLTHRUSTING', 'tta.CDLTRISTAR',
'tta.CDLUNIQUE3RIVER', 'tta.CDLUPSIDEGAP2CROWS', 'tta.CDLXSIDEGAP3METHODS', 'tta.LINEARREG',
'tta.LINEARREG_ANGLE', 'tta.LINEARREG_INTERCEPT', 'tta.LINEARREG_SLOPE', 'tta.STDDEV', 'tta.TSF', 'tta.VAR', 'pta.cdl_doji', 'pta.cdl_inside', 'pta.ha', 'pta.ao', 'pta.apo', 'pta.bias', 'pta.bop', 'pta.brar', 'pta.cci', 'pta.cfo', 'pta.cg', 'pta.cmo', 'pta.coppock', 'pta.er', 'pta.eri',
    'pta.fisher', 'pta.inertia', 'pta.kdj', 'pta.kst', 'pta.macd', 'pta.mom', 'pta.pgo', 'pta.ppo', 'pta.psl', 'pta.pvo', 'pta.qqe', 'pta.roc', 'pta.rsi', 'pta.rvgi', 'pta.slope', 'pta.smi', 'pta.squeeze',
    'pta.stoch', 'pta.stochrsi', 'pta.trix', 'pta.tsi', 'pta.uo', 'pta.willr', 'pta.dema', 'pta.ema', 'pta.fwma', 'pta.hilo', 'pta.hl2', 'pta.hlc3', 'pta.hma', 'pta.kama', 'pta.linreg', 'pta.midpoint',
    'pta.midprice', 'pta.ohlc4', 'pta.pwma', 'pta.rma', 'pta.sinwma', 'pta.sma', 'pta.ssf', 'pta.supertrend', 'pta.swma', 'pta.t3', 'pta.tema', 'pta.trima', 'pta.vidya', 'pta.vwap', 'pta.vwma',
    'pta.wcp', 'pta.wma', 'pta.zlma', 'pta.entropy', 'pta.kurtosis', 'pta.mad', 'pta.median', 'pta.quantile', 'pta.skew', 'pta.stdev', 'pta.variance', 'pta.zscore', 'pta.adx', 'pta.amat',
    'pta.aroon', 'pta.chop', 'pta.cksp', 'pta.decay', 'pta.decreasing', 'pta.increasing', 'pta.psar', 'pta.qstick', 'pta.ttm_trend', 'pta.vortex', 'pta.aberration',
    'pta.accbands', 'pta.atr', 'pta.bbands', 'pta.donchian', 'pta.kc', 'pta.massi', 'pta.natr', 'pta.pdist', 'pta.rvi', 'pta.thermo', 'pta.true_range', 'pta.ui', 'pta.ad', 'pta.adosc',
    'pta.aobv', 'pta.cmf', 'pta.efi', 'pta.eom', 'pta.mfi', 'pta.nvi', 'pta.obv', 'pta.pvi', 'pta.pvol', 'pta.pvt', 'fta.SMA', 'fta.SMM', 'fta.SSMA', 'fta.EMA', 'fta.DEMA', 'fta.TEMA', 'fta.TRIMA', 'fta.TRIX', 'fta.VAMA', 'fta.ER', 'fta.KAMA', 'fta.ZLEMA',
    'fta.WMA', 'fta.HMA', 'fta.EVWMA', 'fta.VWAP', 'fta.SMMA', 'fta.MACD', 'fta.PPO', 'fta.VW_MACD', 'fta.EV_MACD', 'fta.MOM', 'fta.ROC', 'fta.RSI', 'fta.IFT_RSI',
    'fta.TR', 'fta.ATR', 'fta.SAR', 'fta.BBANDS', 'fta.BBWIDTH', 'fta.MOBO', 'fta.PERCENT_B', 'fta.KC', 'fta.DO', 'fta.DMI', 'fta.ADX', 'fta.PIVOT', 'fta.PIVOT_FIB', 'fta.STOCH',
    'fta.STOCHD', 'fta.STOCHRSI', 'fta.WILLIAMS', 'fta.UO', 'fta.AO', 'fta.MI', 'fta.VORTEX', 'fta.KST', 'fta.TSI', 'fta.TP', 'fta.ADL', 'fta.CHAIKIN', 'fta.MFI', 'fta.OBV', 'fta.WOBV',
    'fta.VZO', 'fta.PZO', 'fta.EFI', 'fta.CFI', 'fta.EBBP', 'fta.EMV', 'fta.CCI', 'fta.COPP', 'fta.BASP', 'fta.BASPN', 'fta.CMO', 'fta.CHANDELIER',
    'fta.WTO', 'fta.FISH', 'fta.APZ', 'fta.SQZMI', 'fta.VPT', 'fta.FVE', 'fta.VFI', 'fta.MSD', 'fta.STC',]
    inds.fit(X, y, indicators=indicators, ranges=[(2, 100)], trials=5)


    inds.prune(top=50, studies=5)
    out = inds.transform(X)
    print("done")


    # pta.vp(X.close, X.volume)
import numpy as np
import pandas as pd

def compute_log_returns(series, **kwargs):
    return np.log( series['close'] / series['open'] )


def compute_rsi(series, window=14, **kwargs):

    delta = np.log(series / series.shift(1))
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, - delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1  + rs))

    return pd.Series(rsi, index = series.index)

def normalizeRsi(rsi, **kwargs):
    return (rsi - rsi.mean()) / rsi.std()

def volatitily(series, window=20, **kwargs):
    delta = np.log(series / series.shift(1))
    return delta.rolling(window=window).mean()

def momentum(series, window=10, **kwargs):
    raw_momentum = series - series.shift(window)
    log_monetum = np.log(series / series.shift(window))
    return raw_momentum, log_monetum




data = pd.read_csv("../../data/blueChips/AAPL.csv")
price = data['close']

#Get rsi
rsi = compute_rsi(price)
rsi = normalizeRsi(rsi)

#max rsi
max = rsi.max()

print(max)
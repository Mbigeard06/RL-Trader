import numpy as np
import pandas as pd

def compute_log_returns(df, **kwargs):
    return np.log( df['close'] / df['open'] )

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

def volatility(series, window=20, **kwargs):
    delta = np.log(series / series.shift(1))
    return delta.rolling(window=window).mean()

def momentum(series, window=10, **kwargs):
    raw_momentum = series - series.shift(window)
    log_monentum = np.log(series / series.shift(window))
    return raw_momentum, log_monentum

def zscore_price(series, windows=10):
    return (series - series.rolling(windows=windows).mean()) / series.rolling(windows=windows).std()

def macd(series, span_short=12, span_long=26, signal_span=9, **kwargs):
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd = ema_short - ema_long
    macd_signal = macd.ewm(span = signal_span, adjust=False).mean()
    macd_diff = macd - macd_signal
    return macd, macd_signal, macd_diff

def bollinger(series, span=2, **kwargs):
    sma = series.rolling(window=20).mean()
    std = series.rolling(window=20).std()
    upper_band = sma + span * std
    lower_band = sma - span * std
    width = upper_band - lower_band
    return upper_band, lower_band, width

def atr(df, period, **kwargs):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()

    tr = (pd.concat(high_low, high_close, low_close)).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()

    return atr

def drawdown(series, **kwargs):
    peak = series.cummax()
    return (series / peak ) - 1

def sharpe_ratio(series, risk_free_rate=0.0, period = 252, **kwargs):
    net = series.mean() - risk_free_rate
    volatility = series.std()
    ratio = net / volatility
    return np.sqrt(period)* ratio

def Acceleration(series, **kwargs):
    return series / series.shitf(1)

def entropy(series, window = 20):
    return None

    





data = pd.read_csv("../../data/blueChips/AAPL.csv")
price = data['close']

#Get rsi
macd, signal, diff = macd(price)

#max rsi
print(diff)
import numpy as np
import pandas as pd

# --- Feature Functions ---
def compute_log_returns(df, **kwargs):
    return np.log(df['close'] / df['open'])

def compute_rsi(series, window=14, **kwargs):
    delta = np.log(series / series.shift(1))
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=series.index)

def normalize_rsi(rsi, **kwargs):
    return (rsi - rsi.mean()) / rsi.std()

def volatility(series, window=20, **kwargs):
    delta = np.log(series / series.shift(1))
    return delta.rolling(window=window).mean()

def momentum(series, window=10, **kwargs):
    raw_momentum = series - series.shift(window)
    log_momentum = np.log(series / series.shift(window))
    return raw_momentum, log_momentum

def zscore_price(series, window=10):
    return (series - series.rolling(window=window).mean()) / series.rolling(window=window).std()

def macd(series, span_short=12, span_long=26, signal_span=9, **kwargs):
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd = ema_short - ema_long
    macd_signal = macd.ewm(span=signal_span, adjust=False).mean()
    macd_diff = macd - macd_signal
    return macd, macd_signal, macd_diff

def bollinger(series, span=2, **kwargs):
    sma = series.rolling(window=20).mean()
    std = series.rolling(window=20).std()
    upper_band = sma + span * std
    lower_band = sma - span * std
    width = upper_band - lower_band
    return upper_band, lower_band, width

def atr(df, period=14, **kwargs):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def drawdown(series, **kwargs):
    peak = series.cummax()
    return (series / peak) - 1

def rolling_sharpe(returns, window=20, risk_free_rate=0.0, **kwargs):
    excess_returns = returns - risk_free_rate
    rolling_mean = excess_returns.rolling(window=window).mean()
    rolling_std = excess_returns.rolling(window=window).std()
    sharpe = rolling_mean / rolling_std
    return sharpe * np.sqrt(252)

def acceleration(series, **kwargs):
    returns = np.log(series / series.shift(1))
    accel = returns - returns.shift(1)
    return accel

def entropy(series, window=25, bins=10, **kwargs):
    def entropy_func(arr):
        z = (arr - np.mean(arr)) / np.std(arr)
        hist = np.histogram(z, bins=bins)[0]
        probs = hist / np.sum(hist)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])
    return series.rolling(window).apply(entropy_func, raw=True)

def overnight_return(df, **kwargs):
    return np.log(df['open'] / df['close'].shift(1))

def compute_features(df):
    price = df['close']
    log_return = compute_log_returns(df)
    rsi = normalize_rsi(compute_rsi(price))
    vol = volatility(price)
    raw_mom, log_mom = momentum(price)
    zscore = zscore_price(price)
    macd_line, macd_signal, macd_diff = macd(price)
    boll_upper, boll_lower, boll_width = bollinger(price)
    atr_val = atr(df)
    draw = drawdown(price)
    sharpe = rolling_sharpe(log_return)
    accel = acceleration(price)
    ent = entropy(log_return)
    overnight = overnight_return(df)

    return pd.DataFrame({
        'datetime': df['datetime'],
        'log_return': log_return,
        'rsi': rsi,
        'volatility': vol,
        'momentum_raw': raw_mom,
        'momentum_log': log_mom,
        'zscore_price': zscore,
        'macd': macd_line,
        'macd_signal': macd_signal,
        'macd_diff': macd_diff,
        'bollinger_upper': boll_upper,
        'bollinger_lower': boll_lower,
        'bollinger_width': boll_width,
        'atr': atr_val,
        'drawdown': draw,
        'sharpe_ratio': sharpe,
        'acceleration': accel,
        'entropy': ent,
        'overnight_return': overnight
    }).dropna().reset_index(drop=True)

# --- Example usage ---
data = pd.read_csv("../../data/blueChips/AAPL.csv")
features = compute_features(data)
print(features.head())

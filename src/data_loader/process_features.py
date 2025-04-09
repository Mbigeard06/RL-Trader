import numpy as np
import pandas as pd

def compute_rsi(series, window=14):

    delta = np.log(series / series.shift(1))
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, - delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1  + rs))

    return pd.Series(rsi, index = series.index)

data = pd.read_csv("../../data/blueChips/AAPL.csv")
price = data['close']
rsi = compute_rsi(price)
print(rsi[:15])
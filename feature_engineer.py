import pandas as pd
import numpy as np
import pandas_ta as ta

def add_features_to_data(data):
    if data is None or data.empty: return None
    if not all(c in data.columns for c in ['High', 'Low', 'Close', 'Adj Close']):
        print("FEHLER: Notwendige Spalten für Feature-Erstellung nicht gefunden.")
        return None
    
    df = data.copy()
    price_series = df['Adj Close']
    
    df['daily_return'] = price_series.pct_change()
    df['SMA_10'] = price_series.rolling(window=10).mean()
    df['SMA_50'] = price_series.rolling(window=50).mean()
    df['sma_signal'] = np.where(df['SMA_10'] > df['SMA_50'], 1, 0)
    df['RSI_14'] = ta.rsi(price_series, length=14)
    df.ta.macd(close=price_series, fast=12, slow=26, signal=9, append=True)
    df.ta.atr(high=df['High'], low=df['Low'], close=df['Close'], length=14, append=True)

    for i in range(1, 4):
        df[f'daily_return_lag_{i}'] = df['daily_return'].shift(i)
        df[f'RSI_14_lag_{i}'] = df['RSI_14'].shift(i)

    df.dropna(inplace=True)
    return df

def create_regression_targets(data, future_days=7):
    if data is None: return None
    df = data.copy()
    df[f'future_{future_days}d_low'] = df['Low'].iloc[::-1].rolling(window=future_days).min().iloc[::-1].shift(-future_days)
    df[f'future_{future_days}d_high'] = df['High'].iloc[::-1].rolling(window=future_days).max().iloc[::-1].shift(-future_days)
    df.dropna(inplace=True)
    return df
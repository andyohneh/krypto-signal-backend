import pandas as pd
import numpy as np
import pandas_ta as ta
from data_manager import download_historical_data

def add_features_to_data(data):
    if data is None or data.empty: return None
    print("Erstelle neue Features (inkl. RSI, MACD & ATR)...")
    if not all(col in data.columns for col in ['High', 'Low', 'Close', 'Adj Close']):
        print("FEHLER: Notwendige Spalten für Feature-Erstellung nicht gefunden.")
        return None
    
    data_with_features = data.copy()
    price_series = data_with_features['Adj Close']
    
    data_with_features['daily_return'] = price_series.pct_change()
    data_with_features['SMA_10'] = price_series.rolling(window=10).mean()
    data_with_features['SMA_50'] = price_series.rolling(window=50).mean()
    data_with_features['sma_signal'] = np.where(data_with_features['SMA_10'] > data_with_features['SMA_50'], 1, 0)
    data_with_features['RSI_14'] = ta.rsi(price_series, length=14)
    macd = ta.macd(price_series, fast=12, slow=26, signal=9)
    data_with_features = pd.concat([data_with_features, macd], axis=1)
    data_with_features['ATRr_14'] = ta.atr(data_with_features['High'], data_with_features['Low'], data_with_features['Close'], length=14)
    
    data_with_features.dropna(inplace=True)
    return data_with_features
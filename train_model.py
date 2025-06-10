import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

FEATURES_LIST = [
    'daily_return', 'SMA_10', 'SMA_50', 'sma_signal', 'RSI_14',
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'ATRr_14',
    'daily_return_lag_1', 'RSI_14_lag_1',
    'daily_return_lag_2', 'RSI_14_lag_2',
    'daily_return_lag_3', 'RSI_14_lag_3'
]

def train_regression_model(data, target_column_name):
    if data is None or data.empty: return None, None
    if not all(col in data.columns for col in FEATURES_LIST + [target_column_name]):
        print(f"FEHLER: Notwendige Spalten für Ziel '{target_column_name}' nicht gefunden.")
        return None, None

    X = data[FEATURES_LIST]
    y = data[target_column_name]
    
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    if len(X_test) == 0: return None, None
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler
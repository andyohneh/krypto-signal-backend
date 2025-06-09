import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

FEATURES_LIST = [
    'daily_return', 'SMA_10', 'SMA_50', 'sma_signal', 'RSI_14',
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'ATRr_14'
]

def train_and_evaluate_model(data):
    if data is None or data.empty: return None, None, 0.0
    print("Starte Modelltraining und -bewertung...")
    data['target'] = np.where(data['Adj Close'].shift(-1) > data['Adj Close'], 1, 0)
    data.dropna(inplace=True)
    X = data[FEATURES_LIST]
    y = data['target']
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    if len(X_test) == 0: return None, None, 0.0
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    print(f"Genauigkeit auf Test-Daten: {test_accuracy:.2%}")
    return model, scaler, test_accuracy
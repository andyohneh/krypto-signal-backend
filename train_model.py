import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor # Dein bevorzugtes Modell
from sklearn.preprocessing import StandardScaler # Dein bevorzugter Scaler
from sklearn.model_selection import train_test_split # Für optionale Datenaufteilung

# Diese Liste wird von anderen Skripten importiert, um konsistent zu sein
FEATURES_LIST = [
    'daily_return', 'SMA_10', 'SMA_50', 'sma_signal', 'RSI_14',
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'ATRr_14'
    # 'Close' ist hier NICHT enthalten, da es in deiner ursprünglichen Liste nicht war.
    # WICHTIG: feature_engineer.py muss diese Liste von Features auch erstellen!
]

def train_regression_model(data, target_column_name):
    """
    Trainiert ein Regressions-Modell, um eine bestimmte Ziel-Spalte vorherzusagen.
    Args:
        data (pd.DataFrame): DataFrame mit Features und der Zielspalte.
        target_column_name (str): Der Name der Zielspalte.
    Returns:
        tuple: (trainiertes Regressionsmodell, trainierter StandardScaler) oder (None, None) bei Fehler.
    """
    print(f"\n--- Starte Regressions-Training für Ziel: {target_column_name} ---")

    # Überprüfe, ob alle notwendigen Spalten vorhanden sind
    required_columns = FEATURES_LIST + [target_column_name]
    if not all(col in data.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in data.columns]
        print(f"FEHLER: Notwendige Spalten für Ziel '{target_column_name}' nicht gefunden: {missing_cols}")
        return None, None # Gib None, None zurück, wenn Spalten fehlen

    # Trenne Features (X) und Ziel (y)
    X = data[FEATURES_LIST]
    y = data[target_column_name]

    # Nan-Werte final behandeln (sollten idealerweise schon in feature_engineer.py gelöst sein)
    X = X.fillna(0)
    y = y.fillna(0)

    # Wenn X oder y nach dropna/fillna leer sind, kann nicht trainiert werden
    if X.empty or y.empty:
        print(f"FEHLER: Leere Daten nach Bereinigung für Ziel '{target_column_name}'. Training nicht möglich.")
        return None, None

    # Daten aufteilen (80% Training, 20% Test)
    split_index = int(len(X) * 0.8)
    if split_index == 0: # Stelle sicher, dass mindestens ein Trainingspunkt vorhanden ist
        print(f"WARNUNG: Nicht genügend Daten für Aufteilung in Training/Test für Ziel '{target_column_name}'. Training mit allen Daten.")
        X_train, y_train = X, y
    else:
        X_train, _ = X[:split_index], X[split_index:] # _ ignoriert X_test
        y_train, _ = y[:split_index], y[split_index:] # _ ignoriert y_test

    # Skalierung der Trainingsdaten (Scaler wird hier gelernt)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Das Regressions-Modell
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # Dein Modell

    print("Trainiere Regressor-Modell...")
    model.fit(X_train_scaled, y_train)
    print("Training abgeschlossen.")

    return model, scaler # Gibt Modell UND Scaler zurück
# train_model.py (Finale, korrigierte Helfer-Version)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def train_and_evaluate_model(data):
    """
    Nimmt einen DataFrame mit Features entgegen, erstellt die Zielvariable,
    trainiert ein Modell und gibt das Modell, den Scaler und die
    Test-Genauigkeit zurück.
    """
    if data is None or data.empty:
        print("FEHLER: Leere Daten an train_and_evaluate_model übergeben.")
        return None, None, 0.0

    print("Starte Modelltraining und -bewertung...")

    # KORREKTUR: Die Logik zur Erstellung der Zielvariable wird hier hinzugefügt
    # Wir wollen vorhersagen, ob der Preis am NÄCHSTEN Tag steigt.
    data['target'] = np.where(data['Adj Close'].shift(-1) > data['Adj Close'], 1, 0)
    # Entferne Zeilen, wo das Ziel nicht berechnet werden konnte (die letzte Zeile)
    data.dropna(subset=['target'], inplace=True)

    # 1. Features und Ziel (y) trennen
    features = [
        'daily_return',
        'SMA_10',
        'SMA_50',
        'sma_signal',
        'RSI_14',
        'MACD_12_26_9',
        'MACDh_12_26_9',
        'MACDs_12_26_9'
    ]
    X = data[features]
    y = data['target']

    # 2. Daten in Trainings- und Test-Set aufteilen
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    if len(X_train) == 0 or len(X_test) == 0:
        print("FEHLER: Nicht genügend Daten zum Aufteilen in Training und Test.")
        return None, None, 0.0

    # 3. Daten skalieren
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Daten erfolgreich skaliert.")

    # 4. Modell trainieren
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    print("Modelltraining abgeschlossen.")

    # 5. Modell bewerten
    test_accuracy = model.score(X_test_scaled, y_test)
    print(f"Genauigkeit auf Test-Daten (ungesehen): {test_accuracy:.2%}")

    # 6. Fertige Werkzeuge zurückgeben
    return model, scaler, test_accuracy
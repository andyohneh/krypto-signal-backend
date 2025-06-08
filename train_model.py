# train_model.py (Version 5 - mit RandomForest)

import pandas as pd
import numpy as np
# NEU: Wir importieren den RandomForest anstelle der LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

def train_model_for_asset(input_filename, output_model_name, output_scaler_name):
    """
    Trainiert jetzt ein leistungsfähigeres RandomForest-Modell.
    """
    print(f"\n--- Starte V5 Modelltraining für {output_model_name} (RandomForest) ---")
    try:
        data = pd.read_csv(input_filename, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"FEHLER: Datei '{input_filename}' nicht gefunden.")
        return

    # Datenvorbereitung bleibt exakt gleich
    data['target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    data.dropna(inplace=True)
    features = ['daily_return', 'SMA_10', 'SMA_50', 'sma_signal', 'RSI_14']
    X = data[features]
    y = data['target']

    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- DIE ENTSCHEIDENDE ÄNDERUNG ---
    # Wir ersetzen das einfache Modell durch den leistungsfähigen RandomForest.
    # n_estimators=100 bedeutet, wir erstellen einen "Wald" aus 100 Entscheidungsbäumen.
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    print("Trainiere RandomForest-Modell...")
    model.fit(X_train_scaled, y_train)
    print("Modelltraining abgeschlossen.")

    # Bewertung und Speichern bleibt exakt gleich
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)

    print(f"Genauigkeit auf Trainings-Daten: {train_accuracy:.2%}")
    print(f"Genauigkeit auf Test-Daten (ungesehen): {test_accuracy:.2%}")

    joblib.dump(model, output_model_name)
    joblib.dump(scaler, output_scaler_name)
    print(f"Modell in '{output_model_name}' und Scaler in '{output_scaler_name}' gespeichert.")


if __name__ == '__main__':
    # Prozess für Bitcoin
    train_model_for_asset(
        "btc_data_with_features.csv",
        "trained_btc_model.joblib",
        "btc_scaler.joblib"
    )

    # Prozess für Gold
    train_model_for_asset(
        "gold_data_with_features.csv",
        "trained_gold_model.joblib",
        "gold_scaler.joblib"
    )

    print("\nAlle Modelle wurden mit dem neuen RandomForest-Modell neu trainiert.")
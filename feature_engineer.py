# feature_engineer.py (NEUE VERSION)
import pandas as pd
import numpy as np
# NEU: Importiere die Funktion aus unserer anderen Datei
from data_manager import download_historical_data 

SHORT_SMA_WINDOW = 10
LONG_SMA_WINDOW = 50

def add_features_to_data(data):
    """Nimmt ein DataFrame entgegen und fügt die Feature-Spalten hinzu."""
    if data is None or data.empty:
        return None

    print("Erstelle neue Features...")

    # Sicherstellen, dass die 'Close'-Spalte existiert
    if 'Close' not in data.columns:
        print("FEHLER: 'Close'-Spalte nicht in den Daten gefunden.")
        return None

    # Kopie erstellen, um Warnungen zu vermeiden
    data_with_features = data.copy()

    data_with_features['daily_return'] = data_with_features['Close'].pct_change()
    data_with_features[f'SMA_{SHORT_SMA_WINDOW}'] = data_with_features['Close'].rolling(window=SHORT_SMA_WINDOW).mean()
    data_with_features[f'SMA_{LONG_SMA_WINDOW}'] = data_with_features['Close'].rolling(window=LONG_SMA_WINDOW).mean()
    data_with_features['sma_signal'] = np.where(data_with_features[f'SMA_{SHORT_SMA_WINDOW}'] > data_with_features[f'SMA_{LONG_SMA_WINDOW}'], 1, 0)

    data_with_features.dropna(inplace=True)
    return data_with_features

if __name__ == '__main__':
    # --- Prozess für Bitcoin ---
    # 1. Lade die Daten direkt durch Aufruf der Funktion
    btc_raw_data = download_historical_data("BTC-USD")

    # 2. Füge Features hinzu
    btc_featured_data = add_features_to_data(btc_raw_data)

    if btc_featured_data is not None:
        # 3. Speichere das Ergebnis
        output_filename = "btc_data_with_features.csv"
        print(f"Features erfolgreich erstellt. Speichere Ergebnis in '{output_filename}'...")
        btc_featured_data.to_csv(output_filename)
        print("\n--- Letzte 5 Tage Bitcoin-Daten mit neuen Features ---")
        print(btc_featured_data.tail())

    print("\n" + "="*50 + "\n")

    # --- Prozess für Gold ---
    gold_raw_data = download_historical_data("GC=F")
    gold_featured_data = add_features_to_data(gold_raw_data)
    if gold_featured_data is not None:
        output_filename = "gold_data_with_features.csv"
        print(f"Features erfolgreich erstellt. Speichere Ergebnis in '{output_filename}'...")
        gold_featured_data.to_csv(output_filename)
        print("\n--- Letzte 5 Tage Gold-Daten mit neuen Features ---")
        print(gold_featured_data.tail())
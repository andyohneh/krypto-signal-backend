# feature_engineer.py (Finale Version, die jetzt funktioniert)

import pandas as pd
import numpy as np
import pandas_ta as ta
from data_manager import download_historical_data

SHORT_SMA_WINDOW = 10
LONG_SMA_WINDOW = 50

def add_features_to_data(data):
    if data is None or data.empty:
        return None

    print("Erstelle neue Features (inkl. RSI) auf Basis von 'Adj Close'...")

    price_series = data['Adj Close']

    if price_series.empty:
        print("FEHLER: 'Adj Close'-Spalte ist leer oder nicht vorhanden.")
        return None

    data_with_features = data.copy()

    data_with_features['daily_return'] = price_series.pct_change()
    data_with_features[f'SMA_{SHORT_SMA_WINDOW}'] = price_series.rolling(window=SHORT_SMA_WINDOW).mean()
    data_with_features[f'SMA_{LONG_SMA_WINDOW}'] = price_series.rolling(window=LONG_SMA_WINDOW).mean()
    data_with_features['sma_signal'] = np.where(data_with_features[f'SMA_{SHORT_SMA_WINDOW}'] > data_with_features[f'SMA_{LONG_SMA_WINDOW}'], 1, 0)
    data_with_features['RSI_14'] = ta.rsi(price_series, length=14)

    data_with_features.dropna(inplace=True)
    return data_with_features

if __name__ == '__main__':
    # Prozess für Bitcoin
    btc_raw_data = download_historical_data("BTC-USD")
    btc_featured_data = add_features_to_data(btc_raw_data)
    if btc_featured_data is not None:
        output_filename = "btc_data_with_features.csv"
        print(f"Features erfolgreich erstellt. Speichere Ergebnis in '{output_filename}'...")
        btc_featured_data.to_csv(output_filename)
        print("\n--- Letzte 5 Tage Bitcoin-Daten mit neuen Features ---")
        print(btc_featured_data.tail())

    print("\n" + "="*50 + "\n")

    # Prozess für Gold
    gold_raw_data = download_historical_data("GC=F")
    gold_featured_data = add_features_to_data(gold_raw_data)
    if gold_featured_data is not None:
        output_filename = "gold_data_with_features.csv"
        print(f"Features erfolgreich erstellt. Speichere Ergebnis in '{output_filename}'...")
        gold_featured_data.to_csv(output_filename)
        print("\n--- Letzte 5 Tage Gold-Daten mit neuen Features ---")
        print(gold_featured_data.tail())
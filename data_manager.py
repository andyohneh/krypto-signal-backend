# data_manager.py (FINALE VERSION - Bereinigt die Spalten)

import yfinance as yf
import pandas as pd

def download_historical_data(ticker_symbol, period="1y", interval="1d"):
    """
    Lädt historische Daten und gibt ein sauberes pandas DataFrame
    mit einfachen Spaltennamen zurück.
    """
    print(f"Lade historische Daten für {ticker_symbol}...")
    try:
        data = yf.download(
            ticker_symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False
        )

        if data.empty:
            print(f"Keine Daten für {ticker_symbol} gefunden.")
            return None

        # --- DIE FINALE KORREKTUR ---
        # "Flache" die Spaltenüberschriften ab, falls es ein MultiIndex ist.
        # ('Adj Close', 'BTC-USD') wird zu 'Adj Close'
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        print(f"Erfolgreich {len(data)} Datenpunkte geladen und Spalten bereinigt.")
        return data
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return None
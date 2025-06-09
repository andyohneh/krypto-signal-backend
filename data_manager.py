import yfinance as yf
import pandas as pd

def download_historical_data(ticker_symbol, period="1y", interval="1d"):
    print(f"Lade historische Daten für {ticker_symbol}...")
    try:
        data = yf.download(ticker_symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if data.empty:
            print(f"Keine Daten für {ticker_symbol} gefunden.")
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        print(f"Erfolgreich {len(data)} Datenpunkte geladen und Spalten bereinigt.")
        return data
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return None
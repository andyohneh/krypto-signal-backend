import yfinance as yf
import pandas as pd

def download_historical_data(ticker_symbol, period="1y", interval="1d"):
    try:
        data = yf.download(ticker_symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if data.empty:
            print(f"Keine Daten für {ticker_symbol} gefunden.")
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        print(f"Fehler bei Daten-Download für {ticker_symbol}: {e}")
        return None
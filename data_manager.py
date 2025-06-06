# data_manager.py (VEREINFACHTE VERSION)
import yfinance as yf

def download_historical_data(ticker_symbol, period="1y", interval="1d"):
    """
    Lädt historische Daten und gibt ein sauberes pandas DataFrame zurück.
    """
    print(f"Lade historische Daten für {ticker_symbol}...")
    try:
        # Lade die Daten
        data = yf.download(ticker_symbol, period=period, interval=interval, progress=False)

        if data.empty:
            print(f"Keine Daten für {ticker_symbol} gefunden.")
            return None

        # Wir benennen die Index-Spalte explizit 'Date'
        data.index.name = 'Date'

        print(f"Erfolgreich {len(data)} Datenpunkte für {ticker_symbol} geladen.")
        return data
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return None

# Der if __name__ == '__main__': Block wird entfernt,
# da wir dieses Skript nicht mehr direkt ausführen wollen.
# Es dient jetzt nur noch als Bibliothek für andere Skripte.
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def add_features_to_data(df, asset_name="", skip_scaling=False):
    """
    Fügt einem DataFrame technische Indikatoren als Features hinzu und skaliert diese.
    Gibt den erweiterten DataFrame und die verwendeten Scaler zurück.
    """
    df_copy = df.copy()

    if 'Close' not in df_copy.columns:
        raise ValueError("DataFrame muss eine 'Close'-Spalte enthalten.")

    # Sortieren nach Datum, falls nicht bereits geschehen
    df_copy = df_copy.sort_values(by='Date')

    # 1. Täglicher Return
    df_copy['daily_return'] = df_copy['Close'].pct_change().fillna(0)

    # 2. Gleitende Durchschnitte (SMA_10, SMA_50)
    df_copy['SMA_10'] = df_copy['Close'].rolling(window=10, min_periods=1).mean()
    df_copy['SMA_50'] = df_copy['Close'].rolling(window=50, min_periods=1).mean()

    # 3. SMA Signal (z.B. 1 wenn SMA_10 > SMA_50, sonst 0)
    # Behandelt NaN's nach dem ersten Lauf (z.B. wenn 50 Perioden noch nicht voll sind)
    df_copy['sma_signal'] = np.where(df_copy['SMA_10'] > df_copy['SMA_50'], 1, 0)
    # Fülle die ersten Werte mit 0, wo SMA_50 noch nicht berechnet werden kann
    df_copy['sma_signal'] = df_copy['sma_signal'].fillna(0)


    # 4. Relative Stärke Index (RSI_14)
    delta = df_copy['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_copy['RSI_14'] = 100 - (100 / (1 + rs))
    df_copy['RSI_14'].fillna(0, inplace=True) # Fülle NaN-Werte am Anfang

    # 5. MACD (Moving Average Convergence Divergence) - 12, 26, 9
    ema_12 = df_copy['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df_copy['Close'].ewm(span=26, adjust=False).mean()
    df_copy['MACD_12_26_9'] = ema_12 - ema_26
    df_copy['MACDs_12_26_9'] = df_copy['MACD_12_26_9'].ewm(span=9, adjust=False).mean() # MACD Signal Line
    df_copy['MACDh_12_26_9'] = df_copy['MACD_12_26_9'] - df_copy['MACDs_12_26_9'] # MACD Histogram

    # 6. ATR (Average True Range) - 14
    # True Range (TR)
    high_low = df_copy['High'] - df_copy['Low'] # Assuming 'High' and 'Low' columns exist in your data
    high_prev_close = abs(df_copy['High'] - df_copy['Close'].shift(1))
    low_prev_close = abs(df_copy['Low'] - df_copy['Close'].shift(1))
    
    # Sicherstellen, dass 'High' und 'Low' Spalten existieren.
    # Wenn nicht, musst du sie in den Dummy-Daten erstellen oder von einer API abrufen.
    # Für die Demo: Fügen wir sie hinzu, wenn sie fehlen, mit Werten nahe am 'Close'.
    if 'High' not in df_copy.columns:
        df_copy['High'] = df_copy['Close'] * (1 + np.random.uniform(0.001, 0.005, size=len(df_copy)))
    if 'Low' not in df_copy.columns:
        df_copy['Low'] = df_copy['Close'] * (1 - np.random.uniform(0.001, 0.005, size=len(df_copy)))

    tr = pd.DataFrame({'hl': high_low, 'hpc': high_prev_close, 'lpc': low_prev_close}).max(axis=1)
    df_copy['ATRr_14'] = tr.ewm(span=14, adjust=False).mean() # Smooth mit EMA

    # Fülle alle NaN-Werte, die durch die Indikatorberechnungen entstehen (z.B. am Anfang)
    df_copy.fillna(0, inplace=True) # Oder eine andere geeignete Füllstrategie

    # Features, die skaliert werden sollen
    features_to_scale = [
        'Close', 'daily_return', 'SMA_10', 'SMA_50', 'RSI_14',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'ATRr_14'
    ]

    # Initialisiere Scaler
    scaler_low = MinMaxScaler(feature_range=(0, 1))
    scaler_high = MinMaxScaler(feature_range=(0, 1))
    
    if not skip_scaling:
        # Skaliere die Features. Wir müssen sicherstellen, dass nur die Features, die wir wirklich skalieren wollen,
        # in den Scaler gegeben werden. 'sma_signal' ist z.B. 0 oder 1, das muss nicht skaliert werden.
        df_copy[features_to_scale] = scaler_low.fit_transform(df_copy[features_to_scale])
        _ = scaler_high.fit_transform(df_copy[features_to_scale]) 
    
    return df_copy, scaler_low, scaler_high

# Die create_regression_targets Funktion bleibt gleich
def create_regression_targets(df, target_col):
    """
    Erstellt ein zukünftiges Preis-Ziel als Regressionstarget.
    """
    df_copy = df.copy()
    SHIFT_DAYS = 1
    df_copy[target_col] = df_copy['Close'].shift(-SHIFT_DAYS)
    return df_copy
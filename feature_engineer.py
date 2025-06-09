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

    # --- WICHTIGE KORREKTUR: FÜGE HIGH UND LOW HINZU, BEVOR SIE VERWENDET WERDEN ---
    # Wenn 'High' oder 'Low' nicht in den ursprünglichen Daten sind, erstelle Dummy-Werte.
    # Für echte Anwendungen solltest du sicherstellen, dass deine historischen Daten diese Spalten enthalten.
    if 'High' not in df_copy.columns:
        # Erstelle High-Werte leicht über dem Close-Preis
        df_copy['High'] = df_copy['Close'] * (1 + np.random.uniform(0.001, 0.005, size=len(df_copy)))
        print("WARNUNG: 'High' Spalte nicht gefunden. Dummy-Werte erstellt.")
    if 'Low' not in df_copy.columns:
        # Erstelle Low-Werte leicht unter dem Close-Preis
        df_copy['Low'] = df_copy['Close'] * (1 - np.random.uniform(0.001, 0.005, size=len(df_copy)))
        print("WARNUNG: 'Low' Spalte nicht gefunden. Dummy-Werte erstellt.")
    # -------------------------------------------------------------------------------


    # 1. Täglicher Return
    df_copy['daily_return'] = df_copy['Close'].pct_change().fillna(0)

    # 2. Gleitende Durchschnitte (SMA_10, SMA_50)
    df_copy['SMA_10'] = df_copy['Close'].rolling(window=10, min_periods=1).mean()
    df_copy['SMA_50'] = df_copy['Close'].rolling(window=50, min_periods=1).mean()

    # 3. SMA Signal (z.B. 1 wenn SMA_10 > SMA_50, sonst 0)
    df_copy['sma_signal'] = np.where(df_copy['SMA_10'] > df_copy['SMA_50'], 1, 0)
    df_copy['sma_signal'] = df_copy['sma_signal'].fillna(0)


    # 4. Relative Stärke Index (RSI_14)
    delta = df_copy['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_copy['RSI_14'] = 100 - (100 / (1 + rs))
    df_copy['RSI_14'] = df_copy['RSI_14'].fillna(0) # Direktzuweisung statt inplace


    # 5. MACD (Moving Average Convergence Divergence) - 12, 26, 9
    ema_12 = df_copy['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df_copy['Close'].ewm(span=26, adjust=False).mean()
    df_copy['MACD_12_26_9'] = ema_12 - ema_26
    df_copy['MACDs_12_26_9'] = df_copy['MACD_12_26_9'].ewm(span=9, adjust=False).mean() # MACD Signal Line
    df_copy['MACDh_12_26_9'] = df_copy['MACD_12_26_9'] - df_copy['MACDs_12_26_9'] # MACD Histogram

    # 6. ATR (Average True Range) - 14
    # True Range (TR)
    high_low = df_copy['High'] - df_copy['Low'] # Jetzt sind 'High' und 'Low' garantiert vorhanden
    high_prev_close = abs(df_copy['High'] - df_copy['Close'].shift(1))
    low_prev_close = abs(df_copy['Low'] - df_copy['Close'].shift(1))
    
    tr = pd.DataFrame({'hl': high_low, 'hpc': high_prev_close, 'lpc': low_prev_close}).max(axis=1)
    df_copy['ATRr_14'] = tr.ewm(span=14, adjust=False).mean() # Smooth mit EMA

    # Fülle alle NaN-Werte, die durch die Indikatorberechnungen entstehen (z.B. am Anfang)
    # Beachte: 'inplace=True' wurde hier in 'df_copy = df_copy.fillna(0)' geändert,
    # um die Pandas FutureWarning zu vermeiden.
    df_copy = df_copy.fillna(0) 

    # Features, die skaliert werden sollen
    features_to_scale = [
        'Close', 'daily_return', 'SMA_10', 'SMA_50', 'RSI_14',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'ATRr_14'
    ]

    # Initialisiere Scaler
    scaler_low = MinMaxScaler(feature_range=(0, 1))
    scaler_high = MinMaxScaler(feature_range=(0, 1))
    
    if not skip_scaling:
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
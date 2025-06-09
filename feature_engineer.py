import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random # Falls du random.uniform verwendest

# Definieren des Fensters für gleitende Durchschnitte
WINDOW_SIZE = 10

def add_features_to_data(df, asset_name="", skip_scaling=False):
    """
    Fügt einem DataFrame technische Indikatoren als Features hinzu und skaliert diese.
    Gibt den erweiterten DataFrame und die verwendeten Scaler zurück.
    """
    df_copy = df.copy()

    # Stelle sicher, dass 'Close' existiert
    if 'Close' not in df_copy.columns:
        raise ValueError("DataFrame muss eine 'Close'-Spalte enthalten.")

    # Berechne gleitende Durchschnitte
    df_copy['SMA_short'] = df_copy['Close'].rolling(window=WINDOW_SIZE, min_periods=1).mean()
    df_copy['SMA_long'] = df_copy['Close'].rolling(window=WINDOW_SIZE * 2, min_periods=1).mean()

    # Berechne RSI (Relative Stärke Index) - Vereinfacht für Demo
    delta = df_copy['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=WINDOW_SIZE, min_periods=1).mean()
    avg_loss = loss.rolling(window=WINDOW_SIZE, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df_copy['RSI'] = 100 - (100 / (1 + rs))
    df_copy['RSI'].fillna(0, inplace=True) # Fülle NaN-Werte, die am Anfang entstehen

    # Berechne MACD (Moving Average Convergence Divergence) - Vereinfacht für Demo
    ema_short = df_copy['Close'].ewm(span=WINDOW_SIZE, adjust=False).mean()
    ema_long = df_copy['Close'].ewm(span=WINDOW_SIZE * 2, adjust=False).mean()
    df_copy['MACD'] = ema_short - ema_long

    # Berechne 'relative_price_change' und 'dummy_volume_indicator'
    # Hier werden die Dummy-Features generiert, wie du sie ursprünglich hattest
    # Sie benötigen keinen "previous_price" aus dem DataFrame direkt,
    # sondern simulieren nur eine kleine Volatilität.
    volatility_factor = 0.01
    if "bitcoin" in asset_name.lower():
        volatility_factor = 0.02
    elif "gold" in asset_name.lower():
        volatility_factor = 0.005

    # Erstelle simulierte 'relative_price_change' für jede Zeile
    # Dies ist eine vereinfachte Version, die keine historische Abhängigkeit hat
    # Wenn du echte historische Daten verwenden möchtest, müsstest du dies hier anpassen
    df_copy['relative_price_change'] = df_copy['Close'].pct_change().fillna(0) # Tatsächliche prozentuale Änderung

    # Eine Dummy-Volumen-Indikator, die mit der Preisänderung korreliert
    df_copy['dummy_volume_indicator'] = (df_copy['relative_price_change'].abs() * 10 +
                                         np.random.uniform(0.3, 0.6, size=len(df_copy)))
    df_copy['dummy_volume_indicator'] = df_copy['dummy_volume_indicator'].clip(upper=1.0) # Begrenze auf 1.0

    # Features, die skaliert werden sollen
    features_to_scale = ['Close', 'SMA_short', 'SMA_long', 'RSI', 'MACD', 'relative_price_change', 'dummy_volume_indicator']

    # Initialisiere Scaler
    scaler_low = MinMaxScaler(feature_range=(0, 1)) # Skaliert für "low" Vorhersage (z.B. von 0 bis 1)
    scaler_high = MinMaxScaler(feature_range=(0, 1)) # Skaliert für "high" Vorhersage (z.B. von 0 bis 1)
    
    if not skip_scaling:
        # Skaliere die Features
        # WICHTIG: Fitte die Scaler auf die Daten und transformiere sie
        # Wir müssen separate Scaler für Low und High nutzen, da sie
        # später eventuell unterschiedlich gefittet werden könnten.
        # Für den Anfang können sie auf denselben Daten gefittet werden.
        df_copy[features_to_scale] = scaler_low.fit_transform(df_copy[features_to_scale])
        # Für den high-Scaler wird derselbe fit verwendet, da die Daten identisch sind
        # Wenn du unterschiedliche Skalierungen für low/high haben möchtest,
        # müsstest du hier unterschiedliche Daten für den fit verwenden.
        _ = scaler_high.fit_transform(df_copy[features_to_scale]) # _ um die Rückgabe zu ignorieren

    # Entferne NaN-Werte, die durch die Berechnungen entstanden sein könnten
    df_copy.dropna(inplace=True)

    return df_copy, scaler_low, scaler_high # Gibt den DataFrame und die Scaler zurück

# Die create_regression_targets Funktion bleibt gleich
def create_regression_targets(df, target_col):
    """
    Erstellt ein zukünftiges Preis-Ziel als Regressionstarget.
    """
    df_copy = df.copy()
    # Beispiel: Ziel ist der Preis in 1 Tag (SHIFT_DAYS = 1)
    SHIFT_DAYS = 1
    df_copy[target_col] = df_copy['Close'].shift(-SHIFT_DAYS)
    return df_copy
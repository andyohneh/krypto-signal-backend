# app.py
from flask import Flask, jsonify, request
import requests
import random
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression # Bleibt für Typ-Annotation, aber nicht mehr zum Trainieren hier
import joblib # Wichtig zum Laden des Modells
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = Flask(__name__)

# --- API-Keys aus Umgebungsvariablen lesen ---
BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY')
FMP_API_KEY = os.environ.get('FMP_API_KEY')
# -------------------------------------------

SETTINGS_FILE = 'settings.json'
MODEL_FILENAME = "trading_model.joblib" # Name der Modelldatei

ml_model = None # Globale Variable für das geladene Modell

def load_settings():
    data_dir = os.environ.get('RENDER_DATA_DIR', '.')
    settings_path = os.path.join(data_dir, SETTINGS_FILE)
    if os.path.exists(settings_path):
        try:
            with open(settings_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Fehler: {settings_path} ist beschädigt. Verwende Standardeinstellungen.")
            return get_default_settings()
    return get_default_settings()

def get_default_settings():
    return {
        "bitcoin_tp_percentage": 2.5, "bitcoin_sl_percentage": 1.5,
        "xauusd_tp_percentage": 1.8, "xauusd_sl_percentage": 0.8,
        "update_interval_minutes": 15
    }

def save_settings(settings):
    data_dir = os.environ.get('RENDER_DATA_DIR', '.')
    settings_path = os.path.join(data_dir, SETTINGS_FILE)
    if not os.path.exists(data_dir) and data_dir != '.':
        os.makedirs(data_dir)
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=4)
    print(f"Einstellungen gespeichert in: {settings_path}")

# NEU: Funktion zum Laden des trainierten Modells
def load_trained_model(filename=MODEL_FILENAME):
    model_path = filename # Wenn die Modelldatei im selben Verzeichnis wie app.py liegt
    if os.path.exists(model_path):
        try:
            loaded_model = joblib.load(model_path)
            print(f"Modell erfolgreich aus '{model_path}' geladen.")
            return loaded_model
        except Exception as e:
            print(f"Fehler beim Laden des Modells '{model_path}': {e}")
            return None
    else:
        print(f"Modelldatei '{model_path}' nicht gefunden. Kann keine Vorhersagen machen.")
        return None

# Die alte train_ml_model() Funktion wird ENTFERNT

def create_features_for_prediction(current_price, asset_name=""):
    volatility_factor = 0.01
    if "bitcoin" in asset_name.lower(): volatility_factor = 0.02
    elif "gold" in asset_name.lower(): volatility_factor = 0.005
    simulated_previous_price = current_price * (1 + random.uniform(-volatility_factor, volatility_factor / 2)) 
    relative_price_change = (current_price - simulated_previous_price) / simulated_previous_price
    dummy_volume_indicator = abs(relative_price_change) * 10 + random.uniform(0.3, 0.6)
    dummy_volume_indicator = min(dummy_volume_indicator, 0.9)
    return np.array([[relative_price_change, dummy_volume_indicator]])

# --- Hauptprogrammfluss ---
current_settings = load_settings()
ml_model = load_trained_model() # Modell beim Serverstart laden

@app.route('/')
def home():
    return "Hallo von deinem Flask-Backend! Einstellungen: " + json.dumps(current_settings)

@app.route('/get_signals')
def get_signals():
    global current_settings, ml_model # ml_model ist jetzt global geladen
    bitcoin_data = {}
    gold_data = {}
    global_error_message = ""

    if ml_model is None: # Prüfen, ob das Modell geladen wurde
        global_error_message = "ML-Modell konnte nicht geladen werden. Signale sind nicht verfügbar."
        bitcoin_data = {"price": "Fehler", "signal_type": "Modellfehler"}
        gold_data = {"price": "Fehler", "signal_type": "Modellfehler"}
    elif not BINANCE_API_KEY or not FMP_API_KEY:
        global_error_message = "API-Keys nicht konfiguriert auf dem Server."
        bitcoin_data = {"price": "Fehler", "signal_type": "API Key Fehler"}
        gold_data = {"price": "Fehler", "signal_type": "API Key Fehler"}
    else:
        # Bitcoin
        try:
            binance_url = 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT'
            headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
            response_btc = requests.get(binance_url, headers=headers, timeout=10)
            response_btc.raise_for_status()
            btc_price_data = response_btc.json()
            current_btc_price = float(btc_price_data['price'])
            btc_features = create_features_for_prediction(current_btc_price, "Bitcoin")
            btc_prediction = ml_model.predict(btc_features)[0]
            btc_signal_type = "Verkauf" if btc_prediction == 1 else "Kauf"
            btc_tp_percentage = current_settings.get("bitcoin_tp_percentage", 2.5)
            btc_sl_percentage = current_settings.get("bitcoin_sl_percentage", 1.5)
            calculated_btc_tp = current_btc_price * (1 + (btc_tp_percentage / 100))
            calculated_btc_sl = current_btc_price * (1 - (btc_sl_percentage / 100))
            bitcoin_data = {"price": round(current_btc_price, 2), "entry": round(current_btc_price, 2), "take_profit": round(calculated_btc_tp, 2), "stop_loss": round(calculated_btc_sl, 2), "signal_type": btc_signal_type}
        except requests.exceptions.RequestException as e:
            global_error_message += f"Fehler beim Bitcoin-Abruf: {e}. "
            bitcoin_data = {"price": "Nicht verfügbar", "signal_type": "Fehler"}
        except Exception as e:
            global_error_message += f"Fehler bei Bitcoin-Verarbeitung: {e}. "
            bitcoin_data = {"price": "Nicht verfügbar", "signal_type": "Fehler"}

        # Gold
        try:
            fmp_url = f'https://financialmodelingprep.com/api/v3/quote/XAUUSD?apikey={FMP_API_KEY}'
            response_xauusd = requests.get(fmp_url, timeout=10)
            response_xauusd.raise_for_status()
            xauusd_data_list = response_xauusd.json()
            if xauusd_data_list and isinstance(xauusd_data_list, list) and len(xauusd_data_list) > 0:
                current_xauusd_price = float(xauusd_data_list[0]['price'])
                xauusd_features = create_features_for_prediction(current_xauusd_price, "Gold")
                xauusd_prediction = ml_model.predict(xauusd_features)[0]
                xauusd_signal_type = "Verkauf" if xauusd_prediction == 1 else "Kauf"
                xauusd_tp_percentage = current_settings.get("xauusd_tp_percentage", 1.8)
                xauusd_sl_percentage = current_settings.get("xauusd_sl_percentage", 0.8)
                calculated_xauusd_tp = current_xauusd_price * (1 + (xauusd_tp_percentage / 100))
                calculated_xauusd_sl = current_xauusd_price * (1 - (xauusd_sl_percentage / 100))
                gold_data = {"price": round(current_xauusd_price, 2), "entry": round(current_xauusd_price, 2), "take_profit": round(calculated_xauusd_tp, 2), "stop_loss": round(calculated_xauusd_sl, 2), "signal_type": xauusd_signal_type}
            else:
                global_error_message += "Gold-Daten von FMP sind leer oder ungültig. "
                gold_data = {"price": "Nicht verfügbar", "signal_type": "Fehler"}
        except requests.exceptions.RequestException as e:
            global_error_message += f"Fehler beim Gold-Abruf: {e}. "
            gold_data = {"price": "Nicht verfügbar", "signal_type": "Fehler"}
        except Exception as e:
            global_error_message += f"Fehler bei Gold-Verarbeitung: {e}. "
            gold_data = {"price": "Nicht verfügbar", "signal_type": "Fehler"}

    response_data = {"bitcoin": bitcoin_data, "gold": gold_data, "settings": current_settings}
    if global_error_message:
        response_data["global_error"] = global_error_message.strip()
    return jsonify(response_data)

@app.route('/save_settings', methods=['POST'])
def save_app_settings():
    global current_settings
    data = request.get_json() 
    if data:
        current_settings.update(data)
        save_settings(current_settings)
        # print(f"Einstellungen aktualisiert und gespeichert: {current_settings}") # Debug-Ausgabe
        return jsonify({"status": "success", "message": "Settings saved."})
    else:
        return jsonify({"status": "error", "message": "No JSON data received."}), 400

if __name__ == '__main__':
    print("Flask-Server wird gestartet...")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
# app.py
from flask import Flask, jsonify, request
import requests
import random
import json
import os # Wichtig für Umgebungsvariablen
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = Flask(__name__)

# --- API-Keys aus Umgebungsvariablen lesen (SICHERER!) ---
# Diese Variablen werden auf Render.com gesetzt
BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY')
FMP_API_KEY = os.environ.get('FMP_API_KEY')
# ------------------------------------------------------

SETTINGS_FILE = 'settings.json' # Name der Datei für Einstellungen

ml_model = None

def load_settings():
    # Versuche, Einstellungen aus dem Render-Datenverzeichnis zu laden
    # Render stellt oft einen Pfad für persistente Daten bereit
    data_dir = os.environ.get('RENDER_DATA_DIR', '.') # Standardmäßig aktuelles Verzeichnis
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
        "bitcoin_tp_percentage": 2.5,
        "bitcoin_sl_percentage": 1.5,
        "xauusd_tp_percentage": 1.8,
        "xauusd_sl_percentage": 0.8,
        "update_interval_minutes": 15
    }

def save_settings(settings):
    data_dir = os.environ.get('RENDER_DATA_DIR', '.')
    settings_path = os.path.join(data_dir, SETTINGS_FILE)

    # Erstelle das Datenverzeichnis, falls es nicht existiert (für lokale Tests)
    if not os.path.exists(data_dir) and data_dir != '.':
        os.makedirs(data_dir)

    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=4)
    print(f"Einstellungen gespeichert in: {settings_path}")


def train_ml_model():
    print("Starte ML-Modelltraining...")
    X = np.array([
        [0.01, 0.5], [0.02, 0.7], [-0.01, 0.4], [-0.02, 0.8],
        [0.005, 0.6], [-0.005, 0.3], [0.015, 0.9], [-0.015, 0.7],
    ])
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    model = LogisticRegression()
    model.fit(X, y)
    print("ML-Modelltraining abgeschlossen.")
    return model

def create_features_for_prediction(current_price, asset_name=""):
    volatility_factor = 0.01
    if "bitcoin" in asset_name.lower():
        volatility_factor = 0.02
    elif "gold" in asset_name.lower():
        volatility_factor = 0.005
    simulated_previous_price = current_price * (1 + random.uniform(-volatility_factor, volatility_factor / 2)) 
    relative_price_change = (current_price - simulated_previous_price) / simulated_previous_price
    dummy_volume_indicator = abs(relative_price_change) * 10 + random.uniform(0.3, 0.6)
    dummy_volume_indicator = min(dummy_volume_indicator, 0.9)
    return np.array([[relative_price_change, dummy_volume_indicator]])

current_settings = load_settings() # Einstellungen laden beim Serverstart
ml_model = train_ml_model() # Modell trainieren beim Serverstart

@app.route('/')
def home():
    return "Hallo von deinem Flask-Backend! Einstellungen: " + json.dumps(current_settings)

@app.route('/get_signals')
def get_signals():
    # ... (Rest der get_signals Funktion bleibt gleich, sie nutzt current_settings)
    global current_settings
    bitcoin_data = {}
    gold_data = {}
    global_error_message = ""

    # Sicherstellen, dass API-Keys geladen wurden
    if not BINANCE_API_KEY or not FMP_API_KEY:
        global_error_message = "API-Keys nicht konfiguriert auf dem Server."
        bitcoin_data = {"price": "Fehler", "entry": "Fehler", "take_profit": "Fehler", "stop_loss": "Fehler", "signal_type": "Fehler"}
        gold_data = {"price": "Fehler", "entry": "Fehler", "take_profit": "Fehler", "stop_loss": "Fehler", "signal_type": "Fehler"}
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
            bitcoin_data = {"price": "Nicht verfügbar", "entry": "Nicht verfügbar", "take_profit": "Nicht verfügbar", "stop_loss": "Nicht verfügbar", "signal_type": "Fehler"}
        except Exception as e:
            global_error_message += f"Fehler bei Bitcoin-Verarbeitung: {e}. "
            bitcoin_data = {"price": "Nicht verfügbar", "entry": "Nicht verfügbar", "take_profit": "Nicht verfügbar", "stop_loss": "Nicht verfügbar", "signal_type": "Fehler"}

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
                gold_data = {"price": "Nicht verfügbar", "entry": "Nicht verfügbar", "take_profit": "Nicht verfügbar", "stop_loss": "Nicht verfügbar", "signal_type": "Fehler"}
        except requests.exceptions.RequestException as e:
            global_error_message += f"Fehler beim Gold-Abruf: {e}. "
            gold_data = {"price": "Nicht verfügbar", "entry": "Nicht verfügbar", "take_profit": "Nicht verfügbar", "stop_loss": "Nicht verfügbar", "signal_type": "Fehler"}
        except Exception as e:
            global_error_message += f"Fehler bei Gold-Verarbeitung: {e}. "
            gold_data = {"price": "Nicht verfügbar", "entry": "Nicht verfügbar", "take_profit": "Nicht verfügbar", "stop_loss": "Nicht verfügbar", "signal_type": "Fehler"}

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
        print(f"Einstellungen aktualisiert und gespeichert: {current_settings}")
        return jsonify({"status": "success", "message": "Settings saved."})
    else:
        return jsonify({"status": "error", "message": "No JSON data received."}), 400

if __name__ == '__main__':
    print("Flask-Server wird gestartet...")
    # Render verwendet die PORT Umgebungsvariable. Fallback auf 5000 für lokale Tests.
    port = int(os.environ.get('PORT', 5000))
    # debug=False ist besser für die Produktion. Render setzt dies oft auch.
    # host='0.0.0.0' ist wichtig, damit der Server von außen erreichbar ist.
    app.run(debug=False, host='0.0.0.0', port=port)
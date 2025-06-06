# app.py
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy # NEU
import requests
import random
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = Flask(__name__)

# --- Konfiguration für die Datenbank ---
# Die URL wird aus den Umgebungsvariablen von Render gelesen
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app) # Initialisiert die Datenbank-Verbindung

# --- API-Keys (werden weiterhin aus Umgebungsvariablen gelesen) ---
BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY')
FMP_API_KEY = os.environ.get('FMP_API_KEY')

MODEL_FILENAME = "trading_model.joblib"
ml_model = None

# --- Datenbank-Modell für unsere Einstellungen ---
# Definiert, wie unsere "settings"-Tabelle in der DB aussehen soll
class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True) # Jede Tabelle braucht einen Primärschlüssel
    bitcoin_tp_percentage = db.Column(db.Float, default=2.5)
    bitcoin_sl_percentage = db.Column(db.Float, default=1.5)
    xauusd_tp_percentage = db.Column(db.Float, default=1.8)
    xauusd_sl_percentage = db.Column(db.Float, default=0.8)
    update_interval_minutes = db.Column(db.Integer, default=15)

# --- Neue Funktionen zum Laden und Speichern von Einstellungen aus der DB ---
def load_settings():
    # Finde den ersten (und einzigen) Einstellungs-Eintrag in der DB
    settings = Settings.query.first()
    if settings:
        # Konvertiere das DB-Objekt in ein Dictionary (ähnlich wie unser altes JSON)
        return {
            "bitcoin_tp_percentage": settings.bitcoin_tp_percentage,
            "bitcoin_sl_percentage": settings.bitcoin_sl_percentage,
            "xauusd_tp_percentage": settings.xauusd_tp_percentage,
            "xauusd_sl_percentage": settings.xauusd_sl_percentage,
            "update_interval_minutes": settings.update_interval_minutes,
        }
    # Falls die DB leer ist, erstelle einen Standard-Eintrag
    print("Keine Einstellungen in der DB gefunden, erstelle Standard-Eintrag.")
    default_settings_obj = Settings()
    db.session.add(default_settings_obj)
    db.session.commit()
    return get_default_settings() # Gib die Standardwerte zurück

def get_default_settings():
    return {
        "bitcoin_tp_percentage": 2.5, "bitcoin_sl_percentage": 1.5,
        "xauusd_tp_percentage": 1.8, "xauusd_sl_percentage": 0.8,
        "update_interval_minutes": 15
    }

def save_settings(new_settings):
    # Finde den ersten (und einzigen) Einstellungs-Eintrag
    settings_obj = Settings.query.first()
    if settings_obj:
        # Aktualisiere die Werte des gefundenen Eintrags
        for key, value in new_settings.items():
            if hasattr(settings_obj, key):
                setattr(settings_obj, key, value)
        db.session.commit() # Speichere die Änderungen in der DB
        print(f"Einstellungen in DB aktualisiert.")

def load_trained_model(filename=MODEL_FILENAME):
    if os.path.exists(filename):
        try:
            return joblib.load(filename)
        except Exception as e:
            print(f"Fehler beim Laden des Modells: {e}")
            return None
    print(f"Modelldatei '{filename}' nicht gefunden.")
    return None

def create_features_for_prediction(current_price, asset_name=""):
    # Diese Funktion bleibt unverändert
    volatility_factor = 0.01
    if "bitcoin" in asset_name.lower(): volatility_factor = 0.02
    elif "gold" in asset_name.lower(): volatility_factor = 0.005
    simulated_previous_price = current_price * (1 + random.uniform(-volatility_factor, volatility_factor / 2)) 
    relative_price_change = (current_price - simulated_previous_price) / simulated_previous_price
    dummy_volume_indicator = abs(relative_price_change) * 10 + random.uniform(0.3, 0.6)
    dummy_volume_indicator = min(dummy_volume_indicator, 0.9)
    return np.array([[relative_price_change, dummy_volume_indicator]])

# --- Hauptprogrammfluss ---
# Erstelle die Datenbank-Tabelle(n), falls sie noch nicht existieren
with app.app_context():
    db.create_all()

current_settings = load_settings() # Lade Einstellungen aus der DB
ml_model = load_trained_model()

# --- API-Routen ---
@app.route('/')
def home():
    return "Hallo von deinem Flask-Backend! Einstellungen aus DB: " + json.dumps(current_settings)

@app.route('/get_signals')
def get_signals():
    # Die Logik hier bleibt fast gleich, sie verwendet 'current_settings'
    # ... (kompletter Code der get_signals Funktion von vorher) ...
    global current_settings, ml_model
    bitcoin_data = {}
    gold_data = {}
    global_error_message = ""

    if ml_model is None:
        global_error_message = "ML-Modell konnte nicht geladen werden."
        bitcoin_data = {"price": "Fehler", "signal_type": "Modellfehler"}
        gold_data = {"price": "Fehler", "signal_type": "Modellfehler"}
    elif not BINANCE_API_KEY or not FMP_API_KEY:
        global_error_message = "API-Keys nicht konfiguriert auf dem Server."
        bitcoin_data = {"price": "Fehler", "signal_type": "API Key Fehler"}
        gold_data = {"price": "Fehler", "signal_type": "API Key Fehler"}
    else:
        try: # Bitcoin
            response_btc = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", headers={'X-MBX-APIKEY': BINANCE_API_KEY}, timeout=10)
            response_btc.raise_for_status()
            current_btc_price = float(response_btc.json()['price'])
            btc_features = create_features_for_prediction(current_btc_price, "Bitcoin")
            btc_prediction = ml_model.predict(btc_features)[0]
            btc_signal_type = "Verkauf" if btc_prediction == 1 else "Kauf"
            btc_tp_percentage = current_settings.get("bitcoin_tp_percentage", 2.5)
            btc_sl_percentage = current_settings.get("bitcoin_sl_percentage", 1.5)
            calculated_btc_tp = current_btc_price * (1 + (btc_tp_percentage / 100))
            calculated_btc_sl = current_btc_price * (1 - (btc_sl_percentage / 100))
            bitcoin_data = {"price": round(current_btc_price, 2), "entry": round(current_btc_price, 2), "take_profit": round(calculated_btc_tp, 2), "stop_loss": round(calculated_btc_sl, 2), "signal_type": btc_signal_type}
        except Exception as e:
            global_error_message += f"Fehler Bitcoin: {e}. "
            bitcoin_data = {"price": "N/A", "signal_type": "Fehler"}

        try: # Gold
            response_xauusd = requests.get(f'https://financialmodelingprep.com/api/v3/quote/XAUUSD?apikey={FMP_API_KEY}', timeout=10)
            response_xauusd.raise_for_status()
            xauusd_data_list = response_xauusd.json()
            if xauusd_data_list:
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
                global_error_message += "Gold-Daten leer. "
                gold_data = {"price": "N/A", "signal_type": "Fehler"}
        except Exception as e:
            global_error_message += f"Fehler Gold: {e}. "
            gold_data = {"price": "N/A", "signal_type": "Fehler"}

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
        return jsonify({"status": "success", "message": "Settings saved to DB."})
    return jsonify({"status": "error", "message": "No JSON data received."}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
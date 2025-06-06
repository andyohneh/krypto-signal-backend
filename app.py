# app.py (FINALE VERSION mit ECHTER KI)

import os
import json
import requests
import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import firebase_admin
from firebase_admin import credentials, messaging

# NEU: Importiere unsere Helfer-Funktionen
from data_manager import download_historical_data
from feature_engineer import add_features_to_data

# --- Initialisierungen (App, Firebase, DB) ---
app = Flask(__name__)

try:
    firebase_cred_json_str = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
    if firebase_cred_json_str:
        cred = credentials.Certificate(json.loads(firebase_cred_json_str))
        firebase_admin.initialize_app(cred)
        print("Firebase Admin SDK initialisiert.")
    else:
        print("WARNUNG: Firebase nicht initialisiert.")
except Exception as e:
    print(f"FEHLER bei Firebase-Initialisierung: {e}")

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Lade die neuen, trainierten Modelle ---
try:
    btc_model = joblib.load("trained_btc_model.joblib")
    gold_model = joblib.load("trained_gold_model.joblib")
    print("Echte, trainierte KI-Modelle für BTC und Gold geladen.")
except FileNotFoundError:
    btc_model = None
    gold_model = None
    print("FEHLER: Trainierte Modelldateien nicht gefunden!")

# --- Datenbank-Modell & Funktionen (unverändert) ---
class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # ... (alle Spalten wie vorher)
    bitcoin_tp_percentage = db.Column(db.Float, default=2.5)
    bitcoin_sl_percentage = db.Column(db.Float, default=1.5)
    xauusd_tp_percentage = db.Column(db.Float, default=1.8)
    xauusd_sl_percentage = db.Column(db.Float, default=0.8)
    update_interval_minutes = db.Column(db.Integer, default=15)

def load_settings_from_db():
    # ... (Funktion wie vorher)
    settings = Settings.query.first()
    if settings:
        return { "bitcoin_tp_percentage": settings.bitcoin_tp_percentage, "bitcoin_sl_percentage": settings.bitcoin_sl_percentage, "xauusd_tp_percentage": settings.xauusd_tp_percentage, "xauusd_sl_percentage": settings.xauusd_sl_percentage, "update_interval_minutes": settings.update_interval_minutes }
    print("DB leer, erstelle Standard-Eintrag.")
    default = Settings()
    db.session.add(default)
    db.session.commit()
    return { "bitcoin_tp_percentage": 2.5, "bitcoin_sl_percentage": 1.5, "xauusd_tp_percentage": 1.8, "xauusd_sl_percentage": 0.8, "update_interval_minutes": 15 }

# ... (weitere DB-Funktionen wie save_settings unverändert)
def save_settings(new_settings):
    settings_obj = Settings.query.first()
    if settings_obj:
        for key, value in new_settings.items():
            if hasattr(settings_obj, key): setattr(settings_obj, key, value)
        db.session.commit()

# --- App-Kontext zum Start ---
with app.app_context():
    db.create_all()
    current_settings = load_settings_from_db()
print(f"Einstellungen beim Start geladen: {current_settings}")


# --- NEUE Funktion zur Feature-Erstellung für die Live-Vorhersage ---
def create_live_features(ticker):
    """Holt die neuesten Daten und berechnet die Features für die Vorhersage."""
    # 1. Lade die neuesten historischen Daten
    raw_data = download_historical_data(ticker, period="3mo", interval="1d") # 3 Monate reichen für 50-Tage-SMA
    if raw_data is None:
        return None

    # 2. Berechne die Features
    featured_data = add_features_to_data(raw_data)
    if featured_data is None:
        return None

    # 3. Wähle die letzte Zeile (die aktuellsten Features)
    latest_features = featured_data.iloc[-1]

    # 4. Formatiere sie für das Modell (als 2D-Array)
    model_input = np.array([[
        latest_features['daily_return'],
        latest_features['SMA_10'],
        latest_features['SMA_50'],
        latest_features['sma_signal']
    ]])
    return model_input

# --- API-Routen ---
@app.route('/')
def home():
    return "Krypto Helfer Backend - Echte KI-Modelle sind live!"

@app.route('/get_signals')
def get_signals():
    global current_settings
    bitcoin_data = {}
    gold_data = {}
    global_error_message = ""

    # --- Bitcoin Signal mit ECHTER KI ---
    if btc_model:
        try:
            # 1. Hole Live-Preis
            btc_price_response = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT").json()
            current_btc_price = float(btc_price_response['price'])

            # 2. Erstelle ECHTE Features aus den neuesten historischen Daten
            btc_features = create_live_features("BTC-USD")
            if btc_features is not None:
                # 3. Mache eine Vorhersage mit dem trainierten BTC-Modell
                prediction = btc_model.predict(btc_features)[0]
                btc_signal_type = "Kauf" if prediction == 1 else "Verkauf" # 1=Preis rauf, 0=Preis runter

                # Berechne TP/SL (wie vorher)
                tp_perc = current_settings.get("bitcoin_tp_percentage", 2.5)
                sl_perc = current_settings.get("bitcoin_sl_percentage", 1.5)
                bitcoin_data = {
                    "price": round(current_btc_price, 2), "entry": round(current_btc_price, 2),
                    "take_profit": round(current_btc_price * (1 + tp_perc/100), 2),
                    "stop_loss": round(current_btc_price * (1 - sl_perc/100), 2),
                    "signal_type": btc_signal_type
                }
            else:
                global_error_message += "BTC Feature-Erstellung fehlgeschlagen. "
                bitcoin_data = {"signal_type": "Fehler"}
        except Exception as e:
            global_error_message += f"Fehler bei BTC-Signal: {e}. "
            bitcoin_data = {"signal_type": "Fehler"}
    else:
        global_error_message += "BTC-Modell nicht geladen. "
        bitcoin_data = {"signal_type": "Fehler"}


    # --- Gold Signal mit ECHTER KI ---
    if gold_model:
        # Ähnliche Logik für Gold
        try:
            FMP_API_KEY = os.environ.get('FMP_API_KEY')
            gold_price_response = requests.get(f'https://financialmodelingprep.com/api/v3/quote/XAUUSD?apikey={FMP_API_KEY}').json()
            current_gold_price = float(gold_price_response[0]['price'])

            gold_features = create_live_features("GC=F")
            if gold_features is not None:
                prediction = gold_model.predict(gold_features)[0]
                gold_signal_type = "Kauf" if prediction == 1 else "Verkauf"

                tp_perc = current_settings.get("xauusd_tp_percentage", 1.8)
                sl_perc = current_settings.get("xauusd_sl_percentage", 0.8)
                gold_data = {
                    "price": round(current_gold_price, 2), "entry": round(current_gold_price, 2),
                    "take_profit": round(current_gold_price * (1 + tp_perc/100), 2),
                    "stop_loss": round(current_gold_price * (1 - sl_perc/100), 2),
                    "signal_type": gold_signal_type
                }
            else:
                global_error_message += "Gold Feature-Erstellung fehlgeschlagen. "
                gold_data = {"signal_type": "Fehler"}
        except Exception as e:
            global_error_message += f"Fehler bei Gold-Signal: {e}. "
            gold_data = {"signal_type": "Fehler"}
    else:
        global_error_message += "Gold-Modell nicht geladen. "
        gold_data = {"signal_type": "Fehler"}


    # --- Finale Antwort zusammenbauen ---
    response_data = {"bitcoin": bitcoin_data, "gold": gold_data, "settings": current_settings}
    if global_error_message:
        response_data["global_error"] = global_error_message.strip()
    return jsonify(response_data)


# Die Endpunkte /save_settings und /send_test_notification bleiben unverändert
@app.route('/save_settings', methods=['POST'])
def save_app_settings():
    global current_settings; data = request.get_json()
    if data:
        current_settings.update(data); save_settings(current_settings)
        return jsonify({"status": "success", "message": "Settings saved to DB."})
    return jsonify({"status": "error", "message": "No JSON data received."}), 400

@app.route('/send_test_notification', methods=['POST'])
def send_test_notification():
    data = request.get_json(); token = data.get('token')
    if not token: return jsonify({"status": "error", "message": "Kein Token."}), 400
    try:
        message = messaging.Message(notification=messaging.Notification(title='Test!', body='Funktioniert! 🎉'), token=token)
        response = messaging.send(message)
        return jsonify({"status": "success", "message": f"Nachricht gesendet: {response}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
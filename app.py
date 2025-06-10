import os
import json
import requests
import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import LargeBinary, func
import firebase_admin
from firebase_admin import credentials, messaging
from dotenv import load_dotenv

load_dotenv()

from data_manager import download_historical_data
from feature_engineer import add_features_to_data
from train_model import FEATURES_LIST

app = Flask(__name__)

# --- Robuste Firebase-Initialisierung ---
cred = None
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate("serviceAccountKey.json")
        print("Firebase-Credentials aus lokaler Datei geladen.")
    except FileNotFoundError:
        print("Lokale Schlüsseldatei nicht gefunden. Versuche Umgebungsvariable...")
        try:
            cred_str = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
            if cred_str:
                cred = credentials.Certificate(json.loads(cred_str))
                print("Firebase-Credentials aus Umgebungsvariable geladen.")
            else:
                print("WARNUNG: FIREBASE_SERVICE_ACCOUNT_JSON Variable nicht gefunden.")
        except Exception as e:
            print(f"Fehler beim Parsen der Firebase-Credentials: {e}")
    
    if cred:
        firebase_admin.initialize_app(cred)
        print("Firebase Admin SDK initialisiert.")
    else:
        print("Firebase Admin SDK NICHT initialisiert.")

# --- Datenbank-Konfiguration ---
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Datenbank-Modelle ---
class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    bitcoin_tp_percentage = db.Column(db.Float, default=2.5)
    bitcoin_sl_percentage = db.Column(db.Float, default=1.5)
    xauusd_tp_percentage = db.Column(db.Float, default=1.8)
    xauusd_sl_percentage = db.Column(db.Float, default=0.8)
    update_interval_minutes = db.Column(db.Integer, default=15)
    last_btc_signal = db.Column(db.String(100), default='N/A')
    last_gold_signal = db.Column(db.String(100), default='N/A')

class TrainedModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    data = db.Column(LargeBinary, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())

class Device(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fcm_token = db.Column(db.String(255), unique=True, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())

# --- Globale Variablen für geladene Artefakte ---
current_settings = {}
btc_low_model, btc_high_model, btc_low_scaler, btc_high_scaler = None, None, None, None
gold_low_model, gold_high_model, gold_low_scaler, gold_high_scaler = None, None, None, None

# --- Datenbank- & Helfer-Funktionen ---
def load_artifacts_from_db():
    global btc_low_model, btc_high_model, btc_low_scaler, btc_high_scaler, gold_low_model, gold_high_model, gold_low_scaler, gold_high_scaler
    print("Versuche, Regressions-Modelle und Scaler aus der DB zu laden...")
    try:
        with app.app_context():
            artifacts = TrainedModel.query.all()
            if not artifacts:
                print("WARNUNG: Keine Modelle in der DB gefunden."); return
            
            artifact_map = {artifact.name: pickle.loads(artifact.data) for artifact in artifacts}
            
            btc_low_model=artifact_map.get('btc_low_model'); btc_high_model=artifact_map.get('btc_high_model')
            btc_low_scaler=artifact_map.get('btc_low_scaler'); btc_high_scaler=artifact_map.get('btc_high_scaler')
            gold_low_model=artifact_map.get('gold_low_model'); gold_high_model=artifact_map.get('gold_high_model')
            gold_low_scaler=artifact_map.get('gold_low_scaler'); gold_high_scaler=artifact_map.get('gold_high_scaler')

            if all([btc_low_model, gold_low_model]):
                print(f"Erfolgreich {len(artifacts)} Artefakte geladen.")
            else:
                print("WARNUNG: Einige Regressions-Modelle/Scaler konnten nicht gefunden werden.")
    except Exception as e:
        print(f"FEHLER beim Laden der Artefakte: {e}")

def load_settings_from_db():
    settings = Settings.query.first()
    if not settings:
        settings = Settings(); db.session.add(settings); db.session.commit()
    return {c.name: getattr(settings, c.name) for c in settings.__table__.columns if c.name != 'id'}

def save_settings(new_settings):
    s = Settings.query.first()
    if s:
        for k, v in new_settings.items():
            if hasattr(s, k):
                setattr(s, k, v)
        db.session.commit()

def get_live_features_for_regression(ticker):
    raw_data = download_historical_data(ticker, period="3mo", interval="1d")
    if raw_data is None: return None
    
    featured_data = add_features_to_data(raw_data)
    if featured_data is None or not all(col in featured_data.columns for col in FEATURES_LIST):
        return None
        
    return featured_data[FEATURES_LIST].tail(1)

# --- App-Start ---
with app.app_context():
    db.create_all()
    current_settings = load_settings_from_db()
load_artifacts_from_db()

# --- API-Routen ---
@app.route('/')
def home():
    return "Krypto Helfer 2.0 - Regressions-Modelle sind live!"

@app.route('/get_chart_data/<ticker_symbol>')
def get_chart_data(ticker_symbol):
    try:
        data = download_historical_data(ticker_symbol, period="6mo", interval="1d")
        if data is None: return jsonify({"error": "Rohdaten laden fehlgeschlagen"}), 500
        
        import pandas_ta as ta
        data['SMA_10'] = data['Adj Close'].rolling(window=10).mean()
        data['SMA_50'] = data['Adj Close'].rolling(window=50).mean()
        data['RSI_14'] = ta.rsi(data['Adj Close'], length=14)
        
        chart_columns = ['Adj Close', 'SMA_10', 'SMA_50', 'RSI_14']
        chart_data = data[chart_columns].copy()
        chart_data.rename(columns={'Adj Close': 'price', 'SMA_10': 'sma_short', 'SMA_50': 'sma_long', 'RSI_14': 'rsi'}, inplace=True)
        
        chart_data.reset_index(inplace=True)
        chart_data['Date'] = chart_data['Date'].dt.strftime('%Y-%m-%d')
        
        return jsonify(chart_data.dropna().to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_signals')
def get_signals():
    global current_settings
    bitcoin_data, gold_data, error_msg = {}, {}, ""
    
    if all([btc_low_model, btc_high_model, btc_low_scaler, btc_high_scaler]):
        try:
            current_price = float(requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT").json()['price'])
            latest_features_df = get_live_features_for_regression("BTC-USD")
            
            if latest_features_df is not None:
                live_features_low_scaled = btc_low_scaler.transform(latest_features_df)
                live_features_high_scaled = btc_high_scaler.transform(latest_features_df)
                predicted_low = btc_low_model.predict(live_features_low_scaled)[0]
                predicted_high = btc_high_model.predict(live_features_high_scaled)[0]
                atr_value = latest_features_df['ATRr_14'].iloc[0]
                stop_loss = predicted_low - (atr_value * 1.5)
                bitcoin_data = {"price": round(current_price, 2), "entry": round(predicted_low, 2), "take_profit": round(predicted_high, 2), "stop_loss": round(stop_loss, 2), "signal_type": "Preis-Ziel"}
            else:
                error_msg += "BTC Feature-Erstellung fehlgeschlagen. "
        except Exception as e:
            error_msg += f"BTC Fehler: {e}. "; bitcoin_data={"price":"Fehler", "signal_type":"Fehler"}
    else:
        error_msg += "BTC Regressions-Modelle nicht geladen. "
    
    if all([gold_low_model, gold_high_model, gold_low_scaler, gold_high_scaler]):
        try:
            FMP_API_KEY = os.environ.get('FMP_API_KEY')
            current_price = float(requests.get(f'https://financialmodelingprep.com/api/v3/quote/XAUUSD?apikey={FMP_API_KEY}').json()[0]['price'])
            latest_features_df = get_live_features_for_regression("GC=F")

            if latest_features_df is not None:
                live_features_low_scaled = gold_low_scaler.transform(latest_features_df)
                live_features_high_scaled = gold_high_scaler.transform(latest_features_df)
                predicted_low = gold_low_model.predict(live_features_low_scaled)[0]
                predicted_high = gold_high_model.predict(live_features_high_scaled)[0]
                atr_value = latest_features_df['ATRr_14'].iloc[0]
                stop_loss = predicted_low - (atr_value * 1.5)
                gold_data = {"price": round(current_price, 2), "entry": round(predicted_low, 2), "take_profit": round(predicted_high, 2), "stop_loss": round(stop_loss, 2), "signal_type": "Preis-Ziel"}
            else:
                error_msg += "Gold Feature-Erstellung fehlgeschlagen. "
        except Exception as e:
            error_msg += f"Gold Fehler: {e}. "; gold_data={"price":"Fehler", "signal_type":"Fehler"}
    else:
        error_msg += "Gold Regressions-Modelle nicht geladen. "

    response = {"bitcoin": bitcoin_data, "gold": gold_data, "settings": current_settings}
    if error_msg:
        response["global_error"] = error_msg.strip()
    return jsonify(response)
    
@app.route('/save_settings', methods=['POST'])
def save_app_settings():
    global current_settings
    data = request.get_json()
    if data:
        current_settings.update(data)
        save_settings(current_settings)
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

@app.route('/register_device', methods=['POST'])
def register_device():
    data = request.get_json()
    token = data.get('token')
    if not token:
        return jsonify({"status": "error", "message": "Kein Token erhalten."}), 400
    with app.app_context():
        existing_device = Device.query.filter_by(fcm_token=token).first()
        if existing_device:
            existing_device.timestamp = func.now()
            db.session.commit()
            print(f"Geräte-Token {token[:15]}... bereits vorhanden, Zeitstempel aktualisiert.")
            return jsonify({"status": "success", "message": "Gerät bereits registriert."})
        else:
            new_device = Device(fcm_token=token)
            db.session.add(new_device)
            db.session.commit()
            print(f"Neues Gerät mit Token {token[:15]}... registriert.")
            return jsonify({"status": "success", "message": "Gerät erfolgreich registriert."})

@app.route('/send_test_notification', methods=['POST'])
def send_test_notification():
    data = request.get_json()
    token = data.get('token')
    if not token: return jsonify({"status": "error"}), 400
    try:
        message = messaging.Message(notification=messaging.Notification(title='Test!', body='Funktioniert! 🎉'), token=token)
        response = messaging.send(message)
        return jsonify({"status": "success", "response": str(response)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
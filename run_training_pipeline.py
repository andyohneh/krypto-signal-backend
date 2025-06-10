import os
import json
import pickle
import requests
from dotenv import load_dotenv

# Lade die .env-Datei für die lokale Entwicklung
load_dotenv()

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import LargeBinary, func
import firebase_admin
from firebase_admin import credentials, messaging
import pandas as pd
import numpy as np

# Importiere unsere sauberen Helfer-Funktionen und die Feature-Liste
from data_manager import download_historical_data
from feature_engineer import add_features_to_data, create_regression_targets
from train_model import train_regression_model, FEATURES_LIST

# --- Setup ---
# WICHTIG: Die Flask-App wird hier nur für den Datenbank-Kontext benötigt,
# nicht um einen Server zu starten.
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Datenbankmodelle ---
class Device(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fcm_token = db.Column(db.String(255), unique=True, nullable=False)
    timestamp = db.Column(db.DateTime, default=func.now())

class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    last_btc_signal = db.Column(db.String(100), default='N/A')
    last_gold_signal = db.Column(db.String(100), default='N/A')
    scaler_btc_low = db.Column(LargeBinary)
    model_btc_low = db.Column(LargeBinary)
    scaler_btc_high = db.Column(LargeBinary)
    model_btc_high = db.Column(LargeBinary)
    scaler_gold_low = db.Column(LargeBinary)
    model_gold_low = db.Column(LargeBinary)
    scaler_gold_high = db.Column(LargeBinary)
    model_gold_high = db.Column(LargeBinary)
    model_update_timestamp = db.Column(db.DateTime, default=func.now())

    def update_model(self, asset_type, model_type, scaler, model):
        scaler_col = f'scaler_{asset_type}_{model_type}'
        model_col = f'model_{asset_type}_{model_type}'
        setattr(self, scaler_col, pickle.dumps(scaler))
        setattr(self, model_col, pickle.dumps(model))

    def get_model(self, asset_type, model_type):
        scaler_col = f'scaler_{asset_type}_{model_type}'
        model_col = f'model_{asset_type}_{model_type}'
        scaler = pickle.loads(getattr(self, scaler_col)) if getattr(self, scaler_col) else None
        model = pickle.loads(getattr(self, model_col)) if getattr(self, model_col) else None
        return scaler, model

# Firebase-Initialisierung
if not firebase_admin._apps:
    cred = None
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
        except json.JSONDecodeError as e:
            print(f"FEHLER beim Parsen der Firebase JSON Umgebungsvariable: {e}")
        except Exception as e:
            print(f"Allgemeiner FEHLER beim Laden der Firebase-Credentials aus Umgebungsvariable: {e}")
    except Exception as e:
        print(f"Allgemeiner FEHLER beim Laden der Firebase-Credentials: {e}")

    if cred:
        try:
            # WICHTIG: Füge 'projectId' explizit hinzu
            # Ersetze 'krypto-helfer-app' durch deine tatsächliche Projekt-ID
            firebase_admin.initialize_app(cred, {'projectId': 'krypto-helfer-app'})
            print("Firebase erfolgreich initialisiert mit expliziter Projekt-ID.")
        except ValueError as e:
            print(f"Firebase bereits initialisiert oder Fehler: {e}")
        except Exception as e:
            print(f"FEHLER bei der Firebase-Initialisierung: {e}")
    else:
        print("FEHLER: Firebase-Credentials konnten NICHT geladen werden. Firebase wird NICHT initialisiert.")


# --- Hilfsfunktion für Benachrichtigungen (NUR FÜR DEBUGGING MIT EINZELNACHRICHT) ---
def send_notification(title, body, tokens):
    if not tokens:
        print("Keine Tokens für den Versand von Benachrichtigungen vorhanden.")
        return

    if not firebase_admin._apps:
        print("Firebase ist nicht initialisiert. Nachricht kann nicht gesendet werden.")
        return

    # --- START DEBUGGING-ÄNDERUNG ---
    # Sende nur an das ERSTE Token in der Liste, als einfache Nachricht
    # und nicht als MulticastMessage, um den /batch-Endpunkt zu umgehen
    if tokens: # Stelle sicher, dass mindestens ein Token existiert
        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            token=tokens[0], # Nimm nur das erste Token
        )
        try:
            response = messaging.send(message) # Sende einzelne Nachricht
            print(f"DEBUG: Einzelne Nachricht erfolgreich gesendet: {response}")
        except Exception as e:
            print(f"DEBUG: Fehler beim Senden einer einzelnen Benachrichtigung: {e}")
    else:
        print("DEBUG: Keine Tokens zum Senden einer einzelnen Testnachricht vorhanden.")
    # --- ENDE DEBUGGING-ÄNDERUNG ---

    # Den ursprünglichen MulticastMessage-Code für diesen Test NICHT verwenden:
    # message = messaging.MulticastMessage(
    #     notification=messaging.Notification(title=title, body=body),
    #     tokens=tokens,
    # )
    # try:
    #     response = messaging.send_multicast(message)
    #     print(f"Erfolgreich {response.success_count} Nachrichten gesendet, {response.failure_count} Fehler.")
    #     if response.failure_count > 0:
    #         for resp in response.responses:
    #             if not resp.success:
    #                 print(f"Fehler beim Senden: {resp.exception}")
    # except Exception as e:
    #     print(f"Fehler beim Senden der Benachrichtigung: {e}")


def run_training_pipeline():
    # Der gesamte Code der Pipeline, der zuvor hier war, bleibt unverändert.
    # Er wurde nur um den app.run() Block herum angeordnet.
    with app.app_context():
        db.create_all()

        settings = Settings.query.first()
        if not settings:
            settings = Settings()
            db.session.add(settings)
            db.session.commit()
            print("Initialer Settings-Eintrag erstellt.")

        print("Phase 1: Daten herunterladen...")
        btc_data_path = 'data/btc_historical_data.csv'
        gold_data_path = 'data/gold_historical_data.csv'

        if not os.path.exists(btc_data_path) or not os.path.exists(gold_data_path):
            print("Erstelle Dummy-Historische Daten für BTC und GOLD...")
            os.makedirs(os.path.dirname(btc_data_path), exist_ok=True)

            dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
            btc_prices = np.random.rand(100) * 10000 + 30000
            btc_high = btc_prices * (1 + np.random.uniform(0.001, 0.005, size=len(btc_prices)))
            btc_low = btc_prices * (1 - np.random.uniform(0.001, 0.005, size=len(btc_prices)))
            btc_data = pd.DataFrame({'Date': dates, 'Close': btc_prices, 'High': btc_high, 'Low': btc_low})
            btc_data.to_csv(btc_data_path, index=False)

            gold_prices = np.random.rand(100) * 100 + 1800
            gold_high = gold_prices * (1 + np.random.uniform(0.001, 0.005, size=len(gold_prices)))
            gold_low = gold_prices * (1 - np.random.uniform(0.001, 0.005, size=len(gold_prices)))
            gold_data = pd.DataFrame({'Date': dates, 'Close': gold_prices, 'High': gold_high, 'Low': gold_low})
            gold_data.to_csv(gold_data_path, index=False)
            print("Dummy-Daten erstellt.")
        else:
            print("Historische Daten vorhanden. Verwende bestehende Daten.")

        btc_df = pd.read_csv(btc_data_path)
        gold_df = pd.read_csv(gold_data_path)

        btc_df['Date'] = pd.to_datetime(btc_df['Date'])
        gold_df['Date'] = pd.to_datetime(gold_df['Date'])


        print("Phase 2: Feature Engineering und Target-Erstellung...")
        btc_data_engineered, _, _ = add_features_to_data(btc_df.copy(), asset_name="bitcoin")
        gold_data_engineered, _, _ = add_features_to_data(gold_df.copy(), asset_name="gold")

        btc_data_final_low = create_regression_targets(btc_data_engineered.copy(), 'low_target')
        btc_data_final_high = create_regression_targets(btc_data_engineered.copy(), 'high_target')
        gold_data_final_low = create_regression_targets(gold_data_engineered.copy(), 'low_target')
        gold_data_final_high = create_regression_targets(gold_data_engineered.copy(), 'high_target')

        all_features_and_low_target = FEATURES_LIST + ['low_target']
        all_features_and_high_target = FEATURES_LIST + ['high_target']

        btc_data_final_low.dropna(subset=all_features_and_low_target, inplace=True)
        btc_data_final_high.dropna(subset=all_features_and_high_target, inplace=True)
        gold_data_final_low.dropna(subset=all_features_and_low_target, inplace=True)
        gold_data_final_high.dropna(subset=all_features_and_high_target, inplace=True)

        if btc_data_final_low.empty or btc_data_final_high.empty or gold_data_final_low.empty or gold_data_final_high.empty:
            print("FEHLER: Nicht genügend Daten nach Feature Engineering und Target-Erstellung. Modelle werden nicht trainiert.")
            return

        print("Phase 3: Modelltraining und Speicherung...")

        btc_model_low, btc_scaler_low_from_train = train_regression_model(btc_data_final_low, 'low_target')
        if btc_model_low and btc_scaler_low_from_train:
            settings.update_model('btc', 'low', btc_scaler_low_from_train, btc_model_low)
            print("BTC Low-Modell trainiert und gespeichert.")
        else:
            print("WARNUNG: BTC Low-Modell konnte nicht trainiert werden. Überspringe Speicherung.")

        btc_model_high, btc_scaler_high_from_train = train_regression_model(btc_data_final_high, 'high_target')
        if btc_model_high and btc_scaler_high_from_train:
            settings.update_model('btc', 'high', btc_scaler_high_from_train, btc_model_high)
            print("BTC High-Modell trainiert und gespeichert.")
        else:
            print("WARNUNG: BTC High-Modell konnte nicht trainiert werden. Überspringe Speicherung.")

        gold_model_low, gold_scaler_low_from_train = train_regression_model(gold_data_final_low, 'low_target')
        if gold_model_low and gold_scaler_low_from_train:
            settings.update_model('gold', 'low', gold_scaler_low_from_train, gold_model_low)
            print("GOLD Low-Modell trainiert und gespeichert.")
        else:
            print("WARNUNG: GOLD Low-Modell konnte nicht trainiert werden. Überspringe Speicherung.")

        gold_model_high, gold_scaler_high_from_train = train_regression_model(gold_data_final_high, 'high_target')
        if gold_model_high and gold_scaler_high_from_train:
            settings.update_model('gold', 'high', gold_scaler_high_from_train, gold_model_high)
            print("GOLD High-Modell trainiert und gespeichert.")
        else:
            print("WARNUNG: GOLD High-Modell konnte nicht trainiert werden. Überspringe Speicherung.")

        if (btc_model_low or btc_model_high or gold_model_low or gold_model_high):
            settings.model_update_timestamp = func.now()
            db.session.commit()
            print("Modelle erfolgreich aktualisiert und in Datenbank gespeichert (wenn Training erfolgreich war).")
        else:
            print("KEINE Modelle erfolgreich trainiert oder gespeichert. Zeitstempel nicht aktualisiert.")


        print("Phase 4: Signale generieren und Benachrichtigen...")

        device_tokens = [d.fcm_token for d in Device.query.all()]
        if not device_tokens:
            print("Keine registrierten Geräte-Tokens gefunden. Keine Benachrichtigungen möglich.")

        btc_details = {
            "asset_key": "btc",
            "asset_display_name": "Bitcoin",
            "current_price": btc_df['Close'].iloc[-1],
        }

        gold_details = {
            "asset_key": "gold",
            "asset_display_name": "Gold",
            "current_price": gold_df['Close'].iloc[-1],
        }

        for details in [btc_details, gold_details]:
            asset_key = details["asset_key"]
            asset_display_name = details["asset_display_name"]
            current_price = details["current_price"]
            
            scaler_for_predict, model_for_predict = settings.get_model(asset_key, 'low')
            _, high_model_for_predict = settings.get_model(asset_key, 'high')

            if not all([scaler_for_predict, model_for_predict, high_model_for_predict]):
                print(f"FEHLER: Modelle oder Scaler für {asset_display_name} konnten für Vorhersage nicht geladen werden. Überspringe Signalgenerierung.")
                continue

            dummy_current_df = pd.DataFrame({
                'Date': [pd.to_datetime('today')],
                'Close': [current_price],
                'High': [current_price * (1 + np.random.uniform(0.001, 0.005))],
                'Low': [current_price * (1 - np.random.uniform(0.001, 0.005))]
            })
            
            latest_features_df, _, _ = add_features_to_data(dummy_current_df, asset_display_name.lower(), skip_scaling=True)
            
            X_predict = latest_features_df[FEATURES_LIST]

            X_predict_scaled = scaler_for_predict.transform(X_predict)
            
            predicted_low = model_for_predict.predict(X_predict_scaled)[0]
            predicted_high = high_model_for_predict.predict(X_predict_scaled)[0]
            
            new_signal_text = f"Einstieg: {predicted_low:.2f}, TP: {predicted_high:.2f}"
            
            last_signal = getattr(settings, f'last_{asset_key}_signal')
            print(f"Analyse für {asset_display_name}: Letztes Signal='{last_signal}', Neues Signal='{new_signal_text}'")

            if new_signal_text != last_signal or last_signal == 'N/A':
                print(f"-> Signal für {asset_display_name} hat sich geändert! Sende Benachrichtigung...")
                title = f"Neues Preis-Ziel: {asset_display_name}"
                body = f"Neues Ziel: Einstieg ca. {predicted_low:.2f}, Take Profit ca. {predicted_high:.2f}"
                send_notification(title, body, device_tokens)
                
                setattr(settings, f'last_{asset_key}_signal', new_signal_text)
                db.session.commit()
            else:
                print(f"-> Signal für {asset_display_name} unverändert. Keine Aktion nötig.")
    
    print("Trainings-Pipeline abgeschlossen.")

# --- WICHTIG: Entferne den app.run() Aufruf aus dem Cron Job Skript! ---
# if __name__ == '__main__':
#    app.run(debug=True, host='0.0.0.0', port=os.getenv("PORT", 5000))
# Stattdessen nur die run_training_pipeline() Funktion aufrufen
if __name__ == '__main__':
    run_training_pipeline()
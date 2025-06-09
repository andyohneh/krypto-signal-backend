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
from train_model import train_regression_model, FEATURES_LIST # Stelle sicher, dass FEATURES_LIST hier importiert wird

# --- Setup ---
app = Flask(__name__) # Flask-App für SQLAlchemy-Kontext
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Datenbankmodelle (müssen hier wieder definiert werden, da es ein separates Skript ist) ---
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
        # asset_type sollte hier 'btc' oder 'gold' sein
        scaler_col = f'scaler_{asset_type}_{model_type}'
        model_col = f'model_{asset_type}_{model_type}'
        setattr(self, scaler_col, pickle.dumps(scaler))
        setattr(self, model_col, pickle.dumps(model))

    def get_model(self, asset_type, model_type):
        # asset_type sollte hier 'btc' oder 'gold' sein
        scaler_col = f'scaler_{asset_type}_{model_type}'
        model_col = f'model_{asset_type}_{model_type}'
        scaler = pickle.loads(getattr(self, scaler_col)) if getattr(self, scaler_col) else None
        model = pickle.loads(getattr(self, model_col)) if getattr(self, model_col) else None
        return scaler, model

# --- Firebase-Initialisierung (für dieses Skript) ---
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
            firebase_admin.initialize_app(cred)
            print("Firebase erfolgreich initialisiert.")
        except ValueError as e:
            print(f"Firebase bereits initialisiert oder Fehler: {e}")
        except Exception as e:
            print(f"FEHLER bei der Firebase-Initialisierung: {e}")
    else:
        print("FEHLER: Firebase-Credentials konnten NICHT geladen werden. Firebase wird NICHT initialisiert.")


# --- Hilfsfunktion für Benachrichtigungen ---
def send_notification(title, body, tokens):
    if not tokens:
        print("Keine Tokens für den Versand von Benachrichtigungen vorhanden.")
        return

    if not firebase_admin._apps:
        print("Firebase ist nicht initialisiert. Nachricht kann nicht gesendet werden.")
        return

    message = messaging.MulticastMessage(
        notification=messaging.Notification(title=title, body=body),
        tokens=tokens,
    )
    try:
        response = messaging.send_multicast(message)
        print(f"Erfolgreich {response.success_count} Nachrichten gesendet, {response.failure_count} Fehler.")
        if response.failure_count > 0:
            for resp in response.responses:
                if not resp.success:
                    print(f"Fehler beim Senden: {resp.exception}")
    except Exception as e:
        print(f"Fehler beim Senden der Benachrichtigung: {e}")


# --- Hauptlogik der Trainingspipeline ---
def run_training_pipeline():
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
            # Füge Dummy 'High' und 'Low' hinzu, damit feature_engineer sie findet
            btc_high = btc_prices * (1 + np.random.uniform(0.001, 0.005, size=len(btc_prices)))
            btc_low = btc_prices * (1 - np.random.uniform(0.001, 0.005, size=len(btc_prices)))
            btc_data = pd.DataFrame({'Date': dates, 'Close': btc_prices, 'High': btc_high, 'Low': btc_low})
            btc_data.to_csv(btc_data_path, index=False)

            gold_prices = np.random.rand(100) * 100 + 1800
            # Füge Dummy 'High' und 'Low' hinzu
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
        btc_data_engineered, btc_scaler_low, btc_scaler_high = add_features_to_data(btc_df.copy(), asset_name="bitcoin")
        gold_data_engineered, gold_scaler_low, gold_scaler_high = add_features_to_data(gold_df.copy(), asset_name="gold")

        btc_data_final_low = create_regression_targets(btc_data_engineered.copy(), 'low_target')
        btc_data_final_high = create_regression_targets(btc_data_engineered.copy(), 'high_target')
        gold_data_final_low = create_regression_targets(gold_data_engineered.copy(), 'low_target')
        gold_data_final_high = create_regression_targets(gold_data_engineered.copy(), 'high_target')

        # WICHTIG: Sicherstellen, dass die Zielspalten im DataFrame sind, bevor wir .dropna aufrufen
        # und bevor wir das DataFrame an train_regression_model übergeben.
        # Hier ist ein kleiner Workaround, um sicherzustellen, dass die Targets immer da sind
        # und die Dropna-Funktion damit arbeiten kann. Normalerweise würden diese Zeilen
        # in der create_regression_targets Funktion selbst gehandhabt werden.
        # Aber da wir hier nur die subset von dropna prüfen, sollte es passen.
        all_features_and_low_target = FEATURES_LIST + ['low_target']
        all_features_and_high_target = FEATURES_LIST + ['high_target']

        btc_data_final_low.dropna(subset=all_features_and_low_target, inplace=True)
        btc_data_final_high.dropna(subset=all_features_and_high_target, inplace=True)
        gold_data_final_low.dropna(subset=all_features_and_low_target, inplace=True)
        gold_data_final_high.dropna(subset=all_features_and_high_target, inplace=True)


        if btc_data_final_low.empty or btc_data_final_high.empty or gold_data_final_low.empty or gold_data_final_high.empty:
            print("FEHLER: Nicht genügend Daten nach Feature Engineering und Target-Erstellung.")
            return

        print("Phase 3: Modelltraining und Speicherung...")

        # --- KORREKTUR: ÜBERGIB DAS GESAMTE DATAFRAME UND DEN STRING-NAMEN DES ZIELS ---
        # train_regression_model wird intern die Features und das Target auswählen
        btc_model_low = train_regression_model(btc_data_final_low, 'low_target') # Ganzer DF
        settings.update_model('btc', 'low', btc_scaler_low, btc_model_low)
        print("BTC Low-Modell trainiert und gespeichert.")

        btc_model_high = train_regression_model(btc_data_final_high, 'high_target') # Ganzer DF
        settings.update_model('btc', 'high', btc_scaler_high, btc_model_high)
        print("BTC High-Modell trainiert und gespeichert.")

        gold_model_low = train_regression_model(gold_data_final_low, 'low_target') # Ganzer DF
        settings.update_model('gold', 'low', gold_scaler_low, gold_model_low)
        print("GOLD Low-Modell trainiert und gespeichert.")

        gold_model_high = train_regression_model(gold_data_final_high, 'high_target') # Ganzer DF
        settings.update_model('gold', 'high', gold_scaler_high, gold_model_high)
        print("GOLD High-Modell trainiert und gespeichert.")
        # --------------------------------------------------------------------------------------

        settings.model_update_timestamp = func.now()
        db.session.commit()
        print("Modelle erfolgreich aktualisiert und in Datenbank gespeichert.")

        print("Phase 4: Signale generieren und Benachrichtigen...")

        device_tokens = [d.fcm_token for d in Device.query.all()]
        if not device_tokens:
            print("Keine registrierten Geräte-Tokens gefunden. Keine Benachrichtigungen möglich.")

        # --- KORREKTUR: Trennung von Asset-Key (DB) und Asset-Display-Name (Benutzer) ---
        btc_details = {
            "asset_key": "btc", # Für Datenbank-Spalten (scaler_btc_low)
            "asset_display_name": "Bitcoin", # Für Benachrichtigungen
            "current_price": btc_df['Close'].iloc[-1],
        }

        gold_details = {
            "asset_key": "gold", # Für Datenbank-Spalten (scaler_gold_low)
            "asset_display_name": "Gold", # Für Benachrichtigungen
            "current_price": gold_df['Close'].iloc[-1],
        }

        for details in [btc_details, gold_details]:
            asset_key = details["asset_key"]
            asset_display_name = details["asset_display_name"]
            current_price = details["current_price"]
            
            # Lade die aktuellsten Modelle und Scaler aus der Datenbank
            scaler_low, low_model = settings.get_model(asset_key, 'low') # Verwendet asset_key
            scaler_high, high_model = settings.get_model(asset_key, 'high') # Verwendet asset_key

            if not all([scaler_low, low_model, scaler_high, high_model]):
                print(f"FEHLER: Modelle oder Scaler für {asset_display_name} konnten nicht geladen werden. Überspringe Signalgenerierung.")
                continue

            dummy_current_df = pd.DataFrame({
                'Date': [pd.to_datetime('today')],
                'Close': [current_price],
                # Füge hier Dummy 'High' und 'Low' hinzu, damit add_features_to_data sie findet
                'High': [current_price * (1 + np.random.uniform(0.001, 0.005))],
                'Low': [current_price * (1 - np.random.uniform(0.001, 0.005))]
            })
            
            # Füge Features hinzu (ohne Skalierung, da der Scaler separat angewendet wird)
            # asset_name für add_features_to_data ist immer noch "bitcoin" oder "gold"
            latest_features_df, _, _ = add_features_to_data(dummy_current_df, asset_display_name.lower(), skip_scaling=True)
            
            latest_features_df = latest_features_df[FEATURES_LIST]

            # Vorhersage
            predicted_low = low_model.predict(scaler_low.transform(latest_features_df))[0]
            predicted_high = high_model.predict(scaler_high.transform(latest_features_df))[0]
            
            new_signal_text = f"Einstieg: {predicted_low:.2f}, TP: {predicted_high:.2f}"
            
            last_signal = getattr(settings, f'last_{asset_key}_signal') # Verwendet asset_key
            print(f"Analyse für {asset_display_name}: Letztes Signal='{last_signal}', Neues Signal='{new_signal_text}'")

            if new_signal_text != last_signal or last_signal == 'N/A':
                print(f"-> Signal für {asset_display_name} hat sich geändert! Sende Benachrichtigung...")
                title = f"Neues Preis-Ziel: {asset_display_name}"
                body = f"Neues Ziel: Einstieg ca. {predicted_low:.2f}, Take Profit ca. {predicted_high:.2f}"
                send_notification(title, body, device_tokens)
                
                setattr(settings, f'last_{asset_key}_signal', new_signal_text) # Verwendet asset_key
                db.session.commit()
            else:
                print(f"-> Signal für {asset_display_name} unverändert. Keine Aktion nötig.")
    
    print("Trainings-Pipeline abgeschlossen.")

if __name__ == '__main__':
    run_training_pipeline()
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
    last_btc_signal = db.Column(db.String(100), default='N/A') # Erhöht auf 100
    last_gold_signal = db.Column(db.String(100), default='N/A') # Erhöht auf 100
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

# --- Firebase-Initialisierung (für dieses Skript) ---
if not firebase_admin._apps: # Prüfen, ob Firebase bereits initialisiert wurde (sehr unwahrscheinlich in diesem Skript, aber gute Praxis)
    cred = None # Initialisiere cred hier als None, um NameError zu vermeiden
    try:
        # 1. Versuch: Lade aus lokaler Datei (für deinen PC)
        cred = credentials.Certificate("serviceAccountKey.json")
        print("Firebase-Credentials aus lokaler Datei geladen.")
    except FileNotFoundError:
        # 2. Versuch: Lade aus Umgebungsvariable (für Render)
        print("Lokale Schlüsseldatei nicht gefunden. Versuche Umgebungsvariable...")
        try:
            # Hier den korrekten Namen der Umgebungsvariable verwenden
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
            print(f"Firebase bereits initialisiert oder Fehler: {e}") # Sollte hier nicht passieren
        except Exception as e:
            print(f"FEHLER bei der Firebase-Initialisierung: {e}")
    else:
        print("FEHLER: Firebase-Credentials konnten NICHT geladen werden. Firebase wird NICHT initialisiert.")


# --- Hilfsfunktion für Benachrichtigungen ---
def send_notification(title, body, tokens):
    if not tokens:
        print("Keine Tokens für den Versand von Benachrichtigungen vorhanden.")
        return

    # Firebase-Initialisierung prüfen
    # Die Initialisierung sollte bereits oben im Skript passiert sein.
    # Hier wird nur geprüft, ob die Firebase-App tatsächlich verfügbar ist.
    if not firebase_admin._apps:
        print("Firebase ist nicht initialisiert. Nachricht kann nicht gesendet werden.")
        return

    # Sende Nachrichten in Batches, um API-Limits zu beachten
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
    with app.app_context(): # Flask-App-Kontext ist für SQLAlchemy und DB-Zugriff erforderlich
        db.create_all() # Stellt sicher, dass die Tabellen existieren

        # Hole den Einstellungs-Eintrag oder erstelle ihn
        settings = Settings.query.first()
        if not settings:
            settings = Settings()
            db.session.add(settings)
            db.session.commit()
            print("Initialer Settings-Eintrag erstellt.")

        # Phase 1: Daten herunterladen (fiktiv, da keine echten APIs)
        print("Phase 1: Daten herunterladen...")
        # In einem echten Szenario würden wir hier echte Daten von APIs holen
        btc_data_path = 'data/btc_historical_data.csv'
        gold_data_path = 'data/gold_historical_data.csv'

        # Erstelle Dummy-Daten, wenn sie nicht existieren
        if not os.path.exists(btc_data_path) or not os.path.exists(gold_data_path):
            print("Erstelle Dummy-Historische Daten für BTC und GOLD...")
            # Erstelle ein Verzeichnis, falls es nicht existiert
            os.makedirs(os.path.dirname(btc_data_path), exist_ok=True)

            # Dummy BTC Daten
            dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
            btc_prices = np.random.rand(100) * 10000 + 30000 # Preise um 30k-40k
            btc_data = pd.DataFrame({'Date': dates, 'Close': btc_prices})
            btc_data.to_csv(btc_data_path, index=False)

            # Dummy Gold Daten
            gold_prices = np.random.rand(100) * 100 + 1800 # Preise um 1800-1900
            gold_data = pd.DataFrame({'Date': dates, 'Close': gold_prices})
            gold_data.to_csv(gold_data_path, index=False)
            print("Dummy-Daten erstellt.")
        else:
            print("Historische Daten vorhanden. Verwende bestehende Daten.")

        # Beispielhaftes Laden und Verarbeiten
        btc_df = pd.read_csv(btc_data_path)
        gold_df = pd.read_csv(gold_data_path)

        # Konvertiere 'Date' Spalte zu datetime
        btc_df['Date'] = pd.to_datetime(btc_df['Date'])
        gold_df['Date'] = pd.to_datetime(gold_df['Date'])


        # Phase 2: Feature Engineering und Target-Erstellung
        print("Phase 2: Feature Engineering und Target-Erstellung...")
        btc_data_engineered, btc_scaler_low, btc_scaler_high = add_features_to_data(btc_df.copy(), asset_name="bitcoin")
        gold_data_engineered, gold_scaler_low, gold_scaler_high = add_features_to_data(gold_df.copy(), asset_name="gold")

        # Targets erstellen
        btc_data_final_low = create_regression_targets(btc_data_engineered.copy(), 'low_target')
        btc_data_final_high = create_regression_targets(btc_data_engineered.copy(), 'high_target')
        gold_data_final_low = create_regression_targets(gold_data_engineered.copy(), 'low_target')
        gold_data_final_high = create_regression_targets(gold_data_engineered.copy(), 'high_target')

        # Stelle sicher, dass keine NaN-Werte in Features oder Targets sind
        btc_data_final_low.dropna(subset=FEATURES_LIST + ['low_target'], inplace=True)
        btc_data_final_high.dropna(subset=FEATURES_LIST + ['high_target'], inplace=True)
        gold_data_final_low.dropna(subset=FEATURES_LIST + ['low_target'], inplace=True)
        gold_data_final_high.dropna(subset=FEATURES_LIST + ['high_target'], inplace=True)

        if btc_data_final_low.empty or btc_data_final_high.empty or gold_data_final_low.empty or gold_data_final_high.empty:
            print("FEHLER: Nicht genügend Daten nach Feature Engineering und Target-Erstellung.")
            return

        # Phase 3: Modelltraining und Speicherung
        print("Phase 3: Modelltraining und Speicherung...")

        # BTC Low-Modell
        btc_model_low = train_regression_model(btc_data_final_low[FEATURES_LIST], btc_data_final_low['low_target'])
        settings.update_model('btc', 'low', btc_scaler_low, btc_model_low)
        print("BTC Low-Modell trainiert und gespeichert.")

        # BTC High-Modell
        btc_model_high = train_regression_model(btc_data_final_high[FEATURES_LIST], btc_data_final_high['high_target'])
        settings.update_model('btc', 'high', btc_scaler_high, btc_model_high)
        print("BTC High-Modell trainiert und gespeichert.")

        # GOLD Low-Modell
        gold_model_low = train_regression_model(gold_data_final_low[FEATURES_LIST], gold_data_final_low['low_target'])
        settings.update_model('gold', 'low', gold_scaler_low, gold_model_low)
        print("GOLD Low-Modell trainiert und gespeichert.")

        # GOLD High-Modell
        gold_model_high = train_regression_model(gold_data_final_high[FEATURES_LIST], gold_data_final_high['high_target'])
        settings.update_model('gold', 'high', gold_scaler_high, gold_model_high)
        print("GOLD High-Modell trainiert und gespeichert.")

        # Zeitstempel aktualisieren
        settings.model_update_timestamp = func.now()
        db.session.commit()
        print("Modelle erfolgreich aktualisiert und in Datenbank gespeichert.")

        # Phase 4: Signale generieren und Benachrichtigen
        print("Phase 4: Signale generieren und Benachrichtigen...")

        # Holen der aktuellsten Tokens
        device_tokens = [d.fcm_token for d in Device.query.all()]
        if not device_tokens:
            print("Keine registrierten Geräte-Tokens gefunden. Keine Benachrichtigungen möglich.")

        # Für BTC
        btc_details = {
            "asset_name": "Bitcoin",
            "current_price": btc_df['Close'].iloc[-1], # Letzter bekannter Preis
            "scaler_low_key": 'scaler_btc_low',
            "model_low_key": 'model_btc_low',
            "scaler_high_key": 'scaler_btc_high',
            "model_high_key": 'model_btc_high',
            "last_signal_key": 'last_btc_signal'
        }

        # Für Gold
        gold_details = {
            "asset_name": "Gold",
            "current_price": gold_df['Close'].iloc[-1], # Letzter bekannter Preis
            "scaler_low_key": 'scaler_gold_low',
            "model_low_key": 'model_gold_low',
            "scaler_high_key": 'scaler_gold_high',
            "model_high_key": 'model_gold_high',
            "last_signal_key": 'last_gold_signal'
        }

        for details in [btc_details, gold_details]:
            asset_name = details["asset_name"]
            current_price = details["current_price"]
            
            # Lade die aktuellsten Modelle und Scaler aus der Datenbank
            scaler_low, low_model = settings.get_model(asset_name.lower(), 'low')
            scaler_high, high_model = settings.get_model(asset_name.lower(), 'high')

            if not all([scaler_low, low_model, scaler_high, high_model]):
                print(f"FEHLER: Modelle oder Scaler für {asset_name} konnten nicht geladen werden. Überspringe Signalgenerierung.")
                continue

            # Hier würden in einem echten Szenario aktuelle Daten für die Vorhersage vorbereitet
            # Für die Demo: Wir verwenden die Feature Engineering Logik auf dem aktuellen Preis
            # Erstelle ein Dummy-DataFrame für den aktuellen Preis, um es an add_features_to_data anzupassen
            dummy_current_df = pd.DataFrame({'Date': [pd.to_datetime('today')], 'Close': [current_price]})
            
            # Füge Features hinzu (ohne Skalierung, da der Scaler separat angewendet wird)
            latest_features_df, _, _ = add_features_to_data(dummy_current_df, asset_name.lower(), skip_scaling=True)
            
            # Entferne die 'Date' und 'Close' Spalten und stelle sicher, dass die Features in der richtigen Reihenfolge sind
            latest_features_df = latest_features_df[FEATURES_LIST]

            # Vorhersage
            predicted_low = low_model.predict(scaler_low.transform(latest_features_df))[0]
            predicted_high = high_model.predict(scaler_high.transform(latest_features_df))[0]
            
            # Erstelle den Signal-Text (der jetzt länger ist)
            new_signal_text = f"Einstieg: {predicted_low:.2f}, TP: {predicted_high:.2f}"
            
            last_signal = getattr(settings, details["last_signal_key"])
            print(f"Analyse für {asset_name}: Letztes Signal='{last_signal}', Neues Signal='{new_signal_text}'")

            # Benachrichtigungslogik: Sende, wenn sich das Signal ändert ODER wenn es \"N/A\" ist (erster Lauf)
            if new_signal_text != last_signal or last_signal == 'N/A': # Füge 'N/A' Bedingung hinzu
                print(f"-> Signal für {asset_name} hat sich geändert! Sende Benachrichtigung...")
                title = f"Neues Preis-Ziel: {asset_name}"
                body = f"Neues Ziel: Einstieg ca. {predicted_low:.2f}, Take Profit ca. {predicted_high:.2f}"
                send_notification(title, body, device_tokens)
                
                # Speichere das neue Signal in der Datenbank
                setattr(settings, details["last_signal_key"], new_signal_text)
                db.session.commit()
            else:
                print(f"-> Signal für {asset_name} unverändert. Keine Aktion nötig.")
    
    print("Trainings-Pipeline abgeschlossen.")

if __name__ == '__main__':
    run_training_pipeline()
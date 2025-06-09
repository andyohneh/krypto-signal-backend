# run_training_pipeline.py (Temporärer DEBUG-CODE V2 - Initialisierungstest)
import sys
import os
from dotenv import load_dotenv
load_dotenv()
print("--- Test 1: Skript-Start & .env geladen ---")

try:
    print("\n--- Test 2: Initialisiere Flask App ---")
    from flask import Flask
    app = Flask(__name__)
    print("    [ERFOLG] Flask App initialisiert.")
except Exception as e:
    print(f"    [FEHLER] bei Flask-Initialisierung: {e}")
    sys.exit(1) # Beenden, wenn schon das fehlschlägt

try:
    print("\n--- Test 3: Initialisiere Firebase Admin ---")
    import firebase_admin
    from firebase_admin import credentials
    # Prüfen, ob die App schon initialisiert ist, um Fehler zu vermeiden
    if not firebase_admin._apps:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
    print("    [ERFOLG] Firebase Admin initialisiert.")
except Exception as e:
    print(f"    [FEHLER] bei Firebase-Initialisierung: {e}")
    sys.exit(1)

try:
    print("\n--- Test 4: Initialisiere SQLAlchemy DB ---")
    from flask_sqlalchemy import SQLAlchemy
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL nicht in der Umgebung gefunden!")
    app.config['SQLALCHEMY_DATABASE_URI'] = db_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db = SQLAlchemy(app)
    print("    [ERFOLG] SQLAlchemy DB initialisiert.")
except Exception as e:
    print(f"    [FEHLER] bei SQLAlchemy-Initialisierung: {e}")
    sys.exit(1)

print("\n\nDebug-Skript (Initialisierung) erfolgreich bis zum Ende durchgelaufen!")
# run_training_pipeline.py (Finale korrigierte Version)

from feature_engineer import add_features_to_data
from train_model import train_model_for_asset
from data_manager import download_historical_data

def run_full_pipeline():
    """
    Führt die gesamte Daten- und Trainings-Pipeline aus.
    """
    print("Starte die vollständige Trainings-Pipeline...")

    # Schritt 1 & 2: Daten laden und Features erstellen
    print("\nVerarbeite Bitcoin-Daten...")
    btc_raw_data = download_historical_data("BTC-USD")
    btc_featured_data = add_features_to_data(btc_raw_data)
    if btc_featured_data is not None:
        btc_featured_data.to_csv("btc_data_with_features.csv")
        print("Bitcoin-Daten mit Features erfolgreich gespeichert.")

    print("\nVerarbeite Gold-Daten...")
    gold_raw_data = download_historical_data("GC=F")
    gold_featured_data = add_features_to_data(gold_raw_data)
    if gold_featured_data is not None:
        gold_featured_data.to_csv("gold_data_with_features.csv")
        print("Gold-Daten mit Features erfolgreich gespeichert.")

    # Schritt 3: Modelle mit den neuen Daten trainieren
    print("\nTrainiere Modelle mit den neuen Daten...")

    # KORREKTUR: Hier die kurzen Namen verwenden, nach denen die App sucht
    train_model_for_asset(
        "btc_data_with_features.csv",
        "btc_model",      # Korrekter Name
        "btc_scaler"      # Korrekter Name
    )
    train_model_for_asset(
        "gold_data_with_features.csv",
        "gold_model",     # Korrekter Name
        "gold_scaler"     # Korrekter Name
    )

    print("\n\nPipeline erfolgreich durchgelaufen! Neue Modelle und Scaler wurden erstellt.")

if __name__ == '__main__':
    run_full_pipeline()
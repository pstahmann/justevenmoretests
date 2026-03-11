import os
import torch

# =============================================================================
# 1. UMGEBUNGSERKENNUNG (Kaggle vs. Lokal)
# =============================================================================
IS_KAGGLE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') != ''

if IS_KAGGLE:
    # Kaggle mountet unter /kaggle/input/<dataset-slug>/
    KAGGLE_DATA_DIR = "/kaggle/input/imbalanced-data"
    BASE_OUTPUT_DIR = "/kaggle/working/outputs"
    DATASET_PATHS = {
        "mlg_ulb": os.path.join(KAGGLE_DATA_DIR, "mlg-ulbcreditcardfraud.csv"),
        "creditcard_2023": os.path.join(KAGGLE_DATA_DIR, "credit-card-fraud-detection-dataset-2023.csv"),
        "fraud_data": os.path.join(KAGGLE_DATA_DIR, "Fraud_Data.csv"),
        "train_transaction": os.path.join(KAGGLE_DATA_DIR, "train_transaction_identity.csv"),
    }
else:
    LOCAL_DATA_DIR = "./data"
    BASE_OUTPUT_DIR = "./outputs"
    DATASET_PATHS = {
        "mlg_ulb": os.path.join(LOCAL_DATA_DIR, "mlg-ulbcreditcardfraud.csv"),
        "creditcard_2023": os.path.join(LOCAL_DATA_DIR, "credit-card-fraud-detection-dataset-2023.csv"),
        "fraud_data": os.path.join(LOCAL_DATA_DIR, "Fraud_Data.csv"),
        "train_transaction": os.path.join(LOCAL_DATA_DIR, "train_transaction_identity.csv"),
    }

OUTPUT_DIR = BASE_OUTPUT_DIR

# =============================================================================
# 2. ORDNERSTRUKTUR
# =============================================================================
CHECKPOINT_DIR = os.path.join(BASE_OUTPUT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_OUTPUT_DIR, "results")

# =============================================================================
# 3. HARDWARE CONFIG
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Kein Side-Effect beim Import mehr — explizit aufrufen ---
_initialized = False

def init():
    """Einmalig aufrufen, um Ordner zu erstellen und Umgebung zu loggen."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    if IS_KAGGLE:
        print("Kaggle-Umgebung erkannt. Cloud-Pfade aktiv.")
    else:
        print("Lokale Umgebung erkannt. PC-Pfade aktiv.")

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if torch.cuda.is_available():
        print(f"GPU erkannt: {torch.cuda.get_device_name(0)}")
    else:
        print("Kein CUDA-Gerät gefunden — CPU-Modus.")


# Beim ersten Import initialisieren (Rückwärtskompatibilität)
init()

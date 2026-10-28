"""Configuration de l'application (modifiable par variables d'environnement)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- Modèle -----------------------------------------------------------------
# Dépôt Hugging Face Hub qui contient le fichier du modèle (ex. "MDMAK04/ham500-resnet50")
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "simo21Ms/ham500-resnet50-finetune")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "resnet50_finetune.onnx")
# Chemin local optionnel : s'il existe, on l'utilise au lieu de télécharger (développement local)
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "")
# Token si le dépôt du modèle est privé
HF_TOKEN = os.getenv("HF_TOKEN") or None

# --- Prétraitement ------------------------------------------------------------
# "raw"    : valeurs brutes 0-255 (= ce qui a été utilisé à l'entraînement dans le notebook)
# "resnet" : keras.applications.resnet50.preprocess_input (à utiliser si tu ré-entraînes avec)
PREPROCESS_MODE = os.getenv("PREPROCESS_MODE", "raw")
IMG_SIZE = (224, 224)

# --- API ------------------------------------------------------------------------
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

# Ordre alphabétique = ordre du LabelEncoder du notebook (0..6)
CLASS_CODES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = {
    "akiec": "Kératose actinique / maladie de Bowen",
    "bcc": "Carcinome basocellulaire",
    "bkl": "Lésion de type kératose bénigne",
    "df": "Dermatofibrome",
    "mel": "Mélanome",
    "nv": "Naevus mélanocytaire",
    "vasc": "Lésion vasculaire",
}

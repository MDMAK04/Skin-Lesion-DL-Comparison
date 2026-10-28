"""Prétraitement d'une image et prédiction.

IMPORTANT : le prétraitement doit être IDENTIQUE à celui de l'entraînement.
Dans le notebook : cv2.imread -> BGR2RGB -> cv2.resize(224, 224) -> float32 (0-255), sans normalisation.
"""
import time
from typing import List, Tuple

import cv2
import numpy as np

from app import config
from app.schemas import PredictionItem


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Octets d'une image -> tenseur (1, 224, 224, 3) float32."""
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError("Fichier image illisible ou corrompu.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, config.IMG_SIZE)
    x = img.astype("float32")

    if config.PREPROCESS_MODE == "resnet":
        # équivalent de keras.applications.resnet50.preprocess_input (RGB -> BGR, moyenne ImageNet)
        x = x[..., ::-1] - np.array([103.939, 116.779, 123.68], dtype="float32")
    return np.expand_dims(x, axis=0)


def predict(model, image_bytes: bytes) -> Tuple[List[PredictionItem], float]:
    """Retourne les 7 classes triées par probabilité et le temps d'inférence (ms)."""
    x = preprocess(image_bytes)
    start = time.perf_counter()
    probs = np.asarray(model.predict(x))[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    items = [
        PredictionItem(
            label=code,
            name=config.CLASS_NAMES[code],
            probability=float(probs[i]),
        )
        for i, code in enumerate(config.CLASS_CODES)
    ]
    items.sort(key=lambda it: it.probability, reverse=True)
    return items, elapsed_ms

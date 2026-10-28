"""Chargement du modèle ONNX (téléchargé depuis le Hugging Face Hub au premier démarrage).

On utilise ONNX Runtime plutôt que TensorFlow : ~290 Mo de RAM au lieu de ~600 Mo,
ce qui permet un hébergement gratuit à 512 Mo.
"""
import logging
import os

import numpy as np

from app import config

logger = logging.getLogger("uvicorn.error")
_model = None


class OnnxModel:
    """Petit wrapper autour d'une session ONNX Runtime."""

    def __init__(self, path: str):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # hébergeurs gratuits : 1 CPU partagé
        self.session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, x: np.ndarray) -> np.ndarray:
        """x : (N, 224, 224, 3) float32 -> probabilités (N, 7)."""
        return self.session.run(None, {self.input_name: x.astype("float32")})[0]


def _resolve_model_path() -> str:
    if config.LOCAL_MODEL_PATH and os.path.isfile(config.LOCAL_MODEL_PATH):
        logger.info("Modèle local : %s", config.LOCAL_MODEL_PATH)
        return config.LOCAL_MODEL_PATH

    from huggingface_hub import hf_hub_download

    logger.info("Téléchargement du modèle %s/%s ...", config.MODEL_REPO_ID, config.MODEL_FILENAME)
    return hf_hub_download(
        repo_id=config.MODEL_REPO_ID,
        filename=config.MODEL_FILENAME,
        token=config.HF_TOKEN,
    )


def load_model() -> OnnxModel:
    """Charge le modèle une seule fois."""
    global _model
    if _model is None:
        _model = OnnxModel(_resolve_model_path())
        logger.info("Modèle ONNX chargé.")
    return _model


def get_model():
    return _model

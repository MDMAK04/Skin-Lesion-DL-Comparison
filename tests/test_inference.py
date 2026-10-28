"""Tests du prétraitement et de l'API (sans charger le vrai modèle)."""
import cv2
import numpy as np
import pytest

from app import config, inference


def _png_bytes(h=300, w=400):
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_preprocess_shape_and_range():
    x = inference.preprocess(_png_bytes())
    assert x.shape == (1, 224, 224, 3)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 255.0  # mode "raw" : pas de normalisation


def test_preprocess_resnet_mode(monkeypatch):
    monkeypatch.setattr(config, "PREPROCESS_MODE", "resnet")
    x = inference.preprocess(_png_bytes())
    assert x.shape == (1, 224, 224, 3)
    assert x.min() < 0  # moyenne ImageNet soustraite


def test_preprocess_rejects_garbage():
    with pytest.raises(ValueError):
        inference.preprocess(b"ceci n'est pas une image")


class _FakeModel:
    def predict(self, x):
        p = np.zeros((1, 7), dtype="float32")
        p[0, 4] = 0.7   # mel
        p[0, 5] = 0.3   # nv
        return p


def test_predict_sorted_and_complete():
    items, ms = inference.predict(_FakeModel(), _png_bytes())
    assert len(items) == 7
    assert items[0].label == "mel"
    assert abs(sum(i.probability for i in items) - 1.0) < 1e-5
    assert ms >= 0


def test_api(monkeypatch):
    from fastapi.testclient import TestClient
    from app import main, model_loader

    monkeypatch.setattr(model_loader, "load_model", lambda: _FakeModel())
    monkeypatch.setattr(model_loader, "get_model", lambda: _FakeModel())

    with TestClient(main.app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        assert len(client.get("/classes").json()) == len(config.CLASS_CODES)

        r = client.post("/predict", files={"file": ("a.png", _png_bytes(), "image/png")})
        assert r.status_code == 200
        assert r.json()["top"]["label"] == "mel"

        bad = client.post("/predict", files={"file": ("a.txt", b"x", "text/plain")})
        assert bad.status_code == 415

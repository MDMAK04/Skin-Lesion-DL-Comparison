"""Schémas Pydantic des réponses de l'API."""
from typing import List
from pydantic import BaseModel


class PredictionItem(BaseModel):
    label: str          # code de la classe (ex. "mel")
    name: str           # nom complet
    probability: float  # entre 0 et 1


class PredictionResponse(BaseModel):
    top: PredictionItem
    predictions: List[PredictionItem]  # les 7 classes, triées par probabilité décroissante
    model: str
    inference_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model: str

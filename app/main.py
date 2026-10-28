"""API FastAPI : classification de lésions cutanées (ResNet50 fine-tuné sur HAM500)."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import config, inference, model_loader
from app.schemas import HealthResponse, PredictionResponse

MODEL_LABEL = "ResNet50 Fine-Tuning (HAM500)"


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_loader.load_model()  # chargé une fois au démarrage
    yield


app = FastAPI(
    title="HAM500 - Classification de lésions cutanées",
    description="Atelier Deep Learning (MLAIM) : ResNet50 Fine-Tuning, 7 classes. "
    "Projet pédagogique : ne constitue PAS un outil de diagnostic médical.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model_loader.get_model() is not None,
        model=MODEL_LABEL,
    )


@app.get("/classes", tags=["info"])
def classes():
    return [{"label": c, "name": config.CLASS_NAMES[c]} for c in config.CLASS_CODES]


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(file: UploadFile = File(...)):
    if file.content_type not in config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, "Format non supporté. Utilise JPEG, PNG, WEBP ou BMP.")

    data = await file.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Image trop lourde (max {config.MAX_UPLOAD_MB:g} Mo).")

    model = model_loader.get_model()
    if model is None:
        raise HTTPException(503, "Modèle non chargé.")

    try:
        items, elapsed_ms = inference.predict(model, data)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    return PredictionResponse(
        top=items[0],
        predictions=items,
        model=MODEL_LABEL,
        inference_ms=round(elapsed_ms, 1),
    )


# L'interface web est servie par FastAPI lui-même (à monter EN DERNIER).
app.mount("/", StaticFiles(directory=config.BASE_DIR / "static", html=True), name="static")

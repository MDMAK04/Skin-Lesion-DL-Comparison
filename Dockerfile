FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf \
    MODEL_REPO_ID=simo21Ms/ham500-resnet50-finetune \
    MODEL_FILENAME=resnet50_finetune.onnx

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
RUN chown -R user:user /app
USER user

# Les hébergeurs (Render, Koyeb...) fournissent le port dans $PORT
EXPOSE 10000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]

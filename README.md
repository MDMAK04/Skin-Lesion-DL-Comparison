# 🔬 Classification de lésions cutanées : CNN, VGG16, ResNet50 et ViT (HAM500)

Étude comparative de quatre architectures de Deep Learning (CNN personnalisé, VGG16, ResNet50, Vision Transformer) en **From Scratch** et en **Fine-Tuning**, sur le dataset **HAM500** (500 images, 7 classes de lésions cutanées). Le meilleur modèle, **ResNet50 Fine-Tuning**, est déployé derrière une API **FastAPI** avec une interface web.

> **Atelier Deep Learning** · Master Machine Learning et Intelligence Multimédia (MLAIM) · USMBA, Fès
> Encadrant : Pr. Abdelouahed Sabri · Auteur : Mohammed EL MAKHLOUFI

> ⚠️ **Projet pédagogique. Ce n'est pas un outil de diagnostic médical.** Le modèle est entraîné sur 320 images et reconnaît très mal les classes rares (voir *Limites*).

**Démo en ligne :** `https://ham500-classifier.onrender.com` *(à adapter après le déploiement)*
**Modèle :** <https://huggingface.co/simo21Ms/ham500-resnet50-finetune>

---

## Sommaire
1. [Contexte et objectifs](#contexte-et-objectifs)
2. [Dataset](#dataset)
3. [Protocole expérimental](#protocole-expérimental)
4. [Résultats](#résultats)
5. [Analyse](#analyse)
6. [Limites](#limites)
7. [Structure du dépôt](#structure-du-dépôt)
8. [Lancer l'application en local](#lancer-lapplication-en-local)
9. [API](#api)
10. [Hébergement du modèle et déploiement](#hébergement-du-modèle-et-déploiement)
11. [Reproduire les expériences](#reproduire-les-expériences)
12. [Tests](#tests)

---

## Contexte et objectifs
Comparer sept configurations expérimentales pour la classification multiclasse d'images de lésions cutanées, et analyser l'effet de l'**architecture** et de la **stratégie d'entraînement** (poids aléatoires ou préentraînés sur ImageNet). Il s'agit d'une étude expérimentale sur un dataset précis, **pas** d'une démonstration générale de la supériorité d'une architecture.

## Dataset
HAM500 est un sous-ensemble de HAM10000 : 500 images dermoscopiques, 7 classes, très déséquilibrées.

| Classe | Description | Images | Part |
|---|---|---:|---:|
| `nv` | Naevus mélanocytaire | 332 | 66,4 % |
| `bkl` | Kératose bénigne | 57 | 11,4 % |
| `mel` | Mélanome | 43 | 8,6 % |
| `bcc` | Carcinome basocellulaire | 34 | 6,8 % |
| `akiec` | Kératose actinique | 21 | 4,2 % |
| `vasc` | Lésion vasculaire | 8 | 1,6 % |
| `df` | Dermatofibrome | 5 | 1,0 % |

Division **stratifiée** (`random_state=42`) : **320** images en entraînement (64 %), **80** en validation (16 %), **100** en test (20 %). Les images sont redimensionnées en 224 × 224.

> Un classifieur qui prédit toujours `nv` obtient ~66 % d'accuracy : **l'accuracy seule est trompeuse**, d'où l'usage du F1 macro.

## Protocole expérimental
| Paramètre | Valeur |
|---|---|
| Optimiseur / perte | Adam / Categorical Crossentropy |
| Batch size / époques max | 16 / 30 |
| Early stopping | patience 5 sur `val_loss`, restauration des meilleurs poids |
| Learning rate | CNN : 1e-3 · autres From Scratch : 1e-4 · Fine-Tuning : 1e-5 |
| Tête de classification | GlobalAveragePooling → Dense 128 (ReLU) → Dropout 0,5 → Dense 7 (Softmax) |
| Fine-Tuning | poids ImageNet, **toutes** les couches entraînées |

## Résultats
Évaluation sur le **jeu de test** (100 images).

| Modèle | Accuracy | Precision macro | Recall macro | **F1 macro** |
|---|---:|---:|---:|---:|
| CNN personnalisé · Scratch | 0,66 | 0,09 | 0,14 | 0,11 |
| VGG16 · Scratch | 0,66 | 0,09 | 0,14 | 0,11 |
| VGG16 · Fine-Tuning | 0,68 | 0,22 | 0,21 | 0,21 |
| ResNet50 · Scratch | 0,66 | 0,09 | 0,14 | 0,11 |
| **ResNet50 · Fine-Tuning** | **0,73** | **0,40** | **0,30** | **0,32** |
| ViT-B/16 · Scratch | 0,66 | 0,09 | 0,14 | 0,11 |
| ViT-B/16 · Fine-Tuning | 0,65 | 0,35 | 0,21 | 0,21 |

Détail dans [`results/metrics.csv`](results/metrics.csv). Les courbes et matrices de confusion sont dans le notebook et dans le [rapport](report/rapport_TP_classification_DL.tex).

## Analyse
- **Les 4 modèles From Scratch s'effondrent** sur la classe majoritaire (accuracy 0,66, F1 macro 0,11) : 320 images ne suffisent pas à entraîner des millions de paramètres depuis zéro.
- **Le Fine-Tuning améliore le F1 macro** pour les trois architectures préentraînées ; **ResNet50** est le meilleur modèle.
- **Surapprentissage** net sur ResNet50 Fine-Tuning (entraînement ≈ 98 % vs validation ≈ 74 %), limité par l'early stopping.
- Le **ViT** est le plus coûteux en temps et n'apporte pas d'avantage sur ce petit dataset.
- Le nombre de paramètres n'explique pas les performances.

## Limites
- 100 images de test, **1 à 2 images** pour `df` et `vasc` : métriques par classe très instables.
- **Un seul entraînement par configuration** (pas de variance mesurée) : les écarts de quelques points ne sont pas statistiquement établis.
- Le modèle déployé **prédit surtout `nv`** : rappel de ~0,11 seulement sur le mélanome. **Ne pas utiliser pour un usage clinique.**
- Dans le notebook, les images ont été fournies aux réseaux en valeurs brutes 0-255, **sans `preprocess_input`** propre à chaque architecture. L'application reproduit exactement ce prétraitement (`PREPROCESS_MODE=raw`).
- Pas de séparation par `lesion_id` : une même lésion peut apparaître dans plusieurs ensembles.

## Structure du dépôt
```
skin-lesion-dl-comparison/
├── README.md
├── Dockerfile                  # image de l'application (lit $PORT)
├── render.yaml                 # déploiement gratuit sur Render
├── requirements.txt            # dépendances de l'application (ONNX Runtime, sans TensorFlow)
├── requirements-convert.txt    # TensorFlow + tf2onnx (uniquement pour convertir le modèle)
├── requirements-dev.txt        # + pytest, httpx
├── pytest.ini
├── .gitignore                  # exclut *.keras (modèle trop lourd pour GitHub)
├── .dockerignore
├── app/                        # application FastAPI
│   ├── main.py                 # routes : /health, /classes, /predict + interface web
│   ├── config.py               # paramètres (modèle, prétraitement, classes)
│   ├── model_loader.py         # télécharge et charge le modèle ONNX depuis Hugging Face
│   ├── inference.py            # prétraitement + prédiction
│   ├── schemas.py              # modèles Pydantic des réponses
│   └── static/                 # interface web (HTML / CSS / JS)
├── notebooks/
│   └── classification_supervisee_DL.ipynb   # expériences (Parties 1 à 7)
├── report/
│   ├── rapport_TP_classification_DL.tex     # rapport LaTeX
│   └── figures/                # figures exportées du notebook
├── results/
│   └── metrics.csv             # tableau comparatif
├── scripts/
│   ├── export_model.py         # allège le modèle (273 Mo -> ~92 Mo)
│   ├── convert_to_onnx.py      # Keras -> ONNX (vérifie l'équivalence des sorties)
│   └── upload_to_hf.py         # envoie le modèle sur Hugging Face Hub
└── tests/
    └── test_inference.py
```

## Lancer l'application en local
```bash
git clone https://github.com/MDMAK04/skin-lesion-dl-comparison.git
cd skin-lesion-dl-comparison

python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt

# Option A : le modèle ONNX est téléchargé automatiquement depuis Hugging Face
uvicorn app.main:app --reload --port 8000

# Option B : utiliser un fichier local
LOCAL_MODEL_PATH=./resnet50_finetune.onnx uvicorn app.main:app --reload --port 8000
# Windows PowerShell : $env:LOCAL_MODEL_PATH=".\resnet50_finetune.onnx"; uvicorn app.main:app --reload
```
Interface : <http://localhost:8000> · Documentation interactive de l'API : <http://localhost:8000/docs>

Avec Docker :
```bash
docker build -t ham500 .
docker run -p 10000:10000 ham500      # http://localhost:10000
```

### Variables d'environnement
| Variable | Défaut | Rôle |
|---|---|---|
| `MODEL_REPO_ID` | `simo21Ms/ham500-resnet50-finetune` | dépôt Hugging Face du modèle |
| `MODEL_FILENAME` | `resnet50_finetune.onnx` | nom du fichier du modèle |
| `LOCAL_MODEL_PATH` | *(vide)* | chemin local (prioritaire sur le téléchargement) |
| `HF_TOKEN` | *(vide)* | token, si le dépôt du modèle est privé |
| `PREPROCESS_MODE` | `raw` | `raw` (comme le notebook) ou `resnet` (`preprocess_input`) |
| `MAX_UPLOAD_MB` | `10` | taille maximale d'une image |

> ⚠️ `PREPROCESS_MODE` doit correspondre à l'entraînement. Si tu ré-entraînes le modèle avec `preprocess_input`, passe-le à `resnet`.

## API
| Méthode | Route | Description |
|---|---|---|
| `GET` | `/health` | état du service et du modèle |
| `GET` | `/classes` | liste des 7 classes |
| `POST` | `/predict` | classification d'une image (`multipart/form-data`, champ `file`) |

```bash
curl -X POST http://localhost:8000/predict -F "file=@lesion.jpg"
```
```json
{
  "top": {"label": "nv", "name": "Naevus mélanocytaire", "probability": 0.81},
  "predictions": [
    {"label": "nv",  "name": "Naevus mélanocytaire", "probability": 0.81},
    {"label": "bkl", "name": "Lésion de type kératose bénigne", "probability": 0.09}
  ],
  "model": "ResNet50 Fine-Tuning (HAM500)",
  "inference_ms": 142.3
}
```
*(la vraie réponse contient les 7 classes)*

Codes d'erreur : `400` image illisible · `413` fichier trop lourd · `415` format non supporté · `503` modèle non chargé.

## Hébergement du modèle et déploiement (100 % gratuit)
**Problème :** le `.keras` sauvegardé après l'entraînement pèse ~273 Mo (il contient l'état de l'optimiseur Adam) et TensorFlow demande ~600 Mo de RAM, trop pour un hébergeur gratuit (~512 Mo). GitHub refuse aussi les fichiers de plus de 100 Mo.

**Solution :**
1. **Alléger** le modèle (supprimer l'optimiseur) : 273 Mo → ~92 Mo. → `scripts/export_model.py`
2. **Convertir en ONNX** : l'inférence se fait avec ONNX Runtime (~290 Mo de RAM, ~60 ms par image), sans TensorFlow. → `scripts/convert_to_onnx.py`
3. **Héberger le fichier sur Hugging Face Hub** (gratuit, pas dans Git). → `scripts/upload_to_hf.py`
4. **Déployer l'application sur Render** (plan gratuit, Docker) depuis GitHub : l'app télécharge le modèle au démarrage.

```bash
# 1. Alléger (dans Colab ou en local)
python scripts/export_model.py --src resnet50_finetune.keras --dst resnet50_finetune_light.keras

# 2. Convertir en ONNX (~3 Go de RAM : le plus simple est Google Colab)
pip install -r requirements-convert.txt
python scripts/convert_to_onnx.py --src resnet50_finetune_light.keras --dst resnet50_finetune.onnx

# 3. Envoyer sur le Hub
export HF_TOKEN=hf_xxx        # PowerShell : $env:HF_TOKEN="hf_xxx"
python scripts/upload_to_hf.py --file resnet50_finetune.onnx --repo simo21Ms/ham500-resnet50-finetune
```
4. Sur [render.com](https://render.com) : **New → Blueprint** (ou *Web Service*), connecter le dépôt GitHub, Render lit `render.yaml` et construit l'image `Dockerfile`. Aucune carte bancaire n'est nécessaire pour le plan gratuit.

**Remarques :**
- Plan gratuit : le service **s'endort après ~15 min d'inactivité**, le premier appel suivant prend 1 à 2 minutes (réveil + téléchargement du modèle). Ouvre l'URL quelques minutes avant une démonstration.
- Si le dépôt du modèle est privé, ajouter la variable secrète `HF_TOKEN` dans Render.
- Les mêmes fichiers marchent sur d'autres hébergeurs Docker gratuits (Koyeb, Fly.io…) : le port est lu dans `$PORT`.

## Reproduire les expériences
1. Ouvrir `notebooks/classification_supervisee_DL.ipynb` dans Google Colab (GPU recommandé).
2. Placer le dataset HAM500 sur Google Drive et adapter `DATASET_DIR`.
3. Exécuter les cellules dans l'ordre. Les modèles sont sauvegardés au format `.keras`.
4. Pour compiler le rapport : `report/rapport_TP_classification_DL.tex` (pdfLaTeX), avec les figures dans `report/figures/`.

## Tests
```bash
pip install -r requirements-dev.txt
pytest
```
Les tests couvrent le prétraitement, l'ordre des classes et l'API (avec un faux modèle : aucun fichier de modèle n'est nécessaire).

## Licence et avertissement
Code à usage pédagogique. Le dataset HAM10000 est soumis à sa propre licence (CC BY-NC 4.0). **Les prédictions ne constituent pas un avis médical.**

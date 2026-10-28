"""Convertit le modèle Keras (.keras) en ONNX pour une inférence légère (sans TensorFlow).

Pourquoi : TensorFlow consomme ~600 Mo de RAM, trop pour un hébergeur gratuit (512 Mo).
ONNX Runtime fait la même prédiction avec ~290 Mo (testé : écart de 1e-10 avec Keras).

La conversion elle-même demande ~3 Go de RAM : le plus simple est de la faire dans Google Colab.

Prérequis (uniquement pour convertir) :
    pip install -r requirements-convert.txt

Usage :
    python scripts/convert_to_onnx.py --src resnet50_finetune_light.keras --dst resnet50_finetune.onnx
"""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="modèle .keras (allégé ou complet)")
    parser.add_argument("--dst", default="resnet50_finetune.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    import keras

    model = keras.models.load_model(args.src, compile=False)
    tmp = tempfile.mkdtemp()
    saved = os.path.join(tmp, "saved_model")

    print("1/3  Export SavedModel ...")
    model.export(saved)

    print("2/3  Conversion SavedModel -> ONNX ...")
    import tf2onnx.convert as tf2onnx_convert
    from tf2onnx import optimizer

    # L'étape d'optimisation de tf2onnx dépasse 4 Go de RAM sur ResNet50 : on la saute.
    # Le modèle produit est identique numériquement (vérifié à l'étape 3).
    optimizer.optimize_graph = lambda g, catch_errors=True, **kw: g
    sys.argv = ["tf2onnx.convert", "--saved-model", saved, "--output", args.dst, "--opset", str(args.opset)]
    tf2onnx_convert.main()

    print("3/3  Vérification : Keras vs ONNX ...")
    import onnxruntime as ort

    sess = ort.InferenceSession(args.dst, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.randint(0, 255, (4, 224, 224, 3)).astype("float32")  # échelle 0-255 comme à l'entraînement
    diff = float(np.abs(np.asarray(model(x, training=False)) - sess.run(None, {name: x})[0]).max())
    print(f"Écart maximal des probabilités Keras/ONNX : {diff:.2e}")
    assert diff < 1e-3, "Les sorties Keras et ONNX diffèrent trop : conversion à vérifier."

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"OK : {args.dst} ({os.path.getsize(args.dst) / 1024 / 1024:.0f} Mo)")


if __name__ == "__main__":
    main()

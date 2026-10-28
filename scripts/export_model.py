"""Allège le modèle entraîné (supprime l'état de l'optimiseur Adam).

Pourquoi : un .keras sauvegardé après model.fit() contient aussi l'état d'Adam (2 tenseurs par poids),
soit ~3x la taille des poids. ResNet50 + tête = ~23,8 M de paramètres :
    poids seuls ≈ 95 Mo   |   avec l'état d'Adam ≈ 273 Mo.
Pour l'inférence, seuls les poids sont nécessaires.

Usage :
    python scripts/export_model.py --src resnet50_finetune.keras --dst resnet50_finetune_light.keras
"""
import argparse
import os

import keras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="modèle .keras complet (273 Mo)")
    parser.add_argument("--dst", default="resnet50_finetune_light.keras", help="modèle allégé")
    args = parser.parse_args()

    model = keras.models.load_model(args.src, compile=False)  # compile=False : on ignore l'optimiseur
    model.save(args.dst)

    mb = lambda p: os.path.getsize(p) / 1024 / 1024
    print(f"{args.src} : {mb(args.src):.0f} Mo  ->  {args.dst} : {mb(args.dst):.0f} Mo")
    print(f"Paramètres : {model.count_params():,}")


if __name__ == "__main__":
    main()

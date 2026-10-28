"""Envoie le modèle sur le Hugging Face Hub (hébergement gratuit de fichiers lourds).

Prérequis : un compte Hugging Face et un token d'écriture (Settings > Access Tokens)
    export HF_TOKEN=hf_xxx            (Windows PowerShell : $env:HF_TOKEN="hf_xxx")

Usage :
    python scripts/upload_to_hf.py --file resnet50_finetune_light.keras --repo simo21Ms/ham500-resnet50-finetune
"""
import argparse
import os

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--repo", required=True, help="identifiant 'utilisateur/nom-du-depot'")
    parser.add_argument("--private", action="store_true", help="dépôt privé (il faudra alors un HF_TOKEN côté app)")
    args = parser.parse_args()

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=args.file,
        path_in_repo=os.path.basename(args.file),
        repo_id=args.repo,
        repo_type="model",
    )
    print(f"OK : https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()

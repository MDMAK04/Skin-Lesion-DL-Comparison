#!/usr/bin/env bash
# Déploie l'application sur un Hugging Face Space (SDK Docker).
# Usage : ./scripts/deploy_space.sh https://huggingface.co/spaces/UTILISATEUR/NOM_DU_SPACE
set -euo pipefail

SPACE_URL="${1:?Usage: $0 https://huggingface.co/spaces/UTILISATEUR/NOM_DU_SPACE}"
TMP="$(mktemp -d)"

git clone "$SPACE_URL" "$TMP"
cp -r app Dockerfile requirements.txt "$TMP"/
cp deploy/HF_SPACE_README.md "$TMP/README.md"   # le README d'un Space doit contenir un en-tête YAML

cd "$TMP"
git add .
git commit -m "Déploiement de l'application" || echo "Rien à commiter"
git push
echo "Space mis à jour : $SPACE_URL"

# 📄 Document Extraction and Classification

Une application Streamlit pour l'extraction et la classification de documents avec support d'upload de fichiers et de capture d'images via webcam.

## 🚀 Installation et Lancement

### Prérequis

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (gestionnaire de paquets Python ultra-rapide)

### Installation

1. **Cloner le projet**
   ```bash
   git clone <url-du-repo>
   cd extraction-classification
   ```

2. **Installer les dépendances avec uv**
   ```bash
   uv sync
   ```

### Lancement de l'application

Pour démarrer l'application Streamlit avec uv :

```bash
uv run streamlit run appStreamlit.py
```

L'application sera accessible sur : `http://localhost:8501`

## 📋 Fonctionnalités

- **📁 Upload de fichiers** : Support des formats PNG, JPG, JPEG
- **📷 Capture webcam** : Prise de photo directement depuis l'interface
- **🔄 Bouton de rechargement** : Réinitialisation rapide de l'application
- **🔍 Traitement** : Extraction et classification des documents (à implémenter)

## 🎯 Utilisation

1. **Uploader un fichier** : Utilisez le sélecteur de fichier pour charger une image
2. **Ou prendre une photo** : Cliquez sur "Take a Picture" pour utiliser votre webcam
3. **Traiter le document** : Cliquez sur "Extract and Classify" pour analyser l'image
4. **Recharger** : Utilisez le bouton "Reload" pour repartir de zéro

## 🛠️ Structure du projet

```
extraction-classification/
├── appStreamlit.py      # Application Streamlit principale
├── main.py              # Script principal (optionnel)
├── pyproject.toml       # Configuration du projet et dépendances
├── README.md            # Ce fichier
└── ressources/          # Dossier des ressources
```

## 📦 Dépendances

- `streamlit >= 1.52.0` : Framework web pour applications de données
- `opencv-python >= 4.12.0.88` : Traitement d'images

## 🔧 Développement

### Commandes utiles avec uv

```bash
# Installer une nouvelle dépendance
uv add nom-du-paquet

# Mettre à jour les dépendances
uv sync

# Exécuter un script Python
uv run python script.py

# Activer l'environnement virtuel
uv shell
```

### Lancer en mode développement

```bash
# Avec rechargement automatique
uv run streamlit run appStreamlit.py --server.runOnSave true
```

## 📝 Notes

- L'application utilise le port 8501 par défaut
- Assurez-vous que votre webcam est accessible pour la fonction de capture d'images
- Les sessions sont gérées automatiquement par Streamlit

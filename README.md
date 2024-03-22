# Fashion Diffusion

Un projet d'apprentissage automatique pour prédire la popularité des articles de mode sur la base de leurs images et de leurs métadonnées.

## Introduction

Fashion Diffusion est un projet d'apprentissage automatique qui utilise des techniques de traitement d'image et de modélisation prédictive pour prédire la popularité des articles de mode sur la base de leurs images et de leurs métadonnées. Le projet utilise des données provenant de plusieurs sources, notamment des images de produits, des descriptions de produits et des évaluations de clients.

## Modèles

Le projet utilise plusieurs modèles pour prédire la popularité des articles de mode, notamment :

- **Modèle de régression linéaire (Linear Regression Model)** : utilise les métadonnées des produits pour prédire leur popularité.
- **Modèle de réseau neuronal convolutif (Convolutional Neural Network Model, CNN)** : utilise les images des produits pour prédire leur popularité. Le modèle CNN utilisé dans ce projet est le `clip-vit-large-patch14`.
- **Modèle d'apprentissage en profondeur (Deep Learning Model)** : combine les métadonnées et les images des produits pour prédire leur popularité. Le modèle d'apprentissage en profondeur utilise une architecture de réseau neuronal à plusieurs couches pour extraire les caractéristiques des images et des métadonnées des produits. Le modèle utilise également `ViTImageProcessor` et `AutoTokenizer` pour prétraiter les images et les métadonnées en entrées pour le modèle.

En plus des modèles ci-dessus, le projet utilise également les outils suivants pour le traitement d'image et la génération de descriptions de produits :

- `humanparsing` : un modèle de segmentation sémantique pour identifier les différentes parties du corps humain dans les images de produits.
- `ootd` : un modèle de détection d'objets pour identifier les différents articles de mode dans les images de produits.
- `openpose` : un modèle de pose estimation pour identifier la pose du mannequin dans les images de produits.
- `VisionEncoderDecoderModel` : un modèle de génération de texte à partir d'images pour générer des descriptions de produits à partir des images de produits.

# Utilisation
Pour utiliser le projet, vous devez d'abord cloner le dépôt GitHub et installer les dépendances nécessaires en utilisant pip :

```
git clone https://github.com/nat997/fashion_diffusion.git
```

```sh
conda create -n ootd python==3.10
conda activate ootd
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
```

Ensuite, vous pouvez utiliser les scripts fournis dans le répertoire "scripts" pour télécharger les données, entraîner les modèles et évaluer leurs performances. Par exemple, pour entraîner le modèle d'apprentissage en profondeur, vous pouvez utiliser le script "train_deep_model.py" :

```
python scripts/train_deep_model.py --data_dir data/ --model_dir models/
```
Le script entraînera le modèle sur les données spécifiées et enregistrera les paramètres du modèle dans le répertoire spécifié. Vous pouvez ensuite utiliser le script "evaluate_deep_model.py" pour évaluer les performances du modèle sur un ensemble de données de test :

```
python scripts/evaluate_deep_model.py --data_dir data/ --model_dir models/
```
Le script calculera les métriques d'évaluation appropriées et les affichera à l'écran.

Télécharger [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) into ***checkpoints*** folder

Pour lancer le projet :
```
cd run
python gradio_ootd.py
```
## Demonstration

![image](https://github.com/nat997/fashion_diffusion/assets/67456959/84c6250f-493e-4f5b-a3df-de54aed12732)

## GIF processing 

![Screen Recording 2024-03-22 at 12 14 54](https://github.com/nat997/fashion_diffusion/assets/67456959/923f38b7-440c-471f-9993-58292d515db8)


## Diagram function of the code

![Untitled Diagram drawio](https://github.com/nat997/fashion_diffusion/assets/67456959/39fb47b2-877e-459b-8c86-e89c83872fc7)

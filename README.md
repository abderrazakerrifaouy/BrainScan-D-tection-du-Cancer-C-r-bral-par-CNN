# BrainScan – Détection du Cancer Cérébral par CNN

Modèle de Deep Learning capable de classer des images IRM cérébrales afin d'aider au diagnostic précoce des tumeurs. Le projet combine **CNN**, **TensorFlow/Keras** et **OpenCV**, avec une interface **Streamlit** pour tester les prédictions.

>  Projet pédagogique : cet outil est une aide au diagnostic et ne remplace pas l'avis d'un médecin.

---

##  Contexte

**BrainScan AI** est une startup marocaine spécialisée dans l'imagerie médicale assistée par l'IA. L'objectif est de développer une application intelligente qui :

- analyse et classe les images IRM du cerveau pour détecter la présence de cancer ;
- assiste les médecins dans l'interprétation rapide et fiable des résultats ;
- réduit le temps de diagnostic et les erreurs humaines.

---

##  Technologies utilisées

| Domaine | Outils |
|---|---|
| Langage | Python 3.10+ |
| Deep Learning | TensorFlow / Keras |
| Traitement d'images | OpenCV |
| Calcul & données | NumPy, Pandas |
| Prétraitement ML | scikit-learn (`LabelEncoder`, `train_test_split`, `class_weight`) |
| Visualisation | Matplotlib, Seaborn |
| Interface | Streamlit |
| Versioning | Git / GitHub |
| Environnement | Jupyter Notebook / VS Code |

---

##  Architecture du projet

```
brainscan/
├── data/
│   └── dataset/                # Images IRM (un dossier par classe)
├── notebooks/
│   └── brainscan.ipynb         # Exploration, entraînement, évaluation
├── src/
│   ├── preprocessing.py        # Chargement, nettoyage, resize, normalisation
│   ├── model.py                # Définition et compilation du CNN
│   ├── train.py                # Entraînement + ModelCheckpoint
│   ├── evaluate.py             # Métriques, matrice de confusion
│   └── predict.py              # predict_image(path)
├── models/
│   └── best_model.h5           # Meilleur modèle sauvegardé
├── app/
│   └── app.py                  # Interface Streamlit
├── requirements.txt
└── README.md
```

---

##  Pipeline

```
Dataset IRM
   │
   ▼
Chargement + vérification des extensions (jpeg, jpg, bmp, png) + try/except
   │
   ▼
Redimensionnement 224×224 (OpenCV) → listes images / labels → tableaux NumPy
   │
   ▼
Analyse des classes (distribution, échantillons) + rééquilibrage si besoin
   │
   ▼
Encodage des labels → Split train/test → Normalisation [0, 1]
   │
   ▼
CNN (Conv2D + MaxPooling + Dropout + Dense)
   │
   ▼
Entraînement (Adam, categorical_crossentropy, ModelCheckpoint)
   │
   ▼
Évaluation (accuracy, loss, matrice de confusion, classification report)
   │
   ▼
Sauvegarde du modèle (.h5) → predict_image(path) → Interface Streamlit
```

### 1. Chargement et prétraitement
- Vérification des extensions et suppression des fichiers invalides.
- Gestion des erreurs de chargement avec `try-except`.
- Redimensionnement à **224×224** avec OpenCV.
- Conversion en tableaux NumPy, encodage des labels (`LabelEncoder` / `to_categorical`).
- Vérification de l'équilibre des classes et rééquilibrage si nécessaire.
- Division train/test et normalisation des pixels dans **[0, 1]**.

### 2. Modèle CNN
- Blocs `Conv2D` + `MaxPooling2D`, `Dropout` pour limiter le surapprentissage, couches `Dense` finales.
- Activation `ReLU` pour les couches cachées, `Softmax` pour la sortie.
- Compilation : optimiseur **Adam**, perte **categorical_crossentropy**.
- Vérification avec `model.summary()` et `plot_model()`.
- Durée d'entraînement mesurée avec le module `time`.

### 3. Entraînement et évaluation
- `model.fit()` avec le callback **ModelCheckpoint** (sauvegarde du meilleur modèle).
- Courbes d'apprentissage (accuracy / loss).
- Matrice de confusion et rapport de classification.
- Exemples de prédictions correctes et incorrectes.

### 4. Déploiement
- Fonction `predict_image(path)` pour tester une image.
- Modèle sauvegardé au format `.h5`.
- Interface **Streamlit** pour charger une IRM et obtenir la prédiction.

---

##  Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/<votre-pseudo>/brainscan.git
cd brainscan

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

**requirements.txt (exemple)**

```
tensorflow
opencv-python
numpy
pandas
scikit-learn
matplotlib
seaborn
streamlit
pydot
```

---

##  Utilisation

**Entraîner le modèle**

```bash
python src/train.py
```

**Prédire sur une image**

```python
from src.predict import predict_image

label, confidence = predict_image("chemin/vers/image.jpg")
print(label, confidence)
```

**Lancer l'interface Streamlit**

```bash
streamlit run app/app.py
```

---

## 📊 Résultats

| Métrique | Valeur |
|---|---|
| Accuracy (test) | _à compléter_ |
| Precision | _à compléter_ |
| Recall | _à compléter_ |
| F1-score | _à compléter_ |
| Temps d'entraînement | _à compléter_ |

_Ajouter ici les courbes d'apprentissage et la matrice de confusion._

---

##  Critères de qualité visés

- **Qualité du modèle** : métriques élevées, hyperparamètres optimisés, bonne généralisation.
- **Documentation** : étapes claires et reproductibles.
- **Code** : lisible, structuré, robuste face aux erreurs et aux entrées incorrectes.

---

##  Planification (5 jours)

| Jour | Tâches |
|---|---|
| J1 | Chargement, nettoyage et exploration du dataset |
| J2 | Prétraitement, rééquilibrage, split, normalisation |
| J3 | Conception et entraînement du CNN |
| J4 | Évaluation, optimisation des hyperparamètres, `predict_image` |
| J5 | Interface Streamlit, documentation, préparation de la démo |

---

## 👤 Auteur

**Abderrazak Errifaouy** – Développement Web Full Stack, YouCode (UM6P)
GitHub : [abderrazakerrifaouy](https://github.com/abderrazakerrifaouy)

Projet encadré par **Hamid Oufakir**.

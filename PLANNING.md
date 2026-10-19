# 🗓️ Planification de travail – BrainScan (CNN)

**Période :** du lundi 12/10/2026 au vendredi 16/10/2026 (avant minuit)
**Mode :** individuel – **Durée :** 5 jours
**Livrables :** code source GitHub + planification de travail
**Évaluation :** 30 min (démo 10 min · code review / culture du projet 10 min · mise en situation 10 min)

---

## 📊 Vue d'ensemble

| Jour | Date | Objectif principal | Livrable de fin de journée |
|---|---|---|---|
| J1 | Lun 12/10 | Setup + chargement + exploration du dataset | Repo GitHub, dataset chargé, graphiques d'exploration |
| J2 | Mar 13/10 | Prétraitement complet + rééquilibrage + split | Données prêtes (X_train, X_test, y_train, y_test) |
| J3 | Mer 14/10 | Conception et entraînement du CNN | Modèle entraîné + `best_model.h5` |
| J4 | Jeu 15/10 | Évaluation + optimisation + `predict_image` | Métriques finales, matrice de confusion, fonction de prédiction |
| J5 | Ven 16/10 | Streamlit + documentation + préparation de la démo | Application, README, push final avant minuit |

> 💡 Règle : un **commit Git à la fin de chaque bloc** de travail (au moins 3 commits/jour).

---

## 📅 Jour 1 – Lundi 12/10 : Setup & Exploration

| Horaire | Tâche | Détail |
|---|---|---|
| 09h00 – 10h00 | Lecture du brief | Lister les exigences, critères de performance, livrables |
| 10h00 – 11h00 | Setup du projet | Créer le repo GitHub, structure des dossiers, `.gitignore`, `requirements.txt`, environnement virtuel |
| 11h00 – 12h30 | Téléchargement du dataset | Télécharger, décompresser, vérifier l'arborescence (1 dossier = 1 classe) |
| 14h00 – 15h30 | Chargement + vérification | Importer les bibliothèques, parcourir les images, vérifier les extensions (jpeg, jpg, bmp, png), supprimer les fichiers invalides |
| 15h30 – 16h30 | Gestion des erreurs | Bloc `try-except` pour le chargement, journaliser les images corrompues |
| 16h30 – 17h30 | Exploration | Lister les classes, graphique du nombre d'images par classe, échantillon d'images par classe |
| 17h30 – 18h00 | Bilan + commit | Notes sur le dataset (taille, classes, déséquilibre), push GitHub |

**✅ Checklist J1**
- [x] Repo GitHub créé et structuré
- [x] Dataset téléchargé et exploré
- [ ] Extensions vérifiées, fichiers invalides supprimés
- [ ] `try-except` en place
- [ ] Graphique de distribution des classes
- [ ] Échantillon d'images affiché
- [ ] Premier push

---

## 📅 Jour 2 – Mardi 13/10 : Prétraitement

| Horaire | Tâche | Détail |
|---|---|---|
| 09h00 – 10h30 | Construction des listes | Mélanger les images de toutes les classes dans 2 listes (`images`, `labels`) avec le même indice |
| 10h30 – 12h00 | Redimensionnement OpenCV | `cv2.resize` à 224×224, gérer les images en niveaux de gris / couleur (cohérence des canaux) |
| 14h00 – 14h45 | Conversion NumPy | Listes → `np.array`, vérifier les `shape` et `dtype` |
| 14h45 – 15h45 | Équilibre des classes | Analyser la distribution ; si déséquilibre : `class_weight`, oversampling ou data augmentation |
| 15h45 – 16h30 | Encodage des labels | `LabelEncoder` puis `to_categorical` |
| 16h30 – 17h15 | Split train/test | `train_test_split` avec `stratify` et `random_state` fixé |
| 17h15 – 17h45 | Normalisation | Pixels ÷ 255 → plage [0, 1] |
| 17h45 – 18h15 | Bilan + commit | Vérifier les shapes finales, déplacer le code dans `src/preprocessing.py` |

**✅ Checklist J2**
- [ ] Listes `images` / `labels` alignées
- [ ] Images redimensionnées en 224×224
- [ ] Tableaux NumPy créés
- [ ] Équilibre vérifié (et corrigé si besoin)
- [ ] Labels encodés
- [ ] Split train/test stratifié
- [ ] Pixels normalisés [0, 1]
- [ ] `preprocessing.py` commité

---

## 📅 Jour 3 – Mercredi 14/10 : Modèle CNN & Entraînement

| Horaire | Tâche | Détail |
|---|---|---|
| 09h00 – 10h30 | Architecture du CNN | `Conv2D` + `MaxPooling2D` (3 blocs : 32 → 64 → 128 filtres), `Flatten`, `Dense`, `Dropout` |
| 10h30 – 11h00 | Activations | `ReLU` (couches cachées), `Softmax` (sortie multi-classes) ou `Sigmoid` (binaire) |
| 11h00 – 11h45 | Compilation | Optimiseur `Adam`, perte `categorical_crossentropy`, métrique `accuracy` |
| 11h45 – 12h30 | Vérification | `model.summary()`, `plot_model()` |
| 14h00 – 14h30 | Hyperparamètres | Learning rate (1e-3 au départ), epochs (20–30), batch size (16 ou 32) |
| 14h30 – 15h00 | Callbacks | `ModelCheckpoint` (`save_best_only=True`, monitor `val_accuracy`), `EarlyStopping` |
| 15h00 – 17h30 | Premier entraînement | `model.fit()` avec mesure du temps (`time.time()`), surveiller overfitting / underfitting |
| 17h30 – 18h15 | Bilan + commit | Noter accuracy/loss, sauvegarder `best_model.h5`, `model.py` + `train.py` |

**✅ Checklist J3**
- [ ] Architecture CNN définie
- [ ] Modèle compilé
- [ ] `summary()` et `plot_model()` affichés
- [ ] Hyperparamètres choisis
- [ ] `ModelCheckpoint` configuré
- [ ] Durée d'entraînement mesurée
- [ ] Premier modèle sauvegardé

> ⚠️ Si GPU indisponible : utiliser Google Colab / Kaggle Notebooks pour l'entraînement.

---

## 📅 Jour 4 – Jeudi 15/10 : Évaluation, Optimisation, Prédiction

| Horaire | Tâche | Détail |
|---|---|---|
| 09h00 – 10h00 | Évaluation sur le test | `model.evaluate()`, accuracy et loss |
| 10h00 – 11h00 | Courbes d'apprentissage | Graphiques accuracy / loss (train vs validation) |
| 11h00 – 12h00 | Matrice de confusion | `confusion_matrix` + heatmap Seaborn |
| 12h00 – 12h30 | Rapport de classification | `classification_report` (precision, recall, F1) |
| 14h00 – 15h30 | Optimisation | Tester 2–3 variantes : data augmentation, dropout, learning rate, nombre de filtres, batch size (tableau comparatif) |
| 15h30 – 16h15 | Analyse des erreurs | Afficher des prédictions correctes et incorrectes |
| 16h15 – 17h15 | `predict_image(path)` | Charger l'image, resize 224×224, normaliser, prédire, retourner classe + confiance, gérer les erreurs (fichier absent, format invalide) |
| 17h15 – 17h45 | Sauvegarde finale | Meilleur modèle en `.h5` |
| 17h45 – 18h15 | Bilan + commit | `evaluate.py`, `predict.py` |

**✅ Checklist J4**
- [ ] Accuracy / loss sur le test
- [ ] Courbes d'apprentissage
- [ ] Matrice de confusion
- [ ] Classification report
- [ ] Au moins 2 expériences d'optimisation comparées
- [ ] Exemples de prédictions correctes / incorrectes
- [ ] `predict_image(path)` fonctionnelle et robuste

---

## 📅 Jour 5 – Vendredi 16/10 : Streamlit, Documentation, Soutenance

| Horaire | Tâche | Détail |
|---|---|---|
| 09h00 – 11h00 | Interface Streamlit | Upload d'image, aperçu, bouton « Prédire », affichage classe + score de confiance, gestion des erreurs |
| 11h00 – 12h00 | Tests de l'application | Tester avec images de chaque classe, mauvais formats, image vide |
| 14h00 – 15h30 | Documentation | Finaliser le README (résultats, captures, installation), nettoyer les notebooks |
| 15h30 – 16h30 | Nettoyage du code | Noms clairs, docstrings, suppression du code mort, vérifier `requirements.txt` |
| 16h30 – 17h30 | Préparation de la démo | Répéter la démo de 10 minutes (voir plan ci-dessous) |
| 17h30 – 18h30 | Préparation code review + mise en situation | Revoir les questions probables (voir ci-dessous) |
| 18h30 – 19h00 | Push final | Vérifier que le repo se clone et s'exécute depuis zéro |
| **Avant 23h59** | **Deadline** | Rien de nouveau après le push final |

**✅ Checklist J5**
- [ ] Streamlit fonctionnel
- [ ] README complet avec résultats
- [ ] Code propre et commenté
- [ ] Planification ajoutée au repo
- [ ] Push final vérifié

---

## 🎤 Préparation de l'évaluation (30 min)

### Démo – 10 min
1. (1 min) Présenter le problème et le dataset
2. (2 min) Montrer le prétraitement (resize, normalisation, équilibrage)
3. (2 min) Présenter l'architecture CNN et les hyperparamètres
4. (2 min) Résultats : courbes, matrice de confusion, métriques
5. (3 min) Démo live Streamlit avec 2–3 images

### Code review / culture du projet – 10 min : questions probables
- Pourquoi 224×224 ? Pourquoi normaliser en [0, 1] ?
- Comment fonctionne `Conv2D` ? Rôle de `MaxPooling` et de `Dropout` ?
- Pourquoi `ReLU` et `Softmax` ? Pourquoi `categorical_crossentropy` ?
- Qu'est-ce que l'overfitting ? Comment l'avez-vous détecté et réduit ?
- Pourquoi la matrice de confusion et le recall sont-ils importants en médical (faux négatifs) ?
- Comment avez-vous géré le déséquilibre des classes ?
- Que fait `ModelCheckpoint` ?

### Mise en situation – 10 min : scénarios possibles
- Ajouter une nouvelle classe au modèle
- Améliorer les performances (transfer learning : VGG16, ResNet, MobileNet)
- Déployer l'application (Docker, API FastAPI)
- Expliquer les limites éthiques et médicales du modèle

---

## ⚠️ Risques et plans B

| Risque | Plan B |
|---|---|
| Entraînement trop lent (CPU) | Google Colab / Kaggle GPU, réduire la taille des images ou le batch size |
| Overfitting | Data augmentation, plus de Dropout, EarlyStopping |
| Dataset déséquilibré | `class_weight`, oversampling, métriques par classe |
| Accuracy insuffisante | Transfer learning (MobileNetV2 / VGG16) |
| Retard sur le planning | Prioriser : modèle + évaluation + `predict_image` avant Streamlit |
| Problème Git de dernière minute | Push intermédiaire chaque soir |

---

## ✅ Critères de performance – auto-contrôle

**Qualité des modèles**
- [ ] Métriques élevées sur le test
- [ ] Hyperparamètres optimisés et comparés
- [ ] Bonne généralisation (écart train/val réduit)

**Documentation et reproductibilité**
- [ ] README clair, étapes décrites
- [ ] `requirements.txt` à jour
- [ ] Seed fixé (`random_state`, `tf.random.set_seed`)

**Efficacité du code**
- [ ] Code lisible et organisé en modules
- [ ] Temps d'exécution optimisé
- [ ] Gestion des erreurs et des entrées incorrectes

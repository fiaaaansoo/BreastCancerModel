# Breast Cancer Classification and Clustering

Ce projet repose exclusivement sur `numpy` pour l’implémentation d’algorithmes de classification et de clustering.

## Objectifs

Le projet vise à :
- Implémenter un algorithme de **clustering K-Means** avec évaluation via le **Silhouette Score**.
- Développer un modèle de **classification supervisée** basé sur la **régression logistique multiclasse (softmax)**.
- Évaluer les performances de la classification à partir des données originales et d’une **représentation basée sur les distances aux centroïdes**.

## Dataset

Le jeu de données utilisé est le **Breast Cancer Wisconsin Diagnostic Dataset**, disponible publiquement :

- 30 variables numériques extraites d’images de tumeurs
- Une étiquette binaire : `M` (malin) ou `B` (bénin)

Lien officiel :  
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Le fichier `wdbc.data` doit être placé dans le dossier `./data/`.

## Structure du projet

```
BreastCancerModel/
├── data/                  # Fichier wdbc.data
├── models/                # Fichiers modèles (.pkl)
├── models.py              # ClusteringModel & ClassificationModel
├── wdbc_train.py          # Script d'entraînement pour la moulinette
├── README.md              # Présentation du projet
```

## Description des modèles

### ClusteringModel
- Algorithme K-Means implémenté manuellement
- Paramétrable par le nombre de clusters
- Évaluation par le **Silhouette Score**
- Génère une représentation des données basée sur la distance aux centroïdes

### ClassificationModel
- Régression logistique softmax (multiclasse)
- Apprentissage par descente de gradient
- Sortie probabiliste
- Évaluation par **précision**, **rappel** et **F1-score**

## Entraînement et génération des modèles

Le script `wdbc_train.py` entraîne et sauvegarde les trois modèles attendus :
1. Modèle de clustering (clustering.pkl)
2. Modèle de classification sur données brutes (classif.pkl)
3. Modèle de classification sur représentation centroïdes (classif-centroid.pkl)

Lancer l'entraînement :

```bash
python wdbc_train.py
```

Les fichiers `.pkl` sont enregistrés dans le dossier `./models/`.


## Contraintes respectées

- Utilisation exclusive de `numpy` pour le clustering et la classification
- Aucun appel à `scikit-learn` pour les modèles
- Fichier `models.py` conforme à la signature exigée
- Testabilité complète du projet via `wdbc_train.py` et fichiers `.pkl`

## Auteur

Projet développé par l’utilisateur `fiaaaansoo` (GitHub)

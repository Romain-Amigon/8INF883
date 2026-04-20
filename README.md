# 8INF883

## Utilisation de l'application 

Pour utiliser l'application d'inférence (à la racine du projet) :
```bash
streamlit run app/app.py
```

## Etude GAN

Il est possible de visualiser les étapes de créations du GAN et l'étude menée sur ce modèle via le notebook python dans ``GAN/GAN.ipynb``.

## Rapport

Le rapport du projet est disponible en format pdf au chemin suivant : ``rapport/rapport_projet_8INF883``.

---

## Manipulation d'Attributs Faciaux par Modèles Génératifs (CVAE vs cGAN)

1. Objectif du Projet
Ce projet implémente un système de modification d'attributs faciaux (ajout de lunettes, changement de couleur de cheveux, etc.) basé sur la manipulation de l'espace latent. Le projet vise à comparer des architectures génératives fondamentales pour évaluer leurs capacités respectives en matière de traduction d'images conditionnelle.

2. Jeu de Données
Le modèle est entraîné sur le dataset public CelebA (CelebFaces Attributes Dataset). Ce jeu de données comprend plus de 200 000 images de visages, chacune annotée rigoureusement avec 40 attributs binaires, ce qui en fait le standard optimal pour la génération conditionnelle de visages.

https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

équivalent Kaggle : https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

4. Méthodologie et Architectures
Le projet adopte une démarche comparative en deux étapes pour isoler les forces et faiblesses des différentes approches de modélisation générative :

Modèle de Référence : Auto-encodeur Variationnel Conditionnel (CVAE)
Le CVAE sert de base de référence (baseline). Il permet d'apprendre un espace latent continu où l'arithmétique vectorielle est utilisée pour appliquer les filtres d'attributs.

Modèle Avancé : Réseau Antagoniste Génératif Conditionnel (cGAN)
Une architecture antagoniste (inspirée de cGAN/StarGAN) est implémentée pour résoudre la perte de netteté inhérente aux fonctions de perte du VAE, permettant la génération de modifications d'attributs avec un haut niveau de photoréalisme.

et Transformer si il y a le temps

4. Évaluation des Performances
La qualité des images générées et la robustesse des modèles sont évaluées à l'aide de métriques perceptives et structurelles :

SSIM (Structural Similarity Index Measure) : Mesure la préservation de la structure globale du visage original après l'application du filtre.

LPIPS (Learned Perceptual Image Patch Similarity) : Évalue la qualité visuelle et le réalisme des résultats en se basant sur la perception humaine.

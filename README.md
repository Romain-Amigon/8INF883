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

## Structure du projet

```plaintext
C:.
│   .gitignore
│   Rapport_de_Projet_8INF883.pdf
│   README.md
│   requirements.txt
│
├───.ipynb_checkpoints
│       
│
├───app
│       app.py
│
├───CVAE
│   │   CVAE_comp_512.ipynb ==> Pour entrainer le CVAE
│   │   femme.png
│   │   moi.jpg
│   │   test.ipynb          ==> Pour tester une fois entrainé
│   │
│   ├───models
│   │    
│   │
│   └───old_train
│           CVAE_comp.ipynb
│           VAE_simple.ipynb
│
├───GAN
│   │   CGAN_diagram.png
│   │   GAN.ipynb
│   │   requirements.txt
│   │
│   └───saved_models
│      ...
│
├───rapport
|   ...
│
└───transformer
    │ 
    │   femme.png
    │   test.ipynb    ==> Pour tester une fois entrainé
    │   transf.ipynb  ==> Pour entrainer le ViTCVAE
    │
    └───.ipynb_checkpoints
            
``

## Auto-encodeur Variationnel Conditionnel (Deep CVAE)
Notre modèle de base est un CVAE profond basé sur des réseaux de neurones convolutifs (CNN), conçu pour encoder les images dans une distribution statistique tout en forçant la séparation des attributs.

* **Architecture :** Le réseau est composé de 5 couches de convolution pour l'encodeur et 5 couches de convolution transposée pour le décodeur, totalisant environ 48 millions de paramètres. L'espace latent a été fixé à une dimension de 512 pour maximiser la rétention des détails faciaux en résolution 128x128.
* **Conditionnement :** Les 40 labels binaires de CelebA sont concaténés à l'image en entrée de l'encodeur, ainsi qu'au vecteur latent échantillonné en entrée du décodeur.
* **Observations et Limites :** Le modèle démontre une excellente capacité à reconstruire la structure globale du visage et permet l'édition d'attributs spécifiques (ex: ajout de lunettes). Cependant, il illustre parfaitement les limites théoriques des VAE : une tendance au lissage (flou) due à la fonction de perte basée sur l'erreur de reconstruction, et une difficulté à isoler parfaitement certains attributs fortement corrélés (enchevêtrement latent).

##  Exploration : Vision Transformer CVAE (ViT-CVAE)
Dans une démarche d'exploration de l'état de l'art, nous avons implémenté et testé une variante remplaçant les convolutions par un mécanisme d'Attention.

* **Approche :** L'image est découpée en séquences de patchs de 8x8 pixels (Patchification). Un Transformer Encoder génère l'espace latent, et un Transformer Decoder reconstruit la séquence avant un réassemblage spatial.
* **Résultats de l'étude :** Cette expérimentation a mis en évidence la difficulté d'entraîner des architectures basées sur l'attention depuis zéro (sans pré-entraînement massif). Le modèle a souffert d'instabilités d'entraînement (explosion du gradient) et les images générées présentaient de forts artefacts en grille (effet mosaïque).
* **Conclusion scientifique :** Ce test confirme que l'absence de biais inductif local dans les Transformers nécessite des volumes de données et des temps de calcul largement supérieurs à ceux disponibles pour ce projet, justifiant notre choix de conserver les réseaux convolutifs et de passer au cGAN pour améliorer la netteté.
# Lab 2 : Deep Learning avec PyTorch

Dans ce lab, on a utilisé PyTorch pour construire et comparer plusieurs architectures de vision par ordinateur sur le dataset **MNIST** (chiffres manuscrits, 60 000 images d'entraînement, 10 000 de test, 28×28 pixels en gris). Le travail est divisé en deux parties : les CNN (avec transfer learning) et le Vision Transformer (from scratch).

## Contenu du dépôt

| Fichier | Contenu |
|---|---|
| `lab-2-part-1-cnn-classifier.ipynb` | Partie 1 : CNN, Faster R-CNN, VGG16, AlexNet |
| `lab-2-part-2-vit.ipynb` | Partie 2 : Vision Transformer (from scratch) |
| `README.md` | Ce fichier |

## Partie 1 : CNN Classifier et modèles pré-entraînés

**Objectif :** classifier des chiffres manuscrits (0 à 9) avec plusieurs architectures et comparer leurs performances.

**Ce qu'on a fait :**

- **Préparation :** normalisation avec la moyenne et l'écart-type calculés sur le train uniquement (0.1307 / 0.3081), ajout d'une dimension de canal, `DataLoader` avec batchs de 64 et mélange.
- **CNN from scratch :** 2 blocs Conv → ReLU → MaxPool, puis 3 couches FC (avec Dropout 0.25). Adam, lr=0.001, 5 epochs.
- **Faster R-CNN :** backbone ResNet50 + FPN pré-entraîné. Boîtes englobantes générées à partir des pixels blancs. SGD, lr=0.005, momentum=0.9, 3 epochs.
- **Fine-tuning VGG16 et AlexNet :** images redimensionnées en 224×224, passées en 3 canaux, normalisées avec les stats ImageNet. Dernière couche FC remplacée par `Linear(4096, 10)`. Adam, lr=0.0001, 3 epochs, sur 20 000 images.
- **Évaluation :** accuracy, F1-score macro, matrice de confusion, courbes d'apprentissage, temps d'entraînement et d'inférence.

| Modèle | Accuracy test | F1 macro | Entraînement | Inférence | Paramètres |
|---|---|---|---|---|---|
| **CNN** | 99.03 % | 99.02 % | **14.7 s** | **0.19 s** | **0.1 M** |
| **AlexNet** | **99.09 %** | **99.08 %** | 3.3 min | 16.9 s | 57 M |
| **VGG16** | 98.97 % | 98.97 % | 19.4 min | 60.0 s | 134 M |
| **Faster R-CNN** | 98.77 % | 98.78 % | 94.7 min | 119.2 s | 41 M |

**Conclusion :** un petit CNN sur mesure rivalise avec des modèles pré-entraînés 500 fois plus gros. AlexNet offre le meilleur compromis performance/temps. Faster R-CNN est surdimensionné pour de la classification pure (il est conçu pour la détection d'objets).

## Partie 2 : Vision Transformer (ViT)

**Objectif :** implémenter un ViT **from scratch** en suivant le tutoriel de Brian Pulfer, puis le comparer aux modèles de la Partie 1.

**Ce qu'on a fait :**

- **Architecture :** patch embedding (49 patchs de 4×4), class token `[CLS]`, positional encoding sinusoïdal, 2 blocs Transformer (Multi-Head Self-Attention + MLP), tête de classification.
- **Hyperparamètres :** `n_patches=7`, `n_blocks=2`, `hidden_d=8`, `n_heads=2`, `out_d=10`.
- **Entraînement :** Adam, lr=0.005, 5 epochs, batchs de 128, GPU T4.

| Modèle | Accuracy test | Paramètres | Temps entraînement |
|---|---|---|---|
| **ViT (tuto)** | **74.98 %** | **0.01 M** | 29.5 min |

**Conclusion :** le ViT est **beaucoup moins performant** que les CNN sur MNIST. Deux raisons : il a 10 fois moins de paramètres que le CNN, et le ViT a besoin de millions d'images pour rivaliser avec un CNN. Cette version est **pédagogique** : elle permet de comprendre l'architecture brique par brique.

## Synthèse : ce qu'on a appris

- **Un modèle n'est pas meilleur par sa taille.** Un petit CNN (0.1 M paramètres) atteint 99 % et bat VGG16 (134 M paramètres). Sur une tâche simple, la complexité ne paie pas.
- **Le transfer learning fait gagner un temps énorme.** VGG16 et AlexNet atteignent 99 % en 3 epochs seulement, contre 5 pour un CNN from scratch.
- **Le choix du modèle dépend de la tâche.** Pour de la classification simple : CNN ou AlexNet. Pour de la localisation : Faster R-CNN. Pour des tâches complexes avec beaucoup de données : ViT.
- **Le ViT n'est pas magique.** Il a besoin de beaucoup de données et d'une architecture adaptée. Sur MNIST, un petit ViT est loin d'être compétitif.
- **Toujours comparer à une référence simple.** Ici, un CNN de base suffit à battre des géants pré-entraînés.
